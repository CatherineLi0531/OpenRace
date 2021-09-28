/* Copyright 2021 Coderrect Inc. All Rights Reserved.
Licensed under the GNU Affero General Public License, version 3 or later (“AGPL”), as published by the Free Software
Foundation. You may not use this file except in compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/agpl-3.0.en.html
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <llvm/AsmParser/Parser.h>

#include <catch2/catch.hpp>

#include "Analysis/HappensBeforeGraph.h"
#include "Trace/Event.h"

TEST_CASE("No crash on empty thread", "[regression][happensbefore]") {
  // Previously a bug caused HappensBefore to crash if a spawned thread contained no events

  const char *ModuleString = R"(
%union.pthread_attr_t = type { i64, [48 x i8] }

define i8* @entry(i8*) {
    ret i8* null
}

define void @main() {
  %p_thread = alloca i64
  %1 = call i32 @pthread_create(i64* %p_thread, %union.pthread_attr_t* null, i8* (i8*)* @entry, i8* null)
  %thread = load i64, i64* %p_thread
  %2 = call i32 @pthread_join(i64 %thread, i8** null)
  ret void
}

declare i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)
declare i32 @pthread_join(i64, i8**)
)";

  llvm::LLVMContext Ctx;
  llvm::SMDiagnostic Err;
  auto module = llvm::parseAssemblyString(ModuleString, Err, Ctx);
  if (!module) {
    Err.print("error", llvm::errs());
  }

  race::ProgramTrace program(module.get());

  race::HappensBeforeGraph happensbefore(program);
}

TEST_CASE("Record volatile load/stores", "[regression][ir]") {
  // Volatile cannot be used to prevent races in C++
  // Volatile accesses should be considered for race detection
  // See https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rconc-volatile

  const char *ModuleString = R"(
%union.pthread_attr_t = type { i64, [48 x i8] }

@global = common global i32 0

define i8* @entry(i8* %_arg) {
entry:
  store volatile i32 1, i32* @global
  store volatile i32 2, i32* @global
  ret i8* null
}

define void @main() {
  %t = alloca i64
  %call = call i32 @pthread_create(i64* nonnull %t, %union.pthread_attr_t* null, i8* (i8*)* nonnull @entry, i8* null)
  %1 = load i64, i64* %t
  %call1 = call i32 @pthread_join(i64 %1, i8** null)
  ret void
}

declare i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)
declare i32 @pthread_join(i64, i8**)
)";

  llvm::LLVMContext Ctx;
  llvm::SMDiagnostic Err;
  auto module = llvm::parseAssemblyString(ModuleString, Err, Ctx);
  if (!module) {
    Err.print("error", llvm::errs());
  }

  race::ProgramTrace program(module.get());

  REQUIRE(program.getThreads().size() == 2);

  auto const &thread = program.getThreads().at(1);
  REQUIRE(thread->getEvents().size() == 2);
  CHECK(thread->getEvent(0)->type == race::Event::Type::Write);
}