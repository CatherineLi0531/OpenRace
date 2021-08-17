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
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>

#include <catch2/catch.hpp>

#include "Analysis/LockSet.h"

TEST_CASE("Simple optimization test (avoid duplicate calls)", "[unit]") {
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  llvm::StringRef file = "simple-duplicate-call.ll";
  llvm::StringRef llPath = "integration/optimization/";

  // Read the input file
  auto testfile = llPath.str() + file.str();
  auto module = llvm::parseIRFile(testfile, err, context);
  if (!module) {
    err.print(file.str().c_str(), llvm::errs());
  }
  REQUIRE(module.get() != nullptr);

  race::ProgramTrace program(module.get());

  auto const &threads = program.getThreads();
  REQUIRE(threads.size() == 3);

  auto const &thread1 = threads.at(1);
  auto const &events1 = thread1->getEvents();
  REQUIRE(events1.size() == 5);

  auto const &thread2 = threads.at(2);
  auto const &events2 = thread2->getEvents();
  REQUIRE(events2.size() == 5);
}

TEST_CASE("Optimization test with guards (avoid duplicate calls)", "[unit]") {
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  llvm::StringRef file = "omp-get-thread-num-duplicate-call.ll";
  llvm::StringRef llPath = "integration/optimization/";

  // Read the input file
  auto testfile = llPath.str() + file.str();
  auto module = llvm::parseIRFile(testfile, err, context);
  if (!module) {
    err.print(file.str().c_str(), llvm::errs());
  }
  REQUIRE(module.get() != nullptr);

  race::ProgramTrace program(module.get());

  auto const &threads = program.getThreads();
  REQUIRE(threads.size() == 3);

  auto const &thread1 = threads.at(1);
  auto const &events1 = thread1->getEvents();
  REQUIRE(events1.size() == 9);

  auto const &thread2 = threads.at(2);
  auto const &events2 = thread2->getEvents();
  REQUIRE(events2.size() == 9);
}