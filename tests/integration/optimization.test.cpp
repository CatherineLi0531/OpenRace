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

#include <catch2/catch.hpp>

#include "helpers/ReportChecking.h"

#define TEST_LL(name, file, ...) \
  TEST_CASE(name, "[integration][opt]") { checkTest(file, "integration/optimization/", {__VA_ARGS__}); }

#define EXPECTED(...) __VA_ARGS__
#define NORACE

// Avoid duplicate function traversal during building thread traces
TEST_LL("simple-duplicate-call", "simple-duplicate-call.ll",
        EXPECTED("simple-duplicate-call.c:4:44 simple-duplicate-call.c:4:44"))
TEST_LL("omp-get-thread-num-duplicate-call", "omp-get-thread-num-duplicate-call.ll",
        EXPECTED("omp-get-thread-num-duplicate-call.c:4:44 omp-get-thread-num-duplicate-call.c:4:44"))
