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
  TEST_CASE(name, "[integration][omp]") { checkTest(file, "integration/openmp/", {__VA_ARGS__}); }

#define EXPECTED(...) __VA_ARGS__
#define NORACE

TEST_CASE("OpenMP Integration Tests", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("reduction-no.ll", {}), Oracle("master-iteration-counter-no.ll", {}),
      // Oracle("reduction-yes.ll", {/*TODO*/}), // Need to handle openmp master first
      Oracle("reduction-nowait-yes.ll",
             {
                 "reduction-nowait-yes.c:11:27 reduction-nowait-yes.c:16:27",
                 "reduction-nowait-yes.c:11:27 reduction-nowait-yes.c:16:31",
                 "reduction-nowait-yes.c:16:27 reduction-nowait-yes.c:11:27",
                 "reduction-nowait-yes.c:16:27 reduction-nowait-yes.c:11:31",
             }),
      Oracle("master-used-after-yes.ll", {"master-used-after-yes.c:11:9 master-used-after-yes.c:14:22"}),
      Oracle("single-message-printer.ll",
             {
                 "single-message-printer.c:11:14 single-message-printer.c:11:14",
                 "single-message-printer.c:11:14 single-message-printer.c:11:14",
                 "single-message-printer.c:18:15 single-message-printer.c:18:15",
                 "single-message-printer.c:18:15 single-message-printer.c:18:15",
             }),
      Oracle("single-used-after-no.ll", {}), Oracle("thread-sanitizer-falsepos.ll", {}),
      Oracle("sections-simple-no.ll", {}), Oracle("sections-interproc-no.ll", {}),  // handled by adding 1-callsite PTA
      // Oracle("sections-interproc-no-deep.ll", {}),  // We report FP on the called function, PTA K-callsite limit
      Oracle("sections-interproc-yes.ll", {"sections-interproc-yes.c:3:47 sections-interproc-yes.c:3:47",
                                           "sections-interproc-yes.c:3:47 sections-interproc-yes.c:3:47"}),
      Oracle("duplicate-omp-fork.ll", {}),
      // Oracle("ordered-no.ll", {}), // need support for __kmpc_dispatch_init
      // Oracle("ordered-yes.ll", {"ordered-yes.c:15:30 ordered-yes.c:15:30"}), // need support for __kmpc_dispatch_init
  };

  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP Array Index Analysis Integration Tests", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("array-index-simple.ll", {"array-index-simple.c:8:10 array-index-simple.c:8:12"}),
      Oracle("array-index-inner-yes.ll", {"array-index-inner-yes.c:10:15 array-index-inner-yes.c:10:17"}),
      Oracle("array-index-outer-yes.ll", {"array-index-outer-yes.c:10:15 array-index-outer-yes.c:10:17"}),
      // Oracle("array-multi-dimen-no.ll", {}), // FP on i?
      Oracle("array-stride-2.ll", {}),
  };

  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP Lock Tests", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("lock-set-unset-no.ll", {}),
      Oracle("lock-set-unset-yes.ll",
             {
                 "lock-set-unset-yes.c:11:11 lock-set-unset-yes.c:11:11",
                 "lock-set-unset-yes.c:11:11 lock-set-unset-yes.c:11:11",
             }),

      Oracle("lock-set-unset-yes-2.ll", {"lock-set-unset-yes-2.c:12:19 lock-set-unset-yes-2.c:12:19"})};
  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP get_thread_num", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("get-thread-num-no.ll", {}),
      Oracle("get-thread-num-yes.ll", {"get-thread-num-yes.c:12:14 get-thread-num-yes.c:12:14",
                                       "get-thread-num-yes.c:12:14 get-thread-num-yes.c:12:14"}),
      // Oracle("get-thread-num-interproc-no.ll", {}), // cannot handle interproc yet
      Oracle("get-thread-num-loop-no.ll", {}),
      Oracle("get-thread-num-nested-branch-no.ll", {}),
      Oracle("get-thread-num-double-no.ll", {}),
  };
  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP lastprivate", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("lastprivate-before-yes.ll", {"lastprivate-before-yes.c:13:14 lastprivate-before-yes.c:15:29",
                                           "lastprivate-before-yes.c:15:29 lastprivate-before-yes.c:13:14"}),
      // Cannot pass because there is no race in clang
      // Oracle("last-private-yes.ll", {/*TODO*/}),
      Oracle("lastprivate-no.ll", {}),
      Oracle("lastprivate-loop-split-no.ll", {}),
  };
  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP task", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("task-master-no.ll", {}),
      Oracle("task-single-call.ll", {}),
      Oracle("task-single-no.ll", {}),
      Oracle("task-single-yes.ll", {"task-single-yes.c:15:17 task-single-yes.c:21:17"}),
      Oracle("task-master-single-yes.ll", {"task-master-single-yes.c:18:14 task-master-single-yes.c:14:16",
                                           "task-master-single-yes.c:14:16 task-master-single-yes.c:18:14"}),
      Oracle("task-tid-no.ll", {"task-tid-no.c:15:16 task-tid-no.c:15:16"}),  // cannot identify if condition
      Oracle("task-yes.ll", {"task-yes.c:13:14 task-yes.c:13:14"}),
  };
  checkOracles(oracles, "integration/openmp/");
}

TEST_CASE("OpenMP threadlocal", "[integration][omp]") {
  std::vector<Oracle> oracles = {
      Oracle("threadlocal-no.ll", {}),
  };
  checkOracles(oracles, "integration/openmp/");
}

// set_num_threads and push_num_threads tests
TEST_LL("OpenMP set-num-threads-no", "set-num-threads-no.ll", NORACE)
TEST_LL("OpenMP set-num-threads-reset-yes", "set-num-threads-reset-yes.ll",
        EXPECTED("set-num-threads-reset-yes.c:15:11 set-num-threads-reset-yes.c:15:11"))
// Cannot pass without support for push_num_threads
// TEST_LL("OpenMP push-num-threads-no", "push-num-threads-no.ll" NORACE)
// TEST_LL("OpenMP push-num-threads-yes", "push-num-threads-yes.ll",
//         EXPECTED("push-num-threads-yes.c:11:11 push-num-threads-yes.c:11:11"))
// TEST_LL("OpenMP push-num-threads-2-yes", "push-num-threads-2-yes.ll",
//         EXPECTED("push-num-threads-2-yes.c:14:12 push-num-threads-2-yes.c:14:12"))