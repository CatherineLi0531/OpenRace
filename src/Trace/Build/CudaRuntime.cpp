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

#include "CudaRuntime.h"

using namespace race;

namespace {
// return the spawning grid fork if this is an cuda thread, else return nullptr
const CudaGridFork *isCudaThread(const ThreadTrace &thread) {
  if (!thread.spawnSite) return nullptr;
  return llvm::dyn_cast<CudaGridFork>(thread.spawnSite.value()->getIRInst());
}
}  // namespace

bool CudaRuntime::preVisit(const std::shared_ptr<const IR> &ir, ThreadBuildState &state) {
  // If at device barrier, join all streams
  if (ir->type == IR::Type::CudaDeviceBarrier) {
    return false;
  }

  // If at stream barrier, join stream to calling thread
  if (ir->type == IR::Type::CudaStreamBarrier) {
    return false;
  }

  // If at block barrier, join all block warps & threads
  if (ir->type == IR::Type::CudaBlockBarrier) {
    // for (auto const &thread : gridFork->getBlock->getWarp->getThreads()) {
    //   addJoinEvent(thread, state);
    // }
    // unjoinedTasks.clear();

    return false;
  }

  // TODO: investigate something like a "CudaKernelEnd", which joins up the chain (streams->warps->blocks)
  //   if (ir->type == IR::Type::OpenMPMasterEnd) {
  //     if (isOpenMPMasterThread(state.thread)) {
  //       markMasterEnd(ir->getInst());
  //     }
  //     return false;
  //   }

  return false;
}

void CudaRuntime::preFork(const std::shared_ptr<const ForkIR> &forkIR, const ForkEvent *forkEvent) {
  if (forkIR->type == IR::Type::CudaGridFork) {
    std::shared_ptr<const CudaGridFork> kernel(forkIR, llvm::cast<CudaGridFork>(forkIR.get()));
    // unjoinedKernels.emplace_back(forkEvent, kernel);
  }
}

void CudaRuntime::postFork(const std::shared_ptr<const ForkIR> &forkIR, const ForkEvent *forkEvent) {
  //   if (forkIR->type == IR::Type::CudaGridFork) {
  //     inKernelregion = false;
  //   }
}