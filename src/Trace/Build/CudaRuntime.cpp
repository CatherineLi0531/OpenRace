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
  
  if (ir->type == IR::Type::CudaGridFork) {
    return true; 
  }
  
  
  
  
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

// CREATE FAKE JOINS (for sure) FOR EACH FORK

// Create child forks (maybe) & structs & relations (event & IR)
//    // Kernel invocation implies blocks, warps, & threads

// summary.push_back(std::make_shared<CudaBlockFork>(callInst));
// if (gridFork->hasMultipleBlocks()) {
//   summary.push_back(std::make_shared<CudaBlockFork>(callInst));
// }

// summary.push_back(std::make_shared<CudaWarpFork>(callInst));
// if (gridFork->hasMultipleWarps()) {
//   summary.push_back(std::make_shared<CudaWarpFork>(callInst));
// }

// summary.push_back(std::make_shared<CudaThreadFork>(callInst));
// if (gridFork->hasMultipleThreads()) {
//   summary.push_back(std::make_shared<CudaThreadFork>(callInst));
// }

void CudaRuntime::postFork(const std::shared_ptr<const ForkIR> &forkIR, ThreadBuildState &state,
                           const ForkEvent *forkEvent) {
  // Join child forks & structs & relations (event & IR)
  //    // Kernel invocation implies blocks, warps, & threads

  if (ir->type == IR::Type::CudaThreadFork && llvm::dyn_cast<CudaThreadFork>(ir).isLastThread()) {
    
    ForkEvent* lastThread;

    for (auto event : state.begin(); event != state.end(); event++) {
      if (event == CudaGridFork) { //Create Grid Fork Trace
        state.events.insert(event + 1, CudaBlockFork);
        state.events.insert(event + 2, CudaBlockFork);
        state.childThreads.push_back(CudaBlockFork);
        state.childThreads.push_back(CudaBlockFork);
        state.spawnsite = main;
      }
      else if (event == CudaBlockFork) { //Create Block Fork Trace
        state.events.insert(event + 1, CudaWarpFork);
        state.events.insert(event + 2, CudaWarpFork);
        state.childThreads.push_back(CudaWarpFork);
        state.childThreads.push_back(CudaWarpFork);
        state.spawnsite = CudaGridFork;
      }
      else if (event == CudaWarpFork) { //Create Warp Fork Trace
        state.events.insert(event + 1, CudaThreadFork);
        state.events.insert(event + 2, CudaThreadFork);
        state.childThreads.push_back(CudaThreadFork);
        state.childThreads.push_back(CudaThreadFork);
        state.spawnsite = CudaBlockFork;
      }
      else if (event == CudaThreadFork) { //Link Threads to Warps
        state.spawnsite = CudaWarpFork;
        lastThread = event;
      }
      else if (event == isMainThreadCudaGridFork) { //Remove Fake Forks
        for(int i = 0; i < CudaGridFork.getNumberOfThreads(); i++){
          state.events.remove(event+1);
        }
      }
    }

    //Create Fake Joins
    for (auto event : state.begin(); event != state.end(); event++) {
      if(event = lastThread){
        //Like a stack, this will place them in the trace as Thread, Thread, Warp, Warp, etc.
        state.events.insert(event + 1, CudaJoinGrids);
        state.events.insert(event + 1, CudaJoinBlocks);
        state.events.insert(event + 1, CudaJoinBlocks);
        state.events.insert(event + 1, CudaJoinWarps);
        state.events.insert(event + 1, CudaJoinWarps);
        state.events.insert(event + 1, CudaJoinWarps);
        state.events.insert(event + 1, CudaJoinWarps);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
        state.events.insert(event + 1, CudaJoinThreads);
      }
    }
    

    

  }

  if (forkIR->type == IR::Type::CudaGridFork) {
    
  }
}