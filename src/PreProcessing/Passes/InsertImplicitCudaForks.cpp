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

#include "PreProcessing/Passes/InsertImplicitCudaForks.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/Support/raw_ostream.h>

#include "LanguageModel/Cuda.h"

namespace {

// This creates the forks for the subobjects of the grid (blocks, warps, threads)
void generateForks(llvm::CallBase *cudaGridFork, bool soloBlock, bool soloWarp, bool soloThread) {
  llvm::IRBuilder<> build(cudaGridFork);
  
  // First arg is thread handle of type struct.ident_t *
  // create a new handle so that the two spawned threads can be distinguished from eachother
  //auto ty = arg_list.front()->getType()->getPointerElementType();
  //arg_list[0] = build.CreateAlloca(ty, nullptr, "cr_CudaForkHandle");
  std::string kernelName = cudaGridFork->getName().str();

  int numBlocks =  (soloBlock)  ? 1 : 2;
  int numWarps =   (soloWarp)   ? 1 : 2;
  int numThreads = (soloThread) ? 1 : 2;

  std::string blocks[numBlocks];
  for(int i = 0; i < numBlocks; i++) { blocks[i] = "b"+std::to_string(i); }

  std::string warps[numWarps];
  for(int i = 0; i < numWarps; i++) { warps[i] = "w"+std::to_string(i); }

  std::string threads[numThreads];
  for(int i = 0; i < numThreads; i++) { threads[i] = "t"+std::to_string(i); }

  for(int i = 0; i < numBlocks; i++){
    for(int j = 0; j < numWarps; j++){
      for(int k = 0; k < numThreads; k++){
        auto inst = build.CreateCall(cudaGridFork->getCalledOperand(), llvm::None, kernelName+"_"+blocks[i]+"_"+warps[j]+"_"+threads[k]+"_fork");
        assert(inst);
      }
    }
  }
}
}  // namespace

void insertImplicitCudaForks(llvm::Module &module) {

  // Used to model when grids contain only 1 of either block, warp, or thread
  // We avoid making multiple forks in this case
  bool soloBlock = false;
  bool soloWarp = false;
  bool soloThread = false;

  for (auto &function : module.getFunctionList()) {
    for (auto &basicblock : function.getBasicBlockList()) {
      for (auto &inst : basicblock.getInstList()) {
        auto call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call || !call->getCalledFunction() || !call->getCalledFunction()->hasName()) continue;

        auto const funcName = call->getCalledFunction()->getName();

        if (CudaModel::isForkGrid(funcName)) {
          //TODO: get values of soloBlock, soloWarp, soloThread
          generateForks(call,soloBlock,soloWarp,soloThread);       
        }
      }
    }
  }
}