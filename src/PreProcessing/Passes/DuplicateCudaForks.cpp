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

#include <iostream> //TODO: TEMPORARY

#include "PreProcessing/Passes/DuplicateCudaForks.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/Support/raw_ostream.h>

#include "LanguageModel/Cuda.h"

namespace {

// This creates a duplicate call to kmpc_fork so that PTA can see two threads being spawned
void duplicateForkCall(llvm::CallBase *cudaFork) {
  llvm::IRBuilder<> build(cudaFork);
  std::vector<llvm::Value *> arg_list;

  for (auto const &arg : cudaFork->args()) {
    auto val = llvm::cast<llvm::Value>(arg);
    arg_list.push_back(val);
  }

  // First arg is thread handle of type struct.ident_t *
  // create a new handle so that the two spawned threads can be distinguished from eachother
  auto ty = arg_list.front()->getType()->getPointerElementType();
  arg_list[0] = build.CreateAlloca(ty, nullptr, "FAKE_duplicateCudaForkHandle");

  auto callee = cudaFork->getCalledOperand();

  if (llvm::isa<llvm::CallInst>(cudaFork)) {
    auto inst = build.CreateCall(callee, {arg_list});
    assert(inst);
  } else {
    auto II = llvm::cast<llvm::InvokeInst>(cudaFork);
    auto inst = build.CreateInvoke(callee, II->getNormalDest(), II->getUnwindDest(), {arg_list});
    assert(inst);
  }
}
}  // namespace

void duplicateCudaForks(llvm::Module &module) {
  for (auto &function : module.getFunctionList()) {
    for (auto &basicblock : function.getBasicBlockList()) {
      for (auto &inst : basicblock.getInstList()) {
        auto call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call || !call->getCalledFunction() || !call->getCalledFunction()->hasName()) continue;

        auto const funcName = call->getCalledFunction()->getName();

        if (CudaModel::isKernelLaunch(funcName)) {
          std::cout << "Duplicating Cuda Thread\n";
          duplicateForkCall(call);
        }
      }
    }
  }
}