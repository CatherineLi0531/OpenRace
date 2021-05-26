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

#include "Intrinsic.h"

namespace IntrinsicModel {

// TODO: need different system for storing and organizing these "recognizers"
bool isPrintf(const llvm::StringRef &funcName) { return funcName.equals("printf"); }
bool isLLVMDebug(const llvm::StringRef &funcName) {
  return funcName.equals("llvm.dbg.declare") || funcName.equals("llvm.dbg.value");
}

bool Modeller::addFuncIRRepr(std::vector<std::shared_ptr<const race::IR>>& instructions,
                             llvm::BasicBlock::const_iterator& it, const llvm::CallBase* callInst,
                             const llvm::StringRef& funcName) const {
  if (isPrintf(funcName)) {
    // TODO: model as read?
  } else if (isLLVMDebug(funcName)) {
    // Skip
  } else {
    return false;
  }
  return true;
}

}
