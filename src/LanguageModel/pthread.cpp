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

#include "LanguageModel/pthread.h"

namespace PthreadModel {

bool isPthreadCreate(const llvm::StringRef &funcName) { return funcName.equals("pthread_create"); }

inline bool isPthreadJoin(const llvm::StringRef &funcName) { return funcName.equals("pthread_join"); }
inline bool isPthreadMutexLock(const llvm::StringRef &funcName) { return funcName.equals("pthread_mutex_lock"); }
inline bool isPthreadMutexUnlock(const llvm::StringRef &funcName) { return funcName.equals("pthread_mutex_unlock"); }
inline bool isPthreadSpinLock(const llvm::StringRef &funcName) { return funcName.equals("pthread_spin_lock"); }
inline bool isPthreadSpinUnlock(const llvm::StringRef &funcName) { return funcName.equals("pthread_spin_unlock"); }
inline bool isPthreadOnce(const llvm::StringRef &funcName) { return funcName.equals("pthread_once"); }

bool Modeller::addFuncIRRepr(std::vector<std::shared_ptr<const race::IR>> &instructions,
                             llvm::BasicBlock::const_iterator &it, const llvm::CallBase *callInst,
                             const llvm::StringRef &funcName) const {
  if (isPthreadCreate(funcName)) {
    instructions.push_back(std::make_shared<Create>(callInst));
  } else if (isPthreadJoin(funcName)) {
    instructions.push_back(std::make_shared<Join>(callInst));
  } else if (isPthreadMutexLock(funcName)) {
    instructions.push_back(std::make_shared<MutexLock>(callInst));
  } else if (isPthreadMutexUnlock(funcName)) {
    instructions.push_back(std::make_shared<MutexUnlock>(callInst));
  } else if (isPthreadSpinLock(funcName)) {
    instructions.push_back(std::make_shared<SpinLock>(callInst));
  } else if (isPthreadSpinUnlock(funcName)) {
    instructions.push_back(std::make_shared<PthreadSpinUnlock>(callInst));
  } else {
    return false;
  }
  return true;
}

}  // namespace PthreadModel
