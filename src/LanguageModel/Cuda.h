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

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>

#include <set>

//--Not Mapped--
//         cudaStreamWaitEvent
//         cudaStreamQuery
//         cudaStreamAddCallback
//
//         cudaEventCreate
//         cudaEventDestroy
//         cudaEventSynchronize
//         cudaEventQuery
//
//         cudaSetDevice
//         cudaDeviceCanAccessPeer
//
//         cudaCreateTextureObject
//         cudaDestroyTextureObject
//
//         cudaCreateSurfaceObject
//         cudaDestroySurfaceObject
//
//         __all
//         __any
//         __ballot
//         __shfl
//         __shfl_up
//         __shfl_down
//         __shfl_xor
//
//
//         MEMORY FUNCTIONS
//
// --Analysis Independent (and therefore not supported)--
//         cudaStreamCreateWithPriority
//         cudaDeviceSetLimit ?

// Models CUDA 8.0 (compatibility w/ LLVM & Clang 10.0)

namespace CudaModel {

inline bool isCuda(const llvm::StringRef& funcName) {
  return funcName.startswith("cuda") || funcName.startswith("_ZL9") || funcName.startswith("llvm.nvmm");
}
inline bool isSyncThreads(const llvm::StringRef& funcName) { return funcName.equals("llvm.nvvm.barrier0"); }
inline bool isBlockBarrier(const llvm::StringRef& funcName) { return isSyncThreads(funcName); }

inline bool isDeviceSynchronize(const llvm::StringRef& funcName) { return funcName.equals("cudaDeviceSynchronize"); }

inline bool isStreamCreate(const llvm::StringRef& funcName) { return funcName.equals("cudaStreamCreate"); }
inline bool isStreamBarrier(const llvm::StringRef& funcName) { return funcName.equals("cudaStreamSynchronize"); }
inline bool isStreamDestroy(const llvm::StringRef& funcName) { return funcName.equals("cudaStreamDestroy"); }

inline bool isForkGrid(const llvm::StringRef& funcName) { return funcName.equals("cudaLaunch"); }
inline bool isKernelLaunch(const llvm::StringRef& funcName) { return isForkGrid(funcName); }

const std::set<llvm::StringRef> atomics{"_ZL9atomicAddPjj",
                                        "_ZL9atomicSubPjj",
                                        "_ZL9atomicMinPjj",
                                        "_ZL9atomicMaxPjj",
                                        "_ZL9atomicIncPjj",
                                        "_ZL9atomicDecPjj",
                                        "_ZL10atomicExchPjj",
                                        "_ZL9atomicCASPjjj",
                                        "_ZL9atomicAndPjj",
                                        "_ZL8atomicOrPjj",
                                        "_ZL9atomicXorPjj",
                                        "llvm.nvvm.atomic.load.inc .32.p0i32",
                                        "llvm.nvvm.atomic.load.dec.32.p0i32"};

inline bool isAtomic(const llvm::StringRef& funcName) { return atomics.find(funcName) != atomics.end(); }

// isMemcopy():
//     llvm.memcpy.p0i8.p0i8.i64

}  // namespace CudaModel