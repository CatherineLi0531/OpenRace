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

// Model Progress:
//     --Mapped--
//         kernel<<<>>>
//         cudaDeviceSynchronize
//         __syncthreads
//         cudaMalloc (independent of language model)
//         cudaStreamSynchronize
//
//     --Not Mapped--
//         cudaStreamCreate
//         cudaStreamDestroy
//         cudaStreamWaitEvent
//         cudaStreamQuery
//         cudaStreamAddCallback
//
//         cudaEventCreate
//         cudaEventDestroy
//         cudaEventSynchronize
//         cudaStreamWaitEvent
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
//         atomicAdd
//         atomicSub
//         atomicExch
//         atomicMin
//         atomicMax
//         atomicInc
//         atomicDec
//         atomicCAS
//         atomicAnd
//         atomicOr
//         atomicXor
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
//     --Analysis Independent (and therefore not supported)--
//         cudaStreamCreateWithPriority
//         cudaDeviceSetLimit ?

// Models CUDA 8.0 (compatibility w/ LLVM & Clang 10.0)

namespace CudaModel {

namespace {

// return true of funcName equals any name in names
bool matchesAny(const llvm::StringRef& funcName, const std::vector<llvm::StringRef>& names) {
  for (auto const& name : names) {
    if (funcName.equals(name)) return true;
  }
  return false;
}

}  // namespace

inline bool isSyncThreads(const llvm::StringRef& funcName) { return funcName.equals("llvm.nvvm.barrier0"); }
inline bool isBlockBarrier(const llvm::StringRef& funcName) { return isSyncThreads(funcName); }

inline bool isDeviceSynchronize(const llvm::StringRef& funcName) { return funcName.equals("cudaDeviceSynchronize"); }

inline bool isStreamBarrier(const llvm::StringRef& funcName) { return funcName.equals("cudaStreamSynchronize"); }

inline bool isForkGrid(const llvm::StringRef& funcName) { return funcName.equals("cudaLaunch"); }
inline bool isKernelLaunch(const llvm::StringRef& funcName) { return isForkGrid(funcName); }

}  // namespace CudaModel