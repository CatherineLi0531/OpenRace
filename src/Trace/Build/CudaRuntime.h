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

#include "Trace/Build/RuntimeModel.h"
#include "Trace/Build/TraceBuilder.h"

namespace race {

class CudaRuntime : public Runtime {
  // // NOTE: this ugliness is only needed because there is no way to get the shared_ptr
  // // from the forkEvent. forkEvent->getIRInst() returns a raw pointer instead.

  // struct UnjoinedKernel {
  //   const ForkEvent *forkEvent;
  //   std::shared_ptr<const CudaGridFork> forkIR;

  //   UnjoinedKernel(const ForkEvent *forkEvent, std::shared_ptr<const CudaGridFork> forkIR)
  //       : forkEvent(forkEvent), forkIR(forkIR) {}
  // };

  // // List of unjoined Cuda kernels
  // std::vector<UnjoinedKernel> unjoinedKernels;

  // Add join event to the thread trace for the specified grid fork
  // void addJoinEvents(const UnjoinedKernel &task, ThreadBuildState &state);

 public:
  bool preVisit(const std::shared_ptr<const IR> &ir, ThreadBuildState &state) override;
  void preFork(const std::shared_ptr<const ForkIR> &forkIR, const ForkEvent *forkEvent) override;
  void postFork(const std::shared_ptr<const ForkIR> &forkIR, const ForkEvent *forkEvent) override;
};

}  // namespace race