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