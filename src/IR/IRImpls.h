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
#include <LanguageModel/OpenMP.h>
#include <llvm/IR/CallSite.h>

#include "IR/IR.h"

namespace race {

// ==================================================================
// ================== ReadIR Implementations ========================
// ==================================================================

class Load : public ReadIR {
  const llvm::LoadInst *inst;

 public:
  explicit Load(const llvm::LoadInst *load) : ReadIR(Type::Load), inst(load) {}

  [[nodiscard]] inline const llvm::LoadInst *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getPointerOperand(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::Load; }
};

class APIRead : public ReadIR {
  // Operand that this API call reads
  unsigned int operandOffset;

  const llvm::CallBase *inst;

 public:
  // API call that reads one of it's operands, specified by 'operandOffset'
  APIRead(const llvm::CallBase *inst, unsigned int operandOffset)
      : ReadIR(Type::APIRead), operandOffset(operandOffset), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getOperand(operandOffset); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::APIRead; }
};

// ==================================================================
// ================= WriteIR Implementations ========================
// ==================================================================

class Store : public WriteIR {
  const llvm::StoreInst *inst;

 public:
  explicit Store(const llvm::StoreInst *store) : WriteIR(Type::Store), inst(store) {}

  [[nodiscard]] inline const llvm::StoreInst *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getPointerOperand(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::Store; }
};

class APIWrite : public WriteIR {
  // Operand that this API call reads
  unsigned int operandOffset;

  const llvm::CallBase *inst;

 public:
  // API call that write to one of it's operands, specified by 'operandOffset'
  APIWrite(const llvm::CallBase *inst, unsigned int operandOffset)
      : WriteIR(Type::APIWrite), operandOffset(operandOffset), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override {
    return getInst()->getOperand(operandOffset);
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::APIWrite; }
};

// ==================================================================
// ================== ForkIR Implementations ========================
// ==================================================================

class PthreadCreate : public ForkIR {
  constexpr static unsigned int threadHandleOffset = 0;
  constexpr static unsigned int threadEntryOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit PthreadCreate(const llvm::CallBase *inst) : ForkIR(Type::PthreadCreate), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    return inst->getArgOperand(threadEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::PthreadCreate; }
};

class OpenMPFork : public ForkIR {
  // https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L262
  // @param loc  source location information
  // @param argc  total number of arguments in the ellipsis
  // @param microtask  pointer to callback routine consisting of outlined parallel
  // construct
  // @param ...  pointers to shared variables that aren't global

  constexpr static unsigned int threadHandleOffset = 0;
  constexpr static unsigned int threadEntryOffset = 2;
  const llvm::CallBase *inst;

 public:
  enum class ThreadType { Master, Other };
  const OpenMPFork::ThreadType forkedThreadType;

  explicit OpenMPFork(const llvm::CallBase *inst, ThreadType forkedThreadType = ThreadType::Other)
      : ForkIR(IR::Type::OpenMPFork), inst(inst), forkedThreadType(forkedThreadType) {}

  [[nodiscard]] inline bool isForkingMaster() const { return forkedThreadType == ThreadType::Master; }

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    return inst->getArgOperand(threadEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == IR::Type::OpenMPFork; }
};

class OpenMPTaskFork : public ForkIR {
  // https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_tasking.cpp#L1684
  constexpr static unsigned int taskAllocOffset = 2;
  constexpr static unsigned int taskEntryOffset = 5;
  const llvm::CallBase *inst;

 public:
  explicit OpenMPTaskFork(const llvm::CallBase *inst) : ForkIR(Type::OpenMPTaskFork), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return getThreadEntry(); }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    auto taskAlloc = inst->getArgOperand(taskAllocOffset)->stripPointerCasts();
    auto taskAllocCall = llvm::dyn_cast<llvm::CallBase>(taskAlloc);
    assert(taskAllocCall && "Failed to find task alloc call");
    assert(OpenMPModel::isTaskAlloc(taskAllocCall->getCalledFunction()->getName()) && "failed to find task alloc");

    return taskAllocCall->getArgOperand(taskEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPTaskFork; }
};

class OpenMPForkTeams : public ForkIR {
  // https://github.com/llvm/llvm-project-staging/blob/cc926dc3a87af7023aa9b6c392347a0a8ed6949b/openmp/runtime/src/kmp_csupport.cpp#L392
  // @param loc  source location information
  // @param argc  total number of arguments in the ellipsis
  // @param microtask  pointer to callback routine consisting of outlined parallel
  // construct
  // @param ...  pointers to shared variables that aren't global
  constexpr static unsigned int threadHandleOffset = 0;
  constexpr static unsigned int threadEntryOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit OpenMPForkTeams(const llvm::CallBase *inst) : ForkIR(Type::OpenMPForkTeams), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    return inst->getArgOperand(threadEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPForkTeams; }
};

// Corresponds to cudaStreamCreate(). There is an implicit default stream stream0. Streams themselves have no
// commands, only that they hold grid forks
class CudaStreamFork : public ForkIR {};

// Corresponds to kernel<<<>>>. Grids themselves have no commands, only that they hold block forks
class CudaGridFork : public ForkIR {
  // TODO: URL
  constexpr static unsigned int kernelFunctionOffset = 0;

  constexpr static unsigned int blockDimOffset = 0;
  constexpr static unsigned int threadDimOffset = 1;
  constexpr static unsigned int sharedMemoryOffset = 2;
  constexpr static unsigned int streamMappingOffset = 3;

  // std::shared_ptr<const CudaBlockFork> block1;
  // std::shared_ptr<const CudaBlockFork> block2;

  const llvm::CallBase *inst;

  bool isMultiBlock, isMultiWarp, isMultiThread;

 public:
  explicit CudaGridFork(const llvm::CallBase *inst) : ForkIR(Type::CudaGridFork), inst(inst) {
    // TODO: get blockDim, threadDim and set the 3 booleans in this.
    //    these values are not present in the cudaLaunch call, or either kernel definition
    //    for now, assuming multiple blocks, multiple warps, and multiple threads

    isMultiBlock = true;

    /*
    if (numThreads > 32) {
      isMultiWarp = true;
    }
    */

    isMultiWarp = true;
    isMultiThread = true;
  }

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return getThreadEntry(); }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    // auto gridFork = llvm::dyn_cast<llvm::CallBase>(inst->getArgOperand(kernelFunctionOffset)->stripPointerCasts());
    // auto blockDim = gridFork->getArgOperand(blockDimOffset)->stripPointerCasts();
    // auto threadDim = gridFork->getArgOperand(threadDimOffset)->stripPointerCasts();
    // auto streamMapping = gridFork->getArgOperand(streamMappingOffset)->stripPointerCasts();

    return inst->getArgOperand(kernelFunctionOffset)->stripPointerCasts();
  }

  inline bool hasMultipleBlocks() { return isMultiBlock; }
  inline bool hasMultipleWarps() { return isMultiWarp; }
  inline bool hasMultipleThreads() { return isMultiThread; }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::CudaGridFork; }
};

class CudaBlockFork : public ForkIR {
  // std::shared_ptr<const CudaWarpFork> warp1;
  // std::shared_ptr<const CudaWarpFork> warp2;

  std::shared_ptr<const CudaGridFork> grid;
  llvm::Value *handle;  // make sure this is unique
  llvm::Value *entry;   // created by us because does not exist in code

  //  name (unique id for grid)
  //  name1 (id block 1)
  //    name11 (id warp 1)
  //      name11
  //      name12
  //    name12
  //  name2 (id for block 2)

 public:
  CudaBlockFork(std::shared_ptr<const CudaGridFork> parentGrid, llvm::Value *handle, llvm::Value *entry)
      : ForkIR(Type::CudaBlockFork), grid(parentGrid), handle(handle), entry(entry) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return grid->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return handle; }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override { return entry; }
};

/*

For now, assume all memory is shared and try to verify addrspaces at the end (not during PTA).

CudaRuntime

std::map from CudaGridFork-> struct {
  2 blocks
  4 Warps
  8 Threads (Identical traces)
   -> preprocessing to trick PTA into creating contexts
   -> insert 8 dummy forks
   -> (optional) insert 8 dummy joins
}


=== Preprocessing
CudaGridFork -> insert 8 dummy forks all with entry of cuda grid
dummy_cuda_thread("dummy_thread_1", @the_kernel_entry_func)

=== IRBuilder
when see CudaGridFork, grab the next 8 dummy forks std::array<llvm::Call, 8>;
Add CudaThreadForkIR to summary for those 8 dummy forks

Summary Example
....
call cudaGridFork
call dummycudaThreadfork() x8 [inserted during preprocessing]

=== Runtime
intercept preVisit -> CudaGridFork
  1. Note that this was most recent encountered
  2. Tell trace builder to skip

Next trace builder will see 8 dummy forks for threads
Save gridFork -> list of dummy Forks
Intercept postFork - count how many have been traversed (I am planning to make this an argument in the pre-processing, then read it out?)
After last (8th) dummy thread trace is built, link everything and insert joins
  1. Create trace for grid fork
  2. Create trace for 2 blocks, link to grid fork
  3. Create trace for 4 warps, link to blocks
  4. Traces for threads, exist and have been saved, link to warps vvv

Cleanup the "main thread" Trace with GridFork,
  1. Remove those dummy thread forks
      a. ThreadTrace.h
      b. Update spawnSite of each dummy thread to be the warp
      c. Move event and childthread pointers onto the warps and off of main thread




End Result Trace

--main--
events: cudaFork
childThreads: CudaGrid


--Grid-- x1
events: cudaBlockFork x2
spawnsite: main

--Block-- x2
childThreads: cudaWarp x4
spawnsite: cudaGridFork

--Warp-- x4
events: cudaWarpFork x2
childThreads: cudaWarp x2
spawnsite: block

-- Thread -- x8
events: (build by threadbuilder directly)
childThreads: None
spawnsite: UPDATE to point to the warps







foo(int *a) {
   *a = soemthign;
}

Thread1 (ctx1)
  write to a

Thread2 (ctx1)
  write to a


pthread_create(t1, NULL, entry, NULL);

handle -> t1
entry -> entry


pthread_join(t1)

handle -> t1



kmpc_fork(@omp_entry1, ...)
kmpc_fake_join(@omp_entry1)

*/

class CudaWarpFork : public ForkIR {
  // So that we know about children forks
  // std::shared_ptr<const CudaThreadFork> thread1;
  // std::shared_ptr<const CudaThreadFork> thread2;

  // so we can get parent inst
  std::shared_ptr<const CudaBlockFork> block;

  llvm::Value *handle;  // make sure this is unique
  llvm::Value *entry;   // created by us because does not exist in code

 public:
  CudaWarpFork(std::shared_ptr<const CudaBlockFork> parentBlock, llvm::Value *handle, llvm::Value *entry)
      : ForkIR(Type::CudaWarpFork), block(parentBlock), handle(handle), entry(entry) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return block->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return handle; }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override { return entry; }
};

class CudaThreadFork : public ForkIR {
  // so we can get parent inst
  std::shared_ptr<const CudaWarpFork> warp;

  llvm::Value *handle;  // make sure this is unique
  llvm::Value *entry;   // created by us because does not exist in code

  bool isLastThreadOfGrid = false;
 public:
  CudaThreadFork(std::shared_ptr<const CudaWarpFork> parentThread, llvm::Value *handle, llvm::Value *entry, bool isLast)
      : ForkIR(Type::CudaThreadFork), warp(parentThread), handle(handle), entry(entry), isLastThreadOfGrid(isLast) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return warp->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return handle; }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override { return entry; }

  bool isLastThread() { return isLastThreadOfGrid; }
};
/** Cuda 9+

// Corresponds to cg::partition
class CudaCooperativeGroupFork : public ForkIR {};
*/

// ==================================================================
// ================== JoinIR Implementations ========================
// ==================================================================

class PthreadJoin : public JoinIR {
  const unsigned int threadHandleOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit PthreadJoin(const llvm::CallBase *inst) : JoinIR(Type::PthreadJoin), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::PthreadJoin; }
};

// This actually corresponds to a OpenMP fork instruction, as the fork call acts as both a fork and join in one call
class OpenMPJoin : public JoinIR {
  std::shared_ptr<OpenMPFork> fork;

 public:
  explicit OpenMPJoin(const std::shared_ptr<OpenMPFork> fork) : JoinIR(Type::OpenMPJoin), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPJoin; }
};

class OpenMPTaskJoin : public JoinIR {
  std::shared_ptr<const OpenMPTaskFork> task;

 public:
  explicit OpenMPTaskJoin(const std::shared_ptr<const OpenMPTaskFork> task)
      : JoinIR(Type::OpenMPTaskJoin), task(task) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return task->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return task->getThreadEntry(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPTaskJoin; }
};

// This actually corresponds to a OpenMP forkTeams instruction
// the fork call acts as both a fork and join in one call
class OpenMPJoinTeams : public JoinIR {
  std::shared_ptr<OpenMPForkTeams> fork;

 public:
  explicit OpenMPJoinTeams(const std::shared_ptr<OpenMPForkTeams> fork) : JoinIR(Type::OpenMPJoinTeams), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPJoinTeams; }
};

class CudaJoinGrids : public JoinIR {
  std::shared_ptr<CudaGridFork> fork;

  public:
  explicit CudaJoinGrids(const std::shared_ptr<CudaGridFork> fork) : JoinIR(Type::CudaJoinGrids), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::CudaJoinGrids; }
};

class CudaJoinBlocks : public JoinIR {
  std::shared_ptr<CudaGridFork> fork; //These could point to CudaBlockFork,WarpFork,etc?

  public:
  explicit CudaJoinBlocks(const std::shared_ptr<CudaGridFork> fork) : JoinIR(Type::CudaJoinBlocks), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::CudaJoinBlocks; }
};

class CudaJoinWarps : public JoinIR {
  std::shared_ptr<CudaGridFork> fork;

  public:
  explicit CudaJoinWarps(const std::shared_ptr<CudaGridFork> fork) : JoinIR(Type::CudaJoinWarps), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::CudaJoinWarps; }
};

class CudaJoinThreads : public JoinIR {
  std::shared_ptr<CudaGridFork> fork;

  public:
  explicit CudaJoinThreads(const std::shared_ptr<CudaGridFork> fork) : JoinIR(Type::CudaJoinThreads), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::CudaJoinThreads; }
};

/** Cuda 9+

class CudaJoinCooperativeGroups : public ForkIR {};
*/

// ==================================================================
// ================== LockIR Implementations ========================
// ==================================================================

// LockIRImpl should not be used directly. Instead define a using alias.
// See PthreadMutexLock below as an example.
template <IR::Type T>
class LockIRImpl : public LockIR {
  const unsigned int lockObjectOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit LockIRImpl(const llvm::CallBase *call) : LockIR(T), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(lockObjectOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

class OpenMPCriticalStart : public LockIR {
  // https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L1157
  // @param loc  source location information
  // @param global_tid  global thread number
  // @param crit identity of the critical section. This could be a pointer to a lock
  // associated with the critical section, or some other suitably unique value
  const unsigned int identityOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit OpenMPCriticalStart(const llvm::CallBase *call) : LockIR(Type::OpenMPCriticalStart), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(identityOffset)->stripPointerCasts();
  }

  static inline bool classof(const IR *e) { return e->type == Type::OpenMPCriticalStart; }
};

// NOTE: if a specific API semantic is the same as default impl,
// create a type alias.
using PthreadMutexLock = LockIRImpl<IR::Type::PthreadMutexLock>;
using PthreadSpinLock = LockIRImpl<IR::Type::PthreadSpinLock>;

// https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L2549
using OpenMPSetLock = LockIRImpl<IR::Type::OpenMPSetLock>;
using OpenMPOrderedStart = LockIRImpl<IR::Type::OpenMPOrderedStart>;
// ==================================================================
// ================= UnlockIR Implementations =======================
// ==================================================================

// UnlockIRImpl should not be used directly. Instead define using alias.
// See PthreadMutexUnlock below as an example.
template <IR::Type T>
class UnlockIRImpl : public UnlockIR {
  const unsigned int lockObjectOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit UnlockIRImpl(const llvm::CallBase *call) : UnlockIR(T), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(lockObjectOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

class OpenMPCriticalEnd : public UnlockIR {
  // https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L1512
  // @param loc  source location information
  // @param global_tid  global thread number
  // @param crit identity of the critical section. This could be a pointer to a lock
  // associated with the critical section, or some other suitably unique value
  const unsigned int identityOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit OpenMPCriticalEnd(const llvm::CallBase *call) : UnlockIR(Type::OpenMPCriticalEnd), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(identityOffset)->stripPointerCasts();
  }

  static inline bool classof(const IR *e) { return e->type == Type::OpenMPCriticalEnd; }
};

// NOTE: if a specific API semantic is the same as default impl,
// create a type alias.
using PthreadMutexUnlock = UnlockIRImpl<IR::Type::PthreadMutexUnlock>;
using PthreadSpinUnlock = UnlockIRImpl<IR::Type::PthreadSpinUnlock>;

// https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L2752
using OpenMPUnsetLock = UnlockIRImpl<IR::Type::OpenMPUnsetLock>;
using OpenMPOrderedEnd = UnlockIRImpl<IR::Type::OpenMPOrderedEnd>;
// =================================================================
// ================= Barrier Implementations =======================
// =================================================================

// https://github.com/llvm/llvm-project/blob/d32170dbd5b0d54436537b6b75beaf44324e0c28/openmp/runtime/src/kmp_csupport.cpp#L713
class OpenMPBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit OpenMPBarrier(const llvm::CallBase *call) : BarrierIR(Type::OpenMPBarrier), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

// Corresponds to cudaDeviceSynchronize()
class CudaDeviceBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit CudaDeviceBarrier(const llvm::CallBase *call) : BarrierIR(Type::CudaDeviceBarrier), inst(call) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

// Corresponds to cudaStreamSynchronize()
class CudaStreamBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit CudaStreamBarrier(const llvm::CallBase *call) : BarrierIR(Type::CudaStreamBarrier), inst(call) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

// Corresponds to __syncthreads()
class CudaBlockBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit CudaBlockBarrier(const llvm::CallBase *call) : BarrierIR(Type::CudaBlockBarrier), inst(call) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

/** CUDA 9+

// Corresponds to __syncwarp()
class CudaWarpBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit CudaWarpBarrier(const llvm::CallBase *call) : BarrierIR(Type::CudaWarpBarrier), inst(call) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

// Corresponds to cg::synchronize() / g.sync()
class CudaCooperativeGroupBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit CudaCooperativeGroupBarrier(const llvm::CallBase *call)
      : BarrierIR(Type::CudaCooperativeGroupBarrier), inst(call) {}
  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
}
*/

// =================================================================
// ================= Other Implementations =========================
// =================================================================

// CallIRImpl should not be used directly. Instead define using alias.
// See OpenMPForInit below as an example.
template <const IR::Type T>
class CallIRImpl : public CallIR {
 public:
  explicit CallIRImpl(const llvm::CallBase *inst) : CallIR(inst, T) {}

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

using OpenMPForInit = CallIRImpl<IR::Type::OpenMPForInit>;
using OpenMPForFini = CallIRImpl<IR::Type::OpenMPForFini>;

using OpenMPDispatchInit = CallIRImpl<IR::Type::OpenMPDispatchInit>;
using OpenMPDispatchNext = CallIRImpl<IR::Type::OpenMPDispatchNext>;
using OpenMPDispatchFini = CallIRImpl<IR::Type::OpenMPDispatchFini>;

using OpenMPSingleStart = CallIRImpl<IR::Type::OpenMPSingleStart>;
using OpenMPSingleEnd = CallIRImpl<IR::Type::OpenMPSingleEnd>;

using OpenMPReduce = CallIRImpl<IR::Type::OpenMPReduce>;

using OpenMPMasterStart = CallIRImpl<IR::Type::OpenMPMasterStart>;
using OpenMPMasterEnd = CallIRImpl<IR::Type::OpenMPMasterEnd>;

using OpenMPGetThreadNum = CallIRImpl<IR::Type::OpenMPGetThreadNum>;

using OpenMPTaskWait = CallIRImpl<IR::Type::OpenMPTaskWait>;

using OpenMPGetThreadNumGuardStart = CallIRImpl<IR::Type::OpenMPGetThreadNumGuardStart>;
using OpenMPGetThreadNumGuardEnd = CallIRImpl<IR::Type::OpenMPGetThreadNumGuardEnd>;

}  // namespace race
