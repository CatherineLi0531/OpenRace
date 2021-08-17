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

#include <IR/Builder.h>
#include <llvm/ADT/DenseMap.h>

#include <vector>

#include "IR/IRImpls.h"
//#include "LanguageModel/RaceModel.h"
#include "ThreadTrace.h"
#include "Trace/Event.h"

namespace race {

struct OpenMPState {
  // Track if we are currently in parallel region created from kmpc_fork_teams
  size_t teamsDepth = 0;
  bool inTeamsRegion() const { return teamsDepth > 0; }

  // Track if we are in single region
  bool inSingle = false;

  // Track the start/end instructions of master regions
  std::map<const llvm::CallBase *, const llvm::CallBase *> masterRegions;
  const llvm::CallBase *currentMasterStart = nullptr;

  // record the start of a master
  void markMasterStart(const llvm::CallBase *start) {
    assert(!currentMasterStart && "encountered two master starts in a row");
    currentMasterStart = start;
  }

  // mark the end of a master region
  void markMasterEnd(const llvm::CallBase *end) {
    assert(currentMasterStart && "encountered master end without start");
    masterRegions.insert({currentMasterStart, end});
    currentMasterStart = nullptr;
  }

  // Get the end of a previously encountered master region
  const llvm::CallBase *getMasterRegionEnd(const llvm::CallBase *start) const { return masterRegions.at(start); }

  // NOTE: this ugliness is only needed because there is no way to get the shared_ptr
  // from the forkEvent. forkEvent->getIRInst() returns a raw pointer instead.
  struct UnjoinedTask {
    const ForkEvent *forkEvent;
    std::shared_ptr<const OpenMPTaskFork> forkIR;

    UnjoinedTask(const ForkEvent *forkEvent, std::shared_ptr<const OpenMPTaskFork> forkIR)
        : forkEvent(forkEvent), forkIR(forkIR) {}
  };

  // List of unjoined OpenMP task threads
  std::vector<UnjoinedTask> unjoinedTasks;
};

// record a HB state: current belonging thread + [ a list of seen fork/join events ]
struct HBState {
  std::vector<EventID> forkjoins;

  HBState(std::vector<EventID> forkjoins) { this->forkjoins.assign(forkjoins.begin(), forkjoins.end()); };

  bool operator==(const HBState &other) const {
    return forkjoins.size() == other.forkjoins.size() &&
           std::equal(forkjoins.begin(), forkjoins.end(), other.forkjoins.begin());
  }
};

// record a state of a specific combination of HB relation and lockset
struct LockState {
  std::vector<EventID> lockset;  // a set of held/released locks

  LockState(std::vector<EventID> lockset) { this->lockset.assign(lockset.begin(), lockset.end()); };

  bool operator==(const LockState &other) const {
    return lockset.size() == other.lockset.size() && std::equal(lockset.begin(), lockset.end(), other.lockset.begin());
  }
};

struct HBLockState {
  const ThreadID myTID;

  // track the interesting events during building the thread traces for myTID
  std::vector<EventID> forkJoinEvents;
  std::vector<EventID> lockEvents;

  // the current states
  const HBState *curHB = nullptr;
  const LockState *curLock = nullptr;

  // use both HBState and LockState as the key
  using KeyType = std::pair<const HBState *, const LockState *>;
  llvm::DenseMap<KeyType, std::vector<const llvm::Function *> *> state2fn;

  HBLockState(ThreadID tid) : myTID(tid) {
    curHB = new HBState(forkJoinEvents);
    curLock = new LockState(lockEvents);
  };

  // push into forkJoinEvents/lockEvents to track current state
  void updateEvent(EventID eid, Event::Type typ) {
    switch (typ) {
      case Event::Type::Fork:
      case Event::Type::Join:
      case Event::Type::Barrier: {
        forkJoinEvents.push_back(eid);
        curHB = new HBState(forkJoinEvents);
        break;
      }
      case Event::Type::Lock:
      case Event::Type::Unlock: {
        lockEvents.push_back(eid);
        curLock = new LockState(lockEvents);
        break;
      }
      default:
        llvm::errs() << "Should not call HBLockState::updateEvent for " << typ << "\n";
        break;
    }
  }

  // return true if state2fn has the key (curHB + curLock) for func
  bool existHBLockStateFor(const llvm::Function *func) {
    auto key = std::make_pair(curHB, curLock);
    auto it = state2fn.find(key);
    if (it == state2fn.end()) {  // no such key
      std::vector<const llvm::Function *> *traversedFuncs = new std::vector<const llvm::Function *>();
      traversedFuncs->push_back(func);
      state2fn.insert(std::make_pair(key, traversedFuncs));
      return false;
    }

    auto traversedFuncs = it->second;
    bool found = std::find(traversedFuncs->begin(), traversedFuncs->end(), func) != traversedFuncs->end();
    if (!found) {  // not traversed function for such a state yet
      traversedFuncs->push_back(func);
    }

    return found;
  }

  // clear/release all allocated memory here
  void clear() {
    forkJoinEvents.resize(0);  // or use .clear() ?
    forkJoinEvents.shrink_to_fit();
    lockEvents.resize(0);
    lockEvents.shrink_to_fit();
    curLock = nullptr;
    curHB = nullptr;
    state2fn.shrink_and_clear();
  }
};

struct TraversalState {
  // the HB and lockset state that already recorded
  std::map<const ThreadID, HBLockState *> traversedStates;

  // the HBLockState for the thread id that we are currently building for
  HBLockState *curState = nullptr;

  void findOrCreateCurState(const ThreadID tid) {
    auto it = traversedStates.find(tid);
    if (it == traversedStates.end()) {
      curState = new HBLockState(tid);
      traversedStates.insert(std::make_pair(tid, curState));
    } else {
      assert(it->first == tid);
      curState = it->second;
    }
  }

  // return true if we already traversedStates func for current HB and Lockset state
  bool existStateForFunc(const ThreadID tid, const llvm::Function *func) {
    if (curState == nullptr || curState->myTID != tid) {
      findOrCreateCurState(tid);
    }
    return curState->existHBLockStateFor(func);
  }

  // push into forkJoinEvents/lockEvents in HBLockState for tid to track its current state
  void updateEvent(const ThreadID tid, EventID eid, Event::Type typ) {
    if (curState == nullptr || curState->myTID != tid) {
      findOrCreateCurState(tid);
    }
    curState->updateEvent(eid, typ);
  }

  // remove the record from traversedStates when the build of thread trace is finished
  void removeStateFor(const ThreadID tid) {
    curState->clear();
    curState = nullptr;
    traversedStates.erase(tid);
  }
};

// all included states are ONLY used when building ProgramTrace/ThreadTrace
struct TraceBuildState {
  // Cached function summaries
  FunctionSummaryBuilder builder;

  // the counter of thread id: since we are constructing ThreadTrace while building events,
  // pState.threads.size() will be updated after finishing the construction, we need such a counter
  ThreadID currentTID = 0;

  // When set, skip traversing until this instruction is reached
  const llvm::Instruction *skipUntil = nullptr;

  // Track state specific to OpenMP
  OpenMPState openmp;

  // Track the state of HB relation and lockset for each traversed function
  TraversalState traversal;
};

class ProgramTrace {
  llvm::Module *module;
  std::unique_ptr<ThreadTrace> mainThread;
  std::vector<const ThreadTrace *> threads;

  friend class ThreadTrace;

 public:
  pta::PTA pta;

  [[nodiscard]] inline const std::vector<const ThreadTrace *> &getThreads() const { return threads; }

  [[nodiscard]] const Event *getEvent(ThreadID tid, EventID eid) { return threads.at(tid)->getEvent(eid); }

  // Get the module after preprocessing has been run
  [[nodiscard]] const Module &getModule() const { return *module; }

  explicit ProgramTrace(llvm::Module *module, llvm::StringRef entryName = "main");
  ~ProgramTrace() = default;
  ProgramTrace(const ProgramTrace &) = delete;
  ProgramTrace(ProgramTrace &&) = delete;  // Need to update threads because
                                           // they contain reference to parent
  ProgramTrace &operator=(const ProgramTrace &) = delete;
  ProgramTrace &operator=(ProgramTrace &&) = delete;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ProgramTrace &trace);

}  // namespace race
