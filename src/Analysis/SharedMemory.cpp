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

#include "Analysis/SharedMemory.h"
using namespace race;

SharedMemory::SharedMemory(const ProgramTrace &program) {
  auto const getObjId = [&](const pta::ObjTy *obj) {
    // cppcheck-suppress stlIfFind
    if (auto it = objIDs.find(obj); it != objIDs.end()) {
      return it->second;
    }

    auto id = objIDs.size();
    objIDs[obj] = id;
    return id;
  };

  if (DEBUG_PTA) {
    llvm::outs() << "** SharedMemory **"
                 << "\n";
  }
  for (auto const &thread : program.getThreads()) {
    auto const tid = thread->id;
    if (DEBUG_PTA) {
      llvm::outs() << "------- tid: " << tid << "\n";
    }

    for (auto const &event : thread->getEvents()) {
      switch (event->type) {
        case Event::Type::Read: {
          auto readEvent = llvm::cast<ReadEvent>(event.get());
          auto const &ptsTo = readEvent->getAccessedMemory();
          // TODO: filter?
          for (auto obj : ptsTo) {
            auto &reads = objReads[getObjId(obj)][tid];
            reads.push_back(readEvent);
          }
          break;
        }
        case Event::Type::Write: {
          auto writeEvent = llvm::cast<WriteEvent>(event.get());
          auto const &ptsTo = writeEvent->getAccessedMemory();
          // TODO: filter?
          for (auto obj : ptsTo) {
            auto &writes = objWrites[getObjId(obj)][tid];
            writes.push_back(writeEvent);
          }
          break;
        }
        case Event::Type::Free: {
          auto const freeEvent = llvm::cast<FreeEvent>(event.get());
          auto const &ptsTo = freeEvent->getFreedMemory();
          for (auto const obj : ptsTo) {
            auto &frees = objFrees[getObjId(obj)][tid];
            frees.push_back(freeEvent);
          }
        }
        default:
          // Do Nothing
          break;
      }
    }
  }
}

bool SharedMemory::isShared(ObjID id) const {
  auto const nWriters = numThreadsWrite(id);
  auto const nFrees = numThreadsFree(id);
  auto const nReaders = numThreadsRead(id);

  // Only a few cases where when an obj is shared
  // 1a. multiple threads writing
  // 1b. multiple threads freeing
  // 2a. one thread writing, multiple reading
  // 2b. one thread freeing, multiple reading
  // 3a. one write, one read (from different threads)
  // 3b. one free, one read (from different threads)
  // 3c. one free, one write (from different threads)

  // 1. Multiple threads writing/freeing is potential race
  if (nWriters > 1 || nFrees > 1) {
    return true;
  }

  // 2. Atleast 1 write/free and multiple reads means potential race
  if ((nWriters == 1 || nFrees == 1) && nReaders > 1) {
    return true;
  }

  auto const writingThread = (nWriters == 1) ? std::optional{objWrites.at(id).begin()->first} : std::nullopt;
  auto const readingThread = (nReaders == 1) ? std::optional{objReads.at(id).begin()->first} : std::nullopt;
  auto const freeingThread = (nFrees == 1) ? std::optional{objFrees.at(id).begin()->first} : std::nullopt;

  // 3a. A write and a read from different threads is potential race
  if (writingThread && readingThread && writingThread.value() != readingThread.value()) {
    return true;
  }

  // 3b. A free and a read from different threads is a potential race
  if (freeingThread && readingThread && freeingThread.value() != readingThread.value()) {
    return true;
  }

  // 3c. A free and a write from different threads is a potential race
  if (freeingThread && writingThread && freeingThread.value() != writingThread.value()) {
    return true;
  }

  return false;
}

std::vector<const pta::ObjTy *> SharedMemory::getSharedObjects() const {
  std::vector<const pta::ObjTy *> sharedObjects;
  for (auto const &[obj, objID] : objIDs) {
    if (isShared(objID)) {
      sharedObjects.push_back(obj);
    }
  }
  return sharedObjects;
}
size_t SharedMemory::numThreadsWrite(ObjID id) const {
  auto it = objWrites.find(id);
  if (it == objWrites.end()) return 0;
  return it->second.size();
}
size_t SharedMemory::numThreadsRead(SharedMemory::ObjID id) const {
  auto it = objReads.find(id);
  if (it == objReads.end()) return 0;
  return it->second.size();
}
size_t SharedMemory::numThreadsFree(ObjID id) const {
  auto it = objFrees.find(id);
  if (it == objFrees.end()) return 0;
  return it->second.size();
}
std::map<ThreadID, std::vector<const ReadEvent *>> SharedMemory::getThreadedReads(const pta::ObjTy *obj) const {
  auto id = objIDs.find(obj);
  if (id == objIDs.end()) return {};

  // cppcheck-suppress stlIfFind
  if (auto it = objReads.find(id->second); it != objReads.end()) {
    return it->second;
  }

  return {};
}
std::map<ThreadID, std::vector<const WriteEvent *>> SharedMemory::getThreadedWrites(const pta::ObjTy *obj) const {
  auto id = objIDs.find(obj);
  if (id == objIDs.end()) return {};

  // cppcheck-suppress stlIfFind
  if (auto it = objWrites.find(id->second); it != objWrites.end()) {
    return it->second;
  }

  return {};
}
std::map<ThreadID, std::vector<const FreeEvent *>> SharedMemory::getThreadedFrees(const pta::ObjTy *obj) const {
  auto id = objIDs.find(obj);
  if (id == objIDs.end()) return {};

  // cppcheck-suppress stlIfFind
  if (auto it = objFrees.find(id->second); it != objFrees.end()) {
    return it->second;
  }

  return {};
}
