# If you set LLVM_INSTALL with CMake, these are set to clang 10.0.1 automatically by cmake
# CC=clang
# CXX=clang++
FLAGS=-g -O1 -mllvm -disable-llvm-optzns -fno-discard-value-names -fopenmp -S -emit-llvm

# Check that we are compiling with clang 10.0.1
ifeq ("", "$(shell $(CC) -dumpversion | grep -E '10.0.1')")
$(error "clang 10.0.1 is required to build the test files")
endif
ifeq ("", "$(shell $(CXX) -dumpversion | grep -E '10.0.1')")
$(error "clang 10.0.1 is required to build the test files")
endif

.PHONY: unit integration

all: unit integration
	
%.ll: %.c
	$(CC) $(FLAGS) $< -o $@

%.ll: %.cpp
	$(CXX) $(FLAGS) $< -o $@

unit: unit/Analysis/simpleloop.ll

integration: dataracebench pthreadrace pthread openmp

DRB_C_SRC=$(wildcard integration/dataracebench/*.c)
DRB_C_OUT=$(DRB_C_SRC:.c=.ll)
DRB_CXX_SRC=$(wildcard integration/dataracebench/*.cpp)
DRB_CXX_OUT=$(DRB_CXX_SRC:.cpp=.ll)
dataracebench: $(DRB_C_OUT) $(DRB_CXX_OUT)

PTHREADRACE_C_SRC=$(wildcard integration/pthreadrace/*.c)
PTHREADRACE_CXX_SRC=$(wildcard integration/pthreadrace/*.cpp)
PTHREADRACE_C_OUT=$(PTHREADRACE_C_SRC:.c=.ll)
PTHREADRACE_CXX_OUT=$(PTHREADRACE_CXX_SRC:.cpp=.ll)
pthreadrace: $(PTHREADRACE_C_OUT) $(PTHREADRACE_CXX_OUT)

PTHREAD_C_SRC=$(wildcard integration/pthread/*.c)
PTHREAD_CXX_SRC=$(wildcard integration/pthread/*.cpp)
PTHREAD_C_OUT=$(PTHREAD_C_SRC:.c=.ll)
PTHREAD_CXX_OUT=$(PTHREAD_CXX_SRC:.cpp=.ll)
pthread: $(PTHREAD_C_OUT) $(PTHREAD_CXX_OUT)

OMP_SRC=$(wildcard integration/openmp/*.c)
OMP_OUT=$(OMP_SRC:.c=.ll)
openmp: $(OMP_OUT)

clean:
	@rm -f $(PTHREAD_C_OUT) $(PTHREAD_CXX_OUT) $(PTHREADRACE_C_OUT) $(PTHREADRACE_CXX_OUT) $(DRB_C_OUT) $(DRB_CXX_OUT) $(OMP_OUT)
	