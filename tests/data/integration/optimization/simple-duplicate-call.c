#include <omp.h>
#include <stdio.h>

void write_val(int *dest, int val) { *dest = val; }

int main() {
  int counter = 0;
  int tid = 1;

#pragma omp parallel
  {
    write_val(&counter, tid);
    write_val(&counter, tid);  // this should be skipped
  }

  printf("%d\n", counter);
}