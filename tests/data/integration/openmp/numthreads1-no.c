#include <stdio.h>

int main() {
  int sum = 0;

#pragma omp parallel num_threads(1)
  { sum++; }

  printf("%d\n", sum);
}