#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int* global;

void* write(void* a) {
  (*global)++;
  return NULL;
}
void* freeme(void* a) {
  delete global;
  return NULL;
}

int main() {
  global = new int;
  *global = 0;

  pthread_t t1, t2;
  pthread_create(&t1, NULL, write, NULL);

  pthread_join(t1, NULL);

  printf("%d", *global);
  pthread_create(&t2, NULL, freeme, NULL);
}