#include <pthread.h>
#include <stdlib.h>

int* global;

void* write(void* a) {
  (*global)++;
  return NULL;
}
void* freeme(void* a) {
  free(global);
  return NULL;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, write, NULL);
  pthread_create(&t2, NULL, freeme, NULL);
}