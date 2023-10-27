#include <assert.h>
#include <stdlib.h>
// #include <cmath>
#include <math.h>
#include <iostream>

using namespace std;

unsigned long long split;
unsigned long long ncpus;
unsigned long long r;
unsigned long long k;
unsigned long long pixels = 0;

pthread_mutex_t mutex;

void* thread_sum(void* threadid) {

  unsigned long long* tid = (unsigned long long*) threadid;
  // unsigned long long start = split * (*tid);
  // unsigned long long end = *tid == ncpus-1 ? r : split*(*tid+1);
  unsigned long long split_sum = 0;
  for (unsigned long long i = *tid; i < r; i+= ncpus){
    // split_sum += (unsigned long long)(ceil(sqrtl((r+i) * (r-i))))%k;
    // split_sum += ceil(sqrtl((r+i) * (r-i)));
    split_sum += ceil(sqrtl((r*r) - (i*i)));
  }
  // pthread_mutex_lock(&mutex);
  pixels += split_sum;
  if (pixels >= k)
    pixels %= k;
  // pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(int argc, char** argv) {

  r = atoll(argv[1]);
  k = atoll(argv[2]);

  // getting # of cpus
  cpu_set_t cpuset;
  sched_getaffinity(0, sizeof(cpuset), &cpuset);
  ncpus = CPU_COUNT(&cpuset);

  pthread_t threads[(int)ncpus]; // this is for the thread pool
  unsigned long long ID[(int)ncpus]; // this is for the data that pass into the thread

  split = r / ncpus;
  // cout << "split = " << split << endl;

  for (unsigned long long i = 0; i < ncpus; i++){
    ID[i] = i;
    pthread_create(&threads[i], NULL, thread_sum, (void*)&ID[i]);
  }

  // void* retval;
  for (unsigned long long i = 0; i < ncpus; i++){
    // pthread_join(threads[i], &retval);
    pthread_join(threads[i], NULL);
    // pixels += (unsigned long long)retval;
    // pixels %= k;
  }
  // maybe put pixel add into thread ?

  cout << ((pixels%k)*4)%k << endl;
  pthread_mutex_destroy(&mutex);
  pthread_exit(NULL);
  return 0;
}

