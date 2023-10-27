#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
using namespace std;

int main(int argc, char** argv) {
  unsigned long long r = atoll(argv[1]);
  unsigned long long k = atoll(argv[2]);
  unsigned long long pixels = 0;
  cpu_set_t cpuset;
  sched_getaffinity(0, sizeof(cpuset), &cpuset);
  int ncpus = CPU_COUNT(&cpuset);
  omp_set_num_threads(ncpus);

  // #pragma omp parallel 
  for (unsigned long long i = 0; i < ncpus; i++){
      #pragma omp parallel for schedule(guided, ncpus) reduction(+:pixels)
      for (unsigned long long j = i; j < r; j += ncpus){
        pixels += ceil(sqrtl((r+j) * (r-j)));
      } 
      pixels %= k;
  }

  cout << (4 * pixels%k) % k << endl;
  return 0;
}
