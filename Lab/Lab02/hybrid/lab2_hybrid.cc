#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[]) {
  cpu_set_t cpuset;
  sched_getaffinity(0, sizeof(cpuset), &cpuset);
  int ncpus = CPU_COUNT(&cpuset);
  omp_set_num_threads(ncpus);
  
  unsigned long long r = atoll(argv[1]);
  unsigned long long k = atoll(argv[2]);
  unsigned long long pixels = 0;

  int rank, size;
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

    
  unsigned long long split_sum = 0;

  #pragma omp parallel for schedule(guided, ncpus) reduction(+:split_sum)
  for(unsigned long long i = rank; i < r; i+=size){
    split_sum += ceil(sqrtl((r+i) * (r-i)));
    // split_sum += ceil(sqrtl((r*r) - (i*i)));
  }
  if (split_sum >= k)
    split_sum %= k;

  MPI_Reduce(&split_sum, &pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
    cout << ((pixels%k)*4)%k << endl;
  MPI_Finalize();

  return 0;
}