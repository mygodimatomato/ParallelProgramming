#include <math.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
  
  unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long global_pixels;
  MPI_Init(&argc, &argv);
	

	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unsigned long long local_pixels = 0;
	unsigned long long start = rank * r / size;
	unsigned long long end = (rank + 1) * r / size;
  double starttime, endtime;

	for (unsigned long long x = start; x < end; x++) {
		local_pixels += ceil(sqrtl((r+x) * (r-x)));
	}
  local_pixels %= k;

  MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
	  printf("%llu\n", (4 * global_pixels) % k);
  }
	MPI_Finalize();
}