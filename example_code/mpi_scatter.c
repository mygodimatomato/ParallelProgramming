#include <mpi.h>
#include <stdio.h> 

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv); 

  // Get the rank of the current process.
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create a buffer to store the messages.
  int buffer[nprocs];

  // Scatter the data from the root process to all other processes.
  if (rank == 0) {
    for (int i = 0; i < nprocs; i++) {
      buffer[i] = i;
    }

    // Scatter the data to all other processes.
    MPI_Scatter(buffer, 1, MPI_INT, buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    // Receive the data from the root process.
    MPI_Scatter(NULL, 0, MPI_INT, buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // All processes print the data they received.
  for (int i = 0; i < nprocs; i++) {
    printf("Rank %d received data: %d\n", rank, buffer[i]);
  }

  MPI_Finalize();

  return 0;
}