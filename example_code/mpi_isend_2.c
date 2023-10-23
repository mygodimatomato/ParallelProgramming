#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // Get the rank and number of processes.
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create a buffer to store the messages.
  int buffer[nprocs];

  // Send and receive messages in a ring.
  for (int i = 0; i < nprocs; i++) {
    // Send a message to the next process in the ring.
    int next_rank = (rank + 1) % nprocs;
    MPI_Isend(&rank, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD, NULL);

    // Receive a message from the previous process in the ring.
    int prev_rank = (rank - 1 + nprocs) % nprocs;
    MPI_Status status;
    MPI_Irecv(&buffer[prev_rank], 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, &status);

    // Wait for the receive operation to complete.
    MPI_Wait(&status, MPI_STATUS_IGNORE);
  }

  // Print the received messages.
  for (int i = 0; i < nprocs; i++) {
    printf("Rank %d received message from rank %d: %d\n", rank, i, buffer[i]);
  }

  MPI_Finalize();

  return 0;
}