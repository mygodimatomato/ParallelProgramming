#include <mpi.h>
#include <stdio.h> 

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv); 

  // Get the rank of the current process.
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create a request object.
  MPI_Request request;

  // If the current process is rank 0, send a message to rank 1.
  if (rank == 0) {
    int message = 42;
    MPI_Isend(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);

    printf("halo halo halo\n");

    MPI_Wait(&request, MPI_STATUS_IGNORE);
  } 

  // If the current process is rank 1, receive a message from rank 0.
  else if (rank == 1) {
    int message;
    MPI_Irecv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    printf("Process 1 received message: %d\n", message);
  }
}