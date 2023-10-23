#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id

    if (rank == 0) {
      int message = 123;
      MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }  

    if (rank == 1) {
      int message;
      MPI_Status status;
      MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

      printf("Rank 1 received message: %d\n", message);
    }

    MPI_Finalize();
    return 0;
}