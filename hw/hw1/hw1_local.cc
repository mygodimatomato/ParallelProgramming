#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <iostream>

int compare_floats(const void *a, const void *b)
{
  float fa = *(const float *) a;
  float fb = *(const float *) b;
  return (fa > fb) - (fa < fb);
}

void printArray(float arr[], int n, int rank, bool sort)
{
  // if (sort) {
  //   std::cout << "Rank : " << rank << " , sort"  << std::endl;
  // } else {
  //   std::cout << "Rank : " << rank << " , unsort"  << std::endl;
  // }
  std::cout << "Rank : " << rank << std::endl;
  for (int i = 0; i < n; i++)
    std::cout << arr[i] << " ";
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  int len = atoi(argv[1]);
  char *input_filename = argv[2];
  char *output_filename = argv[3];

  // start MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_File input_file, output_file;

  // deciding the number of elements per process and the start offset
  int local_len = len/size;
  int partner_len = len/size; // the left neighbor's array length
  int offset = local_len * rank;

  if (rank < len % size)
    local_len++;
  if (rank - 1 < len % size)
    offset += rank;
  else if (rank - 1 >= len % size)
    offset += len % size;
  if (rank > 0 && rank - 1 < len % size)
    partner_len++;

  // allocating memory for local array
  // float *local_array = (int *) malloc(local_len * sizeof(float));
  float local_array[local_len];
  float partner_array[partner_len];

  // reading from input file
  MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
  MPI_File_read_at(input_file, sizeof(float) * offset, local_array, local_len, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&input_file);

  // checking the local array value
  // printArray(local_array, local_len, rank, false);

  for (int phase = 0; phase < size; phase++) {
    if (size == 1){
      qsort(local_array, local_len, sizeof(float), compare_floats);
      break;
    }
    // sorting local array
    if ((phase + rank) % 2 == 0 && rank < size - 1) { // odd + odd || even + even 
      
    } else if ((phase + rank) % 2 == 1 && rank > 0) { // odd + even || even + odd
    } 
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // writing to output file
  MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);
  MPI_File_write_at(output_file, sizeof(float) * offset, local_array, local_len, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&output_file);

  MPI_Finalize();
  return 0;
}