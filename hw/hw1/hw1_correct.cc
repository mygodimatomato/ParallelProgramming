#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <algorithm>

#define EVEN_PHASE 0
#define ODD_PHASE 1

using namespace std;

float *tmp;

// nothing wrong
void printArray(float arr[], int n, int rank, bool sort)
{
  if (sort) {
    cout << "Rank : " << rank << " , sort"  << endl;
  } else {
    cout << "Rank : " << rank << " , unsort"  << endl;
  }
  int i;
  for (i = 0; i < n; i++)
    cout << arr[i] << " ";
  cout << endl;
}

void my_merge(float *arr1, float *arr2, int arr1_size, int arr2_size, bool isHigh)
{
  int iter_1, iter_2, iter_3, i;
  if (isHigh){
    if (arr1[0] >= arr2[arr2_size - 1])
      return;
    iter_1 = arr1_size - 1;
    iter_2 = arr2_size - 1;
    iter_3 = arr1_size - 1;
    while(iter_3 >= 0){
      if (arr1[iter_1] > arr2[iter_2]){
        tmp[iter_3--] = arr1[iter_1--];
      } else {
        tmp[iter_3--] = arr2[iter_2--];
      }
    }
  } else {
    if (arr1[arr1_size-1] <= arr2[0])
      return;
    iter_1 = iter_2 = iter_3 = 0;
    while(iter_3 < arr1_size){
      if (arr1[iter_1] < arr2[iter_2]){
        tmp[iter_3++] = arr1[iter_1++];
      } else {
        tmp[iter_3++] = arr2[iter_2++];
      }
    }
  }
  for (i = 0; i < arr1_size; i++){
    arr1[i] = tmp[i];
  }
}

int main (int argc, char **argv) 
{
  int array_size = atoi(argv[1]);
  char *input_filename = argv[2];
  char *output_filename = argv[3];

  int rank, size;
  double start_time, finish_time, loc_elapsed, elapsed;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int split, remainder;
  // need to fix
  remainder = array_size % size;
  split = remainder > 0 ? array_size / size + 1 : array_size / size;

  int start = split * rank < array_size ? split * rank : array_size;
  int end = rank == size-1 ? array_size : split*(rank+1);
  if (end > array_size){
    end = array_size;
  }
  int local_size = end - start;
  
  // for checking
  // cout << "rank " << rank << ", local size is " << local_size << ", start is " << start << " , end is " << end<< endl;

  MPI_File input_file, output_file;

  // need to change to malloc
  float * local_data = new float[local_size];
  float * buffer = new float[split];

  MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
  MPI_File_read_at_all(input_file, sizeof(float) * rank * split, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&input_file);

  if (local_size != 0){
    // qsort(local_data, local_size, sizeof(float), compare);
    sort(local_data, local_data + local_size);

    // get neighbor && neighbor size
    int left, right, left_count, right_count;
    bool last_node = false;
    left = rank - 1 >= 0 ? (rank - 1) : 0;
    left_count = left == rank ? 0 : split;
    right = end == array_size ? rank : rank + 1;
    if (end == array_size) last_node = true;
    if (array_size - end < split * 2){
      right_count = array_size - end;
    } else {
      right_count = split;
    }
    
    // for checking
    // printArray(local_data, local_size, rank, true);
    // cout << "rank " << rank << " , left is " << left << " , right is " << right << " , left count is " << left_count << " , right count is " << right_count << endl;
    tmp = new float[local_size];
    for (int phase = 0; phase < size; phase+=2) {
      // cout << "in phase " << phase << endl;
        if (rank % 2 == 0) { // rank is even
          if (!last_node){
            MPI_Sendrecv(&local_data[local_size-1], 1, MPI_FLOAT, rank+1, EVEN_PHASE, buffer, 1, MPI_FLOAT, rank+1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
            if (local_data[local_size - 1] > buffer[0]){
              MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank+1, EVEN_PHASE, buffer, right_count, MPI_FLOAT, rank+1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              my_merge(local_data, buffer, local_size, right_count, false);
            }
          }
        } else { // rank is odd
          if (left != rank) {
            MPI_Sendrecv(local_data, 1, MPI_FLOAT, rank-1, EVEN_PHASE, buffer, 1, MPI_FLOAT, rank-1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           
            if(local_data[0] < buffer[0]){
              MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank-1, EVEN_PHASE, buffer, left_count, MPI_FLOAT, rank-1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              my_merge(local_data, buffer, local_size, left_count, true);
            } 
          }
        } 
        if (rank % 2 == 1) { // rank is odd
          if (!last_node){
            MPI_Sendrecv(&local_data[local_size-1], 1,MPI_FLOAT, rank+1, ODD_PHASE, buffer, 1, MPI_FLOAT, rank+1, ODD_PHASE, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         
            if (local_data[local_size-1] > buffer[0]){
              MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank+1, ODD_PHASE, buffer, right_count, MPI_FLOAT, rank+1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              my_merge(local_data, buffer, local_size, right_count, false);
            }
          }
        } else {
          if (left != rank) {
            MPI_Sendrecv(local_data, 1, MPI_FLOAT, rank-1, ODD_PHASE, buffer, 1, MPI_FLOAT, rank-1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (local_data[0] < buffer[0]){
              MPI_Sendrecv(local_data, local_size, MPI_FLOAT, rank-1, ODD_PHASE, buffer, left_count, MPI_FLOAT, rank-1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              my_merge(local_data, buffer, local_size, left_count, true);
            }
          }
        }
    }

  } 

  // global swapping
  MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at_all(output_file, sizeof(float) * rank * split, local_data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&output_file);

  MPI_Finalize();
  return 0;
}