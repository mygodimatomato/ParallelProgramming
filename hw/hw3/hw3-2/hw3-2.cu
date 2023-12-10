#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define MY_INF 1073741823
#define BLOCK_SIZE 8

int V, E;
int *adjacency_matrix;

// __global__ void phase1(){

// }

// __global__ void phase2(){

// }

// __global__ void phase3(){

// }

int main(int argc, char *argv[]) {

  // process input 
  FILE *input_file = fopen(argv[1], "r");
  fread(&V, sizeof(int), 1, input_file);
  fread(&E, sizeof(int), 1, input_file);

  int matrix_size = ((V / BLOCK_SIZE) + 1 ) * BLOCK_SIZE; // matrix size must be multiple of BLOCK_SIZE
  adjacency_matrix = (int *)malloc(matrix_size * matrix_size * sizeof(int)); // set the matrix to 1D array for easier data transfer

  cudaMallocHost((void **)&adjacency_matrix, matrix_size * matrix_size * sizeof(int)); // pinned memory for faster data transfer

  for(int i = 0; i < matrix_size; i++) {
    for(int j = 0; j < matrix_size; j++) {
      if (i == j)
        adjacency_matrix[i * matrix_size + j] = 0;
      else 
        adjacency_matrix[i * matrix_size + j] = MY_INF;
    }
  }

  int edge[3];
  for(int i = 0; i < E; i++) {
    fread(edge, sizeof(int), 3, input_file);
    adjacency_matrix[edge[0] * matrix_size + edge[1]] = edge[2];
  }
  // finish processing input

  
  // mygodimatomato : for checking
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++)
      printf("%d ", adjacency_matrix[i * matrix_size + j]);
    printf("\n");
  }


  // Allocate memory for GPU

  // start executing block floyed warshall algorithm
  // for (int r = 0; r < round; r++) {
  //   phase1<<<>>>();
  //   phase2<<<>>>();
  //   phase3<<<>>>();
  // }


  // process output
  FILE *output_file = fopen(argv[2], "w");
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++)
      fwrite(&adjacency_matrix[i * matrix_size + j], sizeof(int), 1, output_file);
  }
  // finish processing output
  

  fclose(input_file);
  fclose(output_file);
  free(adjacency_matrix);

  return 0;
}