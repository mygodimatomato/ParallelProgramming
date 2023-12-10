#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define MY_INF 1073741823
#define BLOCK_SIZE 8

int V, E;
int matrix_size;
int *adjacency_matrix;

int ceil(int a, int b) { return (a + b - 1) / b; }

void input(char* infile) {
  FILE *input_file = fopen(infile, "rb");
  fread(&V, sizeof(int), 1, input_file);
  fread(&E, sizeof(int), 1, input_file);
  
  matrix_size = ((V / BLOCK_SIZE) + 1 ) * BLOCK_SIZE; // matrix size must be multiple of BLOCK_SIZE

  printf("%d, %d, %d\n", V, E, matrix_size); // mygodimatomato : for checking
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
  fclose(input_file);
}

void output(char* outFileName){
  FILE* outfile = fopen(outFileName, "w");
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      if (adjacency_matrix[i * matrix_size + j] > MY_INF)
        adjacency_matrix[i*matrix_size + j] = MY_INF;
      fwrite(&adjacency_matrix[i * matrix_size + j], sizeof(int), 1, outfile);
    }
  }
  fclose(outfile);
}

__global__ void phase1(int* d_dist, int r){

}

__global__ void phase2(){

}

__global__ void phase3(){

}


void block_FW() {
  int round = ceil(V, BLOCK_SIZE);

  
  for (int r = 0; r < round; r++) {
    phase1<<<>>>()
    phase2<<<>>>
    phase3<<<>>>
  }
}

// __global__ void phase1(){

// }


  



  // Allocate memory for GPU

  // start executing block floyed warshall algorithm
  // for (int r = 0; r < round; r++) {
  //   phase1<<<>>>();
  //   phase2<<<>>>();
  //   phase3<<<>>>();
  // }

int main(int argc, char* argv[]) {
  input(argv[1]);

  // mygodimatomato : for checking
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++){
      if(adjacency_matrix[i * matrix_size + j] >= MY_INF)
        printf("INF ");
      else 
        printf("%3d ", adjacency_matrix[i * matrix_size + j]);
    } printf("\n");
  }

  block_FW();
  output(argv[2]);
  return 0;
}