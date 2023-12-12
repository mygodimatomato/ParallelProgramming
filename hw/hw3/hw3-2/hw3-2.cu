#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define MY_INF 1073741823
#define BLOCK_SIZE 8

int V, E;
int matrix_size;
int *adjacency_matrix;
size_t result;
__constant__ int d_matrix_size;

int ceil(int a, int b) { return (a + b - 1) / b; }

void input(char* infile) {
  FILE *input_file = fopen(infile, "rb");
  result = fread(&V, sizeof(int), 1, input_file);
  result = fread(&E, sizeof(int), 1, input_file);
  
  matrix_size = ((V / BLOCK_SIZE) + 1 ) * BLOCK_SIZE; // matrix size must be multiple of BLOCK_SIZE

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
    result = fread(edge, sizeof(int), 3, input_file);
    adjacency_matrix[edge[0] * matrix_size + edge[1]] = edge[2];
  }
  fclose(input_file);
}

void output(char* outFileName){
  FILE* outfile = fopen(outFileName, "w");
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++) {
      if (adjacency_matrix[i * matrix_size + j] >= MY_INF)
        adjacency_matrix[i*matrix_size + j] = MY_INF;
      adjacency_matrix[i * V + j] = adjacency_matrix[i * matrix_size + j];
    }
  }

  fwrite(adjacency_matrix, sizeof(int), V * V, outfile);
  fclose(outfile);
}

__global__ void phase1(int* d_dist, int r){
  // Get index
  int j = threadIdx.x;
  int i = threadIdx.y;
  
  // Copy data from global memory to shared memory
  __shared__ int shared_memory[BLOCK_SIZE * BLOCK_SIZE];
  shared_memory[i * BLOCK_SIZE + j] = d_dist[(i+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  __syncthreads();


  // D(i,j) = min(D(i,j), D(i,k)+D(k,j))
  #pragma unroll 8 // mygodimatomato: should changed by BLOCK_SIZE
  for(int k = 0; k < BLOCK_SIZE; k++){
    int i_2_k = shared_memory[i * BLOCK_SIZE + k];
    int k_2_j = shared_memory[k * BLOCK_SIZE + j];

    if (i_2_k + k_2_j < shared_memory[i * BLOCK_SIZE + j])
      shared_memory[i * BLOCK_SIZE + j] = i_2_k + k_2_j;
  }

  // writing data back to global memory
  d_dist[(i+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[i * BLOCK_SIZE + j];
}

__global__ void phase2(int* d_dist, int r){
  int j = threadIdx.x; // col index
  int i = threadIdx.y; // row index
  int i_offset = 0;
  int j_offset = 0;
  int i_2_k, k_2_j;

  // 0 : row block, 1 : col block, 2 : center block
  __shared__ int shared_memory[3 * BLOCK_SIZE * BLOCK_SIZE];

  shared_memory[i * BLOCK_SIZE + j + (BLOCK_SIZE*BLOCK_SIZE)*2] = d_dist[(i+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];

  if (blockIdx.x == 1) { // col 
    i_offset = BLOCK_SIZE * blockIdx.y; 
    j_offset = BLOCK_SIZE * r;
  } else { // row
    i_offset = BLOCK_SIZE * r;
    j_offset = BLOCK_SIZE * blockIdx.y;
  }

  shared_memory[i * BLOCK_SIZE + j + (BLOCK_SIZE * BLOCK_SIZE) * blockIdx.x] = d_dist[(i+i_offset) * d_matrix_size + j + j_offset];
  __syncthreads();

  #pragma unroll 8 // mygodimatomato: should changed by BLOCK_SIZE
  for (int k = 0; k < BLOCK_SIZE; k++) {
    if (blockIdx.x == 0){
      i_2_k = shared_memory[i * BLOCK_SIZE + k + (BLOCK_SIZE*BLOCK_SIZE)*2];
      k_2_j = shared_memory[k * BLOCK_SIZE + j];
    } else {
      i_2_k = shared_memory[i * BLOCK_SIZE + k + (BLOCK_SIZE * BLOCK_SIZE)];
      k_2_j = shared_memory[k * BLOCK_SIZE + j + (BLOCK_SIZE*BLOCK_SIZE)*2];
    }

    if (shared_memory[i * BLOCK_SIZE + j + (BLOCK_SIZE * BLOCK_SIZE) * blockIdx.x] > i_2_k + k_2_j)
      shared_memory[i * BLOCK_SIZE + j + (BLOCK_SIZE * BLOCK_SIZE) * blockIdx.x] = i_2_k + k_2_j;
  }
    
  d_dist[(i+i_offset) * d_matrix_size + j + j_offset] = shared_memory[i * BLOCK_SIZE + j + (BLOCK_SIZE * BLOCK_SIZE) * blockIdx.x];
}

__global__ void phase3(int* d_dist, int r, int* d_check){
  int j = threadIdx.x; // col index
  int i = threadIdx.y; // row index
  int i_offset = BLOCK_SIZE * blockIdx.x;
  int j_offset = BLOCK_SIZE * blockIdx.y;

  __shared__ int shared_memory[3*BLOCK_SIZE*BLOCK_SIZE];

  // d_dist[(i + i_offset) * d_matrix_size + (j + j_offset)] = blockIdx.y + blockIdx.x;

  shared_memory[i * BLOCK_SIZE + j] = d_dist[(i + i_offset) * d_matrix_size + (j + j_offset)];
  shared_memory[BLOCK_SIZE * (i + BLOCK_SIZE) + j] = d_dist[(i + i_offset) * d_matrix_size + j + r * BLOCK_SIZE];
  shared_memory[BLOCK_SIZE * (i + BLOCK_SIZE*2) + j] = d_dist[(i + r * BLOCK_SIZE) * d_matrix_size + (j + j_offset)];
  __syncthreads();

  #pragma unroll 8
  for(int k = 0; k < BLOCK_SIZE; k++){
    int i_2_k = shared_memory[BLOCK_SIZE * (i + BLOCK_SIZE) + k];
    int k_2_j = shared_memory[BLOCK_SIZE * (k + BLOCK_SIZE*2) + j];
    if (shared_memory[i * BLOCK_SIZE + j] > i_2_k + k_2_j)
      shared_memory[i * BLOCK_SIZE + j] = i_2_k + k_2_j;
  }

  d_dist[(i + i_offset) * d_matrix_size + (j + j_offset)] = shared_memory[i * BLOCK_SIZE + j];
}


void block_FW(int* d_dist, int* d_check) {
  int round = matrix_size/BLOCK_SIZE;
  dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 phase2_num_blocks(2, round); // one for col, one for row, one block will be redundant, but for the whole performance it doesn't really matters
  dim3 phase3_num_blocks(round, round);

  // round = 1; // mygodimatomato: for checking
  for (int r = 0; r < round; r++) {
    phase1<<<1, num_threads, BLOCK_SIZE * BLOCK_SIZE * sizeof(int)>>>(d_dist, r);
    phase2<<<phase2_num_blocks, num_threads, 3 * BLOCK_SIZE * BLOCK_SIZE * sizeof(int)>>>(d_dist, r);
    phase3<<<phase3_num_blocks, num_threads, 3 * BLOCK_SIZE * BLOCK_SIZE * sizeof(int)>>>(d_dist, r, d_check);
  }
}


int main(int argc, char* argv[]) {
  // Read input from input file
  input(argv[1]);

  // Allocate the memory for the matrix in GPU
  int *d_dist;
  cudaMalloc((void**)&d_dist, sizeof(int) * matrix_size * matrix_size);
  cudaMemcpy(d_dist, adjacency_matrix, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_matrix_size, &matrix_size, sizeof(int));
  
  // mygodimatomato: for checking
  int *h_check;
  int *d_check;
  h_check = (int*)malloc(sizeof(int)*BLOCK_SIZE *BLOCK_SIZE);
  cudaMalloc((void**)&d_check, sizeof(int)*BLOCK_SIZE *BLOCK_SIZE);
  
  // Start executing the block Floyed-Warshall
  block_FW(d_dist, d_check);

  // Copy the outcome back to the adjacency_matrix 
  cudaMemcpy(adjacency_matrix, d_dist, sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_check, d_check, sizeof(int)*BLOCK_SIZE*BLOCK_SIZE, cudaMemcpyDeviceToHost);
  
  output(argv[2]);

  // mygodimatomato : for checking
  int k = 0;
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++){
      if(adjacency_matrix[k] == MY_INF)
        printf(" INF ");
      else
        printf("%4d ", adjacency_matrix[k]);
      k++;
    } printf("\n");
  } printf("\n");

  // for(int i = 0; i <BLOCK_SIZE;i++){
  //   for(int j = 0; j <BLOCK_SIZE; j++){
  //     if(h_check[i*BLOCK_SIZE+j] == MY_INF)
  //       printf(" INF ");
  //     else
  //       printf("%4d ", h_check[i*BLOCK_SIZE+j]);
  //   }printf("\n");
  // }printf("\n");

  // Write output to output file
  return 0;
}