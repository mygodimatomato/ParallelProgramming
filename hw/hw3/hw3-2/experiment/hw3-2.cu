#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define MY_INF 1073741823
#define BLOCK_SIZE 64

int V, E;
int matrix_size;
int *adjacency_matrix;
size_t result;
__constant__ int d_matrix_size;

void input(char* infile) {
  FILE *input_file = fopen(infile, "rb");
  result = fread(&V, sizeof(int), 1, input_file);
  result = fread(&E, sizeof(int), 1, input_file);
  
  // int remainder = V%128;
  // matrix_size = (remainder == 0) ? V : V + (128-remainder);
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
  int i = threadIdx.y*4;
  
  // Copy data from global memory to shared memory
  __shared__ int shared_memory[BLOCK_SIZE * BLOCK_SIZE];
  shared_memory[(i+0) * BLOCK_SIZE + j] = d_dist[(i+0+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+1) * BLOCK_SIZE + j] = d_dist[(i+1+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+2) * BLOCK_SIZE + j] = d_dist[(i+2+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+3) * BLOCK_SIZE + j] = d_dist[(i+3+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  __syncthreads();


  // D(i,j) = min(D(i,j), D(i,k)+D(k,j))
  #pragma unroll  // mygodimatomato: should changed by BLOCK_SIZE
  for(int k = 0; k < BLOCK_SIZE; k++){
    shared_memory[(i+0)*BLOCK_SIZE+j] = min(shared_memory[(i+0)*BLOCK_SIZE+j], shared_memory[(i+0) * BLOCK_SIZE + k] +  shared_memory[k * BLOCK_SIZE + j]);
    shared_memory[(i+1)*BLOCK_SIZE+j] = min(shared_memory[(i+1)*BLOCK_SIZE+j], shared_memory[(i+1) * BLOCK_SIZE + k] + shared_memory[k * BLOCK_SIZE + j]);
    shared_memory[(i+2)*BLOCK_SIZE+j] = min(shared_memory[(i+2)*BLOCK_SIZE+j], shared_memory[(i+2) * BLOCK_SIZE + k] + shared_memory[k * BLOCK_SIZE + j]);
    shared_memory[(i+3)*BLOCK_SIZE+j] = min(shared_memory[(i+3)*BLOCK_SIZE+j], shared_memory[(i+3) * BLOCK_SIZE + k]+ shared_memory[k * BLOCK_SIZE + j]);
  }
    __syncthreads();
  // writing data back to global memory
  d_dist[(i+0+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+0) * BLOCK_SIZE + j];
  d_dist[(i+1+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+1) * BLOCK_SIZE + j];
  d_dist[(i+2+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+2) * BLOCK_SIZE + j];
  d_dist[(i+3+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+3) * BLOCK_SIZE + j];
}

__global__ void phase2(int* d_dist, int r){
  // if (blockIdx.y == 0) return;
  int j = threadIdx.x; // col index
  int i = threadIdx.y*4; // row index
  int i_offset = 0;
  int j_offset = 0;
  int i_2_k_0, i_2_k_1, i_2_k_2, i_2_k_3;
  int k_2_j;

  // 0 : row block, 1 : col block, 2 : center block
  __shared__ int shared_memory[3 * BLOCK_SIZE * BLOCK_SIZE];

  shared_memory[(i+0+BLOCK_SIZE*2) * BLOCK_SIZE + j] = d_dist[(i+0+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+1+BLOCK_SIZE*2) * BLOCK_SIZE + j] = d_dist[(i+1+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+2+BLOCK_SIZE*2) * BLOCK_SIZE + j] = d_dist[(i+2+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];
  shared_memory[(i+3+BLOCK_SIZE*2) * BLOCK_SIZE + j] = d_dist[(i+3+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)];


  if (blockIdx.x == 1) { // col 
    i_offset = BLOCK_SIZE * blockIdx.y; 
    j_offset = BLOCK_SIZE * r;
  } else { // row
    i_offset = BLOCK_SIZE * r;
    j_offset = BLOCK_SIZE * blockIdx.y;
  }

  shared_memory[(i+0+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = d_dist[(i+0+i_offset) * d_matrix_size + j + j_offset];
  shared_memory[(i+1+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = d_dist[(i+1+i_offset) * d_matrix_size + j + j_offset];
  shared_memory[(i+2+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = d_dist[(i+2+i_offset) * d_matrix_size + j + j_offset];
  shared_memory[(i+3+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = d_dist[(i+3+i_offset) * d_matrix_size + j + j_offset];

  __syncthreads();

  #pragma unroll // mygodimatomato: should changed by BLOCK_SIZE
  for (int k = 0; k < BLOCK_SIZE; k++) {
    if (blockIdx.x == 0){
      i_2_k_0 = shared_memory[(i+0+BLOCK_SIZE*2) * BLOCK_SIZE + k];
      i_2_k_1 = shared_memory[(i+1+BLOCK_SIZE*2) * BLOCK_SIZE + k];
      i_2_k_2 = shared_memory[(i+2+BLOCK_SIZE*2) * BLOCK_SIZE + k];
      i_2_k_3 = shared_memory[(i+3+BLOCK_SIZE*2) * BLOCK_SIZE + k];
      k_2_j = shared_memory[k * BLOCK_SIZE + j];
    } else {
      i_2_k_0 = shared_memory[(i+0+BLOCK_SIZE) * BLOCK_SIZE + k];
      i_2_k_1 = shared_memory[(i+1+BLOCK_SIZE) * BLOCK_SIZE + k];
      i_2_k_2 = shared_memory[(i+2+BLOCK_SIZE) * BLOCK_SIZE + k];
      i_2_k_3 = shared_memory[(i+3+BLOCK_SIZE) * BLOCK_SIZE + k];
      k_2_j = shared_memory[(k+BLOCK_SIZE*2) * BLOCK_SIZE + j];
    }

    shared_memory[(i+0+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = min(i_2_k_0 + k_2_j, shared_memory[(i+0+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j]);
    shared_memory[(i+1+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = min(i_2_k_1 + k_2_j, shared_memory[(i+1+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j]);
    shared_memory[(i+2+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = min(i_2_k_2 + k_2_j, shared_memory[(i+2+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j]);
    shared_memory[(i+3+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j] = min(i_2_k_3 + k_2_j, shared_memory[(i+3+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j]);
    
  }
    
  d_dist[(i+0+i_offset) * d_matrix_size + j + j_offset] = shared_memory[(i+0+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j];
  d_dist[(i+1+i_offset) * d_matrix_size + j + j_offset] = shared_memory[(i+1+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j];
  d_dist[(i+2+i_offset) * d_matrix_size + j + j_offset] = shared_memory[(i+2+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j];
  d_dist[(i+3+i_offset) * d_matrix_size + j + j_offset] = shared_memory[(i+3+BLOCK_SIZE*blockIdx.x) * BLOCK_SIZE + j];
}

__global__ void phase3(int* d_dist, int r){
  if (blockIdx.x == r || blockIdx.y == r) return;

  __shared__ int row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int col[BLOCK_SIZE][BLOCK_SIZE];

  int i = threadIdx.y*4;
  int j = threadIdx.x;

  int i_offset = blockIdx.y * BLOCK_SIZE;
  int j_offset = blockIdx.x * BLOCK_SIZE;
  int block_round = r * BLOCK_SIZE;

  row[i+0][j] = d_dist[(i+0 + i_offset) * d_matrix_size + block_round + j];
  row[i+1][j] = d_dist[(i+1 + i_offset) * d_matrix_size + block_round + j];
  row[i+2][j] = d_dist[(i+2 + i_offset) * d_matrix_size + block_round + j];
  row[i+3][j] = d_dist[(i+3 + i_offset) * d_matrix_size + block_round + j];
  col[i+0][j] = d_dist[(block_round + i+0)*d_matrix_size + (j_offset)+j];
  col[i+1][j] = d_dist[(block_round + i+1)*d_matrix_size + (j_offset)+j];
  col[i+2][j] = d_dist[(block_round + i+2)*d_matrix_size + (j_offset)+j];
  col[i+3][j] = d_dist[(block_round + i+3)*d_matrix_size + (j_offset)+j];

  int i_2_j_0 = d_dist[(i_offset + i+0)*d_matrix_size + (j_offset)+j];
  int i_2_j_1 = d_dist[(i_offset + i+1)*d_matrix_size + (j_offset)+j];
  int i_2_j_2 = d_dist[(i_offset + i+2)*d_matrix_size + (j_offset)+j];
  int i_2_j_3 = d_dist[(i_offset + i+3)*d_matrix_size + (j_offset)+j];

  __syncthreads();

  #pragma unroll 
  for (int k = 0; k < BLOCK_SIZE; k++){
    i_2_j_0 = min(i_2_j_0, row[i+0][k] + col[k][j]);
    i_2_j_1 = min(i_2_j_1, row[i+1][k] + col[k][j]);
    i_2_j_2 = min(i_2_j_2, row[i+2][k] + col[k][j]);
    i_2_j_3 = min(i_2_j_3, row[i+3][k] + col[k][j]);
  }
  __syncthreads(); // mygodimatomato: add sync to check
  d_dist[(i_offset + i+0)*d_matrix_size + (j_offset)+j] = i_2_j_0;
  d_dist[(i_offset + i+1)*d_matrix_size + (j_offset)+j] = i_2_j_1;
  d_dist[(i_offset + i+2)*d_matrix_size + (j_offset)+j] = i_2_j_2;
  d_dist[(i_offset + i+3)*d_matrix_size + (j_offset)+j] = i_2_j_3;

}


void block_FW(int* d_dist) {
  int round = matrix_size/BLOCK_SIZE;
  dim3 phase2_num_blocks(2, round); // one for col, one for row, one block will be redundant, but for the whole performance it doesn't really matters
  dim3 phase3_num_blocks(round, round);
  dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE/4);

  // round = 1; // mygodimatomato: for checking
  for (int r = 0; r < round; r++) {
    phase1<<<1, num_threads>>>(d_dist, r);
    phase2<<<phase2_num_blocks, num_threads>>>(d_dist, r);
    phase3<<<phase3_num_blocks, num_threads>>>(d_dist, r);
  }
}


int main(int argc, char* argv[]) {
  // Read input from input file

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  input(argv[1]);
  printf("V: %d, E: %d\n", V, E);
  // cudaEventRecord(stop, 0);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Host Input Time: %f ms\n", elapsedTime);
  // Allocate the memory for the matrix in GPU
  int *d_dist;
  cudaMalloc((void**)&d_dist, sizeof(int) * matrix_size * matrix_size);
  cudaMemcpy(d_dist, adjacency_matrix, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_matrix_size, &matrix_size, sizeof(int));
  
  
  // Start executing the block Floyed-Warshall
  block_FW(d_dist);

  // Copy the outcome back to the adjacency_matrix 
  cudaMemcpy(adjacency_matrix, d_dist, sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);

  // cudaEvent_t start2, stop2;
  // cudaEventCreate(&start2);
  // cudaEventCreate(&stop2);
  // cudaEventRecord(start2, 0);
  output(argv[2]);
  // cudaEventRecord(stop2, 0);
  // float elapsedTime2;
  // cudaEventElapsedTime(&elapsedTime2, start2, stop2);
  // printf("Host Output Time: %f ms\n", elapsedTime2);


  // mygodimatomato : for checking
  // int k = 0;
  // for (int i = 0; i < V; i++) {
  //   for (int j = 0; j < V; j++){
  //     if(adjacency_matrix[k] == MY_INF)
  //       printf(" INF ");
  //     else
  //       printf("%4d ", adjacency_matrix[k]);
  //     k++;
  //   } printf("\n");
  // } printf("\n");

  // Write output to output file
  return 0;
}
