#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MY_INF 1073741823
#define BLOCK_SIZE 64

int V, E;
int matrix_size;
int *adjacency_matrix;
size_t result;
int num_gpus;
__constant__ int d_matrix_size;

void input(char* infile) {
  FILE *input_file = fopen(infile, "rb");
  result = fread(&V, sizeof(int), 1, input_file);
  result = fread(&E, sizeof(int), 1, input_file);
  
  int remainder = V%128;
  matrix_size = (remainder == 0) ? V : V + (128-remainder);
  // matrix_size = ((V / BLOCK_SIZE) + 1 ) * BLOCK_SIZE; // matrix size must be multiple of BLOCK_SIZE

  adjacency_matrix = (int *)malloc(matrix_size * matrix_size * sizeof(int)); // set the matrix to 1D array for easier data transfer
 
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

__global__ void phase1(int* d_dist, int r, int d_matrix_size){
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

  // writing data back to global memory
  d_dist[(i+0+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+0) * BLOCK_SIZE + j];
  d_dist[(i+1+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+1) * BLOCK_SIZE + j];
  d_dist[(i+2+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+2) * BLOCK_SIZE + j];
  d_dist[(i+3+r*BLOCK_SIZE) * d_matrix_size + (j+r*BLOCK_SIZE)] = shared_memory[(i+3) * BLOCK_SIZE + j];
}

__global__ void phase2(int* d_dist, int r, int d_matrix_size){
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

__global__ void phase3(int* d_dist, int r, int d_matrix_size, int offset){
  if (blockIdx.x == r || blockIdx.y + offset/BLOCK_SIZE == r) return;

  __shared__ int row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int col[BLOCK_SIZE][BLOCK_SIZE];

  int i = threadIdx.y*4;
  int j = threadIdx.x;

  int i_offset = blockIdx.y * BLOCK_SIZE + offset;
  int j_offset = blockIdx.x * BLOCK_SIZE;
  int block_round = r * BLOCK_SIZE;

  if (i_offset >= d_matrix_size) return;
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

  d_dist[(i_offset + i+0)*d_matrix_size + (j_offset)+j] = i_2_j_0;
  d_dist[(i_offset + i+1)*d_matrix_size + (j_offset)+j] = i_2_j_1;
  d_dist[(i_offset + i+2)*d_matrix_size + (j_offset)+j] = i_2_j_2;
  d_dist[(i_offset + i+3)*d_matrix_size + (j_offset)+j] = i_2_j_3;

}


void block_FW() {
  cudaGetDeviceCount(&num_gpus);
  omp_set_num_threads(num_gpus);
  cudaHostRegister(adjacency_matrix, matrix_size*matrix_size*sizeof(int), cudaHostRegisterDefault);

  int *d_dist[2];
  int offset = 0;
  if ((matrix_size / BLOCK_SIZE) % 2 == 0) {
    offset = matrix_size / 2;
  } else {
    offset = (matrix_size / (2*BLOCK_SIZE) + 1)*BLOCK_SIZE;
  }

  #pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(cpu_thread_id);

    cudaMalloc(&d_dist[cpu_thread_id], sizeof(int) * matrix_size * matrix_size);
    cudaMemcpy(d_dist[cpu_thread_id], adjacency_matrix, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
    dim3 thread_num(BLOCK_SIZE, BLOCK_SIZE/4);
    dim3 phase2_block_num(2, matrix_size/BLOCK_SIZE);
    dim3 phase3_block_num(matrix_size/BLOCK_SIZE, matrix_size/BLOCK_SIZE);

    int round = matrix_size / BLOCK_SIZE;
    for (int i = 0; i < round; i++) {
      phase1<<<1, thread_num>>>(d_dist[cpu_thread_id], i, matrix_size);
      phase2<<<phase2_block_num, thread_num>>>(d_dist[cpu_thread_id], i, matrix_size);
      phase3<<<phase3_block_num, thread_num>>>(d_dist[cpu_thread_id], i, matrix_size, offset*cpu_thread_id);

      cudaDeviceSynchronize();
      #pragma omp barrier

      if (cpu_thread_id == 0 && (i+1) >= offset/BLOCK_SIZE) {
        cudaMemcpy(d_dist[0] + (i+1)*BLOCK_SIZE*matrix_size, d_dist[1] + (i+1)*BLOCK_SIZE*matrix_size, sizeof(int) * matrix_size * BLOCK_SIZE, cudaMemcpyDeviceToDevice);
      }
      if (cpu_thread_id == 1 && (i+1) < offset/BLOCK_SIZE) {
        cudaMemcpy(d_dist[1] + (i+1)*BLOCK_SIZE*matrix_size, d_dist[0] + (i+1)*BLOCK_SIZE*matrix_size, sizeof(int) * matrix_size * BLOCK_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
    if (cpu_thread_id == 0) {
      cudaMemcpy(adjacency_matrix, d_dist[0], sizeof(int) * matrix_size * offset, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(adjacency_matrix+offset*matrix_size, d_dist[1]+offset*matrix_size, sizeof(int) * (matrix_size-offset) * matrix_size, cudaMemcpyDeviceToHost);
    }
  }
}


int main(int argc, char* argv[]) {
  // Read input from input file
  input(argv[1]);
  block_FW();
  output(argv[2]);

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

  return 0;
}