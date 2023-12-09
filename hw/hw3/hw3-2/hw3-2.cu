#include <iostream>
#include <fstream>
#include <thread>
#include <cstdlib>
#include <omp.h>
#include <vector>
#include <cstring>

// MY_INF is 2^30-1
#define MY_INF 1073741823

using namespace std;

int V, E;
int *adjacency_matrix;
int num_cpus;

int main(int argc, char *argv[]) {

  // check the number of CPUs
  char *cpus_env = std::getenv("SLURM_CPUS_PER_TASK");
  num_cpus = atoi(cpus_env);

  // read the file
  char *input_file = argv[1];
  char *output_file = argv[2];

  fstream file;
  file.open(input_file, ios::in | ios::binary);
  file.read((char *)&V, sizeof(int));
  file.read((char *)&E, sizeof(int));

  adjacency_matrix = new int[V*V];
  // initialize the adjacency matrix

  #pragma omp parallel num_threads(num_cpus)
  {
    #pragma omp for schedule(static) collapse(2)
    for (int i = 0; i < V; i++) {
      for (int j = 0; j < V; j++) {
        if (i == j)
          adjacency_matrix[i*V+j] = 0;
        else
          adjacency_matrix[i*V+j] = MY_INF;
      }
    }
  }

  // read the edges
  for (int i = 0; i < E; i++) {
    int source, destination, weight;
    file.read((char *)&source, sizeof(int));
    file.read((char *)&destination, sizeof(int));
    file.read((char *)&weight, sizeof(int));
    adjacency_matrix[source*V+destination] = weight;
  }


  for (int k = 0; k < V; k++) {
    #pragma omp parallel num_threads(num_cpus)
    {
      #pragma omp for schedule(static) collapse(2)
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          if (j == k || j == i) continue;
          if (adjacency_matrix[i*V+k] + adjacency_matrix[k*V+j] < adjacency_matrix[i*V+j])
            adjacency_matrix[i*V+j] = adjacency_matrix[i*V+k] + adjacency_matrix[k*V+j];
        }
      }
    }
  }


  // write the result to the output file
  fstream output;
  output.open(output_file, ios::out | ios::binary);
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++)
      output.write((char*)(&adjacency_matrix[i*V+j]), sizeof(int));
  }

  return 0;
}