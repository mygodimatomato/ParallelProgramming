#include <iostream>
#include <fstream>
#include <thread>
#include <pthread.h>

// MY_INF is 2^30-1
#define MY_INF 1073741823

using namespace std;

int V, E;
int **adjacency_matrix;

int num_cpus = 4;

int main(int argc, char *argv[]) {
  // The first two integers are the number of vertices(V) and the number of deges(E)
  // then, there are E edges. Each edge consists of 3 integers:
  // 1. source vertex id
  // 2. destination vertex id
  // 3. weight of the edge
  // The values of vertex indexes & edge indexes start at 0

  // mygodimatomato : read the file
  char *input_file = argv[1];
  char *output_file = argv[2];

  std::ifstream file(input_file);
  if (file.is_open()) {
    file >> V >> E;
    cout << "V: " << V << ", E: " << E << endl; // mygodimatomato : for checking 

    adjacency_matrix = new int*[V];
    // mygodimatomato : initialize the adjacency matrix
    for (int i = 0; i < V; i++) {
      adjacency_matrix[i] = new int[V];
      for (int j = 0; j < V; j++) {
        if (i == j)
          adjacency_matrix[i][j] = 0;
        else
          adjacency_matrix[i][j] = MY_INF;
      }
    }

    // mygodimatomato : read the edges
    for (int i = 0; i < E; i++) {
      int source, destination, weight;
      file >> source >> destination >> weight;
      adjacency_matrix[source][destination] = weight;
    }

    // mygodimatomato : for testing, print out the adjacency matrix
    for (int i = 0; i < V; i++) {
      for (int j = 0; j < V; j++)
        cout << adjacency_matrix[i][j] << " ";
      cout << endl;
    }
  } else {
    cout << "Unable to open file" << endl;
    return 0;
  }

  // free memory
  for (int i = 0; i < V; i++)
    delete[] adjacency_matrix[i];
  delete[] adjacency_matrix;

  return 0;
}

// void BlockedAllPairs(int **adjacency_matrix, int V, int num_cpus) {

//   for (int round = 1; round <= V; round++) {

//     // self-dependent block
//     for (int k = (round - 1) * num_cpus + 1; k <= round * num_cpus; k++){
//       for (int i = 0; i < V; i++) {
//         for (int j = 0; j < V; j++) {
//           adjacency_matrix[i][j] = min(adjacency_matrix[i][j], adjacency_matrix[i][k] + adjacency_matrix[k][j]);
//         }
//       }
//     }

//     // remaining blocks

//   }
// }