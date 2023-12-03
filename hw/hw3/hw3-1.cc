#include <iostream>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <sched.h>

// MY_INF is 2^30-1
#define MY_INF 1073741823

using namespace std;

int V, E;
int **adjacency_matrix;

int num_cpus = 4;

int main() {
  // The first two integers are the number of vertices(V) and the number of deges(E)
  // then, there are E edges. Each edge consists of 3 integers:
  // 1. source vertex id
  // 2. destination vertex id
  // 3. weight of the edge
  // The values of vertex indexes & edge indexes start at 0

  cin >> V >> E; // read V and E

  // initialize adjacency matrix
  adjacency_matrix = new int*[V];
  for (int i = 0; i < V; i++) {
    adjacency_matrix[i] = new int[V];
    for (int j = 0; j < V; j++) {
      if (i == j)
        adjacency_matrix[i][j] = 0;
      else
        adjacency_matrix[i][j] = MY_INF;
    }
  }

  // read edges
  for (int i = 0; i < E; i++) {
    int source, destination, weight;
    cin >> source >> destination >> weight;
    adjacency_matrix[source][destination] = weight;
  }

  // mygodimatomato : for testing, print out the adjacency matrix
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < V; j++)
      cout << adjacency_matrix[i][j] << " ";
    cout << endl;
  }





  // free memory
  for (int i = 0; i < V; i++)
    delete[] adjacency_matrix[i];
  delete[] adjacency_matrix;

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