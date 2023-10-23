#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <iostream>

#define EVEN_PHASE 0
#define ODD_PHASE 1

using namespace std;

float *tmp;

void printArray(float arr[], int n, int rank, bool sort)
{
  if (sort) {
    cout << "Rank : " << rank << " , sort" << endl;
  } else {
    cout << "Rank : " << rank << " , unsort" << endl;
  }
  int i; 
  for (i = 0; i < n; i++)
    cout << arr[i] << " ";
  cout << endl;
}

void my_merge(float *arr1, float *arr2, int arr1_size, int arr2_size, bool isHigh)
{
  int iter_1, iter_2, iter_3, i;
  if (isHigh) {
    if (arr1[0] >= arr2)
  }
}