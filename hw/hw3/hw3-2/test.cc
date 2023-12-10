#include <iostream>
#include <iomanip>
using namespace std;

#define BLOCK_SIZE 8

int main() {
    const int matrix_size = 16;
    int matrix[matrix_size*matrix_size];
    const int share_size = 8;
    int share[share_size*share_size];

    // Filling the matrix with values from 1 to 100
    for(int i = 0; i < matrix_size*matrix_size; ++i) {
        matrix[i] = i;
    }

    for(int i = 0; i < matrix_size; i++){
        for (int j = 0; j < matrix_size; j++){
            cout <<setw(3)<< matrix[i*matrix_size+j] << " ";
        } cout << endl;
    }cout << endl;


    for(int r = 0; r < 2; r++){
        for(int y = 0; y < 8; y++){
            for(int x = 0; x < 8; x++){
                share[x + y * BLOCK_SIZE] = matrix[(x+r*BLOCK_SIZE) + (y+r*BLOCK_SIZE) * matrix_size];
            }
        }
        for(int i = 0; i < share_size; i++){
            for (int j = 0; j < share_size; j++){
                cout << setw(3) << share[i*BLOCK_SIZE+j] << " ";
            } cout << endl;
        }cout << endl;
    }


    return 0;
}