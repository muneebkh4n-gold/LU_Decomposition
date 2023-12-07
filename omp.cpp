#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <fstream>
using namespace std;

double* matrixMultiplication(double * mat1, double * mat2, int size)
{
    double* resultMatrix = (double*) malloc(sizeof(double )*(size*size));
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            resultMatrix[i*size+j] = 0.00;
            for (int k = 0; k < size; k++){
                resultMatrix[i*size+j] += mat1[i*size+k] * mat2[k*size+j];
            }
        }
    }
    return resultMatrix;
}

void printMatrix(double* matrix, int dim){
    for(int i=0 ; i<dim ; i++){
        for (int j = 0; j < dim; j++)
            printf("%.02lf\t", matrix[i * dim + j]);
        cout<<endl;
    }
}

void permutateMatrix(double * finalMat, int *pVector, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            cout<<finalMat[pVector[i] * size + j]<<"\t";
        }
        cout<<endl;
    }
}

int main(int argc,char *argv[]){
    auto startTime = chrono::high_resolution_clock::now();

    // Parse arguments for matrix size and number of threads
    string size_str(argv[1]);
    int size = stoi(size_str);
    string thread_str(argv[2]);
    int thread = stoi(thread_str);

    // Memory allocation for matrices and permutation vector
    double *Amat = (double*)malloc(sizeof(double)*(size*size));
    int *Pvec = (int*)malloc(sizeof(int)*(size));
    double *Umat = (double*)malloc(sizeof(double)*(size*size));
    double *Lmat = (double*)malloc(sizeof(double)*(size*size));

    // Initialize the matrices and permutation vector
    srand(45);
    #pragma omp parallel num_threads(thread)
    {
        #pragma omp for collapse(2)
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                Amat[i*size+j] = ((double)(rand()%1000)) / 100.0;
                Umat[i*size+j] = (j >= i) ? Amat[i*size+j] : 0.0;
                Lmat[i*size+j] = (j == i) ? 1.0 : (j < i) ? Amat[i*size+j] : 0.0;
            }
        }
    }

    // LU Decomposition with partial pivoting
    for(int k = 0; k < size; k++) {
        // Pivot finding (serial)
        double max = 0.0;
        int kDash = -1;
        for(int i = k; i < size; i++) {
            double val = fabs(Amat[i*size+k]);
            if(val > max) {
                max = val;
                kDash = i;
            }
        }

        // Check for singular matrix
        if(kDash == -1) {
            cout << "\nMatrix is singular" << endl;
            return -1;
        }

        // Swap rows in Amat and update permutation vector
        #pragma omp parallel for num_threads(thread)
        for(int i = 0; i < size; i++) {
            swap(Amat[k*size+i], Amat[kDash*size+i]);
        }

        // Update Lmat and Umat
        Umat[k*size+k] = Amat[k*size+k];
        #pragma omp parallel for num_threads(thread)
        for(int i = k + 1; i < size; i++) {
            Lmat[i*size+k] = Amat[i*size+k] / Umat[k*size+k];
            Umat[k*size+i] = Amat[k*size+i];
        }

        // Update remaining matrix
        #pragma omp parallel for num_threads(thread) collapse(2)
        for(int i = k + 1; i < size; i++) {
            for(int j = k + 1; j < size; j++) {
                Amat[i*size+j] -= Lmat[i*size+k] * Umat[k*size+j];
            }
        }
    }

    // End timing
    auto endTime = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

    // Output and logging (if enabled)
    if(argc == 4 && atoi(argv[3]) == 1) {
        cout << "=============================" << endl;
        cout << "Permutation vector\t";
        for(int i = 0; i < size; i++) {
            cout << Pvec[i] << " ";
        }
        cout << endl;
        cout << "=============================" << endl;
        double *resultMat = matrixMultiplication(Lmat, Umat, size);
        permutateMatrix(resultMat, Pvec, size);
        free(resultMat);
        cout << "=============================" << endl;
    }

    printf("%.3f s\n", (float)total_time.count()/1000);
    float total_t = (float)total_time.count()/1000;
    ofstream fptr;
    fptr.open("logs.txt", ios::app);
    fptr << "OpenMP: " << total_t << "s, threads: " << thread << ", n=" << size << endl;
    fptr.close();

    // Free allocated memory
    free(Amat); free(Lmat); free(Umat); free(Pvec);

    return 0;
}
