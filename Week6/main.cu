#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000
#define TILE 16

__global__ void matMultiplyOnDevice(int* a, int* b, int* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tempSum = 0;
    if(row < m && col < k) {
        for(int i = 0; i < n; i++) {
            tempSum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = tempSum;
    }
}


__global__ void matMultiplyTiled(int* a, int* b, int* c, int m, int n, int k) {

    // declare shared memory for tiles
    __shared__ int shared_a[TILE * TILE];
    __shared__ int shared_b[TILE * TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int tempSum = 0;
    // sweep tile over matrix (same calculation as grid dim)
    for (int i = 0; i < (TILE + n - 1) / TILE; i++) {
        // tile overshoot should be 0
        if ((i * TILE + threadIdx.x) < n && row < m) {
            shared_a[(threadIdx.y * TILE) + threadIdx.x] = a[row * n + i * TILE + threadIdx.x];
        }
        else {
            shared_a[(threadIdx.y * TILE) + threadIdx.x] = 0;
        }

        if ((i * TILE + threadIdx.y) < n && col < k) {
            shared_b[(threadIdx.y * TILE) + threadIdx.x] = b[(i * TILE + threadIdx.y) * k + col];
        }
        else {
            shared_b[(threadIdx.y * TILE) + threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            tempSum += shared_a[(threadIdx.y * TILE) + i] * shared_b[(i * TILE) + threadIdx.x];
        }
        __syncthreads();

    }

    // make sure result isn't outside of c
    if (row < m && col < k) {
        c[row * k + col] = tempSum;
    }

}




void matMultiplyOnHost(int* a, int* b, int* c, int m, int n, int k) {
    for (int row= 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            int tempSum = 0.0;
            for (int h = 0; h < n; h++) {
                tempSum += a[row * n + h] * b[h * k + col];
            }
            c[row * k + col] = tempSum;
        }
    }
}




int main() {
    int* h_a;
    int* h_b;
    int* h_c;
    int* h_check;

    int* d_a;
    int* d_b;
    int* d_c;

    int m = 1000;
    int n = 2000;
    int k = 1500;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    clock_t cpu_startTime, cpu_endTime;
    double cpu_ElapsedTime = 0.0;

    // allocate host memory for operand and resultant matrices
    cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
    cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
    cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
    cudaMallocHost((void**)&h_check, sizeof(int) * m * k);

    printf("Generating Matrices\n");
    // initialize operand matrices
    srand(5);
    for (int i = 0; i < m * n; i++) {
        //printf("%d\n", i);
        h_a[i] = rand() % 1000;
    }

    for (int i = 0; i < n * k; i++) {
        //printf("%d\n", i);
        h_b[i] = rand() % 1000;
    }


    // allocate device memory for operand and resultant matrices
    cudaMalloc((void**)&d_a, sizeof(int) * m * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n * k);
    cudaMalloc((void**)&d_c, sizeof(int) * m * k);

    // copy matrices to device
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    // define block and grid size
    int blockSize = 32;
    unsigned int grid_rows = (m + blockSize - 1) / blockSize;
    unsigned int grid_cols = (k + blockSize - 1) / blockSize;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(blockSize, blockSize);

    cudaEventRecord(start);
    matMultiplyOnDevice<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    //matMultiplyTiled<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaEventRecord(stop);

    cudaMemcpy(h_c, d_c, sizeof(int)* m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU CUDA Time: %.3f ms\n", milliseconds);
/*
    cpu_startTime = clock();
    matMultiplyOnHost(h_a, h_b, h_check, m, n, k);
    cpu_endTime = clock();

    cpu_ElapsedTime = ((cpu_endTime - cpu_startTime)/CLOCKS_PER_SEC);
    printf("CPU Time: %f ms\n", cpu_ElapsedTime);

    for (int j = 0; j < m * k; j++) {
        //printf("h_c[%d] = %d\n", j, h_c[j]);
        //printf("h_check[%d] = %d\n", j, h_check[j]);
        if (h_c[j] != h_check[j]) {
            printf("Some results are wrong\n");
            break;
        }
    }*/

    // free up device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_check);

    printf("Done\n");

    return 0;
}
