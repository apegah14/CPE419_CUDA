#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include <omp.h>

#define LENGTH 5000

__global__ void prefixSum(int *in, int *out, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int temp[2][LENGTH];
    int buffer_in = 0;
    int buffer_out = 1;


    temp[buffer_in][i] = in[i];
    //printf("thread: %d / data sent to shared memory: %d\n", i, temp[buffer_in][i]);

    __syncthreads();

    for(int offset = 1; offset <= length/gridDim.x; offset *= 2) {

        if(i >= offset) {
            temp[buffer_out][i] = temp[buffer_in][i - offset] + temp[buffer_in][i];
        }
        else {
            temp[buffer_out][i] = temp[buffer_in][i];
        }
        __syncthreads();

        // switch in and out buffer portions of temp array
        buffer_out = 1 - buffer_out;
        buffer_in = 1 - buffer_out;
    }
    out[i] = temp[buffer_out][i];

    //__syncthreads();

    
    for(int j = 0; j < gridDim.x; j++) {
        if(i > blockDim.x * j + threadIdx.x) {
            temp[buffer_out][i] += out[blockDim.x * j + blockDim.x - 1];
            //__syncthreads();
        }
    }
    out[i] = temp[buffer_out][i];
}

// prefix sum on CPU (not parallel)
void prefixSumCPU(int *in, int *out, int length) {
    out[0] = in[0];
    for(int i = 1; i < length; i++) {
        out[i] = in[i] + out[i - 1];
    }
}


// initialize array to random value and pad array if necessary
void initArray(int *array, int paddedLength) {
    for(int i = 0; i < LENGTH; i++) {
        array[i] = rand() % 10;
    }
    for(int i = LENGTH; i < paddedLength; i++) {
        array[i] = 0;
    }
}

int main() {
    int *h_in, *h_out, *h_cpu, *d_in, *d_out;

    // pad length for something that isn't a power of 2
    int paddedLength = pow(2, ceil(log2(LENGTH)));
    printf("Length = %d\n", paddedLength);
    int bytes = sizeof(int) * paddedLength;

    

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate host memory for int array and initialize it
    cudaMallocHost((void**)&h_in, bytes);

    cudaMallocHost((void**)&h_out, bytes);
    cudaMallocHost((void**)&h_cpu, bytes);
    initArray(h_in, paddedLength);

    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_in[i]);
    }
    printf("\n\n");

    StartTimer();
    // prefix sum on CPU
    prefixSumCPU(h_in, h_cpu, paddedLength);
    const double tElapsedCPU = GetTimer();

    // allocate device memory for array and copy array from host to device
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // set thread block and grid size (can only use 1 thread block)
    int gridSize = 64;
    int blockSize = LENGTH / gridSize;
    printf("GPU compute\n");

    cudaEventRecord(start);
    prefixSum<<<gridSize, blockSize>>>(d_in, d_out, paddedLength);
    cudaEventRecord(stop);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n\n");


    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_cpu[i]);
    }
    printf("\n\n");

    for(int i = 0; i < LENGTH; i++) {
        if(h_out[i] != h_cpu[i]) {
            printf("Elements are wrong\n");
            break;
        }
    }

    printf("GPU Time: %f ms\n", milliseconds);

    printf("CPU Time: %f ms\n", tElapsedCPU);

    // free memory on device and host
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFreeHost(h_cpu);
}