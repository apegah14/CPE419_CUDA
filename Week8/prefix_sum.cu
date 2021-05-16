#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include <omp.h>

#define LENGTH 70000

__global__ void prefixSum(int *in, int *out, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int temp[4096];
    int buffer_in = 0;
    int buffer_out = 1;
    int tempSum = 0;


    temp[buffer_in * blockDim.x + threadIdx.x] = in[i];
    //printf("thread: %d / data sent to shared memory: %d\n", i, temp[buffer_in * blockDim.x + threadIdx.x]);

    __syncthreads();

    for(int offset = 1; offset <= blockDim.x; offset *= 2) {
        //printf("thread: %d / offset: %d / blockDim.x: %d\n", i, offset, blockDim.x);
        if(threadIdx.x >= offset) {
            temp[buffer_out * blockDim.x + threadIdx.x] = temp[buffer_in * blockDim.x + threadIdx.x - offset] + temp[buffer_in * blockDim.x + threadIdx.x];
        }
        else {
            temp[buffer_out * blockDim.x + threadIdx.x] = temp[buffer_in * blockDim.x + threadIdx.x];
        }
        //printf("thread: %d / data sent to shared memory: %d\n", i, temp[buffer_out * blockDim.x + threadIdx.x]);
        __syncthreads();

        // switch in and out buffer portions of temp array
        buffer_out = 1 - buffer_out;
        buffer_in = 1 - buffer_out;
    }

    out[i] = temp[buffer_out * blockDim.x + threadIdx.x];
    //tempSum = temp[buffer_out * blockDim.x + blockDim.x - 1];
    //printf("thread: %d / data sent to shared memory: %d\n", i, out[i]);
    //printf("thread: %d / data sent to shared memory: %d\n", i, temp[buffer_out * blockDim.x + threadIdx.x]);
    __syncthreads();
    
    for(int j = 0; j < gridDim.x; j++) {
        if(i > blockDim.x * j + threadIdx.x) {
            //printf("j = %d data = %d / %d\n", j, temp[buffer_out * blockDim.x + threadIdx.x], out[blockDim.x * j + blockDim.x - 1]);
            tempSum = out[blockDim.x * j + blockDim.x - 1];
            temp[buffer_out * blockDim.x + threadIdx.x] += tempSum;
        }
        __syncthreads();
        //printf("tempSum = %d\n", tempSum);
    }
    out[i] = temp[buffer_out * blockDim.x + threadIdx.x];
    //printf("thread: %d / data sent to shared memory: %d\n", i, out[i]);
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


    // allocate host memory for int array and initialize it
    cudaMallocHost((void**)&h_in, bytes);

    cudaMallocHost((void**)&h_out, bytes);
    cudaMallocHost((void**)&h_cpu, bytes);
    initArray(h_in, paddedLength);

    printf("Array initialized\n");

    /*
    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_in[i]);
    }*/
    printf("\n\n");

    // prefix sum on CPU
    prefixSumCPU(h_in, h_cpu, paddedLength);

    printf("CPU Done\n");

    // allocate device memory for array and copy array from host to device
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // set thread block and grid size (can only use 1 thread block)
    int gridSize = 64;
    int blockSize = ceil(paddedLength / gridSize);

    prefixSum<<<gridSize, blockSize>>>(d_in, d_out, paddedLength);

    printf("GPU Done\n");

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    /*
    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n\n");


    for(int i = 0; i < LENGTH; i++) {
        printf("%d ", h_cpu[i]);
    }
    printf("\n\n");
    */
    for(int i = 0; i < LENGTH; i++) {
        if(h_out[i] != h_cpu[i]) {
            printf("i: %d / GPU: %d / CPU: %d\n", i, h_out[i], h_cpu[i]);
            printf("Elements are wrong\n");
            break;
        }
    }

    printf("Done\n");
    // free memory on device and host
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFreeHost(h_cpu);
}