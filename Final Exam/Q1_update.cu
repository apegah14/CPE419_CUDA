#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define SAMPLES 256000000

__global__ void sampleCUDA(int *num_inside, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int num_inside_block[1024];

    // initialize each thread's random state
    curandState_t state;
    curand_init(tid, 0, 0, &state);

    // initialize the count to 0
    num_inside_block[threadIdx.x] = 0;

    // each thread does a certain number of samples
    // tried with 1 sample per thread but memory was an issue (plus cuRAND had some crazy performance issues)
    if (tid < SAMPLES / iter) {
        for (int i = 0; i < iter; i++) {
            float x = curand_uniform(&state);
            float y = curand_uniform(&state);
            if (sqrt((x * x) + (y * y)) <= 1) {
                num_inside_block[threadIdx.x] += 1;
            }
        }
    }
    // sync threads before reading from shared memory
    __syncthreads();

    // parallel reduction sum
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;

        if (index < blockDim.x) {
            num_inside_block[index] += num_inside_block[index + i];
        }
        __syncthreads();
    }   

    // throw the reduction sum into global memory for further processing
    if (threadIdx.x == 0) {
        num_inside[blockIdx.x] = num_inside_block[threadIdx.x];
    }
}

// parallel reduction for sample sizes that exceed 1 thread block (same method as Q3)
__global__ void reductionSum(int *num_inside, float *total) {
    __shared__ int reduction[1024];

    // bring in data from global memory to shared
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    reduction[threadIdx.x] = num_inside[tid];
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;

        if (index < blockDim.x) {
            reduction[index] += reduction[index + i];
        }
        __syncthreads();
    }

    // sum up elements atomically (allows it to scale past 1 thread block)
    if (threadIdx.x == 0) {
        atomicAdd(&total[0], reduction[threadIdx.x]);
    }
}

int main() {
    int samples = SAMPLES;
    int block_size = 1024;
    int iter = 1000;
    int grid_size = ((SAMPLES / iter) + block_size - 1) / block_size;
    float *total_cpu = (float*)malloc(sizeof(float));
    float time_ms = 0;

    int *num_inside_gpu;
    float *total;

    cudaMalloc((void**)&num_inside_gpu, sizeof(int) * grid_size);
    cudaMalloc((void**)&total, sizeof(float));

    //timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sampleCUDA<<<grid_size, block_size>>>(num_inside_gpu, iter);

    reductionSum<<<1, 1024>>>(num_inside_gpu, total);
    cudaMemcpy(total_cpu, total, sizeof(float), cudaMemcpyDeviceToHost);

    float pi = (4 * total_cpu[0]) / samples;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    printf("Pi = %.6f\n", pi);

    printf("Time = %.3f ms\n", time_ms);

    return 0;
}