#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define LENGTH 10000000

// parallel reduction for sample sizes that exceed 1 thread block (same method as Q3)
__global__ void reductionSum(int *numbers, int *total) {
    __shared__ int reduction[1024];

    // bring in data from global memory to shared
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    reduction[threadIdx.x] = numbers[tid];
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

int reductionSumCPU(int *numbers) {
    int total = 0;
    for (int i = 0; i < LENGTH; i++) {
        total += numbers[i];
    }
    return total;
}


int main() {
    //timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_gpu;
    double time_seq;

    struct timespec begin, end;

    int *numbers_gpu, *total_gpu, total_cpu;
    int *total_transfer = (int*)malloc(sizeof(int));
    int *numbers = (int*)malloc(sizeof(int) * LENGTH);
    cudaMalloc((void**)&numbers_gpu, sizeof(int) * LENGTH);
    cudaMalloc((void**)&total_gpu, sizeof(int));

    for (int i = 0; i < LENGTH; i++) {
        numbers[i] = rand() % 10;
    }

    cudaMemcpy(numbers_gpu, numbers, sizeof(int) * LENGTH, cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = (LENGTH + block_size - 1) / block_size;
    cudaEventRecord(start);
    reductionSum<<<grid_size, block_size>>>(numbers_gpu, total_gpu);
    cudaEventRecord(stop);
    cudaMemcpy(total_transfer, total_gpu, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);

    printf("Time GPU= %.3f ms\n", time_gpu);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    total_cpu = reductionSumCPU(numbers);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_seq = end.tv_sec - begin.tv_sec;
    time_seq += (end.tv_nsec - begin.tv_nsec) / 1000000.0;
    printf("Time CPU = %.3f ms\n", time_seq);

    printf("CPU result = %d\nGPU result = %d\n", total_cpu, total_transfer[0]);


    return 0;
}