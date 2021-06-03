#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, tid, 0, &state[tid]);
}


__global__ void sampleCUDA(int *num_inside, curandState *states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int num_inside_block[1024];

    num_inside_block[threadIdx.x] = 0;
    __syncthreads();
    float x = curand_uniform(&states[tid]);
    float y = curand_uniform(&states[tid]);
    //printf("thread: %d / x = %.3f / y = %.3f\n", tid, x, y);
    if (sqrt((x * x) + (y * y)) <= 1) {
        num_inside_block[threadIdx.x] = 1;
    }
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;

        if (index < blockDim.x) {
            num_inside_block[index] += num_inside_block[index + i];
        }
        __syncthreads();
    }   

    if (threadIdx.x == 0) {
        num_inside[blockIdx.x] = num_inside_block[threadIdx.x];
        //printf("num_inside: %d\n", num_inside_block[threadIdx.x]);
    }
}

__global__ void reductionSum(int *num_inside, float *total) {
    __shared__ int reduction[1024];

    // bring in data from global memory to shared
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    reduction[threadIdx.x] = num_inside[tid];
    //printf("thread: %d / num_inside: %d\n", tid, reduction[threadIdx.x]);
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;

        if (index < blockDim.x) {
            reduction[index] += reduction[index + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        total[0] = reduction[threadIdx.x];
        //printf("total: %d\n", reduction[threadIdx.x]);
    }
}

int main() {
    int samples = 1024 * 1000;
    int block_size = 1024;
    int grid_size = (samples + block_size - 1) / block_size;
    float *total_cpu = (float*)malloc(sizeof(float));

    int *num_inside_gpu;
    float *total;

    cudaMalloc((void**)&num_inside_gpu, sizeof(int) * grid_size);
    cudaMalloc((void**)&total, sizeof(float));

    // cuRand setup
    curandState_t *states;
    cudaMalloc((void**) &states, samples * sizeof(curandState_t));
    setup_kernel<<<grid_size, block_size>>>(states);

    sampleCUDA<<<grid_size, block_size>>>(num_inside_gpu, states);

    reductionSum<<<1, 1024>>>(num_inside_gpu, total);
    cudaMemcpy(total_cpu, total, sizeof(float), cudaMemcpyDeviceToHost);

    float pi = (4 * total_cpu[0]) / samples;
    printf("Pi = %.6f\n", pi);

    printf("total = %.3f\n", total_cpu[0]);

    return 0;
}