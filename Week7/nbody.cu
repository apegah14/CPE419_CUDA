#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

// CUDA kernel for calculating body force
__global__ void bodyForceGPU(Body *p, float dt, int n) {
    // assigning a portion to each thread can replace CPU outer most for loop
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // make sure block isn't out of bounds
    if (i < n) {
        // same calculation as in omp code
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// CUDA kernel for calculating body force with striding
__global__ void bodyForceStride(Body *p, float dt, int n) {
    int stride = n / (gridDim.x * blockDim.x);
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += stride) {
        // same calculation as in omp code
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        for (int j = 0; j < blockDim.x; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// CUDA kernel for integrating
__global__ void integrate(Body *p, float dt, int n) {
    // assigning a portion to each thread can replace CPU outer most for loop
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure block isn't out of bounds
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

// randomize data
void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 300000;
    cudaDeviceProp props; 
    cudaGetDeviceProperties(&props, 0);
    int sm=props.multiProcessorCount;

    if (argc > 1) nBodies = atoi(argv[1]);
    

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies*sizeof(Body);
    float *h_buf = (float*)malloc(bytes);
    Body *h_p = (Body*)h_buf;

    randomizeBodies(h_buf, 6*nBodies); // Init pos / vel data

    // allocate memory on device
    float *d_buf;
    cudaMalloc(&d_buf, bytes);
    Body *d_p = (Body*)d_buf;

    // set grid and block size
    int blockSize = 128;
    int gridSize = (nBodies + blockSize - 1) / blockSize;


    double totalTime = 0.0;
    cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);

    // iterate through and perform simulation
    for (int iter = 1; iter <= nIters; iter++) {
        StartTimer();

        // body force  and integration CUDA kernel and memory copy

        //bodyForceGPU<<<gridSize, blockSize>>>(d_p, dt, nBodies);
        bodyForceStride<<<sm, 128>>>(d_p, dt, nBodies);
        //integrate<<<gridSize, blockSize>>>(d_p, dt, nBodies);
        //cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();


        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1) { // First iter is warm up
            totalTime += tElapsed;
        }
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    }

    double avgTime = totalTime / (double)(nIters-1);

    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
           nIters, 1/avgTime);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    // free memory
    free(h_buf);
    cudaFree(d_buf);
    return 0;
}
