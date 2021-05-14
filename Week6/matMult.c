#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>

#define NUMTHREADS 4
#define N 1000

int AMat[N * N];
int BMat[N * N];
int seqMult[N * N];
int pthreadMult[N * N];
int ompMult[N * N];

void* pThreadMatMult(void * threadNum) {
    int core = (intptr_t) threadNum;
    int row, col, k;
    int sum;
    for (row = core * N / NUMTHREADS; row < (core + 1) * N / NUMTHREADS; row++) {
        for (col = 0; col < N; col++) {
            sum = 0;
            for (k = 0; k < N; k++) {
                sum += AMat[row * N + k] * BMat[k * N + col];
            }
            pthreadMult[row * N + col] = sum;
        }
    }
}

void seqMatMult() {
    int row, col, k;
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            int tempSum = 0;
            for (k = 0; k < N; k++) {
                tempSum += AMat[row * N + k] * BMat[k * N + col];
            }
            seqMult[row * N + col] = tempSum;
        }
    }
}

void multMatricesOpenMP() {
    int row, col, k;
    #pragma omp parallel num_threads(NUMTHREADS) private(row, col, k)
    {
    #pragma omp for schedule(static)
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                int tempSum = 0;
                for (k = 0; k < N; k++) {
                    tempSum += AMat[row * N + k] * BMat[k * N + col];
                }
                ompMult[row * N + col] = tempSum;
            }
        }
    }
}


int main() {
    int i;
    struct timespec begin, end;
    double seqElapsed, pThreadElapsed, MPStart, MPFinish, MPElapsed;

    pthread_t thread[NUMTHREADS];

    for(i = 0; i < N * N; i++)
    {
        AMat[i] = (int) rand() % 1000;
        BMat[i] = (int) rand() % 1000;
        seqMult[i] = 0;
        pthreadMult[i] = 0;
        ompMult[i] = 0;
    }

    // SEQ
    clock_gettime(CLOCK_MONOTONIC, &begin);
    seqMatMult(AMat, BMat, seqMult, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    seqElapsed = end.tv_sec - begin.tv_sec;
    seqElapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

    
    // PTHREAD
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (i = 0; i < NUMTHREADS; i++)
    {
        pthread_create(&thread[i], NULL, &pThreadMatMult, (void*)(intptr_t) i);
    }
    for (i = 0; i < NUMTHREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    pThreadElapsed = end.tv_sec - begin.tv_sec;
    pThreadElapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

    // OMP
    MPStart = omp_get_wtime();
    multMatricesOpenMP(AMat, BMat, ompMult, N);
    MPFinish = omp_get_wtime();
    MPElapsed = MPFinish - MPStart;

    for(i = 0; i < N * N; i++) {
        if (seqMult[i] != pthreadMult[i] || seqMult[i] != ompMult[i]) {
            printf("Error\n");
            break;
        }
    }

    printf("SEQ Time: %.4f s\n", seqElapsed);
    printf("PThread Time: %.4f s\n", pThreadElapsed);
    printf("OMP Time: %.4f s\n", MPElapsed);

    return 0;
}