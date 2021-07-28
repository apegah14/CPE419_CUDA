#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
 
#define SAMPLES 256000000
#define THREADS 8
int num_inside_pt = 0;

pthread_mutex_t sample_mutex;
pthread_barrier_t sample_barrier;


int sample(int num_samples) {
    int num_inside = 0;
    float x = 0;
    float y = 0;

    unsigned int rand_seed = (int)time(NULL);
    for (int i = 0; i < num_samples; i++) {
        x = (double)rand_r(&rand_seed) / (double)RAND_MAX;
        y = (double)rand_r(&rand_seed) / (double)RAND_MAX;
        if (hypot(x, y) <= 1) {
            num_inside++;
        }
    }
    return num_inside;
}


int sampleOMP(int num_samples) {
    int num_inside = 0;
    int num_inside_thread = 0;
    float x = 0;
    float y = 0;
    int i;
    #pragma omp parallel private(x, y, i) shared(num_samples) num_threads(8)
    {
        unsigned int rand_seed = (int)time(NULL) * omp_get_thread_num();
        #pragma omp for reduction(+:num_inside)
        for (i = 0; i < num_samples; i++) {
            x = (double)rand_r(&rand_seed) / (double)RAND_MAX;
            y = (double)rand_r(&rand_seed) / (double)RAND_MAX;
            if (sqrt((x * x) + (y * y)) <= 1) {
                num_inside += 1;
            }
        }
    }
    return num_inside;
}

// pthreads implementation of Monte Carlo Pi
void* samplePT(void* thread) {
    int thread_num = (intptr_t) thread;
    float x, y;
    int num_inside = 0;

    unsigned int rand_seed = (int)time(NULL) * thread_num;
    for (int i = thread_num * SAMPLES / THREADS; i < ((thread_num + 1) * SAMPLES / THREADS); i++) {
        x = (double)rand_r(&rand_seed) / (double)RAND_MAX;
        y = (double)rand_r(&rand_seed) / (double)RAND_MAX;
        if (sqrt((x * x) + (y * y)) <= 1) {
            num_inside++;
        }
    }

    pthread_mutex_lock(&sample_mutex);
    num_inside_pt += num_inside;
    pthread_mutex_unlock(&sample_mutex);
}


int main() {
    // time stuff
    struct timespec begin, end;
    double seq_elapsed, pt_elapsed, omp_elapsed, omp_start, omp_end;
    float num_inside, pi;
    int samples = SAMPLES;

    pthread_t sample_threads[THREADS];
    pthread_mutex_init(&sample_mutex, NULL);

    // sequential    
    clock_gettime(CLOCK_MONOTONIC, &begin);
    num_inside = sample(samples);
    pi = (4 * num_inside) / samples;
    clock_gettime(CLOCK_MONOTONIC, &end);
    seq_elapsed = end.tv_sec - begin.tv_sec;
    seq_elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("Pi for pthreads= %.6f\n", pi);
    printf("Sequential Time: %.4f sec\n", seq_elapsed);
    
    // pthreads
    clock_gettime(CLOCK_MONOTONIC, &begin);
    // spawn threads
    for (int i = 0; i < THREADS; i++) {
        pthread_create(&sample_threads[i], NULL, &samplePT, (void*)(intptr_t) i);
    }
    for (int i = 0; i < THREADS; i++) {
        pthread_join(sample_threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    seq_elapsed = end.tv_sec - begin.tv_sec;
    seq_elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("Pi for sequential (keep in mind different seed)= %.6f\n", pi);
    printf("p-thread Time: %.4f sec\n", seq_elapsed);

    // OpenMP
    omp_start = omp_get_wtime();
    num_inside = sampleOMP(samples);
    pi = (4 * num_inside) / samples;
    omp_end = omp_get_wtime();
    omp_elapsed = omp_end - omp_start;    
    printf("Pi for OpenMP= %.6f\n", pi);
    printf("OMP Time: %.4f sec\n", omp_elapsed);
    return 0;
}