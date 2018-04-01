#include <cassert>
#include <string>
#include <pthread.h>
#include "stdio.h"

// structure used for passing parameters
typedef	struct {
    int start_idx;
    int n;
    unsigned long *result;
    int *array;
} param;

// each thread will perform this function.
// the primary goal is to add all numbers from array
// and store the result in result.
void *par_sum(void *args) {
    param *params = (param *) args;
    params->result[0] = 0;
    int n = params->start_idx + params->n;
    for (int i = params->start_idx ; i < n ; i++) {
        params->result[0] += params->array[i];
        // this loop is unnecessary, it is just doing more
        // work per element, so you can see the impact of threads
        // when running the program, for example, try this out:
        // time ./prog 1000000 2
        // time ./prog 1000000 20
        for (int i = 0 ; i < 10000 ; i ++);
    }
}

int main (int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: ./prog <n_elem> <n_threads>\n");
        printf("This programs creates an array of `n_elem` with values 1, 2, 3, ...\n");
        printf("then `n_threads` are used to calculate the sum of the array\n");
        return 0;
    }
    // read number of elements and number of threads
    // as command line arguments
    int n_elem = std::stoi(argv[1]);
    int n_threads = std::stoi(argv[2]);

    // stores the total sum
    unsigned long total = 0;
    // stores partial sums calculated by each thread
    unsigned long result[n_threads];

    // creates array of `n_elem` and initializes its values
    int *array = new int[n_elem];
    for (int i = 0 ; i < n_elem ; i ++) {
        array[i] = i + 1;
    }

    // creates array of threads
    pthread_t *threads = new pthread_t[n_threads];
    // number of elements each thread will process
    int elem_per_thread = n_elem / n_threads;

    // creates all threads
    for (int i = 0 ; i < n_threads ; i++ ) {
        // prepares arguments to be passed to a thread
        param *values = new param;
        values->array = array;
        values->result = result + i;
        values->start_idx = i * elem_per_thread;
        values->n = elem_per_thread;
        // creates the thread
        int ret = pthread_create(&threads[i], NULL, par_sum, (void *)values);
        // confirming thread was created
        assert(! ret);
    }

    // sequentially add elements threads didnt process
    // this is because n_elem may not be a multiple of n_threads
    for (int i = n_elem-(n_elem%n_threads) ; i < n_elem ; i++) {
        total += array[i];
    }
    // wait until each threads finishes 
    for (int i = 0 ; i < n_threads ; i ++) {
        pthread_join(threads[i], NULL);
    }
    // add partial sums calculated by all threads
    for (int i = 0 ; i < n_threads ; i ++) {
        total += result[i];
    }

    printf("%ld %ld\n", total, ((unsigned long)n_elem*(n_elem+1))/2);

    // release memory
    delete [] array;
    delete [] threads;

    return 0;
}
