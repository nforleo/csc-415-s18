
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <immintrin.h>

void init_array(float *a, int n) {
	std::srand(std::time(nullptr));
	for (int i = 0 ; i < n ; i++) {
		a[i] = -500. + (float) std::rand() / (float) (RAND_MAX/1000.);
	}
}

void find_max_seq(float *a, int n, float *max) {
	assert(n > 0);
	*max = a[0];
	for (int i = 1 ; i < n ; i++) {
		if (a[i] > *max) {
			*max = a[i];
		}
	}
}

void find_max_avx(float *a, int n, float *max) {
	assert(n > 0);

	// create variables in AVX registers
	__m256 reg;
	__m256 regmax; 

	// calculate number of vectorized operations
	int n_iter = n / 8;

	int st;	
	if (n_iter > 0) {
		// load 256/32=8 values
		regmax = _mm256_load_ps(a);
		// iterate and calculate max values in parallel	
		for (int i = 1, offset = 8 ; i < n_iter ; i++, offset += 8) {
			reg = _mm256_load_ps(a + offset);
			regmax = _mm256_max_ps(regmax, reg);
		}
		// find maximum in register
		float *ptr = (float *) &regmax;
		*max = ptr[0];
		for (int i = 1 ; i < 8 ; i++) {
			if (ptr[i] > *max) {
				*max = ptr[i];
			}
		}
		st = n - (n % 8);
	} else {
		*max = a[0];
		st = 1;
	}

	// process any remaining values sequentially
	for ( ; st < n ; st++) {
		if (a[st] > *max) {
			*max = a[st];
		}
	}
}

int main(int argc, char **argv) {
	char hostname[512];
	gethostname(hostname, 512);
	std::cout << "Running on " << hostname << std::endl;
  
	// define array size
	int n = std::atoi(argv[1]);

	// allocate aligned memory blocks
	float *arr_f = (float *) _mm_malloc(n * sizeof(float), 32);

	// initialize array
	init_array(arr_f, n);

	float max1;
	float max2;

	// find max sequentially
	auto start = std::chrono::high_resolution_clock::now();
	find_max_seq(arr_f, n, &max1);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end -start;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "sequential: " << diff.count() << " sec\t" << max1 << "\t" << n << std::endl;

	// find max using AVX
	start = std::chrono::high_resolution_clock::now();
	find_max_avx(arr_f, n, &max2);
	end = std::chrono::high_resolution_clock::now();
	diff = end -start;
	std::cout << "vectorized: " << diff.count() << " sec\t" << max2 << "\t" << n << std::endl;

	// verify both are equal (check correctness)
	assert(max1 == max2);

	// free memory
	_mm_free(arr_f);

	return 0;
}
