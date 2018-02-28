
#include <iostream>
#include <unistd.h>
#include <immintrin.h>

int main() {
	char hostname[512];
	gethostname(hostname, 512);
	std::cout << "running on " << hostname << std::endl;

	// define registers
	__m256 r0, r1;

	// create some single precision values
	float v1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	float v2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	float u[8];

	// load data into registers
	r0 = __builtin_ia32_loadups256(v1);
	r1 = __builtin_ia32_loadups256(v2);

	// multiply registers
	r0 = __builtin_ia32_mulps256(r0, r1);

	// copy data to output array
	__builtin_ia32_storeups256(u, r0);

	// print array
	for (int i = 0 ; i < 8 ; i++) {
		std::cout << u[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}
