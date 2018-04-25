#define THREADS_PER_BLOCK 128

#include <random>
#include <iostream>

void print_array(float *p, int n) {
    std::cout << n << " elements" << std::endl;
    for (int i = 0 ; i < n ; i++) {
        std::cout << *(p+i) << " ";
    }
    std::cout << std::endl << std::endl;
}

__global__ void convolve(int N, float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        float result = 0;
        for (int i = 0 ; i < 3 ; i++) {
            result += input[index + i];
        }
        output[index] = result / 3.0;
    }
}

int main(int argc, char* argv[]) {
    // define the size of the input array
    int N = 135;

    // create pointers for the CPU arrays
    float *input = new float[N+2];
    float *output = new float[N];

    // generate data randomly and store it in the CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> mydist(1,10);
    for (int i = 0 ; i < (N+2) ; i ++) {
        input[i] = mydist(gen);
    }

    print_array(input, N+2);

    // create pointers for the CUDA arrays
    float *dev_in;
    float *dev_out;

    // allocate space for the arrays in the GPU
    cudaMalloc(&dev_in, sizeof(float) * (N+2));
    cudaMalloc(&dev_out, sizeof(float) * N);

    // transfer data from CPU to GPU
    cudaMemcpy(dev_in, input, sizeof(float) * (N+2), cudaMemcpyHostToDevice);

    // do the work in the GPU
    convolve<<<std::ceil((float)N/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(N, dev_in, dev_out);

    // transfer results from GPU to CPU
    cudaMemcpy(output, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    print_array(output, N);

    // free the memory allocated in the CPU
    delete [] input;
    delete [] output;

    // free the memory allocated in the GPU
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}
