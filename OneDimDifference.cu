#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// constant struct to pass to GPU
struct constants {
    int size;
    double alpha;
    double delta_t;
    double delta_x;
    double T;
};

__constant__ constants gpuConstants;

void printArr(double* arr, int size){
    std::cout << "[";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "]\n";
}

// GPU kernel calculation
__global__ void finite_difference_kernel(double* temp, double* nextIteration) {
    int idx = threadIdx.x + 1;
    nextIteration[idx] = temp[idx] + gpuConstants.delta_t * gpuConstants.alpha / (2*gpuConstants.delta_x) * (temp[idx + 1] + temp[idx - 1] - 2 * temp[idx]);
}

int main() {

    const int size = 5;
    double* temperature = (double*)malloc(size * sizeof(double));

    // defining constants and sending to symbol memory
    constants hostConstants = { 
        size: size,
        alpha: 0.5f,
        delta_t: 0.05f,
        delta_x: 1.0f,
        T: 10.0f};
    cudaError_t err = cudaMemcpyToSymbol(gpuConstants, &hostConstants, sizeof(constants));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to symbol failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    int numBlocks = 1;
    dim3 threadsPerBlock(size - 2, 1, 1);

    // setting initial conditions
    temperature[0] = 50.0;
    temperature[size - 1] = 100.0;
    for (int i = 1; i < size - 1; i++) {
        temperature[i] = 20;
    }

    printArr(temperature, size);

    // allocating memory for arrays on GPU
    double* temp_ptr;
    double* next_iteration;

    cudaMalloc(&temp_ptr, size * sizeof(double));
    cudaMalloc(&next_iteration, size * sizeof(double));
    cudaMemcpy(temp_ptr, temperature, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(next_iteration, temperature, size * sizeof(double), cudaMemcpyHostToDevice);

    int iteration = 0;

    // running iterations of steps delta_t until time T is reached
    for (double t = 0; t < hostConstants.T; t += hostConstants.delta_t) {
        finite_difference_kernel<<<numBlocks, threadsPerBlock>>>(temp_ptr, next_iteration);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
            return -1;
        }

        // swap temperature with next_iteration to reuse on next iteration
        double* tmp = temp_ptr;
        temp_ptr = next_iteration;
        next_iteration = tmp;

        // print out the arr at whole-number t
        if (iteration % 10 == 0){

            cudaMemcpy(temperature, temp_ptr, size * sizeof(double), cudaMemcpyDefault);

            std::cout << "time (s): " << t << "\n";
            printArr(temperature, size);
        }
        iteration++;

    }

    // copy results back to host
    cudaMemcpy(temperature, temp_ptr, size * sizeof(double), cudaMemcpyDefault);

    // output results
    printArr(temperature, size);

    // free GPU memory
    cudaFree(temp_ptr);
    cudaFree(next_iteration);

    return 0;
}
