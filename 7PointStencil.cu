#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

#include <iostream>

// constant struct to pass to GPU
struct constants {
    int size;
    double alpha;
    double delta_t;
    double delta_x;
    double T;
};

__constant__ constants deviceConstants;

void print3DArr(double* arr, int size){

    std::cout << "[";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++){

                int idx = i * size * size + j * size + k;
                std::cout << arr[idx] << " ";

            }
            std::cout << "\n";
        }
        std::cout << "********\n";
    }
    std::cout << "]\n";
}


int main() {
    
    const constants hostConstants = 
    {
        size: 18,
        alpha: 0.1,
        delta_t: 0.1,
        delta_x: 0.1,
        T: 10
    };
    
    int size = hostConstants.size;
    
    // creating temperatures on host memory
    double* temperatures = new double[size * size * size];
     
    // assigning initial conditions for system
    for (int i=0;i < size; i++) {
        for (int j=0; j < size; j++){
            for (int k=0; k < size; k++){

                int idx = i * size * size + j * size + k;

                // temperature for front and bottom faces
                if (j == 0 || i == size - 1) {
                    temperatures[idx] = 50.0;
                }

                // temperature for two side faces
                else if (k == 0 || k == size - 1){
                    temperatures[idx] = 100;
                }

                // temperature for remainder of box
                else{
                    temperatures[idx] = 20;
                }

            }
        }
    }

    // moving constants to device
    cudaError_t err = cudaMemcpyToSymbol(deviceConstants, &hostConstants, sizeof(constants));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to symbol failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // assigning workers
    // (size - 2)^3 total workers (ignoring the far end of each face)
    // total blocks is (size - 2)^3 // 16^3 since each block is 16^3
    dim3 numBlocks((size - 2) / 16, (size - 2) / 16, (size - 2) / 16 );
    dim3 threadsPerBlock(16, 16, 16);

    print3DArr(temperatures, size);


    // define size requirements of temp and next in GPU
    cudaExtent extent = make_cudaExtent(size * sizeof(double), size, size);
    // assign pointers and memory for both
    cudaPitchedPtr temperature_ptr;
    cudaPitchedPtr nextiter_ptr;
    cudaMalloc3D(&temperature_ptr, extent);
    cudaMalloc3D(&nextiter_ptr, extent);

    // Create a host to device copy descriptor for temperatures
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(temperatures, size * sizeof(double), size, size);
    copyParams.dstPtr = temperature_ptr;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    // move temperature data to GPU
    cudaMemcpy3D(&copyParams);

    // now move same data to nextiter
    copyParams.dstPtr = nextiter_ptr;
    cudaMemcpy3D(&copyParams);

    // call kernel here
    /******************



    ******************/

    // freeing up memory
    cudaFree(temperature_ptr.ptr);
    cudaFree(nextiter_ptr.ptr);
    delete[] temperatures;

    return 0;
}