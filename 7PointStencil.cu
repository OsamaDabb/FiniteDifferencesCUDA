#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <array>

#include <iostream>

// constant struct to pass to GPU
struct constants {
    int size = 130;
    double alpha = 0.1;
    double delta_t = 0.1;
    double delta_x = 0.1;
    double T = 10;
};

__constant__ constants deviceConstants;

int main() {
    
    const constants hostConstants;
    
    // creating temperatures on host memory
    double* temperatures[hostConstants.size][hostConstants.size][hostConstants.size];

    // assigning initial conditions for system
    for (int i=0;i < hostConstants.size; i++) {
        for (int j=0; j < hostConstants.size; j++){
            for (int k=0; k < hostConstants.size; k++){

                // temperature for front and bottom faces
                if (j == 0 || i == hostConstants.size - 1) {
                    *temperatures[i][j][k] = 50.0;
                }

                // temperature for two side faces
                else if (k == 0 || k == hostConstants.size - 1){
                    *temperatures[i][j][k] = 100;
                }

                // temperature for remainder of box
                else{
                    *temperatures[i][j][k] = 20;
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
    dim3 numBlocks((hostConstants.size - 2) / 16, (hostConstants.size - 2) / 16, (hostConstants.size - 2) / 16 );
    dim3 threadsPerBlock(16, 16, 16);

    return 0;
}