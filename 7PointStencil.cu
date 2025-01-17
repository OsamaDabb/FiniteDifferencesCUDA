#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

// constant struct to pass to device
struct constants {
    // length of each axis of grid
    int size;
    // number of time-steps
    int T;
    // thermal diffusivity constant
    double alpha;
    // time and length step-size
    double delta_t;
    double delta_x;
};

// initializing device constants
__constant__ constants deviceConstants;

// redefined here for code readability later
int size = 64;

// defining constants for test
const constants hostConstants = 
{
    size:size,
    T: 5000,
    alpha: 1.0,
    delta_t: 0.01,
    delta_x: 0.1
};

/*
Helper for printing the values in a (cubic) 3D array
in a visually clear way.
*/
void print3DArr(const double* arr, int size){

    if (size > 16) return;

    std::cout << std::fixed << std::setprecision(1);

    std::cout << "[\n";
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++){

                int idx = i * size * size + j * size + k;
                std::cout <<    arr[idx] << " ";

            }
            std::cout << "   ";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

/*
helper to write the temperature data to the array
*/
void writeToFile(std::ofstream &file, double* arr, size_t arrSize){

    file.write(reinterpret_cast<char*>(arr), arrSize);

}

/*
@brief
Simulates the heat transfer equations using finite-difference approximation of the derivative.
dT/dt = alpha*dT^2/d(x,y,z)
Does so in parallel for each point using the 7-point stencil. Expects pointers to two (cubic) arrays of equal dimension. 

@param temperature_ptr CUDA 3D ptr to initial condition data
@param gradient_ptr CUDA 3D ptr to memory used for gradient storage
*/
__global__ void calculateGradient(double* temperature_ptr,double* gradient_ptr) {

    // getting unique indices in the three directions
    // (+1 because outer-faces are fixed for dirichlet conditions)
    int x = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int y = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int z = threadIdx.z + blockIdx.z * blockDim.z + 1;

    int size = deviceConstants.size;

    // Out-of-bounds check
    if (x >= size - 1 || y >= size - 1 || z >= size - 1) return; 

    int idx = z * size * size + y * size + x;

    // accessing values for 7-point stencil
    double left   = temperature_ptr[(z * size * size) + y * size + (x - 1)];
    double right  = temperature_ptr[(z * size * size) + y * size + (x + 1)];
    double down   = temperature_ptr[(z * size * size) + (y - 1) * size + x];
    double up     = temperature_ptr[(z * size * size) + (y + 1) * size + x];
    double below  = temperature_ptr[((z - 1) * size * size) + y * size + x];
    double above  = temperature_ptr[((z + 1) * size * size) + y * size + x];
    double center = temperature_ptr[idx];  // current position

    // approximate gradient dT/dt
    gradient_ptr[idx] = deviceConstants.alpha * (1 /(2 * deviceConstants.delta_x)) * 
                                    (left + right + down + up + below + above - 6 * center);

}

__global__ void updateTemperature(double* temperature_ptr,double* gradient_ptr) {

    int x = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int y = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int z = threadIdx.z + blockIdx.z * blockDim.z + 1;
    int size = deviceConstants.size;
    int idx = z * size * size + y * size + x;

    if (x >= size - 1 || y >= size - 1 || z >= size - 1) return; 

    // update temperatures as Tn = Tn-1 + dT/dt * dt
    temperature_ptr[idx] = temperature_ptr[idx] + gradient_ptr[idx] * deviceConstants.delta_t;
}

int main() {

    // begin timing measurement
    auto start = std::chrono::high_resolution_clock::now();
    
    // creating temperatures on host memory
    double* temperatures = new double[size * size * size];
    size_t arrSize = sizeof(double)*size*size*size;

    // creating an output binary
    std::ofstream outputFile("output.bin", std::ios::binary | std::ios::trunc);

    int num_iterations = 1 + hostConstants.T / 10;

    // including array size and # of iterations for future use
    outputFile.write(reinterpret_cast<char*>(&size), sizeof(int)); 
    outputFile.write(reinterpret_cast<char*>(&num_iterations), sizeof(int));
     
    // assigning initial conditions for system
    for (int i=0;i < size; i++) {
        for (int j=0; j < size; j++){
            for (int k=0; k < size; k++){

                int idx = (i * size * size) + (j * size) + k;

                // temperature for front and back faces
                if (i == 0 || i == size - 1){
                    temperatures[idx] = 200.0;
                }
                else if (j == 0 || k == 0){
                    temperatures[idx] = 50.0;
                }

                // temperature for remainder of box
                else {
                    temperatures[idx] = 25.0;
                }

            }
        }
    }

    writeToFile(outputFile, temperatures, arrSize);

    // moving constants to device
    cudaError_t err = cudaMemcpyToSymbol(deviceConstants, &hostConstants, sizeof(constants));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to symbol failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // defining thread/block counts
    // (size - 2)^3 total workers (ignoring the far end of each face)
    // count of blocks is ceil ((size - 2) / 16)^3 since each block is 16^3 threads
    int blockCount = ceil((size - 2) / 8.0);
    dim3 numBlocks(blockCount, blockCount, blockCount);

    int threadsCount = min(8, size - 2);
    dim3 threadsPerBlock(threadsCount, threadsCount, threadsCount);

    // display initial conditions
    print3DArr(temperatures, size);

    // moving the temperature data to the device
    double* temperature_ptr;
    double* gradient_ptr;

    cudaMalloc(&temperature_ptr, arrSize);
    cudaMalloc(&gradient_ptr, arrSize);
    cudaMemcpy(temperature_ptr, temperatures, arrSize, cudaMemcpyDefault);

    // call kernel here
    for (int t = 1; t < hostConstants.T; t += 1) {

        calculateGradient<<<numBlocks, threadsPerBlock>>>(temperature_ptr, gradient_ptr);
        cudaDeviceSynchronize();

        updateTemperature<<<numBlocks, threadsPerBlock>>>(temperature_ptr, gradient_ptr);
        cudaDeviceSynchronize();

        // print results across 10 evenly spaced intervals
        if (t % 10 == 0) {

            std::cout << "Step: " << t << ", Time: " << t*hostConstants.delta_t << "s\n";
            cudaMemcpy(temperatures, temperature_ptr, arrSize, cudaMemcpyDefault);
            print3DArr(temperatures, size);
            writeToFile(outputFile, temperatures, arrSize);
        }
    }

    std::cout << "Final result, Time: " << hostConstants.T*hostConstants.delta_t << "s\n";
    cudaMemcpy(temperatures, temperature_ptr, arrSize, cudaMemcpyDefault);
    print3DArr(temperatures, size);
    writeToFile(outputFile, temperatures, arrSize);


    // free all memory
    cudaFree(temperature_ptr);
    cudaFree(gradient_ptr);
    delete[] temperatures;
    outputFile.close();

    // Calculate elapsed time in milliseconds
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ms" << std::endl;

    return 0;
}