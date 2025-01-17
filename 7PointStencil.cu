#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <chrono>

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

const int size = 128;

/*
Helper for printing the values in a (cubic) 3D array
in a visually clear way.
*/
void print3DArr(const double* arr, int size){

    if (size > 6) return;

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
@brief
Simulates the heat transfer equations using finite-difference approximation of the derivative.
dT/dt = alpha*dT^2/d(x,y,z)
Does so in parallel for each point using the 7-point stencil. Expects pointers to two (cubic) arrays of equal dimension. 

@param temperature_ptr CUDA 3D ptr to initial condition data
@param gradient_ptr CUDA 3D ptr to memory used for gradient storage
*/
__global__ void calculateGradient(double* temperature_ptr,double* gradient_ptr) {

    int size = deviceConstants.size;
    // creating shared memory for repeated access data
    extern __shared__ double s_temperatures[];

    // getting unique indices in the three directions
    // (+1 because outer-faces are fixed for dirichlet conditions)
    int global_x = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int global_y = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int global_z = threadIdx.z + blockIdx.z * blockDim.z + 1;
    int global_idx = global_z * size * size + global_y * size + global_x;
    // Out-of-bounds check
    if (global_x >= size - 1 || global_y >= size - 1 || global_z >= size - 1) return; 

    // moving repeated access memory to shared memory
    int local_idx = threadIdx.z * size * size + threadIdx.y * size + threadIdx.z;
    s_temperatures[local_idx] = temperature_ptr[global_idx];

    // accessing values for 7-point stencil
    // each direction accesses from shared memory if available
    // otherwise from global memory
    double left, right, down, up, inwards, outwards, center;
    // across x
    if (threadIdx.x == 0) {left = temperature_ptr[(global_z * size * size) + global_y * size + (global_x - 1)];}
    else {left = s_temperatures[local_idx - 1];}
    if (threadIdx.x == blockDim.x - 1) {right = temperature_ptr[(global_z * size * size) + global_y * size + (global_x + 1)];}
    else {right = s_temperatures[local_idx + 1];}

    // across y
    if (threadIdx.y == 0) {down = temperature_ptr[(global_z * size * size) + (global_y - 1) * size + global_x];}
    else {down = s_temperatures[local_idx - size * 1];}
    if (threadIdx.y == blockDim.y - 1){ up = temperature_ptr[(global_z * size * size) + (global_y + 1) * size + global_x];}
    else { up = s_temperatures[local_idx + size * 1];}

    // across z
    if (threadIdx.z == 0) { inwards = temperature_ptr[((global_z - 1) * size * size) + global_y * size + global_x];}
    else { inwards = s_temperatures[local_idx - size * size * 1];}
    if (threadIdx.z == blockDim.z - 1){ outwards = temperature_ptr[((global_z + 1) * size * size) + global_y * size + global_x];;}
    else {outwards = s_temperatures[local_idx + size * size * 1];}

    center = s_temperatures[local_idx];  // current position

    // approximate gradient dT/dt
    gradient_ptr[global_idx] = deviceConstants.alpha * (1 /(2 * deviceConstants.delta_x)) * 
                                    (left + right + down + up + inwards + outwards - 6 * center);

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
    
    // defining constants for test
    const constants hostConstants = 
    {
        size:size,
        T: 5000,
        alpha: 0.1,
        delta_t: 0.01,
        delta_x: 0.1
    };
    
    // creating temperatures on host memory
    double* temperatures = new double[size * size * size];
     
    // assigning initial conditions for system
    for (int i=0;i < size; i++) {
        for (int j=0; j < size; j++){
            for (int k=0; k < size; k++){

                int idx = (i * size * size) + (j * size) + k;

                // temperature for front and back faces
                if (i == 0 || i == size - 1) {
                    temperatures[idx] = 80.0;
                }

                // temperature for four side faces
                else if (k == 0 || k == size - 1 || j == 0 || j == size - 1){
                    temperatures[idx] = 50.0;
                }

                // temperature for remainder of box
                else {
                    temperatures[idx] = 20.0;
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

    // defining thread/block counts
    // (size - 2)^3 total workers (ignoring the far end of each face)
    // count of blocks is ceil ((size - 2) / 16)^3 since each block is 16^3 threads
    int blockCount = ceil((size - 2) / 16.0);
    dim3 numBlocks(blockCount, blockCount, blockCount);

    int threadsCount = min(16, size - 2);
    dim3 threadsPerBlock(threadsCount, threadsCount, threadsCount);

    // display initial conditions
    print3DArr(temperatures, size);

    // moving the temperature data to the device
    double* temperature_ptr;
    double* gradient_ptr;

    size_t arrSize = sizeof(double)*size*size*size;
    cudaMalloc(&temperature_ptr, arrSize);
    cudaMalloc(&gradient_ptr, arrSize);
    cudaMemcpy(temperature_ptr, temperatures, arrSize, cudaMemcpyDefault);

    // call kernel here
    for (int t = 1; t < hostConstants.T; t += 1) {

        calculateGradient<<<numBlocks, threadsPerBlock, arrSize>>>(temperature_ptr, gradient_ptr);
        cudaDeviceSynchronize();

        updateTemperature<<<numBlocks, threadsPerBlock>>>(temperature_ptr, gradient_ptr);
        cudaDeviceSynchronize();

        // print results across 10 evenly spaced intervals
        if (t % (hostConstants.T / 10) == 0) {

            std::cout << "Step: " << t << ", Time: " << t*hostConstants.delta_t << "s\n";
            cudaMemcpy(temperatures, temperature_ptr, arrSize, cudaMemcpyDefault);
            print3DArr(temperatures, size);
        }
    }

    std::cout << "Final result, Time: " << hostConstants.T*hostConstants.delta_t << "s\n";
    cudaMemcpy(temperatures, temperature_ptr, arrSize, cudaMemcpyDefault);
    print3DArr(temperatures, size);

    // free all memory
    cudaFree(temperature_ptr);
    cudaFree(gradient_ptr);
    delete[] temperatures;

    // Calculate elapsed time in milliseconds
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ms" << std::endl;

    return 0;
}