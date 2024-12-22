#include<stdint.h>
#include<iostream>

int main(){
    int64_t nvtx_scale = ((int64_t)1)<<14;

    uint64_t* cost = (uint64_t*)malloc(sizeof(uint64_t)*nvtx_scale);

    for(int64_t i=0; i < nvtx_scale; i++)
        cost[i] = uint64_t(123456789);

    uint64_t* dcost;
    cudaMalloc(&dcost, nvtx_scale*sizeof(uint64_t));
    cudaMemcpy(dcost, cost, sizeof(uint64_t)*nvtx_scale, cudaMemcpyHostToDevice);

    memset(cost, 0, sizeof(uint64_t)*nvtx_scale);
    cudaMemcpy(cost, dcost, sizeof(uint64_t)*nvtx_scale, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++) {
        std::cout << i << " " << cost[i] << std::endl;
    }

    return 0;
}