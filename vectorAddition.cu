#include<iostream>

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

   // vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    
    vecAddKernel<<<ceil(1), 2>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    printf("%f %f", *C_h, *(C_h + 1));
}

int main() {
    float A[2] = {1, 2};
    float B[2] = {3, 4};
    float C[2] = {};

    vecAdd(A, B, C, 2);
}