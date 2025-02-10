
#include <iostream>

__global__
void matrixVectorMulKernel(float *b, float *c, float *a, int n) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < n) {
        float sum = 0;

        for(int i = 0; i < n; i++) {
            sum += b[row*n + i] * c[i];
        }
        
        a[row] = sum;
    }

}

int main(void) {
    int n = 1024
    float *a;
    float *b;
    float *c;

    a = (float*)malloc(n*n*sizeof(float));
    b = (float*)malloc(n*sizeof(float));
    c = (float*)malloc(n*sizeof(float));

    for(int i=0; i<n*n; i++) {
        a[i] = i + 1;
    }
    for(i=n*n; i<(n*n + n); i++) {
        b[i] = i + 1;
    }

    float *a_d, *b_d, *c_d;

    cudaMalloc(*a_d, n*n*sizeof(float));
    cudaMalloc(*b_d, n*n*sizeof(float));
    cudaMalloc(*c_d, n*n*sizeof(float));

    cudaMemCpy(a_d, a, n*n*sizeof(float), cudaMemCpyHostToDevice);
    cudaMemCpy(b_d, b, n*sizeof(float), cudaMemCpyHostToDevice);
    
    dim3 dimBlock = 256;
    dim3 dimGrid = ceil(n/256.0);

    matrixVectorMulKernel<<<dimBlock, dimGrid>>>(b_d, c_d, a_d, n);

    cudaMemCpy(c, c_d, n*sizeof(float), cudaMemCpyDeviceToHost);
    
    return 0;
}

