
#include<iostream>

__global__
void matrixMultiplicationKernel(float *m, float *n, float *p, int i, int j, int k) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < i && col < k) {
        int pValue = 0;
        for(int idx = 0; idx < j; idx++) {
            pValue += m[row * j + idx] * n[idx * k + col];
        }

        p[row * k + col] = pValue;
    }

}

int main() {

    int i = 4, j = 6, k = 3;
    float m[24] = {4,5,7,7,6,4,2,1,0,8,7,6,5,6,8,9,0,6,5,7,1,2,1,4};
    float n[18] = {2,2,3,4,7,8,6,8,5,9,6,7,5,4,7,9,5,3};
    float p[12] = {};

    float *m_d, *n_d, *p_d;

    m_d = (float*)malloc(i*j*sizeof(float));
    n_d = (float*)malloc(j*k*sizeof(float));
    p_d = (float*)malloc(i*k*sizeof(float));

    cudaMalloc(&m_d, i*j*sizeof(float));
    cudaMalloc(&n_d, j*k*sizeof(float));
    cudaMalloc(&p_d, i*k*sizeof(float));

    cudaMemcpy(m_d, m, i*j*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(n_d, n, j*k*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimGrid = {static_cast<unsigned int>(ceil(k/16.0)), static_cast<unsigned int>(ceil(i/16.0))};
    dim3 dimBlock = {16, 16};

    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(m_d, n_d, p_d, i, j, k);

    cudaMemcpy(p, p_d, i*k*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(m_d);
    cudaFree(n_d);
    cudaFree(p_d);
    
    for(int idx = 0; idx < 12; idx++) {
        std::cout << p[idx] << " ";
    }

    return 0;
}