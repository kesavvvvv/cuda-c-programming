%%cuda


#include <iostream>

__global__
void matrixVectorMulKernel(float *b, float *c, float *a, int n) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < n) {
        float s = 0.0f;

        for(int i = 0; i < n; i++) {
            s += b[row*n + i] * c[i];
        }
        
        a[row] = s;
    }

}

int main(void) {
    int n = 100024;
    float *a;
    float *b;
    float *c;
    float *d;

    cudaSetDevice(0);
    

    d = (float*)malloc(n*sizeof(float));

    a = (float*)malloc(n*n*sizeof(float));
    b = (float*)malloc(n*sizeof(float));
    c = (float*)malloc(n*sizeof(float));


    for(int i=0; i<n*n; i++) {
        a[i] = 1.0;
    }
    for(int i=0; i<n; i++) {
        b[i] = 1.0;
    }
    /*
    for(int i=0; i<n; i++) {
        float s = 0.0f;
        for(int j=0; j<n; j++) {
            s += a[i*n + j] * b[j];
        }
        d[i] = s;
    }*/

    float *a_d, *b_d, *c_d;

    a_d = (float*)malloc(n*n*sizeof(float));
    b_d = (float*)malloc(n*sizeof(float));
    c_d = (float*)malloc(n*sizeof(float));

    cudaMalloc(&a_d, n*n*sizeof(float));
    cudaMalloc(&b_d, n*sizeof(float));
    cudaMalloc(&c_d, n*sizeof(float));

    cudaMemcpy(a_d, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock = 1024;
    dim3 dimGrid = ceil(n/1024.0);

    std::cout<<ceil(n/256.0);

    cudaDeviceSynchronize();

    matrixVectorMulKernel<<<dimBlock, dimGrid>>>(a_d, b_d, c_d, n);


    cudaMemcpy(c, c_d, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
/*
    for(int i = 0; i<n; i++) {
        std::cout<<d[i]<<" ";
    }
*/
    int fails = 0;
    float smallRange = 0.00001;
    
    for(int i = 0; i < n; i++) {
        if(c[i] != n) {
            fails++;
        }
        //printf("%f\n", c[i]);
    }
    std::cout<<"\n"<<fails<<std::endl;
    printf("sucess");

    return 0;
}