#include <stdio.h>

#define N 1024 // size of matrices
#define THREADS_PER_BLOCK 32

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *a, float *b, float *c, int n)
{
    // Calculate the row and column indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the sum for the current element
    float sum = 0;

    // Perform dot product for the corresponding row of matrix A and column of matrix B
    for (int i = 0; i < n; i++) {
        sum += a[row * n + i] * b[i * n + col];
    }

    // Store the result in the output matrix
    c[row * n + col] = sum;
}

int main()
{
    float *a, *b, *c; // input and output matrices
    float *d_a, *d_b, *d_c; // device input and output matrices

    // Allocate memory for matrices on host
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * N * sizeof(float));
    c = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices on host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i * N + j;
            b[i * N + j] = j * N + i;
            c[i * N + j] = 0;
        }
    }

    // Allocate memory for matrices on device
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel execution
    dim3 gridSize((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // Execute the CUDA kernel for matrix multiplication
    matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy the result matrix from device to host
    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory on host and device
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
