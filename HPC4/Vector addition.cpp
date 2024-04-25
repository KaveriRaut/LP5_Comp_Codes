#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

int main(int argc, char *argv[])
{
    // Size of vectors
    int n = 100000;

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    // Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++)
    {
        // Filling arrays with sine and cosine values squared
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for (i = 0; i < n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}



/*Explanation:

This program performs vector addition using CUDA, which is a parallel computing platform and application programming interface model created by NVIDIA.
The vecAdd CUDA kernel is defined to perform element-wise addition of two arrays a and b and store the result in array c.
In the main function, memory is allocated for host and device vectors (h_a, h_b, h_c, d_a, d_b, d_c).
Values are initialized in host arrays h_a and h_b with the squares of sine and cosine values.
Memory is copied from host to device using cudaMemcpy.
The number of threads per block (blockSize) and the number of blocks in the grid (gridSize) are determined based on the size of the vectors.
The CUDA kernel vecAdd is launched with the specified grid size and block size.
The resulting vector h_c is copied back from the device to the host.
The sum of elements in vector h_c is computed and printed, divided by n, which should be approximately equal to 1 within error.
Finally, memory allocated on both the host and the device is freed.
This program demonstrates the basic concepts of CUDA programming, including memory allocation, kernel invocation, and data transfer between the host and the device.*/
