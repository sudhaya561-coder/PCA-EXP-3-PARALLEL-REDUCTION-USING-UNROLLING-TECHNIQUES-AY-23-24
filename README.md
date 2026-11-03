# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24

<h3>ENTER YOUR NAME:UDHAYA S</h3> 
<h3>ENTER YOUR REGISTER NO : 212225230287</h3> 
<h3>EX. NO : 3</h3> 
<h3>DATE: 03/11/2026</h3>
<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
```python
#Execution Steps in Google Colab:
#Step 1: Open Google Colab and create a new notebook.

#Step 2: Run the following command in the first cell to install the CUDA extension.
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git



#Step 3: Run the following command in the second cell to load the CUDA extension.
%load_ext nvcc4jupyter



#Step 4: Create a new cell and paste the complete CUDA program. Make sure the first line is:
%%cuda
```

```python

%%writefile reduction.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess)                                              \
    {                                                                      \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);             \
        fprintf(stderr, "code: %d, reason: %s\n", error,                   \
                cudaGetErrorString(error));                                \
        exit(1);                                                           \
    }                                                                      \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Reduction using Unrolling 8
__global__ void reduceUnrolling8(int *g_idata, int *g_odata,
                                 unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];

        g_idata[idx] = a1 + a2 + a3 + a4 +
                       b1 + b2 + b3 + b4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// Reduction using Unrolling 16
__global__ void reduceUnrolling16(int *g_idata, int *g_odata,
                                  unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];
        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];

        g_idata[idx] = a1 + a2 + a3 + a4 +
                       b1 + b2 + b3 + b4 +
                       c1 + c2 + c3 + c4 +
                       d1 + d2 + d3 + d4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// CPU Recursive Reduction
int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Starting reduction on device %d: %s\n",
           dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // Initialization
    int size = 1 << 24;

    printf("\nArray size: %d\n", size);

    // Block configuration
    int blocksize = 512;

    if (argc > 1)
        blocksize = atoi(argv[1]);

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    printf("Grid: %d Block: %d\n", grid.x, block.x);

    // Allocate host memory
    size_t bytes = size * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    if (!h_idata || !h_odata || !tmp)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize input data
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // Allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;

    CHECK(cudaMalloc((void **)&d_idata, bytes));
    CHECK(cudaMalloc((void **)&d_odata,
                     grid.x * sizeof(int)));

    // CPU Reduction
    iStart = seconds();

    int cpu_sum = recursiveReduce(tmp, size);

    iElaps = seconds() - iStart;

    printf("CPU reduce elapsed %f sec CPU sum: %d\n",
           iElaps, cpu_sum);

    // Kernel 1: Unrolling 8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling8<<<grid.x / 8, block>>>(
        d_idata, d_odata, size
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata,
                     grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("GPU Unrolling8 elapsed %f sec GPU sum: %d "
           "<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    int sum8 = gpu_sum;

    // Kernel 2: Unrolling 16
    CHECK(cudaMemcpy(d_idata, h_idata, bytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling16<<<grid.x / 16, block>>>(
        d_idata, d_odata, size
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata,
                     grid.x / 16 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 16; i++)
        gpu_sum += h_odata[i];

    printf("GPU Unrolling16 elapsed %f sec GPU sum: %d "
           "<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 16, block.x);

    int sum16 = gpu_sum;

    // Result verification
    printf("\n========== RESULT ==========\n");

    if (cpu_sum == sum8 && cpu_sum == sum16)
        printf("All results match successfully!\n");
    else
        printf("Test failed! Results do not match.\n");

    // Free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // Free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaDeviceReset());

    return 0;
}
```

## OUTPUT:

<img width="1652" height="323" alt="image" src="https://github.com/user-attachments/assets/1d78e619-0f3c-4567-9f4c-7c67e146d7e7" />

<img width="796" height="213" alt="image" src="https://github.com/user-attachments/assets/ac24908c-a692-48a2-b8ae-8914a168a20f" />


## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that unrolling by 16 has executed with less elapsed time than unrolling by 8 with blocks 2048 and 4096.

