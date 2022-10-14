#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <iostream>

constexpr int SHMEM_SIZE = 16 * 16;

__global__
void multiply(int *a, int *b, int *c, int N, int tileSize) {

    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp = 0;

    for (size_t i = 0; i < (N / tileSize); i++)
    {
        A[threadIdx.y * tileSize + threadIdx.x] = a[row * N + (i * tileSize + threadIdx.x)];
        B[threadIdx.y * tileSize + threadIdx.x] = b[(i * tileSize + threadIdx.y) * N + col];

        __syncthreads();

        for (size_t j = 0; j < tileSize; j++)
        {
            tmp += A[threadIdx.y * tileSize + j] * B[j * tileSize + threadIdx.x];
        }

        __syncthreads();
    }   

    c[row * N + col] = tmp;
    
}

void verify_result(int *a, int *b, int *c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

void init_matrices(int *a, int *b, int N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
        a[i * N + j] = rand() % 100;
        b[i * N + j] = rand() % 100;
    }
    
  }
  
}

int main(int argc, char const *argv[])
{
  constexpr int N = 1 << 12;

  constexpr size_t bytes = N * N * sizeof(int);

  int *h_a, *h_b, *h_c;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  int *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  init_matrices(h_a, h_b, N);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int BLOCK_SIZE = 16;

  int GRID_SIZE = (int) ceil(N / BLOCK_SIZE);

  dim3 grid(GRID_SIZE, GRID_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  multiply <<<grid, threads>>> (d_a, d_b, d_c, N, BLOCK_SIZE);


  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  std::cout << "Finish" << std::endl;

  verify_result(h_a, h_b, h_c, N);
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
