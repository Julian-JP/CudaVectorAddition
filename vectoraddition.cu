#include <cassert>
#include <iostream>

__global__
void vectorAddUM(int *a, int *b, int *c, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void init_vector(int *a, int *b, int N) {
    for (size_t i = 0; i < N; i++)
    {
        a[i] = 1;
        b[i] = 9;
    }
    
}

void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
        assert(c[i] == a[i] + b[i]);
    }
  }
}


int main(int argc, char const *argv[])
{
    int id = cudaGetDevice(&id);

    constexpr int N = 1<<20;

    int *a, *b, *c;

    size_t bytes = N * sizeof(float);

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    init_vector(a, b, N);

    constexpr const int BLOCKSIZE = 256;
    const int GRID_SIZE = (int) ceil(N / BLOCKSIZE);

    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    vectorAddUM <<<GRID_SIZE, BLOCKSIZE>>> (a, b, c, N);

    cudaDeviceSynchronize();
    
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    verify_result(a, b, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
