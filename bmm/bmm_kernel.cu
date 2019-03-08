#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Type.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

/*
 * CUDA kernel of batched matrix multiplication:
 * (b, n, m) * (b, m, p)
 */
template <typename scalar_t>
__global__ void bmm_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, const int b, const int n, const int m, const int p) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    for (int x = tx; x < n; x += blockDim.x) {
        for (int y = ty; y < p; y += blockDim.y) {
            scalar_t sum = 0;
            for (int k = 0; k < m; ++k) {
                sum +=  A[((i * n) + x) * m + k] * B[((i * m) + k) * p + y];
            }
            C[((i * n) + x) * p + y] = sum;
        } 
    }
}

} // End of namespace

at::Tensor bmm_cuda_forward(
    const at::Tensor& A,
    const at::Tensor& B) {
    // A: (b, n, m), B: (b, m, p)
    const auto b = A.size(0);
    const auto n = A.size(1);
    const auto m = A.size(2);
    assert(m == B.size(1));
    const auto p = B.size(2);

    auto y = at::zeros({b, n, p}, A.options());

    const dim3 threads(n < 32 ? n: 32, p < 32 ? p: 32);
    const dim3 blocks(b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_cuda_forward", ([&] {
        bmm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            y.data<scalar_t>(),
            b, n, m, p);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> bmm_cuda_backward(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // A: (b, n, m), B: (b, m, p), dy: (b, n, p)
    const auto b = A.size(0);
    const auto n = A.size(1);
    const auto m = A.size(2);
    assert(m == B.size(1));
    const auto p = B.size(2);    

    auto Bt = B.transpose(-1, -2).contiguous(), At = A.transpose(-1, -2).contiguous();
    auto dA = at::zeros_like(A, A.options()), dB = at::zeros_like(B, B.options());
    
    const dim3 blocks(b);
    dim3 threads(32, 32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_cuda_backward_0", ([&] {
        bmm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data<scalar_t>(),
            Bt.data<scalar_t>(),
            dA.data<scalar_t>(),
            b, n, p, m);
    }));

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_cuda_backward_1", ([&] {
        bmm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            At.data<scalar_t>(),
            dy.data<scalar_t>(),
            dB.data<scalar_t>(),
            b, m, n, p);
    }));
    THCudaCheck(cudaGetLastError());
    return {dA, dB};
}


