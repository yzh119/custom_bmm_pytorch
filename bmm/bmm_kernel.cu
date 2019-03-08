/* TODOs
 * - segment_reduce_forward, segment_reduce_backward;
 * - switch backend from aten to dlpack
 */

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
 * CUDA kernel of the forward function for batched matrix multiplication:
 */
template <typename scalar_t>
__global__ void bmm_forward_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ y, const int b, const int n, const int m, const int p) {
/*
    int i = (((blockIdx.x) * blockDim.x) + (threadIdx.x));
    if (i < e) {
        for (int ko = 0; ko < h; ++ko) {
            data_t sum = 0;
            for (int k = 0; k < d; ++k) {
                sum += A[(row[i] * h + ko) * d + k] * Bt[col[i] + ((ko * d + k) * n)];
            }
            y[i * h + ko] = sum;
        }
    }
*/
}

/*
 * CUDA kernel of the backward function for batched matrix multiplication.
 */ 
template <typename scalar_t>
__global__ void bmm_backward_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, const scalar_t* __restrict__ dy, scalar_t* dA, scalar_t* dB, const int b, const int n, const int m, const int p) {
	// TODO
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

    const int threads = 32;
    const dim3 blocks((b + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_cuda_forward", ([&] {
        bmm_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
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

	auto dA = at::zeros_like(A, A.options()), dB = at::zeros_like(B, B.options());
	
	const int threads = 32;
	const dim3 blocks((b + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(A.type(), "bmm_cuda_backward", ([&] {
        bmm_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            dy.data<scalar_t>(),
			dA.data<scalar_t>(),
			dB.data<scalar_t>(),
            b, n, m, p);
    }));
    THCudaCheck(cudaGetLastError());
	return {dA, dB};
}


