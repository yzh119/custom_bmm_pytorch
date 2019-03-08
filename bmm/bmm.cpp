#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor bmm_cuda_forward(
	const at::Tensor& A,
	const at::Tensor& B);

at::Tensor bmm_forward(
	const at::Tensor& A,
	const at::Tensor& B) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	return bmm_cuda_forward(A, B);
}

std::vector<at::Tensor> bmm_cuda_backward(
	const at::Tensor& A,
	const at::Tensor& B,
	const at::Tensor& dy);

std::vector<at::Tensor> bmm_backward(
	const at::Tensor& A,
	const at::Tensor& B,
	const at::Tensor& dy) {
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	CHECK_INPUT(dy);
	return bmm_cuda_backward(A, B, dy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_forward", &bmm_forward, "Batched MM forward");
    m.def("bmm_backward", &bmm_backward, "Batched MM backward");
}
