import torch as th
from bmm import *
from torch.autograd import Function

class BatchedMM(Function):
    @staticmethod
    def forward(ctx, A, B):
        A = A.contiguous()
        B = B.contiguous()
        y = bmm_forward(A, B)
        ctx.save_for_backward(A, B)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B = ctx.saved_tensors
        dA, dB = bmm_backward(A, B, dy)
        return dA, dB

x = th.rand(64, 3, 15, requires_grad=True, device='cuda:0')
y = th.rand(64, 15, 4, requires_grad=True, device='cuda:0')

c1 = x @ y 
c2 = BatchedMM.apply(x, y)

assert th.allclose(c1, c2)

grad = th.rand(64, 3, 4, device='cuda:0')
c1.backward(grad)
x_grad_clone = x.grad.clone()
y_grad_clone = y.grad.clone()
x.grad.zero_()
y.grad.zero_()
c2.backward(grad)
assert th.allclose(x.grad, x_grad_clone) and th.allclose(y.grad, y_grad_clone)


