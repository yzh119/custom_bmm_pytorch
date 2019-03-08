import torch as th
from bmm import *
from torch.autograd import Function

class SparseSoftmax(Function):
    @staticmethod
    def forward(ctx, indptr, eid, x):
        y = sparse_softmax_forward(indptr, eid, x)
        ctx.save_for_backward(indptr, eid, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        indptr, eid, y = ctx.saved_tensors
        return None, None, sparse_softmax_backward(indptr, eid, y, dy)

