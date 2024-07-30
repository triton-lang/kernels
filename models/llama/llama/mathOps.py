import torch.nn.functional as F
import torch
import triton

class MathOps:
    def __init__(self, use_triton=False):
        self.use_triton = use_triton
    
    def matmul(self, x, y):
        if self.use_triton:
            return torch.matmul(x, y)
        else:
            return torch.matmul(x, y)
    def softmax(self,x,dim):
        if self.use_triton:
            return F.softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)
    def argmax(self,x,dim):
        if self.use_triton:
            return torch.argmax(x, dim=-1)
        else:
            return torch.argmax(x, dim=-1)
    
    def cross_entropy(self,input_val,target, reduction, ignore_index):
        if self.use_triton:
            return -F.cross_entropy(
                input=input_val,
                target=target,
                reduction=reduction,
                ignore_index=ignore_index,
            )
        else:
            return -F.cross_entropy(
                input=input_val,
                target=target,
                reduction=reduction,
                ignore_index=ignore_index,
            )
