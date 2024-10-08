import math
import torch.nn.functional as F
import torch
import triton
from typing import Tuple
from torch import nn
from kernels.matmul import matmul
from kernels.cross_entropy import cross_entropy
from kernels.matmul import matmul
from kernels.flash_attention import attention
from kernels.fused_softmax import softmax
from benchmarking import Profiler
import time


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_triton=False):
        super().__init__()
        self.use_triton = use_triton
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def __triton_norm(self, x):
        """
        TODO: Triton kernel added here as needed. remove if we dont want to convert this one
        For now adding the torch version as a placeholder
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @Profiler.profiling_decorator(record_name="RMSNorm")
    def _norm(self, x):
        if not self.use_triton:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            return self.__triton_norm(x)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MathOps:
    printed_attention = False

    def __init__(self, use_triton=False):
        self.use_triton = use_triton

    @Profiler.profiling_decorator("matmul")
    def matmul(self, x, y):
        if self.use_triton:
            return torch.matmul(x, y)
        else:
            return torch.matmul(x, y)

    @Profiler.profiling_decorator("attention")
    def attention(self, xq, keys, values, head_dim, mask):
        scores = self.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = self.softmax(scores.float(), dim=-1).type_as(xq)
        output = self.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        return output

    @Profiler.profiling_decorator("softmax")
    def softmax(self, x, dim):
        if self.use_triton and x.ndim == 2:
            return softmax(x)
        else:
            return F.softmax(x, dim)

    @Profiler.profiling_decorator("argmax")
    def argmax(self, x, dim):
        if self.use_triton:
            return self.triton.language.argmax(x, axis=dim)
        else:
            return torch.argmax(x, dim=dim)

    @Profiler.profiling_decorator("cross_entropy")
    def cross_entropy(self, input_val, target, reduction, ignore_index):
        if self.use_triton:
            return cross_entropy(
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

    def get_rms_norm(self, dim: int, eps: float = 1e-6):
        return _RMSNorm(dim, eps, self.use_triton)

    @Profiler.profiling_decorator("apply_rotary_emb")
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def torch_based(xq, xk, freqs_cis):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def triton_based(xq, xk, freqs_cis):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        if self.use_triton:
            return triton_based(xq, xk, freqs_cis)
        else:
            return torch_based(xq, xk, freqs_cis)

    @Profiler.profiling_decorator(record_name="precompute_freqs_cis")
    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        def torch_based(dim: int, end: int, theta: float = 10000.0):
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def triton_based(dim: int, end: int, theta: float = 10000.0):
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        return torch_based(dim, end, theta)
        if self.use_triton:
            return triton_based(dim, end, theta)
        else:
            return torch_based(dim, end, theta)
