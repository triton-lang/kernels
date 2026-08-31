import torch
import triton
import triton.language as tl
from benchmarking import Profiler

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_C': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'stride_in_c'],
)
@triton.jit
def bn_reduce_kernel(x_ptr, sum_ptr, sumsq_ptr, N, C, stride_in_n, stride_in_c,
                     BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C

    x_ptrs = x_ptr + offs_n[:, None] * stride_in_n + offs_c[None, :] * stride_in_c
    mask_2d = mask_n[:, None] & mask_c[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=0).to(tl.float32)
    psum = tl.sum(x, axis=0)
    psumsq = tl.sum(x * x, axis=0)

    tl.atomic_add(sum_ptr + offs_c, psum, mask=mask_c)
    tl.atomic_add(sumsq_ptr + offs_c, psumsq, mask=mask_c)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_C': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'stride_in_c'],
)
@triton.jit
def bn_norm_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr, sum_ptr, sumsq_ptr,
                   N, C, eps, stride_in_n, stride_in_c, stride_out_n, stride_out_c,
                   BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C

    s = tl.load(sum_ptr + offs_c, mask=mask_c, other=0.).to(tl.float32)
    ss = tl.load(sumsq_ptr + offs_c, mask=mask_c, other=0.).to(tl.float32)
    Nf = tl.full((), N, tl.float32)

    mean = s / Nf
    var = ss / Nf - mean * mean
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.).to(tl.float32)

    scale = gamma * inv_std
    shift = beta - mean * scale

    x_ptrs = x_ptr + offs_n[:, None] * stride_in_n + offs_c[None, :] * stride_in_c
    y_ptrs = y_ptr + offs_n[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
    mask_2d = mask_n[:, None] & mask_c[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=0.).to(tl.float32)
    y = x * scale[None, :] + shift[None, :]
    tl.store(y_ptrs, y, mask=mask_2d)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64,  'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['N', 'C', 'stride_in_c'],
)
@triton.jit
def bn_fused_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr, N, C, eps,
                    stride_in_n, stride_in_c, stride_out_n, stride_out_c,
                    BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.).to(tl.float32)

    s = tl.zeros([BLOCK_C], dtype=tl.float32)
    ss = tl.zeros([BLOCK_C], dtype=tl.float32)

    tl.max_contiguous(offs_c, BLOCK_C)

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x_ptrs = x_ptr + offs_n[:, None] * stride_in_n + offs_c[None, :] * stride_in_c
        m = mask_n[:, None] & mask_c[None, :]
        x = tl.load(x_ptrs, mask=m, other=0.).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)

    Nf = tl.full((), N, tl.float32)
    mean = s / Nf
    var = ss / Nf - mean * mean
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x_ptrs = x_ptr + offs_n[:, None] * stride_in_n + offs_c[None, :] * stride_in_c
        y_ptrs = y_ptr + offs_n[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
        m = mask_n[:, None] & mask_c[None, :]
        x = tl.load(x_ptrs, mask=m, other=0.).to(tl.float32)
        y = x * scale[None, :] + shift[None, :]
        tl.store(y_ptrs, y, mask=m)

def _batchnorm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               output: torch.Tensor, N: int, C: int, eps: float):
    assert input.is_cuda and output.is_cuda and gamma.is_cuda and beta.is_cuda
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32)

    problem_size = N * C
    use_fused = problem_size < (1 << 24)

    if use_fused:
        def grid(meta):
            return (triton.cdiv(C, meta['BLOCK_C']),)
        bn_fused_kernel[grid](
            input, output, gamma, beta,
            N, C, eps,
            input.stride(0), input.stride(1),
            output.stride(0), output.stride(1),
        )
        return

    sum_buf = torch.zeros(C, device=input.device, dtype=torch.float32)
    sumsq_buf = torch.zeros(C, device=input.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(C, meta['BLOCK_C']))

    bn_reduce_kernel[grid](input, sum_buf, sumsq_buf,
                           N, C, input.stride(0), input.stride(1))

    bn_norm_kernel[grid](input, gamma, beta, output,
                         sum_buf, sumsq_buf,
                         N, C, eps,
                         input.stride(0), input.stride(1),
                         output.stride(0), output.stride(1))

@Profiler.profiling_decorator("batchnorm")
def batchnorm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    N, C = input.shape
    output = torch.empty_like(input)
    _batchnorm(input, gamma, beta, output, N, C, eps)
    return output