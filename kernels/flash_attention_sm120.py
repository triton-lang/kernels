"""
Flash Attention Forward for SM120 (Gluon + TMA)
================================================

Flash attention forward pass targeting SM120 GPUs (DGX Spark, RTX 5090, GB10)
using Gluon. SM120 lacks tcgen05/TMEM/WGMMA but has MMAv2 (mma.sync.aligned)
tensor cores and TMA (Tensor Memory Accelerator) for async data movement.

Features:
- TMA for all global memory access: eliminates per-element address computation,
  reduces register pressure, and handles out-of-bounds with zero-fill via
  tensor descriptors
- Double-buffered K pipelining with mbarrier phase tracking:
  K[i+1] prefetch overlaps QK compute + softmax
- Single-buffered V with mbarrier phase tracking:
  V[i] load overlaps QK compute + softmax
- TMA store for output (shared memory -> global via async proxy)
- Causal masking with early loop termination
- BF16 and FP8 (E5M2) support
- Online softmax with numerically stable exp2-based rescaling

Contributed by Second Nature Computing (https://joinsecondnature.com)
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    tma,
    mbarrier,
    fence_async_shared,
    mma_v2,
)


def is_sm12x():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 12


# ---------------------------------------------------------------------------
# Configuration aggregate — defines all layouts and block sizes
# ---------------------------------------------------------------------------

@aggregate
class AttentionConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    NUM_WARPS: gl.constexpr
    IS_FP8: gl.constexpr

    # MMA accumulator layout
    mma_layout: gl.constexpr
    # Operand layouts for Q*K^T
    q_dot_layout: gl.constexpr
    k_dot_layout: gl.constexpr
    # Operand layouts for P*V
    p_dot_layout: gl.constexpr
    v_dot_layout: gl.constexpr
    # Shared memory layouts
    q_smem_layout: gl.constexpr
    kv_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS, IS_FP8):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.IS_FP8 = gl.constexpr(IS_FP8)

        # MMAv2 accumulator layout: mma.sync.aligned.m16n8k{16,32}
        self.mma_layout = gl.constexpr(gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[NUM_WARPS, 1],
            instr_shape=[16, 8],
        ))

        # BF16: 8 elements per 128-bit register group (8*16=128 bits)
        # FP8:  16 elements per 128-bit register group (16*8=128 bits)
        k_width = 16 if IS_FP8 else 8

        # Q*K^T operand layouts
        self.q_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=k_width))
        self.k_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.mma_layout, k_width=k_width))

        # P*V operand layouts
        self.p_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=k_width))
        self.v_dot_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.mma_layout, k_width=k_width))

        # Shared memory layouts (swizzled for bank-conflict-free MMA access)
        smem_dtype = gl.float8e5 if IS_FP8 else gl.bfloat16
        self.q_smem_layout = gl.constexpr(
            gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], smem_dtype))
        self.kv_smem_layout = gl.constexpr(
            gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], smem_dtype))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

RCP_LN2 = gl.constexpr(1.4426950408889634)  # 1/ln(2) for exp2-based softmax


@gluon.jit
def attn_fwd_kernel(
    desc_q, desc_k, desc_v, desc_o,
    sm_scale,
    SEQ_LEN_Q: gl.constexpr,
    SEQ_LEN_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    cfg = AttentionConfig(BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS, IS_FP8)

    SMEM_DTYPE: gl.constexpr = gl.float8e5 if IS_FP8 else gl.bfloat16

    # Program indices: [batch*heads, seq_blocks]
    off_zh = gl.program_id(0)
    off_m = gl.program_id(1) * BLOCK_M

    # --- Allocate shared memory ---
    q_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [BLOCK_M, HEAD_DIM], layout=cfg.q_smem_layout)
    k_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [2, BLOCK_N, HEAD_DIM], layout=cfg.kv_smem_layout)
    v_smem = gl.allocate_shared_memory(
        SMEM_DTYPE, [BLOCK_N, HEAD_DIM], layout=cfg.kv_smem_layout)

    # --- Allocate mbarriers ---
    q_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    k_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    v_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    mbarrier.init(q_bar, count=1)
    for i in gl.static_range(2):
        mbarrier.init(k_bars.index(i), count=1)
    mbarrier.init(v_bar, count=1)

    # --- Load Q block via TMA ---
    q_offset_y = off_zh * SEQ_LEN_Q + off_m
    mbarrier.expect(q_bar, desc_q.block_type.nbytes)
    tma.async_copy_global_to_shared(desc_q, [q_offset_y, 0], q_bar, q_smem)
    mbarrier.wait(q_bar, phase=0)
    mbarrier.invalidate(q_bar)

    q = q_smem.load(cfg.q_dot_layout)

    # --- Initialize online softmax state ---
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32,
                  layout=gl.SliceLayout(1, cfg.mma_layout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32,
                  layout=gl.SliceLayout(1, cfg.mma_layout))
    acc = gl.zeros([BLOCK_M, HEAD_DIM], dtype=gl.float32, layout=cfg.mma_layout)

    # Scaling factor
    qk_scale = sm_scale * RCP_LN2

    # --- Causal loop bound ---
    if IS_CAUSAL:
        loop_end = min(off_m + BLOCK_M, SEQ_LEN_K)
    else:
        loop_end = SEQ_LEN_K

    # Precompute m_offs for causal mask (only used inside loop if IS_CAUSAL)
    m_offs = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.mma_layout))

    # --- Prologue: prefetch K[0] via TMA ---
    kv_base_y = off_zh * SEQ_LEN_K
    mbarrier.expect(k_bars.index(0), desc_k.block_type.nbytes)
    tma.async_copy_global_to_shared(desc_k, [kv_base_y, 0], k_bars.index(0), k_smem.index(0))

    # --- Main loop over K/V blocks ---
    for start_n in range(0, loop_end, BLOCK_N):
        iter_idx = start_n // BLOCK_N
        cur_buf = iter_idx % 2
        k_phase = iter_idx // 2 & 1

        # Issue V[i] load via TMA
        v_phase = iter_idx & 1
        mbarrier.expect(v_bar, desc_v.block_type.nbytes)
        tma.async_copy_global_to_shared(desc_v, [kv_base_y + start_n, 0], v_bar, v_smem)

        # Wait for K[i]
        mbarrier.wait(k_bars.index(cur_buf), k_phase)

        # Load K[i]: [BLOCK_N, HEAD_DIM] -> transposed for Q @ K^T
        k = k_smem.index(cur_buf).permute([1, 0]).load(cfg.k_dot_layout)

        # Prefetch K[i+1] into next buffer
        next_buf = 1 - cur_buf
        next_k_phase = (iter_idx + 1) // 2 & 1
        mbarrier.expect(k_bars.index(next_buf), desc_k.block_type.nbytes)
        tma.async_copy_global_to_shared(
            desc_k, [kv_base_y + start_n + BLOCK_N, 0],
            k_bars.index(next_buf), k_smem.index(next_buf))

        # QK = Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=cfg.mma_layout)
        qk = mma_v2(q, k, qk)

        # Mask out-of-bounds positions
        n_offs = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, cfg.mma_layout))[None, :]
        qk = gl.where(n_offs < SEQ_LEN_K, qk, float("-inf"))

        # Causal mask: mask positions where query position < key position
        if IS_CAUSAL:
            causal_mask = (off_m + m_offs[:, None]) >= n_offs
            qk = gl.where(causal_mask, qk, float("-inf"))

        # --- Online softmax (overlaps V[i] load) ---
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * qk_scale

        q_shifted = qk * qk_scale - m_ij_scaled[:, None]
        p = gl.exp2(q_shifted)

        m_diff = m_i * qk_scale - m_ij_scaled
        alpha = gl.exp2(m_diff)

        l_ij = gl.sum(p, 1)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # --- P * V ---
        p_cast = p.to(SMEM_DTYPE, fp_downcast_rounding="rtz")
        p_dot = gl.convert_layout(p_cast, cfg.p_dot_layout)

        # Wait for V[i]
        mbarrier.wait(v_bar, v_phase)

        v = v_smem.load(cfg.v_dot_layout)

        acc = mma_v2(p_dot, v, acc)

    # --- Invalidate mbarriers ---
    for i in gl.static_range(2):
        mbarrier.invalidate(k_bars.index(i))
    mbarrier.invalidate(v_bar)

    # --- Final normalization ---
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

    # --- Store output via TMA ---
    out = acc.to(SMEM_DTYPE, fp_downcast_rounding="rtz")
    q_smem.store(out)
    fence_async_shared()

    o_offset_y = off_zh * SEQ_LEN_Q + off_m
    tma.async_copy_shared_to_global(desc_o, [o_offset_y, 0], q_smem)
    tma.store_wait(pendings=0)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    return getattr(gl, str(dtype).split('.')[1])


def attention_forward_sm120(q, k, v, sm_scale, causal=False, BLOCK_M=128, BLOCK_N=64, num_warps=4):
    """Flash attention forward pass for SM120 GPUs.

    Args:
        q: [B, H, Sq, D] bfloat16 or float8_e5m2
        k: [B, H, Sk, D] bfloat16 or float8_e5m2
        v: [B, H, Sk, D] bfloat16 or float8_e5m2
        sm_scale: softmax scale (typically 1/sqrt(D))
        causal: whether to apply causal masking

    Returns:
        out: [B, H, Sq, D] same dtype as input
    """
    if not is_sm12x():
        raise RuntimeError(
            "attention_forward_sm120 requires an SM12x NVIDIA GPU "
            "(DGX Spark, RTX 5090, GB10)")

    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    is_fp8 = q.dtype == torch.float8_e5m2
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.bfloat16, torch.float8_e5m2)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)

    triton_dtype = torch_dtype_to_triton(q.dtype)

    def make_desc(t, block_shape):
        total_rows = t.shape[0] * t.shape[1] * t.shape[2]
        layout = gl.NVMMASharedLayout.get_default_for(block_shape, triton_dtype)
        return TensorDescriptor(
            t, shape=[total_rows, D], strides=[D, 1],
            block_shape=block_shape, layout=layout)

    desc_q = make_desc(q, [BLOCK_M, D])
    desc_k = make_desc(k, [BLOCK_N, D])
    desc_v = make_desc(v, [BLOCK_N, D])
    desc_o = make_desc(out, [BLOCK_M, D])

    grid = (B * H, triton.cdiv(Sq, BLOCK_M))

    attn_fwd_kernel[grid](
        desc_q, desc_k, desc_v, desc_o,
        sm_scale,
        SEQ_LEN_Q=Sq,
        SEQ_LEN_K=Sk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        IS_FP8=is_fp8,
        num_warps=num_warps,
    )
    return out
