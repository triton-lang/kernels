"""
Tests for SM120 Gluon Flash Attention (TMA + MMAv2).

Requires an SM12x NVIDIA GPU (DGX Spark, RTX 5090, GB10).

Contributed by Second Nature Computing (https://joinsecondnature.com)
"""

import pytest
import torch

from kernels.flash_attention_sm120 import attention_forward_sm120, is_sm12x


def reference_attention(q, k, v, sm_scale, causal=False):
    """PyTorch reference using scaled_dot_product_attention."""
    if q.dtype == torch.float8_e5m2:
        q_ref = q.to(torch.bfloat16)
        k_ref = k.to(torch.bfloat16)
        v_ref = v.to(torch.bfloat16)
    else:
        q_ref, k_ref, v_ref = q, k, v
    return torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=sm_scale, is_causal=causal)


# --- BF16 non-causal tests ---
@pytest.mark.parametrize("B, H, Sq, Sk, D", [
    (1, 4, 128, 128, 64),
    (1, 4, 256, 256, 64),
    (1, 4, 128, 128, 128),
    (1, 4, 256, 256, 128),
    (2, 8, 128, 128, 64),
    (2, 8, 512, 512, 64),
    (1, 4, 256, 512, 64),   # non-square: Sq < Sk
    (1, 4, 512, 256, 64),   # non-square: Sq > Sk
    (2, 4, 1024, 1024, 64),
    (2, 4, 1024, 1024, 128),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM12x GPU")
def test_attention_bf16(B, H, Sq, Sk, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=False)
    out = attention_forward_sm120(q, k, v, sm_scale, causal=False)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# --- BF16 causal tests ---
@pytest.mark.parametrize("B, H, S, D", [
    (1, 4, 128, 64),
    (1, 4, 256, 64),
    (1, 4, 128, 128),
    (1, 4, 256, 128),
    (2, 8, 512, 64),
    (2, 4, 1024, 64),
    (2, 4, 1024, 128),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM12x GPU")
def test_attention_bf16_causal(B, H, S, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=True)
    out = attention_forward_sm120(q, k, v, sm_scale, causal=True)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# --- FP8 non-causal tests ---
@pytest.mark.parametrize("B, H, Sq, Sk, D", [
    (1, 4, 128, 128, 64),
    (1, 4, 256, 256, 64),
    (1, 4, 128, 128, 128),
    (1, 4, 256, 256, 128),
    (2, 8, 512, 512, 64),
    (1, 4, 256, 512, 64),
    (2, 4, 1024, 1024, 64),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM12x GPU")
def test_attention_fp8(B, H, Sq, Sk, D):
    torch.manual_seed(42)
    # Generate in BF16 then cast — FP8 has limited range
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=False)
    out = attention_forward_sm120(q, k, v, sm_scale, causal=False)

    # FP8 E5M2 has only 2 mantissa bits — errors compound through matmuls
    out_bf16 = out.to(torch.bfloat16)
    ref_bf16 = ref.to(torch.bfloat16)
    torch.testing.assert_close(out_bf16, ref_bf16, rtol=1e-1, atol=4e-1)


# --- FP8 causal tests ---
@pytest.mark.parametrize("B, H, S, D", [
    (1, 4, 128, 64),
    (1, 4, 256, 64),
    (1, 4, 128, 128),
    (1, 4, 256, 128),
    (2, 8, 512, 64),
    (2, 4, 1024, 64),
])
@pytest.mark.skipif(not is_sm12x(), reason="Requires SM12x GPU")
def test_attention_fp8_causal(B, H, S, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e5m2)
    sm_scale = 1.0 / (D ** 0.5)

    ref = reference_attention(q, k, v, sm_scale, causal=True)
    out = attention_forward_sm120(q, k, v, sm_scale, causal=True)

    out_bf16 = out.to(torch.bfloat16)
    ref_bf16 = ref.to(torch.bfloat16)
    torch.testing.assert_close(out_bf16, ref_bf16, rtol=1e-1, atol=4e-1)
