import pytest
import torch

import kernels.topology_sparse_attention as topology_sparse_attention
from kernels.topology_sparse_attention import (
    build_dense_causal_block_schedule,
    build_topology_block_schedule,
    dense_masked_attention,
    scheduled_attention,
)


def test_benchmark_utils_defers_optional_pandas_import(monkeypatch):
    import builtins
    import importlib
    import sys

    sys.modules.pop("benchmarking", None)
    sys.modules.pop("benchmarking.benchmark_utils", None)
    real_import = builtins.__import__

    def import_without_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("No module named 'pandas'", name="pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pandas)

    benchmark_utils = importlib.import_module("benchmarking.benchmark_utils")
    with pytest.raises(ModuleNotFoundError, match="pandas"):
        benchmark_utils.compare_benchmarks({"triton": {"kernel": 1.0}})


def test_scheduled_attention_is_exported_from_kernels_package():
    from kernels import scheduled_attention

    assert scheduled_attention is topology_sparse_attention.scheduled_attention


def test_dense_causal_block_schedule_is_lower_triangular_csr():
    offsets, indices = build_dense_causal_block_schedule(4)

    assert offsets.tolist() == [0, 1, 3, 6, 10]
    assert indices.tolist() == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def test_topology_schedule_adds_persistent_block_outside_local_window():
    centroids = torch.tensor(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [101.0, 0.0],
            [102.0, 0.0],
        ]
    )
    keys = centroids.repeat_interleave(2, dim=0)

    local_offsets, local_indices = build_topology_block_schedule(
        keys,
        block_size=2,
        local_radius_blocks=0,
        sink_blocks=0,
        topk_topology_blocks=0,
    )
    topology_offsets, topology_indices = build_topology_block_schedule(
        keys,
        block_size=2,
        local_radius_blocks=0,
        sink_blocks=0,
        topk_topology_blocks=1,
    )

    assert local_indices[local_offsets[3] : local_offsets[4]].tolist() == [3]
    assert topology_indices[topology_offsets[3] : topology_offsets[4]].tolist() == [
        0,
        3,
    ]


def test_dense_causal_block_schedule_rejects_non_positive_block_count():
    with pytest.raises(ValueError, match="num_blocks"):
        build_dense_causal_block_schedule(0)


def test_topology_block_schedule_rejects_negative_structural_parameters():
    keys = torch.empty((8, 4))

    with pytest.raises(ValueError, match="local_radius_blocks"):
        build_topology_block_schedule(
            keys,
            block_size=2,
            local_radius_blocks=-1,
            sink_blocks=0,
            topk_topology_blocks=0,
        )
    with pytest.raises(ValueError, match="sink_blocks"):
        build_topology_block_schedule(
            keys,
            block_size=2,
            local_radius_blocks=0,
            sink_blocks=-1,
            topk_topology_blocks=0,
        )
    with pytest.raises(ValueError, match="topk_topology_blocks"):
        build_topology_block_schedule(
            keys,
            block_size=2,
            local_radius_blocks=0,
            sink_blocks=0,
            topk_topology_blocks=-1,
        )


def test_scheduled_attention_rejects_non_positive_block_size():
    q = torch.empty((64, 32), dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="block_size"):
        scheduled_attention(q, q, q, offsets, indices, 0)


def test_scheduled_attention_rejects_non_power_of_two_block_size():
    q = torch.empty((96, 32), dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="block_size"):
        scheduled_attention(q, q, q, offsets, indices, 24)


def test_scheduled_attention_rejects_unsupported_head_dimension():
    q = torch.empty((64, 24), dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="head dimension"):
        scheduled_attention(q, q, q, offsets, indices, 16)


def test_scheduled_attention_rejects_mismatched_csr_offsets():
    q = torch.empty((64, 32), dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(3)

    with pytest.raises(ValueError, match="offsets"):
        scheduled_attention(q, q, q, offsets, indices, 16)


def test_scheduled_attention_rejects_cpu_inputs():
    q = torch.empty((64, 32), dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="CUDA"):
        scheduled_attention(q, q, q, offsets, indices, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_rejects_cpu_schedule_for_cuda_inputs():
    q = torch.empty((64, 32), device="cuda", dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="same device"):
        scheduled_attention(q, q, q, offsets, indices, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_rejects_non_integer_schedule_tensors():
    q = torch.empty((64, 32), device="cuda", dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(4)

    with pytest.raises(ValueError, match="integer"):
        scheduled_attention(
            q,
            q,
            q,
            offsets.to(device="cuda", dtype=torch.float32),
            indices.to(device="cuda"),
            16,
        )
    with pytest.raises(ValueError, match="integer"):
        scheduled_attention(
            q,
            q,
            q,
            offsets.to(device="cuda"),
            indices.to(device="cuda", dtype=torch.float32),
            16,
        )


def test_topology_sparse_attention_benchmark_formats_markdown_row():
    from benchmarking.topology_sparse_attention import format_markdown_row

    result = {
        "seq": 1024,
        "scheduled_blocks": 59,
        "dense_blocks": 136,
        "block_reduction": 0.566,
        "dense_masked_ms": 2.341,
        "sdpa_ms": 0.085,
        "triton_dense_csr_ms": 0.098,
        "triton_scheduled_ms": 0.094,
        "sparse_vs_dense_csr": 1.042,
        "sparse_vs_sdpa": 0.904,
        "max_abs_error": 0.0009,
    }

    assert format_markdown_row(result) == (
        "| 1024 | 59 / 136 | 56.6% | 2.341 | 0.085 | 0.098 | 0.094 | "
        "1.04x | 0.90x | 0.0009 |"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topology_schedule_preserves_key_device_for_direct_kernel_use():
    torch.manual_seed(3)
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k.float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )

    assert offsets.device == k.device
    assert indices.device == k.device

    expected = dense_masked_attention(
        q.float(),
        k.float(),
        v.float(),
        offsets.cpu(),
        indices.cpu(),
        block_size,
    )
    actual = scheduled_attention(q, k, v, offsets, indices, block_size)

    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_matches_dense_masked_reference():
    torch.manual_seed(0)
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k.float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )

    expected = dense_masked_attention(
        q.float(), k.float(), v.float(), offsets, indices, block_size
    )
    actual = scheduled_attention(
        q,
        k,
        v,
        offsets.to(device="cuda"),
        indices.to(device="cuda"),
        block_size,
    )

    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_preserves_input_dtype():
    torch.manual_seed(2)
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_dense_causal_block_schedule(seq // block_size)

    actual = scheduled_attention(
        q,
        k,
        v,
        offsets.to(device="cuda"),
        indices.to(device="cuda"),
        block_size,
    )

    assert actual.dtype == q.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scheduled_attention_supports_batch_and_heads():
    torch.manual_seed(1)
    batch = 2
    heads = 3
    seq = 128
    head_dim = 32
    block_size = 32
    q = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq, head_dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k[0, 0].float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=1,
    )

    expected = torch.empty_like(q, dtype=torch.float32)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            expected[batch_idx, head_idx] = dense_masked_attention(
                q[batch_idx, head_idx].float(),
                k[batch_idx, head_idx].float(),
                v[batch_idx, head_idx].float(),
                offsets,
                indices,
                block_size,
            )
    actual = scheduled_attention(
        q,
        k,
        v,
        offsets.to(device="cuda"),
        indices.to(device="cuda"),
        block_size,
    )

    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
