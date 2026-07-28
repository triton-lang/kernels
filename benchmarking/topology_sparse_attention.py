import argparse
import statistics

import torch
import torch.nn.functional as F
import triton

from kernels.topology_sparse_attention import (
    build_dense_causal_block_schedule,
    build_topology_block_schedule,
    dense_masked_attention,
    scheduled_attention,
)


HEADER = (
    "| seq | scheduled / dense blocks | block reduction | dense masked ms | "
    "full causal SDPA ms | Triton dense CSR ms | Triton scheduled ms | "
    "Triton sparse vs dense CSR | Triton vs SDPA | max abs error |"
)
SEPARATOR = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


def timed_cuda(fn, rounds):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def run_case(seq, dim, block_size, rounds):
    torch.manual_seed(20260629 + seq)
    q = torch.randn((seq, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((seq, dim), device="cuda", dtype=torch.float16)
    v = torch.randn((seq, dim), device="cuda", dtype=torch.float16)
    offsets, indices = build_topology_block_schedule(
        k.float(),
        block_size=block_size,
        local_radius_blocks=1,
        sink_blocks=1,
        topk_topology_blocks=max(1, seq // block_size // 8),
    )
    dense_offsets, dense_indices = build_dense_causal_block_schedule(seq // block_size)
    offsets_cuda = offsets.to("cuda")
    indices_cuda = indices.to("cuda")
    dense_offsets_cuda = dense_offsets.to("cuda")
    dense_indices_cuda = dense_indices.to("cuda")

    expected = dense_masked_attention(
        q.float(), k.float(), v.float(), offsets, indices, block_size
    )
    actual = scheduled_attention(q, k, v, offsets_cuda, indices_cuda, block_size)
    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
    dense_expected = dense_masked_attention(
        q.float(), k.float(), v.float(), dense_offsets, dense_indices, block_size
    )
    dense_actual = scheduled_attention(
        q, k, v, dense_offsets_cuda, dense_indices_cuda, block_size
    )
    torch.testing.assert_close(
        dense_actual.float(), dense_expected, rtol=3e-2, atol=3e-2
    )

    dense_masked_ms = timed_cuda(
        lambda: dense_masked_attention(
            q.float(), k.float(), v.float(), offsets, indices, block_size
        ),
        rounds,
    )
    sdpa_ms = timed_cuda(
        lambda: F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], is_causal=True
        ),
        rounds,
    )
    triton_scheduled_ms = timed_cuda(
        lambda: scheduled_attention(q, k, v, offsets_cuda, indices_cuda, block_size),
        rounds,
    )
    triton_dense_csr_ms = timed_cuda(
        lambda: scheduled_attention(
            q, k, v, dense_offsets_cuda, dense_indices_cuda, block_size
        ),
        rounds,
    )
    return {
        "seq": seq,
        "scheduled_blocks": indices.numel(),
        "dense_blocks": dense_indices.numel(),
        "block_reduction": 1.0 - indices.numel() / dense_indices.numel(),
        "dense_masked_ms": dense_masked_ms,
        "sdpa_ms": sdpa_ms,
        "triton_dense_csr_ms": triton_dense_csr_ms,
        "triton_scheduled_ms": triton_scheduled_ms,
        "sparse_vs_dense_csr": triton_dense_csr_ms / triton_scheduled_ms,
        "sparse_vs_sdpa": sdpa_ms / triton_scheduled_ms,
        "max_abs_error": (actual.float() - expected).abs().max().item(),
    }


def format_markdown_row(result):
    return (
        "| {seq} | {scheduled_blocks} / {dense_blocks} | {block_reduction:.1%} | "
        "{dense_masked_ms:.3f} | {sdpa_ms:.3f} | {triton_dense_csr_ms:.3f} | "
        "{triton_scheduled_ms:.3f} | {sparse_vs_dense_csr:.2f}x | "
        "{sparse_vs_sdpa:.2f}x | {max_abs_error:.4f} |"
    ).format(**result)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark topology-derived CSR scheduled sparse attention."
    )
    parser.add_argument("--seq", type=int, nargs="*", default=[1024, 2048, 4096])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(HEADER)
    print(SEPARATOR)
    for seq in args.seq:
        result = run_case(
            seq=seq, dim=args.dim, block_size=args.block_size, rounds=args.rounds
        )
        print(format_markdown_row(result))


if __name__ == "__main__":
    main()
