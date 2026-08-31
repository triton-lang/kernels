from typing import List
import os

import fire
import torch

from models.llama import llama_example_chat_completion, llama_example_text_completion
from benchmarking import Profiler, compare_benchmarks
import kernels
import pprint


def main(operation: str, profile=False, benchmark=False, **kwargs):
    """
    all kwargs are passed to the operation you choose.

    The profile and benchmark flags can be set independently of each other
    *but* if you set both then profiling will be done on both sets
    """
    p = Profiler(profile, benchmark)
    profiles = {}
    benchmarks = {}
    if benchmark:
        # warm_up
        torch.cuda.empty_cache()
        kwargs["suppress_prints"] = True
        p = Profiler(False, False)
        runner(operation, kwargs)

        kwargs["use_triton"] = False
        Profiler.reset()
        p = Profiler(profile, benchmark)
        torch.cuda.empty_cache()
        runner(operation, kwargs)
        benchmarks["non_triton"] = Profiler.get_benchmark_vals()
        profiles["non_triton"] = Profiler.get_profiling_data()
        Profiler.reset()
        p = Profiler(profile, benchmark)

        kwargs["use_triton"] = True
        kwargs["suppress_prints"] = False
        Profiler.reset()
        p = Profiler(profile, benchmark)
        torch.cuda.empty_cache()
        runner(operation, kwargs)
        benchmarks["triton"] = Profiler.get_benchmark_vals()
        profiles["triton"] = Profiler.get_profiling_data()
    elif profile:
        runner(operation, kwargs)
        data = Profiler.get_profiling_data()
        if kwargs["use_triton"]:
            profiles["triton"] = data
        else:
            profiles["non_triton"] = data
    else:
        runner(operation, kwargs)

    if profile:
        for k, v in profiles.items():
            print(f"Profile for {k}")
            pprint.pprint(v, width=160)
            print("\n==================================\n")
    if benchmark:
        print("Benchmark results")
        output = compare_benchmarks(benchmarks)
        print(output)
        print("\n==================================\n")


@Profiler.profiling_decorator("batchnorm_benchmark")
def batchnorm_benchmark(batch_size=512, channels=2048, use_triton=True, suppress_prints=False):
    """Benchmark batchnorm implementation"""
    if not suppress_prints:
        print(f"Running BatchNorm benchmark: {batch_size}x{channels}, use_triton={use_triton}")
    
    device = 'cuda'
    eps = 1e-5
    
    # Create test data
    input_tensor = torch.randn(batch_size, channels, device=device, dtype=torch.float32)
    gamma = torch.ones(channels, device=device, dtype=torch.float32)
    beta = torch.zeros(channels, device=device, dtype=torch.float32)
    
    if use_triton:
        # Use custom Triton implementation
        output = kernels.batchnorm(input_tensor, gamma, beta, eps)
    else:
        # Use PyTorch baseline
        bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=False, device=device)
        bn.weight.data = gamma
        bn.bias.data = beta
        bn.train()
        with torch.no_grad():
            output = bn(input_tensor)
    
    if not suppress_prints:
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    return output


def runner(operation: str, kwargs):
    if operation == "llama_chat_completion":
        llama_example_chat_completion(**kwargs)
    elif operation == "llama_text_completion":
        llama_example_text_completion(**kwargs)
    elif operation == "batchnorm_benchmark":
        batchnorm_benchmark(**kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    fire.Fire(main)
