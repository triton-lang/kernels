import pytest
import torch
import kernels
from kernels import batchnorm, _batchnorm

def create_test_data(batch_size: int, channels: int, device: str = 'cuda'):
    input_tensor = torch.randn(batch_size, channels, device=device, dtype=torch.float32)
    gamma = torch.ones(channels, device=device, dtype=torch.float32)
    beta = torch.zeros(channels, device=device, dtype=torch.float32)
    return input_tensor, gamma, beta

@pytest.mark.parametrize("batch_size,channels", [
    (32, 128),
    (64, 256),
    (128, 512),
    (256, 1024),
    (512, 2048),
])
def test_batchnorm_correctness(batch_size, channels):
    eps = 1e-5
    input_tensor, gamma, beta = create_test_data(batch_size, channels)
    
    output_custom = batchnorm(input_tensor, gamma, beta, eps)

    bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=False, device=input_tensor.device)
    bn.weight.data = gamma
    bn.bias.data = beta
    bn.train()
    with torch.no_grad():
        output_pytorch = bn(input_tensor)

    max_diff = torch.max(torch.abs(output_custom - output_pytorch)).item()
    rel_error = max_diff / (torch.max(torch.abs(output_pytorch)).item() + 1e-8)
    
    assert max_diff < 1e-3 or rel_error < 1e-3, f"Max diff: {max_diff:.2e}, Rel error: {rel_error:.2e}"

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_batchnorm_dtypes(dtype):
    batch_size, channels = 128, 256
    eps = 1e-5
    
    input_tensor = torch.randn(batch_size, channels, device='cuda', dtype=dtype)
    gamma = torch.ones(channels, device='cuda', dtype=dtype)
    beta = torch.zeros(channels, device='cuda', dtype=dtype)
    
    output = batchnorm(input_tensor, gamma, beta, eps)
    assert output.dtype == dtype
    assert output.shape == input_tensor.shape

def test_batchnorm_backward_compatibility():
    batch_size, channels = 64, 128
    eps = 1e-5
    input_tensor, gamma, beta = create_test_data(batch_size, channels)
    
    N, C = input_tensor.shape
    output = torch.empty_like(input_tensor)
    _batchnorm(input_tensor, gamma, beta, output, N, C, eps)
    
    bn = torch.nn.BatchNorm1d(C, eps=eps, affine=True, track_running_stats=False, device=input_tensor.device)
    bn.weight.data = gamma
    bn.bias.data = beta
    bn.train()
    with torch.no_grad():
        expected = bn(input_tensor)
    
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("use_small_problem", [True, False])
def test_fused_vs_two_pass(use_small_problem):
    if use_small_problem:
        batch_size, channels = 32, 64
    else:
        batch_size, channels = 8192, 1024
    
    eps = 1e-5
    input_tensor, gamma, beta = create_test_data(batch_size, channels)
    
    output = batchnorm(input_tensor, gamma, beta, eps)
    
    bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=False, device=input_tensor.device)
    bn.weight.data = gamma
    bn.bias.data = beta
    bn.train()
    with torch.no_grad():
        expected = bn(input_tensor)
    
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)