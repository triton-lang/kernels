import torch
import torch.nn.functional as F
import pytest
from kernels.fused_softmax import triton_softmax

@pytest.mark.parametrize("input_size", [(1024, 1024), (512, 512), (2048, 512)])
def test_softmax_equivalence(input_size):
    # Create random input tensor of specified size
    x = torch.randn(*input_size).cuda()

    # Compute softmax using PyTorch
    pytorch_softmax = F.softmax(x, dim=-1)

    # Compute softmax using Triton
    triton_output = triton_softmax(x)

    # Assert that both outputs are approximately equal
    assert torch.allclose(pytorch_softmax, triton_output, atol=1e-5), \
        f"Triton softmax output doesn't match PyTorch softmax for input size {input_size}"
