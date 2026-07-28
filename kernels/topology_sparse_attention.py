"""Forward attention over topology-derived causal CSR block schedules."""

import math

import torch
import triton
import triton.language as tl


_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}
_SCHEDULE_DTYPES = {torch.int32, torch.int64}


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _zero_dim_persistence_salience(centroids):
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_blocks, dim]")
    num_blocks = centroids.shape[0]
    if num_blocks == 0:
        raise ValueError("at least one block is required")
    if num_blocks == 1:
        return torch.ones((1,), dtype=centroids.dtype, device=centroids.device)

    cpu_centroids = centroids.detach().to("cpu", torch.float64)
    distances = torch.cdist(cpu_centroids, cpu_centroids)
    edges = []
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            edges.append((float(distances[i, j]), i, j))
    edges.sort(key=lambda item: item[0])

    parent = list(range(num_blocks))
    members = {i: {i} for i in range(num_blocks)}
    salience = torch.zeros((num_blocks,), dtype=torch.float64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for distance, left, right in edges:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if len(members[root_left]) > len(members[root_right]):
            root_left, root_right = root_right, root_left

        for block in members[root_left]:
            salience[block] = distance
        parent[root_left] = root_right
        members[root_right].update(members[root_left])
        del members[root_left]

    return salience.to(device=centroids.device, dtype=centroids.dtype)


def build_dense_causal_block_schedule(num_blocks):
    """Build a lower-triangular CSR schedule over causal key blocks."""
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    offsets = [0]
    indices = []
    for q_block in range(num_blocks):
        indices.extend(range(q_block + 1))
        offsets.append(len(indices))
    return torch.tensor(offsets, dtype=torch.int64), torch.tensor(
        indices, dtype=torch.int64
    )


def build_topology_block_schedule(
    keys,
    block_size,
    local_radius_blocks,
    sink_blocks,
    topk_topology_blocks,
):
    """Build a causal CSR schedule from sink, local, and topological key blocks."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape [seq, dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if keys.shape[0] % block_size != 0:
        raise ValueError("sequence length must be divisible by block_size")
    if local_radius_blocks < 0:
        raise ValueError("local_radius_blocks must be non-negative")
    if sink_blocks < 0:
        raise ValueError("sink_blocks must be non-negative")
    if topk_topology_blocks < 0:
        raise ValueError("topk_topology_blocks must be non-negative")

    num_blocks = keys.shape[0] // block_size
    centroids = keys.reshape(num_blocks, block_size, keys.shape[1]).mean(dim=1)
    salience = _zero_dim_persistence_salience(centroids)
    topk = min(max(topk_topology_blocks, 0), num_blocks)
    if topk:
        topology_blocks = set(torch.topk(salience, k=topk).indices.tolist())
    else:
        topology_blocks = set()

    offsets = [0]
    indices = []
    for q_block in range(num_blocks):
        allowed = set(range(min(sink_blocks, num_blocks)))
        allowed.update(range(max(0, q_block - local_radius_blocks), q_block + 1))
        allowed.update(block for block in topology_blocks if block <= q_block)
        allowed = {block for block in allowed if block <= q_block}
        if not allowed:
            allowed.add(q_block)

        indices.extend(sorted(allowed))
        offsets.append(len(indices))

    return torch.tensor(offsets, dtype=torch.int64, device=keys.device), torch.tensor(
        indices, dtype=torch.int64, device=keys.device
    )


def dense_masked_attention(q, k, v, offsets, indices, block_size):
    """Reference attention for a CSR schedule, implemented with PyTorch masking."""
    num_blocks = q.shape[0] // block_size
    allow = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    for q_block in range(num_blocks):
        q_slice = slice(q_block * block_size, (q_block + 1) * block_size)
        start = int(offsets[q_block].item())
        end = int(offsets[q_block + 1].item())
        for block in indices[start:end]:
            block = int(block.item())
            k_slice = slice(block * block_size, (block + 1) * block_size)
            allow[q_slice, k_slice] = True

    positions = torch.arange(q.shape[0], device=q.device)
    allow &= positions[None, :] <= positions[:, None]
    logits = (q @ k.T) / math.sqrt(q.shape[1])
    logits = logits.masked_fill(~allow, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


@triton.jit
def _scheduled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offsets_ptr,
    indices_ptr,
    out_ptr,
    seq: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    q_block = tl.program_id(0)
    batch_block = tl.program_id(1)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr
        + batch_block * stride_qb
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < seq,
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    schedule_ptr = tl.load(offsets_ptr + q_block)
    schedule_end = tl.load(offsets_ptr + q_block + 1)
    while schedule_ptr < schedule_end:
        k_block = tl.load(indices_ptr + schedule_ptr)
        k_pos = k_block * BLOCK_N + offs_n

        k_tile = tl.load(
            k_ptr
            + batch_block * stride_kb
            + k_pos[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=k_pos[:, None] < seq,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k_tile)) * scale
        valid = (
            (offs_m[:, None] < seq)
            & (k_pos[None, :] < seq)
            & (k_pos[None, :] <= offs_m[:, None])
        )
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        v_tile = tl.load(
            v_ptr
            + batch_block * stride_vb
            + k_pos[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=k_pos[:, None] < seq,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_tile.to(tl.float32))
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
        schedule_ptr += 1

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + batch_block * stride_ob
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        out,
        mask=offs_m[:, None] < seq,
    )


def scheduled_attention(q, k, v, offsets, indices, block_size):
    """Run forward causal attention over the key/value blocks in a CSR schedule."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same shape")
    if q.ndim not in (2, 4):
        raise ValueError(
            "q, k, and v must have shape [seq, dim] or [batch, heads, seq, dim]"
        )
    if not _is_power_of_two(block_size):
        raise ValueError("block_size must be a positive power of two")
    if q.shape[-2] % block_size != 0:
        raise ValueError("sequence length must be divisible by block_size")

    seq, head_dim = q.shape[-2:]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError("head dimension must be one of 16, 32, 64, or 128")
    if offsets.ndim != 1 or offsets.numel() != seq // block_size + 1:
        raise ValueError("offsets must have shape [num_query_blocks + 1]")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if offsets.dtype not in _SCHEDULE_DTYPES or indices.dtype not in _SCHEDULE_DTYPES:
        raise ValueError("offsets and indices must be integer tensors")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if offsets.device != q.device or indices.device != q.device:
        raise ValueError("offsets and indices must be on the same device as q")

    output_shape = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    offsets = offsets.contiguous()
    indices = indices.contiguous()
    q_flat = q.reshape(-1, seq, head_dim)
    k_flat = k.reshape(-1, seq, head_dim)
    v_flat = v.reshape(-1, seq, head_dim)
    out = torch.empty_like(q_flat)
    _scheduled_attention_kernel[(seq // block_size, q_flat.shape[0])](
        q_flat,
        k_flat,
        v_flat,
        offsets,
        indices,
        out,
        seq,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1.0 / math.sqrt(head_dim),
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out.reshape(output_shape)
