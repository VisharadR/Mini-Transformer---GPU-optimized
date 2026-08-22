# AMD ROCm variant of kernels/attention_triton.py.
#
# Audit notes (why this file exists instead of just reusing the original
# as-is on ROCm):
#   - The original kernel body has no hardcoded warp-size assumptions --
#     there's no literal 32 anywhere tied to lane/warp math, and the
#     online-softmax + tl.dot logic is expressed in terms of BLOCK_M/BLOCK_N
#     tiles, not raw threads. Triton's backend compiler is the thing that
#     maps a kernel onto CDNA's 64-wide wavefronts vs NVIDIA's 32-wide warps
#     via `num_warps`, and it does that automatically -- so the kernel math
#     itself needed no changes for correctness.
#   - What *does* need AMD-specific tuning is BLOCK_M/BLOCK_N/num_warps: the
#     original file has no autotuning at all (block sizes must be supplied
#     by the caller), so there were no NVIDIA-tuned constants to inherit or
#     fix, just a gap to fill. This file adds a `triton.autotune` sweep with
#     configs sized around a 64-wide wavefront instead of guessing NVIDIA
#     numbers and hoping they're fine on CDNA/RDNA.
#   - The original kernel is also currently *dead code* in this repo:
#     engine/fused_attention.py imports `flash_attn` but
#     TritonSelfAttention._attn_triton() just calls the PyTorch SDPA path
#     ("temporary until Triton is finalized") and no launcher/grid
#     computation exists anywhere. This file adds both an autotuned kernel
#     and the missing launcher so it's actually runnable/testable on ROCm.
#
# Kernel body is otherwise unchanged from kernels/attention_triton.py.

import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from engine.device import get_device_type

# Block-size candidates sized for CDNA's 64-wide wavefront (vs NVIDIA's
# 32-wide warp) -- num_warps here counts Triton "warps" (wavefronts on
# ROCm), so equal num_warps still means 2x the threads-per-block on AMD.
# These are starting points, not hand-tuned-to-the-last-percent numbers;
# triton.autotune benchmarks all of them per (seq_len, head_dim) and keeps
# whichever wins on the actual GPU. See docs/AMD_ROCM.md for real numbers
# once this has run on hardware.
_ROCM_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_ROCM_AUTOTUNE_CONFIGS, key=["seq_len", "head_dim"])
@triton.jit
def flash_attn_rocm(Q, K, V, O, stride_q, stride_k, stride_v, stride_o,
                     n_heads, head_dim, seq_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)  # which block of queries
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, head_dim)

    # Pointers to Q block
    q = tl.load(Q + off_m[:, None] * stride_q + off_d[None, :], mask=off_m[:, None] < seq_len)

    # online softmax stats
    m_i = tl.full([BLOCK_M], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], tl.float32)

    # iterate over K/V tiles
    for n in range(0, seq_len, BLOCK_N):
        off_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(K + off_n[:, None] * stride_k + off_d[None, :], mask=off_n[:, None] < seq_len)
        v = tl.load(V + off_n[:, None] * stride_v + off_d[None, :], mask=off_n[:, None] < seq_len)

        # scores: [M, N] = q @ k^T (implicit via dot + broadcasting)
        scores = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32)) / (head_dim ** 0.5)

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        p = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc = acc * (tl.exp(m_i - m_ij))[:, None] + tl.dot(p, v.to(tl.float32))
        l_i = l_i * tl.exp(m_i - m_ij) + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]
    tl.store(O + off_m[:, None] * stride_o + off_d[None, :], out)


def flash_attention_rocm(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Launcher for flash_attn_rocm. q, k, v: [B, H, L, Dh] contiguous fp32.

    The underlying kernel (unchanged from the original) only takes a single
    stride per tensor, i.e. it operates on one contiguous [L, Dh] slice per
    launch -- there's no batch/head stride param. So this launcher loops over
    (batch, head) in Python and issues one autotuned kernel launch per slice,
    matching what the kernel actually supports rather than changing its
    signature.
    """
    B, H, L, Dh = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            qs = q[b, h].contiguous()
            ks = k[b, h].contiguous()
            vs = v[b, h].contiguous()
            os_ = out[b, h]
            grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]),)
            flash_attn_rocm[grid](
                qs, ks, vs, os_,
                qs.stride(0), ks.stride(0), vs.stride(0), os_.stride(0),
                H, Dh, L,
            )
    return out


if __name__ == "__main__":
    device_type = get_device_type()
    if device_type not in ("rocm", "cuda"):
        print(f"[attention_triton_rocm] No GPU backend detected (got '{device_type}'); skipping correctness test.")
    else:
        torch.manual_seed(0)
        B, H, L, Dh = 2, 4, 128, 64
        q = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)

        out_triton = flash_attention_rocm(q, k, v)
        out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        max_err = (out_triton - out_ref).abs().max().item()
        print(f"[attention_triton_rocm] backend={device_type} max|delta vs SDPA|={max_err:.6f}")
        assert max_err < 1e-2, "Triton kernel output diverges from scaled_dot_product_attention reference"
        print("[attention_triton_rocm] correctness test passed")
