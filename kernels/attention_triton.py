# Triton implementation of Flash Attention
# This is a custom kernel that performs the attention operation in a memory-efficient way
# It uses Triton to compile and run the kernel on GPU
# It is designed to be used as a drop-in replacement for PyTorch's scaled dot-product attention

import triton
import triton.language as tl

@triton.jit
def flash_attn(Q, K, V, O, stride_q, stride_k, stride_v, stride_o,
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
