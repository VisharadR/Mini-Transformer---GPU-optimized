import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Trying out triton kernel for fused attention
_TRITON_OK = False
try:
    from kernels.attention_triton import flash_attn as _flash_attn
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False



class TritonSelfAttention(nn.Module):
    #This class will be serving as substitute for TinySelfAttention while being wrapper for triton kernel
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)


    def _attn_pytorch(self, q, k, v, attn_mask=None):
        # PyTorch SDPA wants [B*H, L, Dh] or [B,H,L,Dh] with batch_first
        B, H, L, Dh = q.shape
        q = q.transpose(1,2) # [B, L, H, Dh]
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # uses fast path (Flash/Memory-efficient) if available
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        ) # [B, L, H, Dh]
        return out.transpose(1,2) # back to [B, H, L, Dh]
    
    def _attn_triton(self, q, k, v):
        # q, k, v: [B, L, H, Dh] -> flatten heads anc call kernel per head block
        return self._attn_pytorch(q, k, v) # temporary until Triton is finalized
    
    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x) # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        H, Dh = self.n_heads, self.d_head
        q = q.view(B, L, H, Dh).transpose(1,2) # [B, H, L, Dh]
        k = k.view(B, L, H, Dh).transpose(1,2)
        v = v.view(B, L, H, Dh).transpose(1,2)

        if _TRITON_OK and x.is_cuda:
            out = self._attn_triton(q, k, v)
        else:
            out = self._attn_pytorch(q, k, v, attn_mask)

        out = out.transpose(1, 2).contiguous().view(B, L, D) # [B, L, D]
        return self.proj(out) # [B, L, D]