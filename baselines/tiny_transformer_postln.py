import torch, torch.nn as nn
import torch.nn.functional as F
import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinySelfAttention, TinyMLP, TinyTransformer
from engine.fused_lnresidual import ln_residual


class TinyTransformerBlockPostLN(nn.Module):
    def __init__(self, d_model=256, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.attn = TinySelfAttention(d_model, n_heads)
        self.mlp = TinyMLP(d_model, mlp_ratio)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        # y1 = LN(x + Attn(x)) - baseline PyTorch LN
        a_res = x +self.attn(x)
        y1= F.layer_norm(
            a_res, a_res.shape[-1:],
            self.ln1.weight, 
            self.ln1.bias
        )
        
        # y2 = LN(y1 + MLP(y1)) - baseline PyTorch LN
        b_res = y1 +self.mlp(y1)
        y2 = F.layer_norm(
            b_res, b_res.shape[-1:],
            self.ln2.weight, 
            self.ln2.bias
        )

        # # y1 = LN(x + Attn(x))  — fused LN+residual
        # a_res = x+self.attn(x)
        # y1 = ln_residual(
        #     a_res.float().contiguous(),
        #     torch.zeros_like(a_res, dtype=torch.float32),
        #     self.ln1.weight.float().contiguous(),
        #     self.ln1.bias.float().contiguous(),
        # ).to(a_res.dtype)

        # # y2 = LN(y1+ MLP(y1)) - fused LN+residual
        # b_res = y1 +self.mlp(y1)
        # y2 = ln_residual(
        #     b_res.float().contiguous(),
        #     torch.zeros_like(b_res, dtype=torch.float32),
        #     self.ln2.weight.float().contiguous(),
        #     self.ln2.bias.float().contiguous(),
        # ).to(b_res.dtype)
        
        return y2
    

class TinyTransformerPostLN(nn.Module):
    def __init__(self, d_model=256, n_heads=4, depth=4, vocab_size=32000, max_len=4096):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([TinyTransformerBlockPostLN(d_model, n_heads) for _ in range(depth)])
        self.head = nn.Linear(d_model, vocab_size)


    def forward(self, idx):
        B, L  = idx.shape
        x = self.tok(idx) + self.pos[:, :L, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)