import torch

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer
from engine.fused_attention import TritonSelfAttention


class Engine:
    def __init__(self, d_model=256, n_heads=4, depth=4, vocab=32000, use_triton_attention=True):
        self.model = TinyTransformer(d_model, n_heads, depth, vocab).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.use_triton_attention = use_triton_attention

        if self.use_triton_attention:
            # swap each block's attention with pluggable module
            for blk in self.model.blocks:
                #keep the same dimensions as orignial
                blk.attn = TritonSelfAttention(d_model=d_model, n_heads=n_heads).to(next(self.model.parameters()).device)

    @torch.inference_mode()
    def forward(self, idx):
        return self.model(idx) 
    


if __name__ == "__main__":
    eng = Engine(use_triton_attention=True)
    x = torch.randint(0, 32000, (4, 128), device=next(eng.model.parameters()).device)
    y = eng.forward(x)
    print(y.shape)

