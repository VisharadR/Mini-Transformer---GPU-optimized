import torch

torch.backends.cuda.matmul.allow_tf32 = True     # enable TF32
torch.backends.cudnn.benchmark = True            # pick best algos for your shapes
torch.set_float32_matmul_precision("high")       # allow tensor cores to kick in

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer
try:
    from engine.fused_attention import TritonSelfAttention
except Exception:
    TritonSelfAttention = None

class Engine:
    def __init__(self, d_model=256, n_heads=4, depth=4, vocab=32000, max_len=4096, use_triton_attention=False, use_amp=True, compile_mode=None):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TinyTransformer(d_model, n_heads, depth, vocab, max_len).to(device).eval()
        
        # swap attention if requested
        if use_triton_attention and TritonSelfAttention is not None:
            for blk in self.model.blocks:
                blk.attn = TritonSelfAttention(d_model=d_model, n_heads=n_heads).to(device)

        # optional torch.compile
        self.compiled = False
        if compile_mode:
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                self.compiled = True
            except Exception:
                pass

        self.use_amp = bool(use_amp)

    @torch.inference_mode()
    def forward(self, idx):
        dev = next(self.model.parameters()).device
        if dev.type == "cuda" and self.use_amp:
            # half-precision matmul/attention on tensor cores
            with torch.amp.autocast('cuda' ,dtype=torch.float16):
                return self.model(idx)
        return self.model(idx)
    


# if __name__ == "__main__":
#     eng = Engine(use_triton_attention=True)
#     x = torch.randint(0, 32000, (4, 128), device=next(eng.model.parameters()).device)
#     y = eng.forward(x)
#     print(y.shape)

