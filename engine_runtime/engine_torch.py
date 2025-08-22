import torch, torch.nn.functional as F
from engine_runtime.engine_iface import Engine
from engine_runtime.kv_cache import KVMem

class TrochEngine(Engine):
    def __init__(self, model, tokenizer, max_seq=2048, amp=True):
        self.m =model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer
        self.dev = next(self.m.parameters()).device
        self.amp = amp

        self.n_layers = len(self.m.blocks)
        self.n_heads = self.m.blocks[0].attn.n_heads
        self.head_dim = self.m.blocks[0].attn.d_head


    def _amp_ctx(self):
        return torch.cuda.amp.autocast(dtype=torch.float16) if (self.dev.type == "cuda" and self.amp) else torch.autocast("cpu", enabled=False)
    
    def prefill(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        """B×L -> B×V logits; initializes internal KV via model forward that writes cache (modify your model to expose hooks)."""
        with torch.inference_mode(), self._amp_ctx():
            logits = self.m(input_ids.to(self.dev))
        return logits[:, -1, :]
    
    def decode_step(self, input_ids: torch.Tensor):
        with torch.inference_mode(), self._amp_ctx():
            logits = self.m(input_ids.to(self.dev))
        return logits[:, -1, :]
    