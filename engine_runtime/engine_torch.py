from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional
from engine_runtime.engine_iface import Engine


class TorchEngine(Engine):
    """
    Minimal wrapper around your TinyTransformer.
    - prefill(B,L) -> logits(B,V) for last token
    - decode_step(B,1) -> logits(B,V)
    NOTE: This baseline recomputes full forward; hook KV later for speed.
    """
    def __init__(self, model, amp: bool = True):
        self.m = model.eval()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m.to(self.dev)
        self.amp = bool(amp)
        # Try to infer vocab size if model exposes it
        self._vocab = getattr(self.m, "tok", None).num_embeddings if hasattr(self.m, "tok") else None

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab

    def _amp_ctx(self):
        use = (self.dev.type == "cuda" and self.amp)
        return torch.amp.autocast('cuda', dtype=torch.float16) if use else torch.autocast(device_type="cpu", enabled=False)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = input_ids.to(self.dev, dtype=torch.long, non_blocking=True)
        # building a simple attention mask for prefill
        if attention_mask is None:
            attention_mask = torch.ones_like(x, dtype=torch.long, device=self.dev)
        with self._amp_ctx():
            logits = self.m(x)  # expect (B, L, V)
        return logits[:, -1, :]

    @torch.inference_mode()
    def decode_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Baseline: full forward on growing sequence (KV cache to be added later)
        x = input_ids.to(self.dev, dtype=torch.long, non_blocking=True)
        with self._amp_ctx():
            logits = self.m(x)
        return logits[:, -1, :]

    def reset(self) -> None:
        pass