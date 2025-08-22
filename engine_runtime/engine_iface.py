from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch

Tensor = torch.Tensor

class Engine(ABC):
    """
    Minimal interface for a token-by-token inference engine.

    Conventions
    ----------
    - input_ids: Long (int64) tensor of shape [B, L] (prefill) or [B, 1] (decode).
    - attention_mask: Optional Long/Bool tensor of shape [B, L].
    - Returns logits of shape [B, V] (V = vocab size) for the *last* position.
    - Implementations may keep an internal KV cache across calls.
    """

    # ---- Introspection helpers (optional but useful) ----
    @property
    def device(self) -> torch.device:
        """Device the model/engine runs on (default: best-effort cuda/cpu)."""
        try:
            return next(self.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def vocab_size(self) -> Optional[int]:
        """Return vocab size if known (used by samplers/UI)."""
        return None

    @property
    def max_seq_len(self) -> Optional[int]:
        """Return max supported context length (for schedulers/UI)."""
        return None

    @property
    def supports_kv_cache(self) -> bool:
        """Whether decode_step uses an incremental KV cache."""
        return False

    # ---- Core API the scheduler relies on ----
    @abstractmethod
    def prefill(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Run the first pass over a prompt (B, L) and initialize any KV cache.
        Returns logits for the last token: shape [B, V].
        """
        
    @abstractmethod
    def decode_step(self, input_ids: Tensor) -> Tensor:
        """
        Generate one next-token step for active sequences (B, 1).
        Returns logits for the new position: shape [B, V].
        """
        
    # ---- Lifecycle hooks (optional) ----
    def reset(self) -> None:
        """Clear internal state/KV cache for all sequences."""
        return

    def reset_batch(self, batch_indices: Tensor) -> None:
        """
        Clear state for specific batch slots (scheduler may call this when
        requests finish). Default no-op.
        """
        return

__all__ = ["Engine"]
