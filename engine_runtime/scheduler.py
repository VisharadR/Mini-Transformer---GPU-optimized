from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any
import time, numpy as np, torch
from engine_runtime.sampling import sample_greedy, sample_topk

@dataclass
class Request:
    prompt_ids: torch.Tensor      # (1, L0) int64
    max_new_tokens: int = 32
    stop_id: Optional[int] = None
    on_token: Optional[Callable[[int], None]] = None  # streaming callback
    tag: Optional[str] = None

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Scheduler:
    """
    Synchronous token-level batcher with separate timing buckets:
      - prefill_times: one per request
      - decode_times: one per token step (batched)
    """
    def __init__(self, engine, batch_cap: int = 16, sampler: str = "greedy", topk: int = 50, temperature: float = 1.0):
        self.engine = engine
        self.batch_cap = batch_cap
        self.sampler = sampler
        self.topk = topk
        self.temperature = temperature
        self.active: List[dict] = []
        self.prefill_times: List[float] = []
        self.decode_times: List[float] = []

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sampler == "greedy":
            return sample_greedy(logits)
        return sample_topk(logits, k=self.topk, temperature=self.temperature)

    def _attach(self, req: Request):
        # PREFILL
        _sync(); t0 = time.perf_counter()
        logits = self.engine.prefill(req.prompt_ids)   # (1,V)
        _sync(); self.prefill_times.append((time.perf_counter() - t0) * 1000.0)
        nxt = self._sample(logits).item()
        if req.on_token: req.on_token(nxt)
        ids = torch.cat([req.prompt_ids.cpu(), torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        self.active.append({"ids": ids, "remaining": req.max_new_tokens - 1, "stop": req.stop_id, "cb": req.on_token, "tag": req.tag})

    def submit_many(self, requests: List[Request]) -> Dict[str, Any]:
        self.prefill_times.clear()
        self.decode_times.clear()
        for r in requests:
            self._attach(r)

        _sync(); start = time.perf_counter()
        while self.active:
            last_tokens = [a["ids"][:, -1:] for a in self.active]
            inp = torch.cat(last_tokens, dim=0)                 # (B,1)
            _sync(); t0 = time.perf_counter()
            logits = self.engine.decode_step(inp)               # (B,V)
            _sync(); self.decode_times.append((time.perf_counter() - t0) * 1000.0)
            next_ids = self._sample(logits)

            to_remove = []
            for i, a in enumerate(self.active):
                tid = int(next_ids[i].item())
                a["ids"] = torch.cat([a["ids"], torch.tensor([[tid]], dtype=torch.long)], dim=1)
                if a["cb"]: a["cb"](tid)
                a["remaining"] -= 1
                if (a["stop"] is not None and tid == a["stop"]) or a["remaining"] <= 0:
                    to_remove.append(i)
            for i in reversed(to_remove):
                self.active.pop(i)

        _sync(); total_ms = (time.perf_counter() - start) * 1000.0

        def _stats(arr):
            if not arr: return {"p50_ms": 0.0, "p95_ms": 0.0, "count": 0}
            return {
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "count": len(arr)
            }

        return {
            "timings": {
                "prefill": _stats(self.prefill_times),
                "decode":  _stats(self.decode_times),
                "total_ms": round(total_ms, 2),
            }
        }
