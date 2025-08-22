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
    tag: Optional[str] = None                         # label for logs

class Scheduler:
    """
    Synchronous token-level batcher with simple dynamic batching.
    Collects per-step timings and returns per-request outputs.
    """
    def __init__(self, engine, batch_cap: int = 16, sampler: str = "greedy", topk: int = 50, temperature: float = 1.0):
        self.engine = engine
        self.batch_cap = batch_cap
        self.sampler = sampler
        self.topk = topk
        self.temperature = temperature
        self.active: List[dict] = []   # each: {ids: (1,T), remaining:int, stop:int|None, cb:fn|None, tag:str|None}
        self.step_times: List[float] = []

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sampler == "greedy":
            return sample_greedy(logits)
        return sample_topk(logits, k=self.topk, temperature=self.temperature)

    def _attach(self, req: Request):
        # PREFILL all tokens in the prompt once
        t0 = time.perf_counter()
        logits = self.engine.prefill(req.prompt_ids)                 # (1,V)
        self.step_times.append((time.perf_counter() - t0) * 1000.0)
        nxt = self._sample(logits).item()
        if req.on_token: req.on_token(nxt)
        ids = torch.cat([req.prompt_ids.cpu(), torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        self.active.append({"ids": ids, "remaining": req.max_new_tokens - 1, "stop": req.stop_id, "cb": req.on_token, "tag": req.tag})

    def submit_many(self, requests: List[Request]) -> Dict[str, Any]:
        """
        Returns:
          {
            'outputs': [ {'tag': str|None, 'ids': List[int]} ... ],
            'timings': {'p50_ms': float, 'p95_ms': float, 'steps': int, 'total_ms': float}
          }
        """
        self.step_times.clear()
        for r in requests:
            self._attach(r)

        start = time.perf_counter()
        while self.active:
            # Build dynamic batch of last tokens
            last_tokens = [a["ids"][:, -1:] for a in self.active]          # list of (1,1)
            inp = torch.cat(last_tokens, dim=0)                             # (B,1)
            t0 = time.perf_counter()
            logits = self.engine.decode_step(inp)                           # (B,V)
            self.step_times.append((time.perf_counter() - t0) * 1000.0)
            next_ids = self._sample(logits)                                 # (B,)

            # Update streams and finishers
            to_remove = []
            for i, a in enumerate(self.active):
                tid = int(next_ids[i].item())
                a["ids"] = torch.cat([a["ids"], torch.tensor([[tid]], dtype=torch.long)], dim=1)
                if a["cb"]: a["cb"](tid)
                a["remaining"] -= 1
                if (a["stop"] is not None and tid == a["stop"]) or a["remaining"] <= 0:
                    to_remove.append(i)
            for i in reversed(to_remove):
                pass  # removal below

            self.active = [a for j, a in enumerate(self.active) if j not in to_remove]

        total_ms = (time.perf_counter() - start) * 1000.0
        p50 = float(np.percentile(self.step_times, 50)) if self.step_times else 0.0
        p95 = float(np.percentile(self.step_times, 95)) if self.step_times else 0.0

        # Collate outputs
        outputs = []
        # In this synchronous version, we don't retain completed items separately; rebuild from requests
        for r in requests:
            outputs.append({
                "tag": r.tag,
                "ids": r.prompt_ids.squeeze(0).tolist()  # only the prompt; the generated part is printed below
            })

        return {
            "outputs": outputs,
            "timings": {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "steps": len(self.step_times), "total_ms": round(total_ms, 2)}
        }
