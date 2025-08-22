from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import torch
from engine_runtime.sampling import sample_greedy

@dataclass
class Request:
    prompt_ids: torch.Tensor      # (1, L0) int64
    max_new_tokens: int = 32
    stop_id: Optional[int] = None
    on_token: Optional[Callable[[int], None]] = None  # streaming callback

class Scheduler:
    """
    Synchronous token-level batcher:
      - Prefills all new requests together.
      - Then loops token-by-token; dynamically batches all active requests.
    """
    def __init__(self, engine, batch_cap: int = 16):
        self.engine = engine
        self.batch_cap = batch_cap
        self.active: List[dict] = []   # each: {ids: (1,T), remaining:int, stop:int|None, cb:fn|None}

    def _attach(self, req: Request):
        # prefill
        logits = self.engine.prefill(req.prompt_ids)                 # (1,V)
        nxt = sample_greedy(logits).item()
        if req.on_token: req.on_token(nxt)
        ids = torch.cat([req.prompt_ids.cpu(), torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        self.active.append({"ids": ids, "remaining": req.max_new_tokens - 1, "stop": req.stop_id, "cb": req.on_token})

    def submit_many(self, requests: List[Request]):
        for r in requests: self._attach(r)
        # decode loop
        while self.active:
            # build batch of next-token inputs: last token of each active seq
            last_tokens = [a["ids"][:, -1:] for a in self.active]              # list of (1,1)
            inp = torch.cat(last_tokens, dim=0)                                # (B,1)
            logits = self.engine.decode_step(inp)                               # (B,V)
            next_ids = sample_greedy(logits)                                    # (B,)
            # update streams
            to_remove = []
            for i, a in enumerate(self.active):
                tid = int(next_ids[i].item())
                # append token to sequence (kept on CPU to keep it simple)
                a["ids"] = torch.cat([a["ids"], torch.tensor([[tid]], dtype=torch.long)], dim=1)
                if a["cb"]: a["cb"](tid)
                a["remaining"] -= 1
                if (a["stop"] is not None and tid == a["stop"]) or a["remaining"] <= 0:
                    to_remove.append(i)
            # remove finished (from back to front)
            for i in reversed(to_remove):
                self.active.pop(i)