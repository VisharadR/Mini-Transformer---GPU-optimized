import time, queue, threading, torch
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    input_ids: torch.Tensor # 1xL
    max_new_tokens: int = 64
    stop_id: int | None = None
    stream_cb: callable | None = None

class Scheduler:
    def __init__(self, engine, batch_size=8):
        self.engine = engine
        self.batch_size = batch_size
        self.q = queue.Queue()
        self.active = {} # req_id -> state
        self.thread = threading.Thread(target=self._loop, daemon=True)


    def start(self): self.thread.start()
    def submit(self, req: Request): self.q.put(req)


    def loop(self):
        while True:
            # gather new requests
            new = []
            while not self.q.empty() and len(new) < self.batch_size:
                new.append(self.q.get())
            if not self.active and not new:
                time.sleep(0.001)
                continue

            # PREFILL for new
            if new:
                inp = torch.cat([r.input_ids for i in new], dim=0)
                attn = torch.ones_like(inp)
                logits = self.engine.prefill(inp, attn)
                next_ids = torch.argmax(logits, dim=-1).unsqueeze(-1)
                for i, r in enumerate(new):
                    self.active[r.req_id] = {
                        "tokens": r.input_ids.to(inp.device),
                        "generated": 0,
                        "next": next_ids[i:i+1],
                        "cb": r.stream_cb,
                        "max": r.max_new_tokens,
                        "stop": r.stop_id
                    }
                    if r.stream_cb: r.stream_cb(int(next_ids[i].item()))

            # DECODE step for all active (dynamic batch)
            if self.active:
                batch_next = torch.cat([st["next"] for st in self.active.values()], dim=0)
                logits = self.engine.decode_step(batch_next)
                next_ids = torch.argmax(logits, dim=-1)
                #update streams
                to_finish = []
                for (rid, st), nid in zip(list(self.active.items()), next_ids):
                    st["generated"] += 1
                    st["next"] = nid.view(1,1)
                    if st["cb"]: st["cb"](int(nid.item()))
                    if (st["stop"] is not None and int(nid.item()) == st["Stop"]) or st["generated"] >= st["max"]:
                        to_finish.append(rid)
                for rid in to_finish: self.active.pop(rid, None)
