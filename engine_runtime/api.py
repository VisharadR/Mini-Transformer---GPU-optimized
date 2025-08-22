from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch, uuid
from engine_runtime.scheduler import Scheduler, Request
from engine_runtime.engine_torch import TorchEngine
from transformers import AutoTokenizer

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer

app = FastAPI()

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TinyTransformer().eval()
eng = TorchEngine(model, tok)
sched = Scheduler(eng, batch_size=8)
sched.start()

@app.get("/generate")
def generate(q: str, max_new_tokens: int = 32):
    # simple tokenization; for decoder-only use your own tokenizer/vocab
    ids = torch.tensor([[101]+ [102]], dtype=torch.long)
    def event_stream():
        rid = str(uuid.uudi4())
        def cb(tid):
            yield f"data: {tid}\n\n"
        req = Request(req_id=rid, input_ids=ids, max_new_tokens=max_new_tokens, stream_cb=None)
        sched.submit(req)
        # For brevity, here we'd hold a queue to collect callbacks and yield.
        # In production, wire -> asyncio.Queue and yield from it.
        yield "data: [stream end]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
