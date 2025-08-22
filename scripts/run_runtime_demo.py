import time, argparse, torch

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer
from engine_runtime.engine_torch import TorchEngine
from engine_runtime.scheduler import Scheduler, Request

def make_prompt_ids(vocab: int, text: str, max_len: int = 32) -> torch.Tensor:
    """
    Tiny demo tokenizer: map chars to pseudo-token IDs (very rough).
    For your real model, replace with your vocab/tokenizer.
    """
    ids = [(ord(c) % (vocab-10)) + 10 for c in text] 
    ids = ids[:max_len]
    if not ids: ids = [10]
    return torch.tensor([ids], dtype=torch.long) # (1,L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=["hello world", "lorem ipsum", "gpu kernels are fun"])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--batch_cap", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()

    print("[int] building TinyTransformer...")
    model = TinyTransformer(d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
                            vocab_size=args.vocab)
    eng = TorchEngine(model, amp=True)
    sched = Scheduler(eng, batch_cap=args.batch_cap)

    # streaming callback prints tokens as they come 
    def mk_cb(tag):
        def _cb(tid: int):
            print(f"[{tag}] token{tid}")
        return _cb
    
    requests = []
    for i, p in enumerate(args.prompts):
        pid = make_prompt_ids(args.vocab, p)
        requests.append(Request(prompt_ids=pid, max_new_tokens=args.max_new, stop_id=None, on_token=mk_cb(f"req{i+1}")))

    t0 = time.time()
    sched.submit_many(requests)
    dt = time.time() - t0
    print(f"\n[done] streamed {len(requests)} requests in {dt:.2f}s")

if __name__ == "__main__":
    main()