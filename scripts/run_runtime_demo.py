import os, sys, time, json, argparse, torch
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer
from engine_runtime.engine_torch import TorchEngine
from engine_runtime.scheduler import Scheduler, Request

def make_prompt_ids(vocab: int, text: str, max_len: int = 64) -> torch.Tensor:
    # Demo tokenizer: map chars to IDs (placeholder). Replace with real tokenizer if available.
    ids = [ (ord(c) % (vocab-10)) + 10 for c in text ][:max_len]
    if not ids: ids = [10]
    return torch.tensor([ids], dtype=torch.long)

def detok(ids):
    # Demo detokenizer: join IDs as numbers. Replace with real detokenizer if you have one.
    return " ".join(str(i) for i in ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=["hello gpu", "deep learning is fun", "cuda kernels"])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--batch_cap", type=int, default=8)
    ap.add_argument("--sampler", type=str, default="greedy", choices=["greedy","topk"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save_outputs", type=str, default=None, help="path to JSON file with results")
    args = ap.parse_args()

    print("[init] building TinyTransformer…")
    model = TinyTransformer(d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
                            vocab_size=args.vocab)
    eng = TorchEngine(model, amp=not args.no_amp)

    # Build requests with streaming callbacks
    all_generated = { }  # tag -> list[int] (generated only)
    def mk_cb(tag):
        def _cb(tid: int):
            all_generated.setdefault(tag, []).append(int(tid))
            print(f"[{tag}] token={tid}")
        return _cb

    requests = []
    for i, text in enumerate(args.prompts):
        tag = f"req{i+1}"
        pid = make_prompt_ids(args.vocab, text)
        requests.append(Request(prompt_ids=pid, max_new_tokens=args.max_new, stop_id=None, on_token=mk_cb(tag), tag=tag))

    # Run scheduler
    sched = Scheduler(eng, batch_cap=args.batch_cap, sampler=args.sampler, topk=args.topk, temperature=args.temperature)

    total_toks = sum(r.prompt_ids.shape[1] + args.max_new for r in requests)
    t0 = time.perf_counter()
    stats = sched.submit_many(requests)
    total_ms = (time.perf_counter() - t0) * 1000.0
    toks_per_s = total_toks / (total_ms / 1000.0)

    # Pretty summary
    print("\n=== RUNTIME SUMMARY ===")
    print(f"Requests: {len(requests)} | Max new tokens per request: {args.max_new}")
    print(f"Engine device: {eng.dev.type.upper()} | AMP: {not args.no_amp}")
    print(f"Sampler: {args.sampler} (topk={args.topk}, temp={args.temperature})")
    print(f"Step latency: p50={stats['timings']['p50_ms']} ms, p95={stats['timings']['p95_ms']} ms, steps={stats['timings']['steps']}")
    print(f"End-to-end: total={stats['timings']['total_ms']} ms  |  tokens/s (approx)={toks_per_s:,.1f}")

    # Reconstruct outputs and optionally save JSON
    results = []
    for i, text in enumerate(args.prompts):
        tag = f"req{i+1}"
        prompt_ids = requests[i].prompt_ids.squeeze(0).tolist()
        gen_ids = all_generated.get(tag, [])
        results.append({
            "tag": tag,
            "prompt_text": text,
            "prompt_ids": prompt_ids,
            "generated_ids": gen_ids,
            "generated_text_demo": detok(gen_ids)  # placeholder detok
        })

    if args.save_outputs:
        os.makedirs(os.path.dirname(args.save_outputs), exist_ok=True) if os.path.dirname(args.save_outputs) else None
        with open(args.save_outputs, "w", encoding="utf-8") as f:
            json.dump({
                "config": vars(args),
                "summary": {
                    "p50_ms": stats["timings"]["p50_ms"],
                    "p95_ms": stats["timings"]["p95_ms"],
                    "steps": stats["timings"]["steps"],
                    "total_ms": stats["timings"]["total_ms"],
                    "tokens_per_s": round(float(toks_per_s), 1),
                },
                "results": results
            }, f, indent=2)
        print(f"[ok] wrote {args.save_outputs}")

if __name__ == "__main__":
    main()