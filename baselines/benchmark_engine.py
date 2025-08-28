import time, numpy as np, torch, csv, sys, argparse

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from engine.perf_toggles import enable_fast_paths
from engine.runner import Engine

enable_fast_paths()

@torch.inference_mode()
def bench(use_triton, use_amp, compile_mode, seq_lens, batch_sizes, warmup, iters):
    eng = Engine(use_triton_attention=use_triton, use_amp=use_amp, compile_mode=compile_mode)
    device = next(eng.model.parameters()).device
    rows = []
    tag = ("FUSED" if use_triton else "BASE") + ("/AMP" if use_amp else "/FP32") + (f"/{compile_mode}" if compile_mode else "")
    print(f"=== {tag} ===")
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device=device)
            for _ in range(warmup): _ = eng.forward(x)
            lat = []
            for _ in range(iters):
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.time(); _ = eng.forward(x)
                if device.type == "cuda": torch.cuda.synchronize()
                lat.append((time.time() - t0) * 1000)
            p50 = float(np.percentile(lat, 50)); p95 = float(np.percentile(lat, 95))
            toks = B * L; tps = toks / (np.mean(lat) / 1000.0)
            row = {"tag": tag, "seq": L, "batch": B,
                   "p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
                   "tokens_per_s": float(tps)}
            rows.append(row)
            print({**row, "tokens_per_s": round(tps,1)})
    return rows

def compare_rows(base, fused):
    # index by (seq,batch)
    b = {(r["seq"], r["batch"]): r for r in base}
    f = {(r["seq"], r["batch"]): r for r in fused}
    print("\n=== COMPARISON (Δp50 = (base-fused)/base, ΔTPS = (fused-base)/base) ===")
    print(f"{'seq':>5} {'batch':>5} | {'p50_base':>8} {'p50_fused':>10} {'Δp50%':>8} | {'tps_base':>10} {'tps_fused':>10} {'ΔTPS%':>8}")
    for key in sorted(b.keys()):
        rb, rf = b[key], f[key]
        dp50 = (rb["p50_ms"] - rf["p50_ms"]) / rb["p50_ms"] * 100.0
        dtps = (rf["tokens_per_s"] - rb["tokens_per_s"]) / rb["tokens_per_s"] * 100.0
        print(f"{key[0]:5d} {key[1]:5d} | {rb['p50_ms']:8.2f} {rf['p50_ms']:10.2f} {dp50:8.1f} | {rb['tokens_per_s']:10.1f} {rf['tokens_per_s']:10.1f} {dtps:8.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=str, default="128,512,1024,2048")
    ap.add_argument("--batch", type=str, default="1,4,16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no_amp", action="store_true", help="disable AMP")
    ap.add_argument("--compile", type=str, default=None, choices=[None, "reduce-overhead", "max-autotune"])
    ap.add_argument("--csv", type=str, default=None, help="write results to CSV")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seq.split(",")]
    batch_sizes = [int(b) for b in args.batch.split(",")]
    use_amp = not args.no_amp

    base = bench(use_triton=False, use_amp=use_amp, compile_mode=args.compile,
                 seq_lens=seq_lens, batch_sizes=batch_sizes, warmup=args.warmup, iters=args.iters)
    fused = bench(use_triton=True, use_amp=use_amp, compile_mode=args.compile,
                  seq_lens=seq_lens, batch_sizes=batch_sizes, warmup=args.warmup, iters=args.iters)
    compare_rows(base, fused)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(base[0].keys()))
            w.writeheader(); w.writerows(base + fused)
        print(f"\nSaved CSV to {args.csv}")