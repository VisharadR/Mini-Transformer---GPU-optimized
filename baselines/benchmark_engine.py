import time, numpy as np, torch

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from engine.perf_toggles import enable_fast_paths
from engine.runner import Engine


@torch.inference_mode()
def bench_engine(seq_lens=(128,512,1024,2048), batch_sizes=(1,4,16), warmup=10, iters=50, use_triton=True):
    eng = Engine(use_triton_attention=use_triton)
    device = next(eng.model.parameters()).device
    rows = []
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device=device)
            for _ in range(warmup):
                _ = eng.forward(x)
            lat = [] # Latencies
            for _ in range(iters):
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                _ = eng.forward(x)
                if device.type == "cuda": torch.cuda.synchronize()
                lat.append((time.time()-t0)*1000)
            p50 = float(np.percentile(lat,50)); p95 = float(np.percentile(lat,95))
            toks = B*L; tps = toks / (np.mean(lat)/1000.0)
            row = {"seq":L,"batch":B,"p50_ms":round(p50,2),"p95_ms":round(p95,2),"tokens_per_s":float(tps)}
            rows.append(row)
            print(("FUSED " if use_triton else "BASE  "), {**row, "tokens_per_s": np.round(tps,1)})
    return rows

def compare():
    base = bench_engine(use_triton=False)
    fused = bench_engine(use_triton=True)

    # index by (seq,batch)
    key = lambda r: (r["seq"], r["batch"])
    bmap = {key(r): r for r in base}
    fmap = {key(r): r for r in fused}

    print("\n=== COMPARISON (speedup = BASE / FUSED for latency; TPS = FUSED / BASE) ===")
    print(f"{'seq':>5} {'batch':>5} | {'p50_base':>8} {'p50_fused':>10} {'p50_gain%':>9} | {'tps_base':>10} {'tps_fused':>10} {'tps_gain%':>9}")
    for k in sorted(bmap.keys()):
        b, f = bmap[k], fmap[k]
        p50_gain = (b["p50_ms"] - f["p50_ms"]) / b["p50_ms"] * 100.0
        tps_gain = (f["tokens_per_s"] - b["tokens_per_s"]) / b["tokens_per_s"] * 100.0
        print(f"{k[0]:5d} {k[1]:5d} | {b['p50_ms']:8.2f} {f['p50_ms']:10.2f} {p50_gain:9.1f} | {b['tokens_per_s']:10.1f} {f['tokens_per_s']:10.1f} {tps_gain:9.1f}")

if __name__ == "__main__":
    print("=== BASELINE ==="); _ = bench_engine(use_triton=False)
    print("=== FUSED ==="); _ = bench_engine(use_triton=True)
    compare()
