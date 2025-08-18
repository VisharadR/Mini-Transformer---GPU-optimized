import time, numpy as np, torch

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from engine.runner import Engine

@torch.inference_mode()
def bench_engine(seq_lens=(128,512), batch_sizes=(1,4,16), warmup=10, iters=50, use_triton=True):
    eng = Engine(use_triton_attention=use_triton)
    device = next(eng.model.parameters()).device
    results = []
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device=device)
            for _ in range(warmup):
                _ = eng.forward(x)
            lat = [] # latencies
            for _ in range(iters):
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()
                _  = eng.forward(x)
                if device.type == 'cuda': torch.cuda.synchronize()
                lat.append((time.time()-t0)*1000)
            p50 = float(np.percentile(lat,50))
            p95 = float(np.percentile(lat, 95))
            toks = B*L 
            tps = toks / (np.mean(lat)/1000.0)
            row = {"seq":L, "batch":B, "p50_ms":round(p50,2), "p95_ms":round(p95,2), "tokens_per_s":round(tps,1)}
            results.append(row)
            print(("FUSED" if use_triton else " BASE "), row)
    return results


if __name__ == "__main__":
    # A/B in one go:
    print("=== BASELINE ===")
    bench_engine(use_triton=False)
    print("=== FUSED ===")
    bench_engine(use_triton=True)

