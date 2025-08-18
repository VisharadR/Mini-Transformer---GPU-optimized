import time, numpy as np, torch

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from engine.perf_toggles import enable_fast_paths
from baselines.tiny_transformer_postln import TinyTransformerPostLN

enable_fast_paths()


@torch.inference_mode()
def bench(seq_lens=(128, 512, 1024, 2048), batch_sizes=(1, 4, 16), warmup=10, iters=50):
    m = TinyTransformerPostLN().to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device=dev)
            # warmup
            for _ in range(warmup): _ = m(x)
            lat = []  # Latencies
            for _ in range(iters):
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                _ = m(x)
                if dev.type == "cuda": torch.cuda.synchronize()
                lat.append((time.time() - t0) * 1000)
            p50 = float(np.percentile(lat, 50))
            p95 = float(np.percentile(lat, 95))
            toks = B * L
            tps = toks / (np.mean(lat) / 1000.0)
            print(f"seq={L}, batch={B}, p50={p50:.2f}ms, p95={p95:.2f}ms, tokens/s={tps:.1f}")


if __name__ == "__main__":
    print("=== Benchmarking TinyTransformerPostLN ===")
    bench()