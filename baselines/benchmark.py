import time 
import torch 
import numpy as np 
from tiny_transformer_torch import TinyTransformer

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from engine.perf_toggles import enable_fast_paths
enable_fast_paths()


@torch.inference_mode()
def bench(model, seq_lens=(128, 512), batch_sizes=(1, 4, 16), warmup=10, iters=50):
    model.eval().cuda()
    results = []
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device='cuda')
            # warmup
            for _ in range(warmup):
                _ = model(x)
            # measure
            latencies = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model(x)
                torch.cuda.synchronize()
                latencies.append((time.time()-t0)*1000)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            toks = B*L
            tps = toks / (np.mean(latencies)/1000.0)
            results.append({"seq": L, "batch": B, "p50_ms":round(p50,2), "p95_ms":round(p95,2), "tokens_per_s":round(tps,1)})
            print(results[-1])
    return results



if __name__ == "__main__":
    model = TinyTransformer(d_model=256, n_heads=4, depth=4)
    model.cuda()
    bench(model)



 