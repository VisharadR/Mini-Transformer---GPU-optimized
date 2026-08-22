import time
import torch
import numpy as np

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from baselines.tiny_transformer_torch import TinyTransformer
from engine.device import get_device_type
from engine.perf_toggles_rocm import enable_fast_paths

# Import the ROCm kernel ports too, so running this script also acts as a
# smoke check that the ported modules load cleanly on the current box.
try:
    from engine import fused_lnresidual_rocm  # noqa: F401
    from kernels import attention_triton_rocm  # noqa: F401
    _ROCM_KERNELS_IMPORTABLE = True
except Exception as e:
    _ROCM_KERNELS_IMPORTABLE = False
    print(f"[benchmark_rocm] Warning: ROCm kernel modules failed to import: {e}")

enable_fast_paths()


@torch.inference_mode()
def bench(model, seq_lens=(128, 512), batch_sizes=(1, 4, 16), warmup=10, iters=50):
    device_type = get_device_type()
    device = "cuda" if device_type in ("cuda", "rocm") else "cpu"
    model.eval().to(device)
    results = []
    for L in seq_lens:
        for B in batch_sizes:
            x = torch.randint(0, 32000, (B, L), device=device)
            # warmup
            for _ in range(warmup):
                _ = model(x)
            # measure
            latencies = []
            for _ in range(iters):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.time() - t0) * 1000)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            toks = B * L
            tps = toks / (np.mean(latencies) / 1000.0)
            row = {
                "backend": device_type,
                "seq": L,
                "batch": B,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "tokens_per_s": round(tps, 1),
            }
            results.append(row)
            print(row)
    return results


if __name__ == "__main__":
    device_type = get_device_type()
    if device_type != "rocm":
        print(
            f"[benchmark_rocm] Warning: detected backend='{device_type}', not 'rocm'. "
            "Running anyway (useful for sanity-checking this script on a CUDA/CPU box "
            "before AMD hardware is available), but these numbers are NOT ROCm numbers."
        )
    print(f"[benchmark_rocm] backend={device_type} rocm_kernels_importable={_ROCM_KERNELS_IMPORTABLE}")

    model = TinyTransformer(d_model=256, n_heads=4, depth=4)
    bench(model)
