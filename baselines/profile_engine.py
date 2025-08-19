import torch
from torch.profiler import profile, ProfilerActivity

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from engine.perf_toggles import enable_fast_paths
from engine.runner import Engine

enable_fast_paths()

def profile_one(tag, fused, B=4, L=512, steps=30):
    eng = Engine(use_triton_attention=fused)
    dev = next(eng.model.parameters()).device
    x = torch.randint(0, 32000, (B, L), device=dev)
    # warmup
    for _ in range(10): _ = eng.forward(x)
    if dev.type == "cuda": torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(steps):
            _ = eng.forward(x)
    print(f"\n== {tag} ==")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

if __name__ == "__main__":
    profile_one("BASE (Pre-LN)", fused=False, B=4, L=512)
    profile_one("FUSED (SDPA/attention)", fused=True,  B=4, L=512)

