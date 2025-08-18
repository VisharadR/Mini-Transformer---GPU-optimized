import torch
from torch.profiler import profile, ProfilerActivity

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


from engine.runner import Engine
from engine.perf_toggles import enable_fast_paths  
enable_fast_paths()  # Enable performance optimizations

for tag, fused in [("BASE", False), ("FUSED", True)]:
    eng = Engine(use_triton_attention=fused)
    dev = next(eng.model.parameters()).device
    x = torch.randint(0, 32000, (4, 512), device=dev)
    #warmup
    for _ in range(10): _ = eng.forward(x)
    if dev.type == "cuda": torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(30): _ = eng.forward(x)
    print(f"== {tag} == ")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

