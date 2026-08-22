import os
import torch


def enable_fast_paths():
    # MIOpen (ROCm's cuDNN-equivalent conv/gemm autotuner) hooks into the same
    # torch.backends.cudnn.* toggles PyTorch uses for cuDNN, so this still works
    # as-is on ROCm builds.
    torch.backends.cudnn.benchmark = True

    # Deliberately NOT setting torch.backends.cuda.matmul.allow_tf32 /
    # cudnn.allow_tf32 here (unlike engine/perf_toggles.py). TF32 is an
    # NVIDIA Ampere+ tensor-core numeric format -- it doesn't exist on AMD
    # GPUs and MIOpen has no equivalent flag, so setting it would be a no-op
    # at best and misleading at worst.

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # MIOpen find-mode: controls how aggressively MIOpen searches for the
    # fastest conv/gemm algorithm the first time a given shape is seen. This
    # is ROCm's rough analogue of cuDNN benchmark mode's search behavior.
    # "2" = hybrid (fast heuristic + limited search) -- a reasonable default
    # for inference workloads with a small number of distinct shapes.
    # See docs/AMD_ROCM.md for the other MIOPEN_FIND_MODE values.
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ.setdefault("MIOPEN_FIND_ENFORCE", "3")

    print(
        "[PerfToggles-ROCm] cuDNN/MIOpen benchmark ON, TF32 skipped (N/A on AMD), "
        f"matmul_precision=high, MIOPEN_FIND_MODE={os.environ['MIOPEN_FIND_MODE']}"
    )
