import torch

def enable_fast_paths():
     # Enable cuDNN auto-tuner for optimal kernels
    torch.backends.cudnn.benchmark = True  

    # Allow TensorFloat-32 (faster matmul on Ampere+ GPUs, minimal accuracy loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Use high precision for float32 matmul (improves throughput)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("[PerfToggles] cuDNN benchmark ON, TF32 allowed, matmul_precision=high")
