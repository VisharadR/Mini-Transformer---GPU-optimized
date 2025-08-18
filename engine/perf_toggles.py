import torch

def enable_fast_paths():
    # allow TensorFloat32 on tensor cores for matrix multiplications (safe for inference)
    torch.backends.cuda.matmul.allow_tf32 = True
    # pick best algorithms for your input shapes 
    torch.backends.cudnn.benchmark = True
    # allow tensor cores to kick in for float32 matmuls
    try:
        torch.set_float32_matmul_precision("high") # "medium" if you want to be conservative
    except Exception:
        pass