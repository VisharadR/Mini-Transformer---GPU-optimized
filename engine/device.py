import torch

# Single source of truth for "which GPU backend am I actually running on".
#
# The trap: on a ROCm-built PyTorch, torch.cuda.is_available() is still True
# and tensors still report device.type == "cuda" -- HIP is exposed through the
# same "cuda" namespace for API-compatibility reasons. So `is_cuda` alone does
# NOT mean "this is an NVIDIA GPU". torch.version.hip is set (non-None) only on
# ROCm builds, so that's the actual discriminator.


def get_device_type() -> str:
    """Returns "rocm", "cuda", or "cpu"."""
    if torch.version.hip is not None:
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_rocm() -> bool:
    return get_device_type() == "rocm"


def is_cuda() -> bool:
    return get_device_type() == "cuda"


if __name__ == "__main__":
    print(f"detected device type: {get_device_type()}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
