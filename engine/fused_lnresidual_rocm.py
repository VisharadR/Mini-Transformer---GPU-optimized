import os
import sys
import warnings
import torch
from torch.utils.cpp_extension import load, ROCM_HOME

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from engine.device import get_device_type

# ROCm/HIP port of engine/fused_lnresidual.py's raw CUDA kernel.
#
# This is hand-adapted, not run through hipify_torch's automatic .cu -> .hip
# conversion. The original kernel leans on a few things that are worth
# checking by hand rather than trusting an auto-converter:
#   - rsqrtf(): confirmed present in HIP's device math library with the same
#     name/signature/semantics as CUDA's fast approximate reciprocal sqrt
#     (ROCm 6.x docs, hip/hip_math.h).
#   - extern __shared__ dynamic shared memory + __syncthreads() tree
#     reduction: supported identically under HIP, no changes needed.
#   - Kernel launch: swapped CUDA's <<<grid, block, smem>>> triple-chevron
#     syntax for the explicit hipLaunchKernelGGL(...) macro. Both are
#     accepted by hipcc, but the macro form makes the HIP-vs-CUDA launch
#     path visually explicit here instead of relying on hipcc silently
#     accepting CUDA syntax.
#   - TORCH_CHECK(x.is_cuda(), ...) guards are left as-is: on ROCm-built
#     PyTorch, HIP tensors report is_cuda() == True (see engine/device.py),
#     so these checks are correct unchanged.
#
# The source below is written directly in HIP syntax (#include
# <hip/hip_runtime.h>, hipLaunchKernelGGL) and saved with a .cu extension so
# torch.utils.cpp_extension.load() routes it through its existing CUDA/HIP
# build path. PyTorch's build machinery still runs .cu sources through
# hipify_python before invoking hipcc on ROCm -- since this source is already
# HIP-native rather than CUDA-native, that pass is a no-op (verified by
# diffing the hipified output PyTorch writes under its build temp dir against
# this source; they were identical). If you change this source, re-check that
# diff rather than assuming hipify leaves HIP-native code untouched.
_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void ln_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ res,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int hidden) {

  int row = blockIdx.x;             // each block handles one (B*L) row
  int tid = threadIdx.x;

  extern __shared__ float sm[];
  float* red = sm;                  // reduction buffer length = blockDim.x

  // reduce mean
  float sum = 0.f;
  for (int i = tid; i < hidden; i += blockDim.x) {
    sum += x[row * hidden + i];
  }
  red[tid] = sum; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float mean = red[0] / hidden;

  // reduce variance
  float vsum = 0.f;
  for (int i = tid; i < hidden; i += blockDim.x) {
    float d = x[row * hidden + i] - mean;
    vsum += d * d;
  }
  red[tid] = vsum; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float var = red[0] / hidden + 1e-5f;
  float inv = rsqrtf(var);

  // normalize, affine, add residual
  for (int i = tid; i < hidden; i += blockDim.x) {
    float xn = (x[row * hidden + i] - mean) * inv;
    float out = xn * gamma[i] + beta[i];
    y[row * hidden + i] = out + res[row * hidden + i];
  }
}

torch::Tensor ln_residual(torch::Tensor x, torch::Tensor res,
                          torch::Tensor gamma, torch::Tensor beta) {
  TORCH_CHECK(x.is_cuda(), "x must be on a HIP/CUDA device");
  TORCH_CHECK(res.is_cuda(), "res must be on a HIP/CUDA device");
  TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be on a HIP/CUDA device");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(res.dtype() == torch::kFloat32, "res must be float32");
  TORCH_CHECK(gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32, "gamma/beta must be float32");
  TORCH_CHECK(x.dim()==3, "x must be [B,L,H]");
  TORCH_CHECK(res.sizes() == x.sizes(), "res must match x");
  TORCH_CHECK(gamma.numel() == x.size(2) && beta.numel() == x.size(2), "gamma/beta length must equal H");

  const int B = x.size(0);
  const int L = x.size(1);
  const int H = x.size(2);

  auto y = torch::empty_like(x);

  dim3 grid(B * L);
  dim3 block(256);
  size_t smem = 256 * sizeof(float);

  hipLaunchKernelGGL(
      ln_residual_kernel, grid, block, smem, 0,
      x.data_ptr<float>(),
      res.data_ptr<float>(),
      gamma.data_ptr<float>(),
      beta.data_ptr<float>(),
      y.data_ptr<float>(),
      H);

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_residual", &ln_residual, "Fused LayerNorm + Residual (HIP/ROCm)");
}
"""

_BUILD_DIR = os.path.join(_PROJ_ROOT, "cpp_build")
_HIP_SRC_PATH = os.path.join(_BUILD_DIR, "ln_residual_hip.cu")


def _write_source():
    os.makedirs(_BUILD_DIR, exist_ok=True)
    if not os.path.exists(_HIP_SRC_PATH):
        with open(_HIP_SRC_PATH, "w", newline="\n") as f:
            f.write(_SRC)


def _build_hip_extension():
    """Compiles the HIP extension via hipcc. Raises a clear, specific error
    (caught by the caller, which then falls back to the pure-PyTorch path)
    instead of letting a missing-toolchain failure surface as a cryptic
    ninja/subprocess traceback."""
    if ROCM_HOME is None:
        raise RuntimeError(
            "ROCm dev tools not found (torch.utils.cpp_extension.ROCM_HOME is "
            "None). Install the ROCm HIP SDK (hipcc + headers, ROCm 6.x) to "
            "build the fused HIP kernel, or use the pure-PyTorch fallback."
        )
    _write_source()
    return load(
        name="ln_residual_hip_ext",
        sources=[_HIP_SRC_PATH],
        extra_cflags=[],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def _ln_residual_pytorch_fallback(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Portable backup path for environments without HIP dev tools. Same
    numerics as the fused kernel (mean/var over the last dim, eps=1e-5) --
    just two ops instead of one fused kernel launch, so it's a correct
    drop-in, not a fast one."""
    hidden = x.shape[-1]
    normed = torch.nn.functional.layer_norm(x, (hidden,), weight=gamma, bias=beta, eps=1e-5)
    return normed + residual


_ext = None
_ext_error = None

if get_device_type() == "rocm":
    try:
        _ext = _build_hip_extension()
    except Exception as e:
        _ext_error = e
        warnings.warn(
            "[fused_lnresidual_rocm] Falling back to the pure-PyTorch "
            f"LN+residual implementation -- HIP extension build failed: {e}",
            RuntimeWarning,
        )


def ln_residual(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Drop-in replacement for engine.fused_lnresidual.ln_residual, usable on
    ROCm. Uses the compiled HIP kernel when available, otherwise the
    pure-PyTorch fallback -- same signature and numerics either way."""
    if _ext is not None:
        return _ext.ln_residual(x, residual, gamma, beta)
    return _ln_residual_pytorch_fallback(x, residual, gamma, beta)


if __name__ == "__main__":
    # Sanity check: fused output vs the unfused PyTorch reference computation.
    # To validate numerically against the *original CUDA* kernel
    # (engine/fused_lnresidual.py), run this same script on an NVIDIA box with
    # CUDA_VISIBLE flipped off / that module imported instead, save both
    # `y` tensors, and diff them -- see docs/AMD_ROCM.md for the exact steps.
    torch.manual_seed(0)
    B, L, H = 2, 8, 256
    x = torch.randn(B, L, H)
    res = torch.randn(B, L, H)
    gamma = torch.randn(H)
    beta = torch.randn(H)

    device_type = get_device_type()
    if device_type in ("rocm", "cuda"):
        x, res, gamma, beta = (t.to("cuda") for t in (x, res, gamma, beta))
        y = ln_residual(x, res, gamma, beta)
        y_ref = _ln_residual_pytorch_fallback(x, res, gamma, beta)
        max_err = (y - y_ref).abs().max().item()
        print(
            f"[fused_lnresidual_rocm] backend={device_type} "
            f"using_hip_ext={_ext is not None} max|delta vs reference|={max_err:.6f}"
        )
    else:
        print(f"[fused_lnresidual_rocm] No GPU backend detected (got '{device_type}'); skipping.")
