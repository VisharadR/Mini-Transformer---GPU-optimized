import os
import torch
from torch.utils.cpp_extension import load

# CUDA implementation for fused layer normalization and residual connection
_SRC = r"""
#include <torch/extension.h>

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
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(res.is_cuda(), "res must be CUDA");
  TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
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

  // launch
  ln_residual_kernel<<<grid, block, smem>>>(
      x.data_ptr<float>(),
      res.data_ptr<float>(),
      gamma.data_ptr<float>(),
      beta.data_ptr<float>(),
      y.data_ptr<float>(),
      H);

  // no explicit cudaDeviceSynchronize(); PyTorch syncs when needed
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_residual", &ln_residual, "Fused LayerNorm + Residual (CUDA)");
}
""";

# Load the CUDA extension
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp_build")
os.makedirs(_BUILD_DIR, exist_ok=True)
_CU = os.path.join(_BUILD_DIR, "ln_residual.cu")


# write the CUDA source code to a file
if not os.path.exists(_CU):
    with open(_CU, "w", newline="\n") as f:
        f.write(_SRC)


# Helpful on first buil with long ppaths on windows
extra_inlcude_paths = []
extra_cflags = []
extra_cuda_cflags = ["-O3"]
extra_ldflags = []

# Load the extension
_ext = load(
    name="ln_residual_ext",
    sources = [_CU],
    extra_include_paths = extra_inlcude_paths,
    extra_cflags = extra_cflags,
    extra_cuda_cflags = extra_cuda_cflags,
    extra_ldflags = extra_ldflags,
    verbose=True
)

def ln_residual(x: torch.tensor, residual:torch.Tensor, gamma:torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return _ext.ln_residual(x, residual, gamma, beta)