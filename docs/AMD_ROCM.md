# AMD ROCm Support

This document covers the AMD/ROCm backend added alongside the existing
NVIDIA/CUDA path. It is purely additive: every `_rocm`-suffixed module lives
next to its CUDA counterpart, and none of the original CUDA/CPU files were
modified. See the "What's verified vs. untested" section at the bottom before
relying on any of this in production -- most of it has been validated on CPU
only, not on real AMD hardware.

## File map

| CUDA / backend-agnostic | ROCm counterpart | Notes |
|---|---|---|
| `engine/fused_lnresidual.py` | `engine/fused_lnresidual_rocm.py` | Hand-ported HIP kernel + pure-PyTorch fallback |
| `engine/perf_toggles.py` | `engine/perf_toggles_rocm.py` | Drops TF32 flags, adds MIOpen find-mode |
| `kernels/attention_triton.py` | `kernels/attention_triton_rocm.py` | Adds AMD-tuned autotune configs + a launcher + a correctness test (none of that existed for the original kernel either -- see below) |
| `baselines/benchmark.py` | `baselines/benchmark_rocm.py` | Same p50/p95/tokens-per-s harness, tags output with detected backend |
| n/a (new) | `engine/device.py` | Single source of truth for `"cuda"` / `"rocm"` / `"cpu"` detection |

## Install

```bash
# 1. Install a ROCm build of PyTorch (check https://pytorch.org/get-started/locally/
#    for the current wheel index matching your ROCm version -- this project was
#    written against ROCm 6.x wheels).
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 2. Triton ships ROCm support in the same `triton` package as CUDA (no
#    separate ROCm wheel) as of Triton 2.1+. It requires ROCm's HIP runtime
#    to be installed on the host regardless of how PyTorch was installed.
pip install triton

# 3. numpy for the benchmark harness
pip install numpy
```

Supported GPUs (per AMD's ROCm compatibility matrix -- verify against the
ROCm release notes for whatever ROCm version you install, this list changes
between releases):
- CDNA: MI100, MI200-series (MI210/MI250/MI250X), MI300-series
- RDNA: a subset of RDNA3 desktop/workstation cards under ROCm's "HIP SDK"
  consumer support (narrower than the CDNA datacenter list)

## Backend detection

```python
from engine.device import get_device_type
get_device_type()  # "cuda", "rocm", or "cpu"
```

The important trap this exists to avoid: on a ROCm build of PyTorch,
`torch.cuda.is_available()` is still `True` and tensors still report
`device.type == "cuda"` -- HIP is exposed through the same `"cuda"` namespace
for API compatibility. `torch.version.hip is not None` is the actual
discriminator. Anything in this repo that needs to tell CUDA and ROCm apart
should go through `get_device_type()` rather than re-deriving this check.

## `engine/fused_lnresidual_rocm.py`

Two paths, chosen automatically:

1. **Compiled HIP kernel** (used when `get_device_type() == "rocm"` and
   `torch.utils.cpp_extension.ROCM_HOME` is set, i.e. hipcc + ROCm headers
   are installed). The HIP source is hand-adapted from the original CUDA
   kernel, not run through hipify's automatic `.cu -> .hip` conversion --
   see the comment block at the top of the file for exactly what was
   manually checked (`rsqrtf`, shared-memory reduction, launch syntax).
2. **Pure-PyTorch fallback** (`torch.nn.functional.layer_norm(...) +
   residual`), used when HIP dev tools aren't available. Same function
   signature, same numerics, just not fused into one kernel launch. This is
   also what runs today in this repo's environment (macOS, no ROCm), and
   was verified to match a manual LayerNorm+residual computation exactly
   (max abs diff `0.0` on a `[2, 8, 256]` test tensor).

If the HIP build fails, you get a `RuntimeWarning` naming the reason (missing
`ROCM_HOME`, compile error, etc.) and a fallback to the PyTorch path -- not a
silent wrong answer and not an opaque crash.

### Validating the compiled HIP kernel against the original CUDA kernel

This has **not** been done -- there's no ROCm GPU in this environment. To do
it on real hardware:

```bash
# On an NVIDIA box:
python -c "
import torch
from engine.fused_lnresidual import ln_residual
torch.manual_seed(0)
x = torch.randn(2, 8, 256, device='cuda')
res = torch.randn(2, 8, 256, device='cuda')
gamma = torch.randn(256, device='cuda')
beta = torch.randn(256, device='cuda')
y = ln_residual(x, res, gamma, beta)
torch.save({'x': x.cpu(), 'res': res.cpu(), 'gamma': gamma.cpu(), 'beta': beta.cpu(), 'y': y.cpu()}, 'cuda_ref.pt')
"

# On an AMD box, load the same inputs and compare:
python -c "
import torch
from engine.fused_lnresidual_rocm import ln_residual
ref = torch.load('cuda_ref.pt')
y = ln_residual(*(ref[k].cuda() for k in ('x', 'res', 'gamma', 'beta')))
print('max|delta| vs CUDA reference:', (y.cpu() - ref['y']).abs().max().item())
"
```

Expect differences at roughly fp32 accumulation-order noise (~1e-5 to 1e-4),
not larger.

## `kernels/attention_triton_rocm.py`

The original `kernels/attention_triton.py` was audited for NVIDIA-only
assumptions (e.g. a hardcoded warp size of 32). It has none -- the kernel is
expressed in BLOCK_M/BLOCK_N tiles and Triton's compiler maps that onto
CDNA's 64-wide wavefronts vs NVIDIA's 32-wide warps on its own via
`num_warps`. So the kernel math needed **no changes** to be portable.

What the original file doesn't have at all -- on either backend -- is
autotuning, a launcher, or a test: the block sizes are unset parameters, and
`engine/fused_attention.py` imports `flash_attn` but never actually calls it
(`TritonSelfAttention._attn_triton()` just calls PyTorch's SDPA, "temporary
until Triton is finalized"). So `kernels/attention_triton_rocm.py` fills that
gap rather than un-doing an NVIDIA-specific choice: it adds a
`triton.autotune` sweep over AMD-appropriate block sizes, a Python launcher
(looping over batch/head, since the kernel itself only handles one `[L, Dh]`
slice per launch), and a correctness test against
`F.scaled_dot_product_attention`.

Given the kernel body is unchanged, this file is **not** a fix for
ROCm-breaking code -- it's the autotune/launcher/test scaffolding the
original never had, written against AMD block-size defaults instead of
NVIDIA ones so it's not silently mistuned if someone does wire it in on an
AMD box.

Run the correctness test:

```bash
python kernels/attention_triton_rocm.py
```

## `engine/perf_toggles_rocm.py`

- Keeps `torch.backends.cudnn.benchmark = True` -- MIOpen (ROCm's cuDNN
  equivalent) intercepts the same toggle.
- Does **not** set `torch.backends.cuda.matmul.allow_tf32` /
  `cudnn.allow_tf32`. TF32 is an NVIDIA Ampere+ tensor-core format with no
  AMD equivalent; MIOpen doesn't implement it.
- Sets `MIOPEN_FIND_MODE=2` (hybrid search) and `MIOPEN_FIND_ENFORCE=3` as
  defaults (only if not already set in the environment) -- MIOpen's rough
  analogue of cuDNN's benchmark-mode algorithm search. Other
  `MIOPEN_FIND_MODE` values (per MIOpen's docs): `1` = fast heuristic only
  (lower one-time search cost, potentially slower steady-state kernels), `3`
  = exhaustive search (slower first call, best steady-state perf) -- worth
  trying `3` if warmup time isn't a concern for your workload.

## Running the benchmark

```bash
python baselines/benchmark_rocm.py
```

Same harness as `baselines/benchmark.py` (p50/p95 latency, tokens/s across a
grid of sequence lengths and batch sizes), tagged with the detected backend
and importing the `_rocm` kernel modules as an import smoke check. Safe to
run on a CUDA or CPU box too (it'll just label the output accordingly) for
sanity-checking the script itself before AMD hardware is available.

## Benchmark comparison table

**Not yet measured on real hardware.** Placeholders below -- fill in after
running `benchmark_rocm.py` / `benchmark.py` on the respective GPUs. Do not
treat these numbers as real until replaced.

| Backend | seq=128, batch=1 p50 (ms) | seq=128, batch=1 tok/s | seq=512, batch=16 p50 (ms) | seq=512, batch=16 tok/s |
|---|---|---|---|---|
| PyTorch – CUDA (NVIDIA) | TBD | TBD | TBD | TBD |
| PyTorch – ROCm (AMD) | TBD | TBD | TBD | TBD |
| Triton – CUDA (NVIDIA) | TBD | TBD | TBD | TBD |
| Triton – ROCm (AMD) | TBD | TBD | TBD | TBD |

## CI

`.github/workflows/rocm-smoke.yml` runs an **import-only** smoke test on a
CPU-only GitHub-hosted runner (there is no free AMD GPU runner on GitHub
Actions, and this project isn't faking one). It checks that
`engine.device`, `engine.fused_lnresidual_rocm`, `engine.perf_toggles_rocm`,
and (if `triton` installs on that runner) `kernels.attention_triton_rocm`
import cleanly when `get_device_type()` resolves to `"cpu"`. It does not
build the HIP extension, launch a GPU kernel, or run
`baselines/benchmark_rocm.py`. Real ROCm testing -- HIP kernel build, Triton
kernel correctness/perf, end-to-end benchmark numbers -- has to be run
manually on AMD hardware.

## What's verified vs. untested

Verified in this environment (macOS, CPU-only, no ROCm/CUDA GPU, no Triton
available):
- `engine/device.py` correctly reports `"cpu"` on this box.
- `engine/perf_toggles_rocm.py` runs without error, sets the documented env
  vars, and skips the TF32 flags.
- `engine/fused_lnresidual_rocm.py`'s pure-PyTorch fallback path produces
  output that exactly matches (`max abs diff = 0.0`) a manual
  `layer_norm(...) + residual` computation, and gracefully falls back
  without building anything since `get_device_type() != "rocm"` here.
- `baselines/benchmark_rocm.py` imports cleanly and correctly reports that
  the ROCm kernel modules aren't importable (no `triton` on this box),
  without crashing.
- All new files pass `python -m py_compile`.

**Not** verified -- no AMD GPU was available to test with:
- The HIP kernel in `fused_lnresidual_rocm.py` has never been compiled or
  run. `rsqrtf`, the shared-memory reduction, and the
  `hipLaunchKernelGGL(...)` launch syntax are believed correct based on
  ROCm's documented HIP/CUDA API parity, but this needs an actual `hipcc`
  build + kernel launch to confirm, and a numeric diff against the CUDA
  kernel's output (see the validation steps above).
- `kernels/attention_triton_rocm.py` has never been run through Triton's
  ROCm backend. The autotune configs are reasonable starting points for a
  64-wide wavefront, not hardware-measured winners.
- No performance numbers of any kind exist for ROCm -- the comparison table
  above is 100% placeholders.
- MIOpen find-mode env vars are set based on MIOpen's documented behavior,
  not measured on this workload.
