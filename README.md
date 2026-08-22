# Mini Transformer – GPU Optimized


This project is about building a **tiny version of a Transformer model** (the same kind of model used in ChatGPT, BERT, and GPT models) — but in a way that’s **fast, lightweight, and optimized for GPUs**.  

It also shows how to make the model **portable** so it can run not just in PyTorch (where it is trained), but also in **ONNX** (for faster deployment) and **CoreML** (for Apple devices like iPhones and Macs).  

Think of this project as a **mini-lab for AI model optimization**:  
- Train and test the model 🏋️  
- Convert it into formats for different platforms 🔄  
- Make it smaller and faster ⚡  
- Verify that it works correctly ✅  

---

## 🌟 Why This Project Matters  

Modern AI models are **big** and **slow**. They work great on powerful servers but are hard to deploy on smaller devices. This project shows:  

- How to **shrink models** without losing accuracy.  
- How to **move models** between platforms (PyTorch → ONNX → CoreML).  
- How to **make them faster** for real-world applications like mobile apps, websites, or edge devices.  

In short: *learn how AI goes from the lab to your phone.* 📱  

---

## ✨ Features

- **Custom CUDA Kernels**: Implemented FlashAttention-style kernels for efficient attention on long sequences.
- **Fused Ops**: Combined residual + layer normalization into a single GPU kernel for lower overhead.
- **Export Pipelines**:
  - PyTorch → ONNX → CoreML
  - Supports CPU, GPU, and Apple Silicon deployment
- **Benchmark Tools**:
  - Compare PyTorch vs. ONNX vs. CoreML performance
  - p50 / p95 latency profiling
- **Mini Runtime (vLLM-style)**:
  - Dynamic batching of multiple prompts
  - Streaming token callbacks
  - Torch backend (ONNX optional)
  - End-to-end demo for interactive text generation

---

## 🧩 How It Works

- **Model Optimizations**:
  - Attention kernels avoid quadratic cost on long sequences.
  - Fused LayerNorm+Residual reduces memory reads/writes.

- **Export Pipelines**:
  - Convert PyTorch model → ONNX → CoreML.
  - Verify parity at the logits level to ensure correctness.

- **Runtime**:
  - Accepts multiple prompts.
  - Splits into prefill (process full prompt) and decode (one token at a time).
  - Uses a simple scheduler for dynamic batching.
  - Streams outputs back with callbacks.

---

## 📦 Installation

Clone the repository and create a virtual environment:
```bash
git clone https://github.com/VisharadR/Mini-Transformer---GPU-optimized.git
cd mini-transformer-gpu-optimized
```
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt 
```
## ⚡ Quickstart
1. Run Baseline Benchmark

Benchmark the PyTorch model performance:
```bash
python baselines/benchmark.py
```

Example output:
```bash
=== Benchmarking TinyTransformerPostLN ===
seq=128, batch=1, p50=4.00ms, p95=4.01ms, tokens/s=35356.6
```

## 🔄 Export Models

Export to ONNX
```bash
python scripts/export_onnx.py
```

This produces models/model.onnx.

Verify ONNX Parity
```bash
python scripts/verify_onnx.py
```

Output should show very small differences (max|Δprobs| ≲ 1e-4):
```bash
'great movie!': max|Δlogits|=0.00000, max|Δprobs|=0.00000, py=1, onnx=1
```
Benchmark ONNX Runtime
```bash
python scripts/bench_onnx.py
```

Run the Runtime Demo
- Try interactive generation with the custom runtime:
```bash
python scripts/run_runtime_demo.py --prompts "hello gpu" "deep learning" --max_new 12
```

## 🍏 Export to CoreML (Mac only)

We provide two workflows:

A. Windows/Linux users: Skip CoreML export (not supported).

B. macOS users with Apple Silicon
Install CoreML tools:
```bash
pip install coremltools==7.2 numpy==1.26.4  # version compatible with numpy<2.0
python scripts/export_coreml.py
```

## ⚠️ Note: Torch 2.5.x is not fully tested with CoreML. For best results use Torch 2.2.x on macOS.


🔧 Quantization

We support ONNX quantization (dynamic).

Run:
```bash
python scripts/quantize_onnx.py
```

This generates models/model.int8.onnx.
To verify:
```bash
python scripts/verify_onnx.py --model models/model.int8.onnx
```
## 🧪 Continuous Integration (CI)

CI is set up to:

- Export the ONNX model.

- Run parity checks (PyTorch ↔ ONNX).

- Upload ONNX artifacts for download.

- No benchmarking is done in CI (benchmarks should be run locally).

## 📊 Project Structure
```bash
baselines/           # PyTorch model + benchmark
  ├── benchmark_engine.py
  ├── benchmark_postln.py
  ├── benchmark.py
  ├── profile_engine.py
  ├── tiny_transformer_postln.py
  ├── tiny_transformer_torch.py
engine/
  ├── fused_attention.py
  ├── fused_lnresidual.py
  ├── perf_toggles.py
  ├── runner.py
engine_runtime/
  ├── api.py
  ├── engine_iface.py
  ├── engine_torch.py
  ├── kv_cache.py
  ├── sampling.py
  ├── scheduler.py
kernels/
  ├── attention_triton.py
models/              # Exported models (ONNX, CoreML, quantized)
  ├── intent_classifier.int8.onnx
  ├── intent_classifier.onnx
scripts/
  ├── export_onnx.py
  ├── verify_onnx.py
  ├── bench_onnx.py
  ├── quantize_onnx.py
  ├── export_coreml.py 
```
## ✅ Roadmap

- Baseline PyTorch benchmark

- Export & verify ONNX

- ONNX Runtime benchmark

- ONNX quantization

- CoreML export (Mac only)

- Minimal CI parity check job

- Add more transformer variants (GPT-style, encoder-decoder)

## Run the Runtime

## 📌 Notes

- Ensure NumPy < 2.0 for CoreML compatibility.

- CoreML export requires macOS + Apple Silicon.

- ONNX Runtime runs smoothly on CPU/GPU across platforms.

## AMD ROCm Support

An AMD/ROCm backend lives alongside the CUDA path, added additively (every
ROCm-specific module is a new `_rocm`-suffixed file; nothing in the
CUDA/CPU path was changed). Full details, install steps, and verification
notes are in [`docs/AMD_ROCM.md`](docs/AMD_ROCM.md) — summary below.

**Install:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install triton numpy
```

**New modules:**
```bash
engine/device.py                    # get_device_type() -> "cuda" | "rocm" | "cpu"
engine/fused_lnresidual_rocm.py     # HIP kernel port + pure-PyTorch fallback
engine/perf_toggles_rocm.py         # MIOpen benchmark mode, no NVIDIA-only TF32 flags
kernels/attention_triton_rocm.py    # AMD-tuned autotune configs + launcher + correctness test
baselines/benchmark_rocm.py         # same p50/p95/tokens-per-s harness, tagged by backend
```

**Run the ROCm benchmark:**
```bash
python baselines/benchmark_rocm.py
```

**Backend comparison** (placeholders — not yet measured on real AMD hardware; see `docs/AMD_ROCM.md`):

| Backend | seq=128, batch=1 p50 (ms) | tokens/s | seq=512, batch=16 p50 (ms) | tokens/s |
|---|---|---|---|---|
| PyTorch – CUDA (NVIDIA) | TBD | TBD | TBD | TBD |
| PyTorch – ROCm (AMD) | TBD | TBD | TBD | TBD |
| Triton – CUDA (NVIDIA) | TBD | TBD | TBD | TBD |
| Triton – ROCm (AMD) | TBD | TBD | TBD | TBD |

**Known limitations:**
- No AMD GPU was available in this environment — the HIP kernel build and
  the Triton ROCm kernel have been reviewed and syntax/import-tested (see
  `docs/AMD_ROCM.md`), but never compiled or run on real ROCm hardware.
- GitHub Actions has no free AMD GPU runners, so CI (`.github/workflows/rocm-smoke.yml`)
  only does an import-only smoke test on CPU. Real ROCm benchmarking and
  correctness checks must be run manually on AMD hardware.
