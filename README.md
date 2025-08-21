# Mini Transformer – GPU Optimized

```bash
🚀 A lightweight, GPU-optimized implementation of transformer-based models with support for PyTorch, ONNX Runtime, CoreML, and quantization.

The goal of this project is to demonstrate:
How to build and benchmark compact transformer models.
How to export models across frameworks (ONNX, CoreML).
How to validate parity between frameworks.
How to optimize runtime performance on CPUs, GPUs, and Apple Silicon.

## 📦 Installation

Clone the repository and create a virtual environment:
git clone https://github.com/VisharadR/Mini-Transformer---GPU-optimized.git
cd mini-transformer-gpu-optimized

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

## ⚡ Quickstart
1. Run Baseline Benchmark

Benchmark the PyTorch model performance:

python baselines/benchmark.py


Example output:

=== Benchmarking TinyTransformerPostLN ===
seq=128, batch=1, p50=4.00ms, p95=4.01ms, tokens/s=35356.6
...

## 🔄 Export Models

Export to ONNX
python scripts/export_onnx.py


This produces models/model.onnx.

Verify ONNX Parity
python scripts/verify_onnx.py


Output should show very small differences (max|Δprobs| ≲ 1e-4):

'great movie!': max|Δlogits|=0.00000, max|Δprobs|=0.00000, py=1, onnx=1

Benchmark ONNX Runtime
python scripts/bench_onnx.py

## 🍏 Export to CoreML (Mac only)


We provide two workflows:

A. Windows/Linux users
Skip CoreML export (not supported).

B. macOS users with Apple Silicon
Install CoreML tools:

pip install coremltools==7.2  # version compatible with numpy<2.0


Run:

python scripts/export_coreml.py


## ⚠️ Note: Torch 2.5.x is not fully tested with CoreML. For best results use Torch 2.2.x on macOS.


🔧 Quantization

We support ONNX quantization (dynamic).

Run:

python scripts/quantize_onnx.py


This generates models/model.int8.onnx.
To verify:

python scripts/verify_onnx.py --model models/model.int8.onnx

## 🧪 Continuous Integration (CI)

CI is set up to:

Export the ONNX model.

Run parity checks (PyTorch ↔ ONNX).

Upload ONNX artifacts for download.

No benchmarking is done in CI (benchmarks should be run locally).

## 📊 Project Structure

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

## ✅ Roadmap


 Baseline PyTorch benchmark

 Export & verify ONNX

 ONNX Runtime benchmark

 ONNX quantization

 CoreML export (Mac only)

 Minimal CI parity check job

 Add more transformer variants (GPT-style, encoder-decoder)

## 📌 Notes


Ensure NumPy < 2.0 for CoreML compatibility.

CoreML export requires macOS + Apple Silicon.

ONNX Runtime runs smoothly on CPU/GPU across platforms.
