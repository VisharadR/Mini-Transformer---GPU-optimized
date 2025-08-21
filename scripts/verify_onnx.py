import argparse, numpy as np, torch, onnxruntime as ort

import os, sys
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="models/intent_classifier.onnx")
    ap.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    ap.add_argument("--texts", type=str, nargs="*", default=[
        "this is awesome!",
        "this is terrible...",
        "i'm not sure how i feel about this",
        "great movie!",
        "utterly disappointing plot but decent acting",
    ])
    ap.add_argument("--provider", type=str, default="CPU", choices=["CPU", "CUDA"])
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    assert onnx_path.exists(), f"ONNX not found: {onnx_path}"

    providers = ["CPUExecutionProvider"] if args.provider=="CPU" else ["CUDAExecutionProvider","CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    tok = AutoTokenizer.from_pretrained(args.model)
    pt_model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()


    print(f"[info] Using ONNX providers: {sess.get_providers()}")


    diffs = []
    for t in args.texts:
        enc_pt = tok(t, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_len)
        with torch.no_grad():
            pt_logits = pt_model(**enc_pt).logits.cpu().numpy()     # [B, C]
        onnx_logits = sess.run(
            ["logits"],
            {"input_ids": enc_pt["input_ids"].numpy(),
            "attention_mask": enc_pt["attention_mask"].numpy()}
        )[0]

        # compare raw logits probabilities
        logit_diff = np.max(np.abs(pt_logits - onnx_logits))
        prob_diff  = np.max(np.abs(softmax(pt_logits) - softmax(onnx_logits)))
        diffs.append((logit_diff, prob_diff))
        print(f"'{t[:48]}{'...' if len(t)>48 else ''}': max|Δlogits|={logit_diff:.5f}, max|Δprobs|={prob_diff:.5f}, "
            f"py={pt_logits.argmax(-1)[0]}, onnx={onnx_logits.argmax(-1)[0]}")

    
    lg, pb = np.mean([d[0] for d in diffs]), np.mean ([d[1] for d in diffs])
    print(f"\n[summary] mean max|Δlogits|={lg:.5f}, mean max|Δprobs|={pb:.5f}")
    print("[ok] Parity looks good if max|Δprobs| ≲ 1e-4")


if __name__ == "__main__":
    main()