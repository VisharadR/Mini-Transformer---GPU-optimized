import argparse, time, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

def run_once(sess, tok, texts, max_len=128):
    enc = tok(texts, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    t0 = time.time()
    out = sess.run(["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0]
    dt = time.time() - t0
    return out, dt, enc["input_ids"].shape

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="models/intent_classifier.onnx")
    ap.add_argument("--provider", type=str, default="CPU", choices=["CPU","CUDA"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    providers = ["CPUExecutionProvider"] if args.provider=="CPU" else ["CUDAExecutionProvider","CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    texts = [f"sample sentence {i}" for i in range(args.batch)]

    # warmup
    for _ in range(10): run_once(sess, tok, texts, args.seq)

    times=[]
    for _ in range(args.iters):
        _, dt, shape = run_once(sess, tok, texts, args.seq)
        times.append(dt)

    b, seqlen = shape[0], shape[1]
    p50 = np.percentile(times, 50)*1000
    p95 = np.percentile(times, 95)*1000
    tps = (b*seqlen) / (np.mean(times))
    print(f"[{args.provider}] batch={b} seq={seqlen}  p50={p50:.2f}ms p95={p95:.2f}ms  tokens/s={tps:,.1f}")
