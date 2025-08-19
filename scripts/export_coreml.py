import os, torch

def as_long_tensor(x):
    return x.to(dtype=torch.long, device="cpu") if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

def export_onnx(model, sample, out_path="models/intent_classifier.onnx"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    input_ids      = as_long_tensor(sample["input_ids"])
    attention_mask = as_long_tensor(sample["attention_mask"])
    torch.onnx.export(
        model, (input_ids, attention_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attention_mask": {1: "seq"},
            # DistilBERT sequence-classification logits are [B, C] (no seq dim)
        },
        opset_version=14
    )
    print(f"[OK] Saved ONNX to {out_path}")

def try_export_coreml(ts, sample, out_path="ios_app/IntentClassifier.mlmodel"):
    try:
        import coremltools as ct
    except Exception as e:
        print("[WARN] coremltools not available; skipping Core ML. Error:", e)
        return False
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        input_ids      = as_long_tensor(sample["input_ids"])
        attention_mask = as_long_tensor(sample["attention_mask"])
        mlmodel = ct.convert(
            ts,
            inputs=[
                ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=int),
                ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=int),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16
        )
        mlmodel.save(out_path)
        print(f"[OK] Saved Core ML to {out_path}")
        return True
    except Exception as e:
        print("[WARN] Core ML conversion failed; will export ONNX instead. Error:", e)
        return False

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).eval()

    sample = tok("great movie!", return_tensors="pt")
    input_ids      = as_long_tensor(sample["input_ids"])
    attention_mask = as_long_tensor(sample["attention_mask"])

    class Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, C]

    wrap = Wrapper(m).eval()
    with torch.no_grad():
        _ = wrap(input_ids, attention_mask)

    ts = torch.jit.trace(wrap, (input_ids, attention_mask))

    ok = try_export_coreml(ts, {"input_ids": input_ids, "attention_mask": attention_mask})
    if not ok:
        export_onnx(wrap, {"input_ids": input_ids, "attention_mask": attention_mask})