import argparse, os
from onnxruntime.quantization import quantize_dynamic, QuantType

def do_quantize(src: str, dst: str, weight_type: str = "QInt8"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    qt = QuantType.QInt8 if weight_type.upper() == "QINT8" else QuantType.QUInt8

    # Some ORT versions support optimize_model, others don't.
    kwargs = dict(model_input=src, model_output=dst, weight_type=qt)
    try:
        # Try newer signature first
        quantize_dynamic(**kwargs, optimize_model=True)  # may raise TypeError on older ORT
    except TypeError:
        quantize_dynamic(**kwargs)  # compatible path

    print(f"[OK] Wrote {dst} (weights: {weight_type})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="models/intent_classifier.onnx")
    ap.add_argument("--dst", type=str, default="models/intent_classifier.int8.onnx")
    ap.add_argument("--weight_type", type=str, default="QInt8", choices=["QInt8", "QUInt8"])
    args = ap.parse_args()
    do_quantize(args.src, args.dst, args.weight_type)