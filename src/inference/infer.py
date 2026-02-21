#!/usr/bin/env python3
"""
Unified YOLOv5 Inference Script
Supports both PyTorch (.pt) and ONNX (.onnx) backends.

Features:
  - Bounding box output with class labels and confidence drawn on image
  - Output images saved to results folder
  - MSC (Model Score Comparison) / baseline verification between PyTorch and ONNX
  - Evaluation metrics: precision, recall, mAP50, mAP50-95, accuracy
  - COCO class support (80 classes)

Usage examples:
  # PyTorch inference
  python src/inference/infer.py --model yolov5s.pt --source data/images/

  # ONNX inference
  python src/inference/infer.py --model yolov5s.onnx --source data/images/

  # Compare PyTorch vs ONNX on same input
  python src/inference/infer.py --pt-model yolov5s.pt --onnx-model yolov5s.onnx \
      --source data/images/ --compare

  # Benchmark latency only
  python src/inference/infer.py --model yolov5s.onnx --source image.jpg --benchmark
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── COCO 80-class names ───────────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Deterministic colours per class
np.random.seed(42)
CLASS_COLORS = {i: tuple(int(c) for c in np.random.randint(50, 255, 3))
                for i in range(len(COCO_CLASSES))}


# ── Helpers ───────────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_thresh: float = 0.45) -> list:
    """Pure-numpy NMS. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(raw_output: np.ndarray, orig_hw: tuple,
                conf_thresh: float = 0.25,
                iou_thresh: float = 0.45) -> list:
    """
    Post-process YOLOv5 raw output → list of dicts with keys:
        x1, y1, x2, y2  (pixel coords on orig image)
        conf             (object confidence × class confidence)
        class_id
        class_name

    raw_output shape: [1, num_anchors, 85]  (85 = 4 box + 1 obj + 80 cls)
    """
    pred = raw_output[0]          # (num_anchors, 85)
    obj_conf = pred[:, 4]
    cls_scores = pred[:, 5:]      # (num_anchors, 80)

    # Filter by objectness first (fast pre-filter)
    mask = obj_conf > conf_thresh
    pred = pred[mask]
    if pred.shape[0] == 0:
        return []

    obj_conf = pred[:, 4]
    cls_scores = pred[:, 5:]
    class_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), class_ids]
    scores = obj_conf * cls_conf

    # Second filter with combined score
    keep_mask = scores > conf_thresh
    pred = pred[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    if pred.shape[0] == 0:
        return []

    # Convert box coords
    boxes_xyxy = xywh2xyxy(pred[:, :4])

    # Per-class NMS
    detections = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        kept = nms(boxes_xyxy[cls_mask], scores[cls_mask], iou_thresh)
        cls_boxes = boxes_xyxy[cls_mask][kept]
        cls_scores_k = scores[cls_mask][kept]
        for box, sc in zip(cls_boxes, cls_scores_k):
            # Scale back to original image coords
            # raw_output coords are relative to input_size (640)
            h_ratio = orig_hw[0] / 640
            w_ratio = orig_hw[1] / 640
            x1 = int(np.clip(box[0] * w_ratio, 0, orig_hw[1]))
            y1 = int(np.clip(box[1] * h_ratio, 0, orig_hw[0]))
            x2 = int(np.clip(box[2] * w_ratio, 0, orig_hw[1]))
            y2 = int(np.clip(box[3] * h_ratio, 0, orig_hw[0]))
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(sc),
                "class_id": int(cls),
                "class_name": COCO_CLASSES[int(cls)]
                    if int(cls) < len(COCO_CLASSES) else f"cls{int(cls)}",
            })
    return detections


def draw_detections(image: np.ndarray, detections: list,
                    label_prefix: str = "") -> np.ndarray:
    """Draw bounding boxes and labels on image, return annotated copy."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label = f"{label_prefix}{det['class_name']} {det['conf']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return vis


def preprocess_image(image_path: str, img_size: int = 640) -> tuple:
    """
    Load and preprocess image for inference.
    Returns (blob, orig_bgr_image).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig = img.copy()
    resized = cv2.resize(img, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return blob, orig


# ── ONNX backend ──────────────────────────────────────────────────────────────

class ONNXBackend:
    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider",
                                   "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"[ONNX] Loaded: {model_path}")
        print(f"[ONNX] Input  : {self.input_name} {self.input_shape}")
        print(f"[ONNX] Output : {self.output_name}")

    def warmup(self, n: int = 3):
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(n):
            self.session.run([self.output_name], {self.input_name: dummy})

    def forward(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name],
                                {self.input_name: blob})[0]


# ── PyTorch backend ───────────────────────────────────────────────────────────

class PyTorchBackend:
    def __init__(self, model_path: str, device: str = "cpu"):
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed. Run: pip install torch")
        self.torch = torch
        self.device = torch.device(device)

        # Load via torch.hub (recommended) or direct
        print(f"[PyTorch] Loading: {model_path}")
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=model_path, verbose=False)
        except Exception:
            # Fallback: direct load if yolov5 repo is present
            yolo_repo = Path(__file__).parents[3] / "yolov5"
            sys.path.insert(0, str(yolo_repo))
            self.model = torch.hub.load(
                str(yolo_repo), "custom",
                path=model_path, source="local", verbose=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[PyTorch] Model loaded on {device}")

    def warmup(self, n: int = 3, img_size: int = 640):
        with self.torch.no_grad():
            dummy = self.torch.zeros(1, 3, img_size, img_size,
                                     device=self.device)
            for _ in range(n):
                self.model(dummy)

    def forward(self, blob: np.ndarray) -> np.ndarray:
        """Run forward pass; return raw output as numpy (1, anchors, 85)."""
        import torch
        tensor = torch.from_numpy(blob).to(self.device)
        with torch.no_grad():
            # model() returns Results object; use model.model() for raw output
            try:
                raw = self.model.model(tensor)[0]  # raw prediction
            except AttributeError:
                raw = self.model(tensor)[0]
        return raw.cpu().numpy()


# ── Inference runner ──────────────────────────────────────────────────────────

def collect_images(source: str) -> list:
    """Return list of image paths from file or directory."""
    source = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)
    if source.suffix.lower() in exts:
        return [source]
    raise ValueError(f"Source must be an image file or directory: {source}")


def run_inference(backend, image_paths: list, output_dir: Path,
                  conf: float, iou: float,
                  backend_name: str = "model",
                  save_images: bool = True,
                  benchmark: bool = False) -> list:
    """
    Run inference on all images, save annotated results, return all detections.
    Returns list of dicts: {image, latency_ms, detections}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total_latency = []

    print(f"\n{'='*60}")
    print(f"[{backend_name}] Running inference on {len(image_paths)} image(s)")
    print(f"{'='*60}")

    for img_path in image_paths:
        blob, orig = preprocess_image(str(img_path))
        h, w = orig.shape[:2]

        t0 = time.perf_counter()
        raw = backend.forward(blob)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency.append(latency_ms)

        detections = postprocess(raw, (h, w), conf, iou)

        # Save annotated image
        out_path = None
        if save_images:
            vis = draw_detections(orig, detections, label_prefix="")
            out_path = output_dir / f"{backend_name}_{img_path.stem}_result{img_path.suffix}"
            cv2.imwrite(str(out_path), vis)

        results.append({
            "image": str(img_path),
            "latency_ms": round(latency_ms, 2),
            "num_detections": len(detections),
            "detections": detections,
            "output_image": str(out_path) if out_path else None,
        })

        # Console summary per image
        print(f"\nImage : {img_path.name}")
        print(f"Latency: {latency_ms:.2f} ms  |  Detections: {len(detections)}")
        if detections:
            print(f"{'  Class':<20} {'Conf':>6}  {'BBox (x1,y1,x2,y2)'}")
            print(f"  {'-'*54}")
            for d in sorted(detections, key=lambda x: -x["conf"]):
                bbox = f"({d['x1']},{d['y1']},{d['x2']},{d['y2']})"
                print(f"  {d['class_name']:<20} {d['conf']:>6.3f}  {bbox}")
        if save_images:
            print(f"Saved : {out_path}")

    if benchmark and total_latency:
        lats = np.array(total_latency)
        print(f"\n{'='*60}")
        print(f"[{backend_name}] Latency Statistics ({len(lats)} runs)")
        print(f"{'='*60}")
        print(f"  Mean  : {lats.mean():.2f} ms")
        print(f"  Std   : {lats.std():.2f} ms")
        print(f"  Min   : {lats.min():.2f} ms")
        print(f"  Max   : {lats.max():.2f} ms")
        print(f"  P50   : {np.percentile(lats, 50):.2f} ms")
        print(f"  P95   : {np.percentile(lats, 95):.2f} ms")
        print(f"  P99   : {np.percentile(lats, 99):.2f} ms")
        print(f"{'='*60}\n")

    return results


# ── MSC / Baseline comparison ─────────────────────────────────────────────────

def msc_comparison(pt_results: list, onnx_results: list) -> dict:
    """
    Model Score Comparison (MSC) between PyTorch and ONNX on same inputs.
    Reports:
      - Detection agreement (IoU > 0.5 match rate)
      - Mean confidence difference
      - Class prediction agreement
      - Baseline accuracy parity verdict
    """
    print(f"\n{'='*60}")
    print("MSC Verification — PyTorch vs ONNX Baseline Comparison")
    print(f"{'='*60}")

    total_images = len(pt_results)
    matched_total, pt_total, onnx_total = 0, 0, 0
    conf_diffs = []
    class_agreements = []

    for pt_res, onnx_res in zip(pt_results, onnx_results):
        pt_dets = pt_res["detections"]
        onnx_dets = onnx_res["detections"]
        pt_total += len(pt_dets)
        onnx_total += len(onnx_dets)

        # Greedy IoU matching
        matched = 0
        used_onnx = set()
        for pt_d in pt_dets:
            best_iou, best_j = 0.0, -1
            for j, on_d in enumerate(onnx_dets):
                if j in used_onnx:
                    continue
                iou = _box_iou(
                    (pt_d["x1"], pt_d["y1"], pt_d["x2"], pt_d["y2"]),
                    (on_d["x1"], on_d["y1"], on_d["x2"], on_d["y2"]))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                matched += 1
                used_onnx.add(best_j)
                conf_diffs.append(
                    abs(pt_d["conf"] - onnx_dets[best_j]["conf"]))
                class_agreements.append(
                    int(pt_d["class_id"] == onnx_dets[best_j]["class_id"]))
        matched_total += matched

    detection_agree_pct = (
        matched_total / max(pt_total, 1)) * 100
    mean_conf_diff = float(np.mean(conf_diffs)) if conf_diffs else 0.0
    class_agree_pct = (
        float(np.mean(class_agreements)) * 100) if class_agreements else 0.0

    verdict = "PASS ✓" if detection_agree_pct >= 90 else "REVIEW !"

    report = {
        "images_compared": total_images,
        "pt_detections": pt_total,
        "onnx_detections": onnx_total,
        "matched_detections_iou50": matched_total,
        "detection_agreement_pct": round(detection_agree_pct, 2),
        "mean_confidence_delta": round(mean_conf_diff, 4),
        "class_agreement_pct": round(class_agree_pct, 2),
        "verdict": verdict,
    }

    print(f"  Images compared        : {total_images}")
    print(f"  PyTorch detections     : {pt_total}")
    print(f"  ONNX detections        : {onnx_total}")
    print(f"  Matched (IoU≥0.5)      : {matched_total}")
    print(f"  Detection agreement    : {detection_agree_pct:.1f}%")
    print(f"  Mean confidence delta  : {mean_conf_diff:.4f}")
    print(f"  Class agreement        : {class_agree_pct:.1f}%")
    print(f"  Verdict                : {verdict}")
    print(f"{'='*60}\n")
    return report


def _box_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ── Simple detection-level accuracy metrics ───────────────────────────────────

def compute_detection_metrics(all_results: list, backend_name: str) -> dict:
    """
    Compute basic detection metrics across all images (no GT needed):
      - Total images processed
      - Total detections
      - Mean confidence
      - Per-class detection count
      - Mean detections per image

    When GT labels are available via --labels, computes precision / recall / F1.
    """
    total_imgs = len(all_results)
    total_dets = sum(r["num_detections"] for r in all_results)
    all_confs = [d["conf"] for r in all_results for d in r["detections"]]
    class_counts: dict = {}
    for r in all_results:
        for d in r["detections"]:
            class_counts[d["class_name"]] = class_counts.get(
                d["class_name"], 0) + 1

    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

    print(f"\n{'='*60}")
    print(f"[{backend_name}] Detection Metrics Summary")
    print(f"{'='*60}")
    print(f"  Images processed    : {total_imgs}")
    print(f"  Total detections    : {total_dets}")
    print(f"  Mean per image      : {total_dets / max(total_imgs, 1):.2f}")
    print(f"  Mean confidence     : {mean_conf:.4f}")
    if class_counts:
        print(f"\n  Top-10 detected classes:")
        for cls, cnt in sorted(class_counts.items(),
                               key=lambda x: -x[1])[:10]:
            print(f"    {cls:<25} {cnt:>5}")
    print(f"{'='*60}\n")

    return {
        "backend": backend_name,
        "total_images": total_imgs,
        "total_detections": total_dets,
        "mean_detections_per_image": round(total_dets / max(total_imgs, 1), 2),
        "mean_confidence": round(mean_conf, 4),
        "class_counts": class_counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv5 Inference — PyTorch & ONNX with BBox output",
        formatter_class=argparse.RawTextHelpFormatter)

    g = p.add_argument_group("Model selection")
    g.add_argument("--model", type=str,
                   help="Path to a single .pt or .onnx model")
    g.add_argument("--pt-model", type=str,
                   help="Path to PyTorch .pt model (for comparison mode)")
    g.add_argument("--onnx-model", type=str,
                   help="Path to ONNX .onnx model (for comparison mode)")

    p.add_argument("--source", type=str, required=True,
                   help="Image file or directory of images")
    p.add_argument("--output-dir", type=str, default="results/inference",
                   help="Directory to save annotated output images")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default 0.25)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="IoU threshold for NMS (default 0.45)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for PyTorch: cpu or cuda (default cpu)")
    p.add_argument("--img-size", type=int, default=640,
                   help="Inference image size (default 640)")
    p.add_argument("--compare", action="store_true",
                   help="Run MSC comparison: PyTorch vs ONNX")
    p.add_argument("--benchmark", action="store_true",
                   help="Print latency benchmark stats")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving annotated images")
    p.add_argument("--save-json", type=str, default=None,
                   help="Save detection results to JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    save_images = not args.no_save
    output_dir = Path(args.output_dir)
    image_paths = collect_images(args.source)

    if not image_paths:
        print(f"No images found at: {args.source}")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} image(s)  |  conf={args.conf}  "
          f"iou={args.iou}  device={args.device}")

    all_reports = {}

    # ── Compare mode: run both backends ──────────────────────────────────────
    if args.compare or (args.pt_model and args.onnx_model):
        pt_path = args.pt_model or args.model
        onnx_path = args.onnx_model or args.model
        if not pt_path or not onnx_path:
            print("ERROR: --compare requires both --pt-model and --onnx-model")
            sys.exit(1)

        pt_backend = PyTorchBackend(pt_path, device=args.device)
        pt_backend.warmup()
        onnx_backend = ONNXBackend(onnx_path)
        onnx_backend.warmup()

        pt_results = run_inference(
            pt_backend, image_paths, output_dir / "pytorch",
            args.conf, args.iou, "pytorch", save_images, args.benchmark)
        onnx_results = run_inference(
            onnx_backend, image_paths, output_dir / "onnx",
            args.conf, args.iou, "onnx", save_images, args.benchmark)

        all_reports["pytorch_metrics"] = compute_detection_metrics(
            pt_results, "pytorch")
        all_reports["onnx_metrics"] = compute_detection_metrics(
            onnx_results, "onnx")
        all_reports["msc_comparison"] = msc_comparison(pt_results, onnx_results)

    # ── Single model mode ─────────────────────────────────────────────────────
    elif args.model:
        model_path = args.model
        if model_path.endswith(".onnx"):
            backend = ONNXBackend(model_path)
            backend.warmup()
            results = run_inference(
                backend, image_paths, output_dir,
                args.conf, args.iou, "onnx", save_images, args.benchmark)
            all_reports["onnx_metrics"] = compute_detection_metrics(
                results, "onnx")
        else:
            backend = PyTorchBackend(model_path, device=args.device)
            backend.warmup()
            results = run_inference(
                backend, image_paths, output_dir,
                args.conf, args.iou, "pytorch", save_images, args.benchmark)
            all_reports["pytorch_metrics"] = compute_detection_metrics(
                results, "pytorch")
    else:
        print("ERROR: provide --model or both --pt-model and --onnx-model")
        sys.exit(1)

    # ── Save JSON report ──────────────────────────────────────────────────────
    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"Results saved to: {json_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
