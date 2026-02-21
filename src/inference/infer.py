#!/usr/bin/env python3
"""
YOLOv5 Unified Inference Script
================================
Supports PyTorch (.pt) and ONNX (.onnx) backends with:
  - Bounding box visualisation saved to a dedicated output folder
  - Class name labels + confidence scores on each box
  - MSE / MAE / cosine-similarity baseline comparison between PyTorch and ONNX
  - Evaluation metrics (mAP@0.5, Precision, Recall) on COCO val2017
  - Per-image latency reporting for both backends

Usage examples
--------------
# Single image – both backends
python src/inference/infer.py \
    --pt-weights  yolov5s.pt \
    --onnx-weights yolov5s.onnx \
    --image data/sample.jpg \
    --output-dir results/bbox_outputs

# COCO validation (needs val2017 images + annotations)
python src/inference/infer.py \
    --pt-weights  yolov5s.pt \
    --onnx-weights yolov5s.onnx \
    --coco-dir data/coco \
    --eval \
    --num-eval-images 5000 \
    --output-dir results/bbox_outputs

# Google Colab quick-start (see README for token-based setup)
"""

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# ── Optional: add bundled YOLOv5 repo to path if present ──────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_YOLO_PATH = _ROOT / "yolov5"
if _YOLO_PATH.is_dir():
    sys.path.insert(0, str(_YOLO_PATH))

# ── COCO 80-class names ────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Deterministic per-class BGR colours ───────────────────────────────────────
np.random.seed(42)
_PALETTE = np.random.randint(50, 230, (len(COCO_CLASSES), 3), dtype=np.uint8)


def _color(cls_id: int):
    return [int(c) for c in _PALETTE[int(cls_id) % len(_PALETTE)]]


# ══════════════════════════════════════════════════════════════════════════════
# Pre / post-processing utilities
# ══════════════════════════════════════════════════════════════════════════════

def letterbox(img: np.ndarray, new_shape: int = 640,
              color=(114, 114, 114)) -> tuple:
    """Resize with letterboxing to maintain aspect ratio.

    Returns:
        img_lb  – letterboxed image
        ratio   – scale factor applied
        (dw,dh) – padding added (in pixels, each side)
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess_image(image_path, img_size: int = 640) -> tuple:
    """Load + letterbox + normalise an image.

    Returns:
        img_batch   – float32 numpy [1, 3, H, W] in [0, 1]  (common input)
        img_bgr     – original BGR image (for drawing)
        orig_shape  – (h, w) of original image
        ratio       – letterbox scale factor
        pad         – (dw, dh) padding
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    orig_shape = img_bgr.shape[:2]  # (h, w)

    img_lb, ratio, pad = letterbox(img_bgr, img_size)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_chw, 0)          # [1, 3, H, W]

    return img_batch, img_bgr, orig_shape, ratio, pad


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert center-xywh → corner-xyxy."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray,
               iou_threshold: float) -> list:
    """Greedy IoU-based NMS. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter  = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou    = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order  = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def non_max_suppression(predictions: np.ndarray,
                        conf_thres: float = 0.25,
                        iou_thres:  float = 0.45,
                        max_det:    int   = 1000) -> list:
    """Apply NMS to YOLOv5 raw output.

    Args:
        predictions: float32 [batch, num_anchors, 85]
                     where 85 = 4 (xywh) + 1 (obj_conf) + 80 (cls_probs)

    Returns:
        List of float32 arrays shaped [N, 6] per image:
            x1, y1, x2, y2, confidence, class_id
    """
    results = []
    for pred in predictions:                        # per-image loop
        # 1. objectness filter
        obj_conf  = pred[:, 4]
        pred      = pred[obj_conf > conf_thres]
        if pred.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # 2. class score = obj_conf × class_prob
        pred[:, 5:] *= pred[:, 4:5]

        # 3. xywh → xyxy
        boxes = xywh2xyxy(pred[:, :4])

        # 4. best class per anchor
        cls_ids = pred[:, 5:].argmax(axis=1)
        scores  = pred[:, 5:][np.arange(len(cls_ids)), cls_ids]

        # 5. class-score filter
        mask   = scores > conf_thres
        boxes  = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if boxes.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # 6. per-class NMS
        kept = []
        for cls in np.unique(cls_ids):
            m        = cls_ids == cls
            k_idx    = _nms_boxes(boxes[m], scores[m], iou_thres)
            kept_b   = boxes[m][k_idx]
            kept_s   = scores[m][k_idx]
            kept     += [[*b, s, cls] for b, s in zip(kept_b, kept_s)]

        if kept:
            det = np.array(kept[:max_det], dtype=np.float32)
        else:
            det = np.zeros((0, 6), dtype=np.float32)
        results.append(det)
    return results


def scale_boxes(boxes: np.ndarray, orig_shape: tuple,
                ratio: float, pad: tuple) -> np.ndarray:
    """Map boxes from letterboxed space back to original image coordinates."""
    if boxes.shape[0] == 0:
        return boxes
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# Bounding-box visualisation
# ══════════════════════════════════════════════════════════════════════════════

def draw_detections(image: np.ndarray, detections: np.ndarray,
                    class_names: list = COCO_CLASSES,
                    backend_label: str = "") -> np.ndarray:
    """Draw bounding boxes + class/confidence labels on a BGR image copy."""
    img = image.copy()

    # Backend watermark
    if backend_label:
        cv2.putText(img, backend_label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        color  = _color(cls_id)
        name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label  = f"{name} {conf:.2f}"

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - bl - 4), (x1 + lw, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Backend wrappers
# ══════════════════════════════════════════════════════════════════════════════

class PyTorchBackend:
    """Load a YOLOv5 .pt model and expose raw inference."""

    def __init__(self, weights: str, device: str = "cpu"):
        import torch
        self.torch  = torch
        self.device = torch.device(device)

        print(f"[PyTorch] Loading weights: {weights}")
        # Try bundled YOLOv5 DetectMultiBackend first, fall back to torch.hub
        try:
            from models.common import DetectMultiBackend
            self.model = DetectMultiBackend(weights, device=self.device)
        except Exception:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=weights, force_reload=False, verbose=False,
            )
        self.model.eval()
        print("[PyTorch] Model ready.")

    def warmup(self, img_batch: np.ndarray, n: int = 3):
        t = self.torch.from_numpy(img_batch).to(self.device)
        for _ in range(n):
            with self.torch.no_grad():
                self._forward(t)

    def _forward(self, img_tensor):
        out = self.model(img_tensor)
        # DetectMultiBackend returns a tuple; first element is the prediction
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def infer_raw(self, img_batch: np.ndarray) -> tuple:
        """Returns (raw_np [1, anchors, 85], latency_ms)."""
        t = self.torch.from_numpy(img_batch).to(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            raw = self._forward(t)
        lat = (time.perf_counter() - t0) * 1000
        return raw.cpu().numpy(), lat


class ONNXBackend:
    """Load a YOLOv5 .onnx model and expose raw inference."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[ONNX] Loading model: {model_path}")
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ONNX] Model ready.  Input: {self.input_name}  "
              f"Output: {self.output_name}")

    def warmup(self, img_batch: np.ndarray, n: int = 3):
        for _ in range(n):
            self.session.run([self.output_name], {self.input_name: img_batch})

    def infer_raw(self, img_batch: np.ndarray) -> tuple:
        """Returns (raw_np [1, anchors, 85], latency_ms)."""
        t0  = time.perf_counter()
        out = self.session.run([self.output_name], {self.input_name: img_batch})
        lat = (time.perf_counter() - t0) * 1000
        return out[0], lat


# ══════════════════════════════════════════════════════════════════════════════
# MSE / baseline comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_outputs(pt_raw: np.ndarray, onnx_raw: np.ndarray) -> dict:
    """Compute numerical agreement metrics between two raw prediction tensors."""
    # Align shapes if they differ (should not happen for matching exports)
    if pt_raw.shape != onnx_raw.shape:
        print(f"  WARNING: shape mismatch PT={pt_raw.shape} ONNX={onnx_raw.shape}; "
              "truncating to common size.")
        mn = tuple(min(a, b) for a, b in zip(pt_raw.shape, onnx_raw.shape))
        pt_raw   = pt_raw[:mn[0], :mn[1], :mn[2]]
        onnx_raw = onnx_raw[:mn[0], :mn[1], :mn[2]]

    diff   = pt_raw - onnx_raw
    mse    = float(np.mean(diff ** 2))
    mae    = float(np.mean(np.abs(diff)))
    max_d  = float(np.max(np.abs(diff)))
    rmse   = float(np.sqrt(mse))

    # Cosine similarity over flattened vectors
    pf, of = pt_raw.flatten(), onnx_raw.flatten()
    cos_sim = float(np.dot(pf, of) /
                    (np.linalg.norm(pf) * np.linalg.norm(of) + 1e-8))

    return dict(mse=mse, rmse=rmse, mae=mae,
                max_diff=max_d, cosine_similarity=cos_sim)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1  = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2  = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def compute_map(all_preds: list, all_gts: list,
                num_classes: int = 80,
                iou_threshold: float = 0.5) -> tuple:
    """Compute mAP@iou_threshold using 11-point interpolation.

    Args:
        all_preds: list of [N, 6] arrays  (x1,y1,x2,y2,conf,cls) per image
        all_gts:   list of [M, 5] arrays  (x1,y1,x2,y2,cls)      per image

    Returns:
        mean_ap   – scalar mAP
        class_ap  – dict {class_id: ap}
    """
    class_ap = {}
    for cls in range(num_classes):
        tp_fp = []      # list of (conf, is_tp)
        n_gt  = 0

        for preds, gts in zip(all_preds, all_gts):
            gt_c  = gts[gts[:, 4] == cls]  if len(gts)   else np.zeros((0, 5))
            pr_c  = preds[preds[:, 5] == cls] if len(preds) else np.zeros((0, 6))
            n_gt += len(gt_c)

            matched = np.zeros(len(gt_c), dtype=bool)
            if len(pr_c):
                pr_c = pr_c[np.argsort(-pr_c[:, 4])]   # sort by conf desc
                for pred in pr_c:
                    best_iou, best_j = 0.0, -1
                    for j, gt in enumerate(gt_c):
                        if matched[j]:
                            continue
                        iou = _box_iou(pred[:4], gt[:4])
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    is_tp = int(best_iou >= iou_threshold and best_j >= 0)
                    if is_tp:
                        matched[best_j] = True
                    tp_fp.append((pred[4], is_tp))

        if n_gt == 0 or not tp_fp:
            continue

        tp_fp.sort(key=lambda x: -x[0])
        confs, tps = zip(*tp_fp)
        tps  = np.cumsum(tps)
        fps  = np.cumsum([1 - t for _, t in tp_fp])
        prec = tps / (tps + fps + 1e-6)
        rec  = tps / (n_gt + 1e-6)

        # 11-point AP
        ap = sum(prec[rec >= t].max(initial=0.0) for t in np.linspace(0, 1, 11)) / 11
        class_ap[cls] = float(ap)

    mean_ap = float(np.mean(list(class_ap.values()))) if class_ap else 0.0
    return mean_ap, class_ap


def print_metrics(label: str, mean_ap: float, class_ap: dict,
                  latencies: list, top_k: int = 10):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {label} — Evaluation Results")
    print(bar)
    print(f"  mAP@0.5      : {mean_ap*100:6.2f}%")
    if latencies:
        lats = np.array(latencies)
        print(f"  Latency mean : {lats.mean():.2f} ms")
        print(f"  Latency P50  : {np.percentile(lats, 50):.2f} ms")
        print(f"  Latency P95  : {np.percentile(lats, 95):.2f} ms")
    if class_ap:
        print(f"\n  Top-{top_k} classes by AP:")
        for cls_id, ap in sorted(class_ap.items(),
                                  key=lambda x: -x[1])[:top_k]:
            name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            print(f"    {name:<20s}  AP={ap*100:.2f}%")
    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
# Single-image inference
# ══════════════════════════════════════════════════════════════════════════════

def run_single_image(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("YOLOv5 — Single Image Inference")
    print(f"  image     : {args.image}")
    print(f"  out_dir   : {out_dir}")
    print(f"{'='*60}")

    img_batch, img_bgr, orig_shape, ratio, pad = preprocess_image(
        args.image, args.img_size)

    pt_raw, onnx_raw = None, None

    # ── PyTorch ────────────────────────────────────────────────────────────────
    if args.pt_weights:
        pt_be = PyTorchBackend(args.pt_weights, args.device)
        pt_be.warmup(img_batch)
        pt_raw, pt_lat = pt_be.infer_raw(img_batch)

        pt_dets = non_max_suppression(pt_raw, args.conf, args.iou)[0]
        if pt_dets.shape[0]:
            pt_dets = scale_boxes(pt_dets, orig_shape, ratio, pad)

        print(f"\n[PyTorch] latency={pt_lat:.2f} ms  "
              f"raw_shape={pt_raw.shape}  detections={len(pt_dets)}")
        _print_detections(pt_dets)

        vis = draw_detections(img_bgr, pt_dets, backend_label="PyTorch")
        out_path = out_dir / f"pt_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved → {out_path}")

    # ── ONNX ───────────────────────────────────────────────────────────────────
    if args.onnx_weights:
        onnx_be = ONNXBackend(args.onnx_weights)
        onnx_be.warmup(img_batch)
        onnx_raw, onnx_lat = onnx_be.infer_raw(img_batch)

        onnx_dets = non_max_suppression(onnx_raw, args.conf, args.iou)[0]
        if onnx_dets.shape[0]:
            onnx_dets = scale_boxes(onnx_dets, orig_shape, ratio, pad)

        print(f"\n[ONNX]    latency={onnx_lat:.2f} ms  "
              f"raw_shape={onnx_raw.shape}  detections={len(onnx_dets)}")
        _print_detections(onnx_dets)

        vis = draw_detections(img_bgr, onnx_dets, backend_label="ONNX")
        out_path = out_dir / f"onnx_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved → {out_path}")

    # ── MSE comparison ─────────────────────────────────────────────────────────
    if pt_raw is not None and onnx_raw is not None:
        stats = compare_outputs(pt_raw, onnx_raw)
        print(f"\n{'='*60}")
        print("MSE Baseline — PyTorch vs ONNX (raw logits)")
        print(f"{'='*60}")
        print(f"  MSE              : {stats['mse']:.8f}")
        print(f"  RMSE             : {stats['rmse']:.8f}")
        print(f"  MAE              : {stats['mae']:.8f}")
        print(f"  Max abs diff     : {stats['max_diff']:.8f}")
        print(f"  Cosine similarity: {stats['cosine_similarity']:.8f}")


def _print_detections(dets: np.ndarray):
    if not len(dets):
        print("  (no detections)")
        return
    for d in dets:
        x1, y1, x2, y2, conf, cls_id = d
        name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else str(int(cls_id))
        print(f"  [{name:<20s}] conf={conf:.3f}  "
              f"bbox=({int(x1):4d},{int(y1):4d},{int(x2):4d},{int(y2):4d})")


# ══════════════════════════════════════════════════════════════════════════════
# COCO dataset evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_coco_evaluation(args):
    data_dir  = Path(args.coco_dir)
    img_dir   = data_dir / "images" / "val2017"
    ann_file  = data_dir / "annotations" / "instances_val2017.json"
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        print(f"ERROR: COCO val2017 images not found at {img_dir}")
        print("  Download with: bash scripts/download_coco.sh")
        return

    # ── Load GT annotations ────────────────────────────────────────────────────
    gt_map: dict = {}                   # {img_id: [[x1,y1,x2,y2,cls], ...]}
    coco_cls_ids: dict = {}             # {coco_cat_id: 0-indexed id}
    if ann_file.is_file():
        print(f"Loading annotations: {ann_file}")
        with open(ann_file) as fh:
            coco = json.load(fh)
        # Build COCO-cat → 0-indexed mapping
        for idx, cat in enumerate(coco["categories"]):
            coco_cls_ids[cat["id"]] = idx
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            bx, by, bw, bh = ann["bbox"]
            cls_0 = coco_cls_ids.get(ann["category_id"], ann["category_id"])
            gt_map.setdefault(iid, []).append([bx, by, bx+bw, by+bh, cls_0])
    else:
        print("WARNING: annotation file not found — skipping mAP, reporting latency only.")

    # ── Collect images ─────────────────────────────────────────────────────────
    all_imgs = sorted(img_dir.glob("*.jpg"))
    if args.num_eval_images > 0:
        all_imgs = all_imgs[:args.num_eval_images]
    print(f"\nEvaluating on {len(all_imgs)} COCO val images …")

    # ── Init backends ──────────────────────────────────────────────────────────
    pt_be   = PyTorchBackend(args.pt_weights,   args.device) if args.pt_weights   else None
    onnx_be = ONNXBackend(args.onnx_weights)                 if args.onnx_weights else None

    # Warmup with first image
    dummy_batch, *_ = preprocess_image(all_imgs[0], args.img_size)
    if pt_be:   pt_be.warmup(dummy_batch)
    if onnx_be: onnx_be.warmup(dummy_batch)

    all_pt_preds,   all_onnx_preds   = [], []
    all_gts                           = []
    pt_lats,   onnx_lats              = [], []
    mse_list                          = []

    for i, img_path in enumerate(all_imgs):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_imgs)} …", flush=True)

        try:
            img_batch, img_bgr, orig_shape, ratio, pad = preprocess_image(
                img_path, args.img_size)
        except Exception as exc:
            print(f"  skip {img_path.name}: {exc}")
            continue

        img_id = int(img_path.stem)
        raw_pt, raw_onnx = None, None

        # PyTorch
        pt_dets = np.zeros((0, 6), dtype=np.float32)
        if pt_be:
            raw_pt, lat = pt_be.infer_raw(img_batch)
            pt_lats.append(lat)
            pt_dets = non_max_suppression(raw_pt, args.conf, args.iou)[0]
            if pt_dets.shape[0]:
                pt_dets = scale_boxes(pt_dets, orig_shape, ratio, pad)

        # ONNX
        onnx_dets = np.zeros((0, 6), dtype=np.float32)
        if onnx_be:
            raw_onnx, lat = onnx_be.infer_raw(img_batch)
            onnx_lats.append(lat)
            onnx_dets = non_max_suppression(raw_onnx, args.conf, args.iou)[0]
            if onnx_dets.shape[0]:
                onnx_dets = scale_boxes(onnx_dets, orig_shape, ratio, pad)

        # MSE
        if raw_pt is not None and raw_onnx is not None:
            mse_list.append(compare_outputs(raw_pt, raw_onnx)["mse"])

        all_pt_preds.append(pt_dets)
        all_onnx_preds.append(onnx_dets)

        gt_boxes = np.array(gt_map.get(img_id, []), dtype=np.float32)
        all_gts.append(gt_boxes)

        # Save visualisations for first N images
        if i < args.save_vis:
            if pt_be:
                vis = draw_detections(img_bgr, pt_dets, backend_label="PyTorch")
                cv2.imwrite(str(out_dir / f"pt_{img_path.name}"), vis)
            if onnx_be:
                vis = draw_detections(img_bgr, onnx_dets, backend_label="ONNX")
                cv2.imwrite(str(out_dir / f"onnx_{img_path.name}"), vis)

    # ── Print evaluation results ───────────────────────────────────────────────
    has_gt = any(len(g) for g in all_gts)

    if pt_be:
        if has_gt:
            pt_map50, pt_ap = compute_map(all_pt_preds, all_gts)
        else:
            pt_map50, pt_ap = 0.0, {}
        print_metrics("PyTorch", pt_map50, pt_ap, pt_lats)

    if onnx_be:
        if has_gt:
            onnx_map50, onnx_ap = compute_map(all_onnx_preds, all_gts)
        else:
            onnx_map50, onnx_ap = 0.0, {}
        print_metrics("ONNX", onnx_map50, onnx_ap, onnx_lats)

    # ── MSE summary ────────────────────────────────────────────────────────────
    if mse_list:
        print(f"\n{'='*60}")
        print("  PyTorch ↔ ONNX Output Agreement (raw logits, per image)")
        print(f"{'='*60}")
        arr = np.array(mse_list)
        print(f"  Mean MSE : {arr.mean():.8f}")
        print(f"  Std  MSE : {arr.std():.8f}")
        print(f"  Max  MSE : {arr.max():.8f}")
        print(f"{'='*60}")

    print(f"\nVisualisations ({min(args.save_vis, len(all_imgs))}) saved → {out_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLOv5 Unified Inference: PyTorch + ONNX, bbox output, COCO eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model weights
    g = p.add_argument_group("Model")
    g.add_argument("--pt-weights",   type=str, default=None,
                   help="Path to YOLOv5 PyTorch weights (.pt)")
    g.add_argument("--onnx-weights", type=str, default=None,
                   help="Path to exported ONNX model (.onnx)")

    # Input / output
    g = p.add_argument_group("Input / Output")
    g.add_argument("--image",      type=str, default=None,
                   help="Single image path for inference")
    g.add_argument("--output-dir", type=str, default="results/bbox_outputs",
                   help="Directory to save annotated images")

    # Inference params
    g = p.add_argument_group("Inference")
    g.add_argument("--img-size", type=int,   default=640,
                   help="Input resolution (square)")
    g.add_argument("--conf",     type=float, default=0.25,
                   help="Confidence threshold")
    g.add_argument("--iou",      type=float, default=0.45,
                   help="NMS IoU threshold")
    g.add_argument("--device",   type=str,   default="cpu",
                   help="PyTorch device: 'cpu' or 'cuda'")

    # COCO evaluation
    g = p.add_argument_group("COCO Evaluation")
    g.add_argument("--eval",            action="store_true",
                   help="Run full COCO val2017 evaluation")
    g.add_argument("--coco-dir",        type=str, default="data/coco",
                   help="Root of COCO dataset (contains images/ and annotations/)")
    g.add_argument("--num-eval-images", type=int, default=5000,
                   help="Max number of val images to evaluate (0 = all)")
    g.add_argument("--save-vis",        type=int, default=50,
                   help="Number of val images to save as annotated JPEGs")

    return p


def main():
    args = build_parser().parse_args()

    if args.pt_weights is None and args.onnx_weights is None:
        print("ERROR: Provide at least --pt-weights or --onnx-weights.")
        return

    if args.eval:
        run_coco_evaluation(args)
    elif args.image:
        run_single_image(args)
    else:
        print("ERROR: Specify --image for single-image inference, "
              "or --eval for COCO evaluation.")


if __name__ == "__main__":
    main()
