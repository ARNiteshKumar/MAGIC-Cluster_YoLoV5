#!/usr/bin/env python3
"""
COCO Full-Dataset Evaluation Script for YOLOv5
Validates both PyTorch and ONNX models on COCO val2017.

Metrics reported:
  - mAP@0.5        (COCO primary)
  - mAP@0.5:0.95   (COCO standard)
  - Precision, Recall, F1  (at conf threshold)
  - Per-class AP
  - Latency stats

Usage:
  # ONNX only
  python src/inference/coco_eval.py \
      --onnx-model artifacts/exports/yolov5s.onnx \
      --data-dir data/coco

  # PyTorch only
  python src/inference/coco_eval.py \
      --pt-model yolov5s.pt \
      --data-dir data/coco

  # Both (generates comparison table)
  python src/inference/coco_eval.py \
      --pt-model yolov5s.pt \
      --onnx-model artifacts/exports/yolov5s.onnx \
      --data-dir data/coco

Directory layout expected:
  data/coco/
    images/val2017/
    annotations/instances_val2017.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Shared helpers from infer.py
sys.path.insert(0, str(Path(__file__).parent))
from infer import (
    COCO_CLASSES, ONNXBackend, PyTorchBackend,
    preprocess_image, postprocess, _box_iou,
)

# COCO category_id → 0-based index mapping (official 80-class subset)
COCO_CAT_TO_IDX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
COCO_IDX_TO_CAT = {v: k for k, v in COCO_CAT_TO_IDX.items()}


# ── COCO GT loader ────────────────────────────────────────────────────────────

def load_coco_gt(ann_file: str) -> dict:
    """Load COCO annotation JSON and build image_id → list[GT box] map."""
    print(f"Loading COCO annotations: {ann_file}")
    with open(ann_file) as f:
        data = json.load(f)

    gt: dict = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt:
            gt[img_id] = []
        x, y, w, h = ann["bbox"]
        cat_idx = COCO_CAT_TO_IDX.get(ann["category_id"], -1)
        if cat_idx < 0:
            continue
        gt[img_id].append({
            "x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "class_id": cat_idx,
            "area": ann["area"],
        })

    images = {img["id"]: img["file_name"] for img in data["images"]}
    print(f"  → {len(images)} images, annotations for {len(gt)} images")
    return gt, images


# ── Per-class AP (VOC 11-point + COCO 101-point) ──────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray,
               method: str = "interp") -> float:
    """Compute average precision via 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def match_predictions(pred_boxes: list, gt_boxes: list,
                      iou_thresh: float = 0.5) -> tuple:
    """
    Match prediction list to GT list at given IoU threshold.
    Returns (tp_array, fp_array, num_gt) for one image.
    """
    n_pred = len(pred_boxes)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)
    matched_gt = set()

    for i, pred in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if gt["class_id"] != pred["class_id"]:
                continue
            if j in matched_gt:
                continue
            iou = _box_iou(
                (pred["x1"], pred["y1"], pred["x2"], pred["y2"]),
                (gt["x1"], gt["y1"], gt["x2"], gt["y2"]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            matched_gt.add(best_j)
        else:
            fp[i] = 1

    return tp, fp, len(gt_boxes)


# ── Main evaluator ────────────────────────────────────────────────────────────

class COCOEvaluator:
    def __init__(self, gt: dict, images: dict,
                 img_dir: str,
                 conf: float = 0.001,
                 iou: float = 0.6,
                 max_images: int = None):
        self.gt = gt
        self.images = images
        self.img_dir = Path(img_dir)
        self.conf = conf
        self.iou = iou
        self.img_ids = list(images.keys())
        if max_images:
            self.img_ids = self.img_ids[:max_images]

    def evaluate(self, backend, backend_name: str,
                 iou_thresh: float = 0.5) -> dict:
        """
        Run inference on all val images and compute:
          mAP@0.5, mAP@0.5:0.95, precision, recall, F1
        """
        print(f"\n{'='*60}")
        print(f"Evaluating [{backend_name}] on {len(self.img_ids)} COCO val images")
        print(f"conf={self.conf}  nms_iou={self.iou}  eval_iou={iou_thresh}")
        print(f"{'='*60}")

        # Per-class accumulators: class_id → [(score, tp, fp), ...]
        cls_preds: dict = {i: [] for i in range(len(COCO_CLASSES))}
        cls_n_gt: dict = {i: 0 for i in range(len(COCO_CLASSES))}
        latencies = []

        for idx, img_id in enumerate(self.img_ids):
            fname = self.images[img_id]
            img_path = self.img_dir / fname
            if not img_path.exists():
                continue

            try:
                blob, orig = preprocess_image(str(img_path))
            except Exception:
                continue

            h, w = orig.shape[:2]
            t0 = time.perf_counter()
            raw = backend.forward(blob)
            latencies.append((time.perf_counter() - t0) * 1000)

            detections = postprocess(raw, (h, w), self.conf, self.iou)
            gt_boxes = self.gt.get(img_id, [])

            # Accumulate per class
            for gt_b in gt_boxes:
                cls_n_gt[gt_b["class_id"]] = cls_n_gt.get(
                    gt_b["class_id"], 0) + 1

            # Sort detections by confidence (high → low)
            detections = sorted(detections, key=lambda x: -x["conf"])
            tp, fp, _ = match_predictions(detections, gt_boxes, iou_thresh)

            for i, det in enumerate(detections):
                cid = det["class_id"]
                if cid < len(COCO_CLASSES):
                    cls_preds[cid].append((det["conf"], tp[i], fp[i]))

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx+1}/{len(self.img_ids)} ...")

        # Compute per-class AP
        aps = []
        cls_ap = {}
        for cid in range(len(COCO_CLASSES)):
            preds = cls_preds[cid]
            n_gt = cls_n_gt.get(cid, 0)
            if n_gt == 0 or not preds:
                cls_ap[COCO_CLASSES[cid]] = 0.0
                continue
            preds.sort(key=lambda x: -x[0])
            tp_cum = np.cumsum([p[1] for p in preds])
            fp_cum = np.cumsum([p[2] for p in preds])
            recall = tp_cum / (n_gt + 1e-9)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap = compute_ap(recall, precision)
            cls_ap[COCO_CLASSES[cid]] = round(ap, 4)
            aps.append(ap)

        mAP50 = float(np.mean(aps)) if aps else 0.0

        # mAP@0.5:0.95 (average over 10 IoU thresholds)
        map_vals = []
        for thresh in np.arange(0.5, 1.0, 0.05):
            aps_t = []
            for cid in range(len(COCO_CLASSES)):
                preds = cls_preds[cid]
                n_gt = cls_n_gt.get(cid, 0)
                if n_gt == 0 or not preds:
                    continue
                preds.sort(key=lambda x: -x[0])
                tp_cum = np.cumsum([p[1] for p in preds])
                fp_cum = np.cumsum([p[2] for p in preds])
                recall = tp_cum / (n_gt + 1e-9)
                precision = tp_cum / (tp_cum + fp_cum + 1e-9)
                aps_t.append(compute_ap(recall, precision))
            map_vals.append(float(np.mean(aps_t)) if aps_t else 0.0)
        mAP5095 = float(np.mean(map_vals)) if map_vals else 0.0

        # Overall precision / recall / F1 at conf threshold
        all_preds_flat = []
        for cid_preds in cls_preds.values():
            all_preds_flat.extend(cid_preds)
        total_gt = sum(cls_n_gt.values())
        total_tp = sum(p[1] for p in all_preds_flat)
        total_fp = sum(p[2] for p in all_preds_flat)
        precision_overall = total_tp / (total_tp + total_fp + 1e-9)
        recall_overall = total_tp / (total_gt + 1e-9)
        f1 = (2 * precision_overall * recall_overall /
              (precision_overall + recall_overall + 1e-9))

        lats = np.array(latencies)
        lat_stats = {
            "mean_ms": round(lats.mean(), 2),
            "p50_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_ms": round(float(np.percentile(lats, 99)), 2),
        } if len(lats) else {}

        result = {
            "backend": backend_name,
            "images_evaluated": len(latencies),
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP5095, 4),
            "precision": round(float(precision_overall), 4),
            "recall": round(float(recall_overall), 4),
            "f1": round(float(f1), 4),
            "latency": lat_stats,
            "per_class_ap": cls_ap,
        }

        self._print_results(result)
        return result

    @staticmethod
    def _print_results(r: dict):
        print(f"\n{'='*60}")
        print(f"[{r['backend']}] Evaluation Results")
        print(f"{'='*60}")
        print(f"  Images evaluated : {r['images_evaluated']}")
        print(f"  mAP@0.5          : {r['mAP50']:.4f}  ({r['mAP50']*100:.2f}%)")
        print(f"  mAP@0.5:0.95     : {r['mAP50_95']:.4f}  ({r['mAP50_95']*100:.2f}%)")
        print(f"  Precision        : {r['precision']:.4f}")
        print(f"  Recall           : {r['recall']:.4f}")
        print(f"  F1               : {r['f1']:.4f}")
        if r.get("latency"):
            lat = r["latency"]
            print(f"  Latency (mean)   : {lat['mean_ms']} ms")
            print(f"  Latency (P95)    : {lat['p95_ms']} ms")
        top_cls = sorted(r["per_class_ap"].items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 classes by AP@0.5:")
        for cls, ap in top_cls:
            bar = "#" * int(ap * 40)
            print(f"    {cls:<22} {ap:.4f}  {bar}")
        print(f"{'='*60}\n")


def print_comparison_table(pt_res: dict, onnx_res: dict):
    """Side-by-side metric table for PyTorch vs ONNX."""
    print(f"\n{'='*70}")
    print(f"{'Metric':<25} {'PyTorch':>12} {'ONNX':>12} {'Delta':>12}")
    print(f"{'-'*70}")
    metrics = [("mAP@0.5", "mAP50"),
               ("mAP@0.5:0.95", "mAP50_95"),
               ("Precision", "precision"),
               ("Recall", "recall"),
               ("F1", "f1"),
               ("Lat mean (ms)", "latency.mean_ms"),
               ("Lat P95 (ms)", "latency.p95_ms")]
    for label, key in metrics:
        def _get(d, k):
            parts = k.split(".")
            v = d
            for p in parts:
                v = v.get(p, {}) if isinstance(v, dict) else {}
            return v if isinstance(v, float) else 0.0
        pt_v = _get(pt_res, key)
        on_v = _get(onnx_res, key)
        delta = on_v - pt_v
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<23} {pt_v:>12.4f} {on_v:>12.4f} {sign+f'{delta:.4f}':>12}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="COCO Full-Dataset Evaluation — PyTorch & ONNX YOLOv5")
    p.add_argument("--pt-model", type=str, default=None,
                   help="Path to PyTorch .pt weights")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="Path to ONNX model")
    p.add_argument("--data-dir", type=str, required=True,
                   help="COCO root dir (expects images/val2017/ and annotations/)")
    p.add_argument("--ann-file", type=str, default=None,
                   help="Override annotation JSON path")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Confidence threshold for eval (default 0.001)")
    p.add_argument("--iou", type=float, default=0.6,
                   help="NMS IoU threshold (default 0.6)")
    p.add_argument("--eval-iou", type=float, default=0.5,
                   help="IoU threshold for TP matching (default 0.5)")
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit number of val images (useful for quick tests)")
    p.add_argument("--device", type=str, default="cpu",
                   help="PyTorch device (cpu / cuda)")
    p.add_argument("--save-json", type=str, default="results/coco_eval.json",
                   help="Output JSON path for metrics")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    ann_file = args.ann_file or str(
        data_dir / "annotations" / "instances_val2017.json")
    img_dir = str(data_dir / "images" / "val2017")

    if not Path(ann_file).exists():
        print(f"ERROR: annotation file not found: {ann_file}")
        print("Download COCO val2017 annotations from https://cocodataset.org/")
        sys.exit(1)

    gt, images = load_coco_gt(ann_file)
    evaluator = COCOEvaluator(
        gt, images, img_dir,
        conf=args.conf, iou=args.iou,
        max_images=args.max_images)

    all_results = {}

    # ── PyTorch evaluation ────────────────────────────────────────────────────
    if args.pt_model:
        pt_backend = PyTorchBackend(args.pt_model, device=args.device)
        pt_backend.warmup()
        all_results["pytorch"] = evaluator.evaluate(
            pt_backend, "pytorch", iou_thresh=args.eval_iou)

    # ── ONNX evaluation ───────────────────────────────────────────────────────
    if args.onnx_model:
        onnx_backend = ONNXBackend(args.onnx_model)
        onnx_backend.warmup()
        all_results["onnx"] = evaluator.evaluate(
            onnx_backend, "onnx", iou_thresh=args.eval_iou)

    # ── Comparison table ──────────────────────────────────────────────────────
    if "pytorch" in all_results and "onnx" in all_results:
        print_comparison_table(all_results["pytorch"], all_results["onnx"])

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Evaluation results saved to: {out}")

    if not all_results:
        print("ERROR: provide at least one of --pt-model or --onnx-model")
        sys.exit(1)


if __name__ == "__main__":
    main()
