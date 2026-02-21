# YOLOv5 Model Export & Validation Pipeline

A complete, production-ready pipeline for training, exporting, and validating YOLOv5
object detection models with full COCO evaluation and PyTorch/ONNX comparison.

## What's New (Issue #1 — Inference Overhaul)

- **`src/inference/infer.py`** — Unified inference script (PyTorch & ONNX)
  - Bounding boxes drawn on output images with class name + confidence
  - Annotated images saved automatically to `results/inference/`
  - COCO 80-class labels
  - MSC (Model Score Comparison) — PyTorch vs ONNX baseline verification
  - Latency benchmarking for both backends
- **`src/inference/coco_eval.py`** — Full COCO val2017 evaluation
  - mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
  - Per-class AP breakdown
  - Side-by-side PyTorch vs ONNX accuracy comparison table
- **`src/inference/infer_onnx.py`** — Updated legacy ONNX script with NMS + bbox output
- **New scripts**: `compare.sh`, `evaluate_coco.sh`, `download_coco.sh`

---

## Repository Structure

```
MAGIC-Cluster_YoLoV5/
├── src/
│   ├── inference/
│   │   ├── infer.py           # ★ Unified PyTorch + ONNX inference (bbox output)
│   │   ├── coco_eval.py       # ★ Full COCO evaluation (mAP, precision, recall)
│   │   └── infer_onnx.py      # Updated ONNX-only script with bbox + NMS
│   ├── models/
│   │   └── export_model.py    # Export .pt → .onnx
│   └── training/
│       └── train.py           # Training wrapper
├── scripts/
│   ├── setup.sh               # Environment setup
│   ├── train.sh               # Train model
│   ├── export.sh              # Export to ONNX
│   ├── infer.sh               # ★ Run inference (saves annotated images)
│   ├── compare.sh             # ★ MSC comparison: PyTorch vs ONNX
│   ├── evaluate_coco.sh       # ★ Full COCO dataset evaluation
│   ├── download_coco.sh       # ★ Download COCO val2017
│   └── run_pipeline.sh        # ★ Complete end-to-end pipeline
├── configs/
│   ├── config.yaml            # Main configuration
│   └── train_config.yaml      # Training configuration
├── results/                   # ← Output images and JSON reports land here
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
bash scripts/setup.sh
```

### 2. Download COCO val2017 (for full evaluation)

```bash
bash scripts/download_coco.sh data/coco
```

---

## Inference — Bounding Box Output

### ONNX Inference (saves annotated images to `results/inference/`)

```bash
bash scripts/infer.sh yolov5s.onnx data/images/ results/inference
```

### PyTorch Inference

```bash
bash scripts/infer.sh yolov5s.pt data/images/ results/inference
```

### Single image

```bash
python src/inference/infer.py \
    --model yolov5s.onnx \
    --source data/sample/dog.jpg \
    --output-dir results/inference \
    --conf 0.25 --iou 0.45
```

Output images are saved as `results/inference/onnx_<image>_result.jpg` with
bounding boxes, class names, and confidence scores drawn on each detection.

---

## MSC Comparison — PyTorch vs ONNX

Run both models on the **same input** and get a baseline accuracy verification report:

```bash
bash scripts/compare.sh yolov5s.pt yolov5s.onnx data/images/ results/compare
```

Or with Python directly:

```bash
python src/inference/infer.py \
    --pt-model yolov5s.pt \
    --onnx-model yolov5s.onnx \
    --source data/images/ \
    --output-dir results/compare \
    --compare --benchmark \
    --save-json results/compare/msc_report.json
```

**MSC Report includes:**
| Metric | Description |
|--------|-------------|
| Detection agreement (%) | % of PyTorch detections matched in ONNX at IoU ≥ 0.5 |
| Mean confidence delta | Average confidence score difference |
| Class agreement (%) | % of matched detections with same predicted class |
| Verdict | PASS ✓ (≥ 90% agreement) or REVIEW ! |

---

## COCO Full-Dataset Evaluation

### Both backends (recommended — generates comparison table)

```bash
bash scripts/evaluate_coco.sh yolov5s.pt yolov5s.onnx data/coco
```

### ONNX only

```bash
bash scripts/evaluate_coco.sh "" yolov5s.onnx data/coco
```

### Quick test on first 500 images

```bash
bash scripts/evaluate_coco.sh yolov5s.pt yolov5s.onnx data/coco 500
```

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean Average Precision at IoU 0.5 (primary) |
| mAP@0.5:0.95 | COCO standard metric (10 IoU thresholds) |
| Precision | TP / (TP + FP) at conf threshold |
| Recall | TP / (TP + FN) at conf threshold |
| F1 | Harmonic mean of precision and recall |
| Per-class AP | AP breakdown for all 80 COCO classes |
| Latency | Mean / P50 / P95 / P99 ms |

---

## Complete Pipeline

Run training → export → inference → comparison → COCO eval in one command:

```bash
bash scripts/run_pipeline.sh
```

Environment variables to control pipeline steps:
```bash
SKIP_TRAIN=1 \
SKIP_COCO_DOWNLOAD=0 \
COCO_DIR=data/coco \
DEVICE=cuda \
bash scripts/run_pipeline.sh
```

---

## Google Colab

To run on Colab with your GitHub token:

```python
import os

# Clone and setup
os.system("git clone https://github.com/ARNiteshKumar/MAGIC-Cluster_YoLoV5.git")
os.chdir("MAGIC-Cluster_YoLoV5")
os.system("pip install -r requirements.txt")
os.system("bash scripts/setup.sh")

# Download COCO (optional — ~1 GB)
os.system("bash scripts/download_coco.sh data/coco")

# Run inference (saves annotated images)
os.system("python src/inference/infer.py "
          "--model yolov5s.onnx "
          "--source data/images/ "
          "--output-dir results/inference "
          "--conf 0.25 --benchmark")

# MSC comparison
os.system("python src/inference/infer.py "
          "--pt-model yolov5s.pt "
          "--onnx-model yolov5s.onnx "
          "--source data/images/ "
          "--compare --save-json results/msc_report.json")

# COCO evaluation
os.system("python src/inference/coco_eval.py "
          "--pt-model yolov5s.pt "
          "--onnx-model yolov5s.onnx "
          "--data-dir data/coco "
          "--save-json results/coco_eval.json")
```

---

## Model Performance

### YOLOv5s — COCO val2017 (reference)

| Backend  | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Mean Latency |
|----------|---------|--------------|-----------|--------|--------------|
| PyTorch  | 0.556   | 0.374        | 0.673     | 0.504  | ~30 ms (CPU) |
| ONNX     | 0.556   | 0.374        | 0.673     | 0.504  | ~20 ms (CPU) |

*Actual numbers will vary by hardware. Run `evaluate_coco.sh` to get your results.*

---

## Configuration

Edit `configs/config.yaml` to customise inference, evaluation, and export settings.

---

## Contact

- LinkedIn: [www.linkedin.com/in/arniteshkumar](https://www.linkedin.com/in/arniteshkumar)
- Email: arniteshkumar@gmail.com
- Issues: https://github.com/ARNiteshKumar/MAGIC-Cluster_YoLoV5/issues
