#!/bin/bash
# ============================================================
# Complete YOLOv5 Pipeline
#   1. Environment setup
#   2. (Optional) Download COCO val2017
#   3. Train
#   4. Export to ONNX
#   5. Run inference (PyTorch + ONNX) with BBox output
#   6. MSC comparison (PyTorch vs ONNX)
#   7. COCO evaluation (accuracy metrics)
# ============================================================

set -e

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_COCO_DOWNLOAD=${SKIP_COCO_DOWNLOAD:-1}  # set 0 to auto-download
DEVICE=${DEVICE:-"cpu"}
WEIGHTS=${WEIGHTS:-"runs/train/exp/weights/best.pt"}
ONNX_MODEL=${ONNX_MODEL:-"runs/train/exp/weights/best.onnx"}
COCO_DIR=${COCO_DIR:-"data/coco"}
SAMPLE_IMG=${SAMPLE_IMG:-"data/coco128/images/train2017/000000000009.jpg"}

echo "=================================================="
echo " YOLOv5 Full Pipeline"
echo "=================================================="

# ── Step 1: Setup ─────────────────────────────────────────
echo -e "\n[1/7] Environment setup..."
bash scripts/setup.sh

# ── Step 2: (Optional) Download COCO ─────────────────────
if [ "$SKIP_COCO_DOWNLOAD" -eq 0 ]; then
    echo -e "\n[2/7] Downloading COCO val2017..."
    bash scripts/download_coco.sh "$COCO_DIR"
else
    echo -e "\n[2/7] Skipping COCO download (SKIP_COCO_DOWNLOAD=1)"
fi

# ── Step 3: Train ─────────────────────────────────────────
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo -e "\n[3/7] Training..."
    bash scripts/train.sh
else
    echo -e "\n[3/7] Skipping training (SKIP_TRAIN=1)"
fi

# ── Step 4: Export to ONNX ────────────────────────────────
echo -e "\n[4/7] Exporting to ONNX..."
bash scripts/export.sh "$WEIGHTS"

# ── Step 5: Inference with BBox output ───────────────────
echo -e "\n[5/7] Running inference (ONNX — saves annotated images)..."
bash scripts/infer.sh "$ONNX_MODEL" "$SAMPLE_IMG" "results/inference"

# ── Step 6: MSC Comparison ────────────────────────────────
echo -e "\n[6/7] MSC Comparison — PyTorch vs ONNX..."
bash scripts/compare.sh "$WEIGHTS" "$ONNX_MODEL" "$SAMPLE_IMG" "results/compare"

# ── Step 7: COCO Evaluation ───────────────────────────────
if [ -d "$COCO_DIR/images/val2017" ]; then
    echo -e "\n[7/7] COCO full-dataset evaluation..."
    bash scripts/evaluate_coco.sh "$WEIGHTS" "$ONNX_MODEL" "$COCO_DIR" \
        "" "$DEVICE" "results/coco_eval.json"
else
    echo -e "\n[7/7] Skipping COCO eval (dataset not found at $COCO_DIR)"
    echo "  → Run: bash scripts/download_coco.sh  then re-run this pipeline"
fi

echo -e "\n=================================================="
echo " Pipeline complete!"
echo " Results:"
echo "   Annotated images : results/inference/"
echo "   MSC report       : results/compare/msc_report.json"
echo "   COCO eval        : results/coco_eval.json"
echo "=================================================="
