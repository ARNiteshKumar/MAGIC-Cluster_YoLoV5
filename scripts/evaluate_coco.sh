#!/bin/bash
# ============================================================
# COCO Full-Dataset Evaluation
# Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
# for both PyTorch and ONNX backends with accuracy comparison.
#
# Prerequisites:
#   1. Download COCO val2017 images:
#        bash scripts/download_coco.sh
#   2. Have either a .pt or .onnx model ready
#
# Usage:
#   # Both backends (recommended — generates comparison table)
#   bash scripts/evaluate_coco.sh [PT_MODEL] [ONNX_MODEL] [COCO_DIR]
#
#   # ONNX only
#   bash scripts/evaluate_coco.sh "" yolov5s.onnx data/coco
#
#   # PyTorch only
#   bash scripts/evaluate_coco.sh yolov5s.pt "" data/coco
#
#   # Quick test on first 500 images
#   bash scripts/evaluate_coco.sh yolov5s.pt yolov5s.onnx data/coco 500
# ============================================================

set -e

PT_MODEL=${1:-"yolov5s.pt"}
ONNX_MODEL=${2:-"artifacts/exports/yolov5s.onnx"}
COCO_DIR=${3:-"data/coco"}
MAX_IMAGES=${4:-""}          # leave empty for full dataset
DEVICE=${5:-"cpu"}
OUTPUT_JSON=${6:-"results/coco_eval.json"}

echo "=================================================="
echo " COCO Evaluation — YOLOv5 PyTorch & ONNX"
echo "=================================================="
echo "  COCO dir      : $COCO_DIR"
[ -n "$PT_MODEL" ]    && echo "  PyTorch model : $PT_MODEL"
[ -n "$ONNX_MODEL" ]  && echo "  ONNX model    : $ONNX_MODEL"
[ -n "$MAX_IMAGES" ]  && echo "  Max images    : $MAX_IMAGES"
echo "  Device        : $DEVICE"
echo "  Output JSON   : $OUTPUT_JSON"
echo "=================================================="

# Build argument list dynamically
ARGS="--data-dir $COCO_DIR --device $DEVICE --save-json $OUTPUT_JSON"
[ -n "$PT_MODEL" ]   && ARGS="$ARGS --pt-model $PT_MODEL"
[ -n "$ONNX_MODEL" ] && ARGS="$ARGS --onnx-model $ONNX_MODEL"
[ -n "$MAX_IMAGES" ] && ARGS="$ARGS --max-images $MAX_IMAGES"

python src/inference/coco_eval.py $ARGS

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_JSON"
