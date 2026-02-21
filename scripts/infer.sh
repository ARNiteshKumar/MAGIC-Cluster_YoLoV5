#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Unified inference runner — supports single-image and COCO evaluation modes
#
# Usage:
#   bash scripts/infer.sh [--eval] [extra args passed to infer.py]
#
# Examples:
#   # Single image (PyTorch + ONNX, saves bbox images to results/bbox_outputs/)
#   bash scripts/infer.sh \
#       --pt-weights  artifacts/models/yolov5s.pt \
#       --onnx-weights artifacts/exports/yolov5s.onnx \
#       --image data/coco128/images/train2017/000000000009.jpg
#
#   # COCO val2017 evaluation
#   bash scripts/infer.sh --eval \
#       --pt-weights  artifacts/models/yolov5s.pt \
#       --onnx-weights artifacts/exports/yolov5s.onnx \
#       --coco-dir data/coco \
#       --num-eval-images 5000
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=========================================="
echo "  YOLOv5 Unified Inference"
echo "=========================================="

# Default paths (override via CLI args)
PT_WEIGHTS=${PT_WEIGHTS:-"artifacts/models/yolov5s.pt"}
ONNX_WEIGHTS=${ONNX_WEIGHTS:-"artifacts/exports/yolov5s.onnx"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/bbox_outputs"}
IMAGE=${IMAGE:-"data/coco128/images/train2017/000000000009.jpg"}

python src/inference/infer.py \
    --pt-weights   "$PT_WEIGHTS" \
    --onnx-weights "$ONNX_WEIGHTS" \
    --output-dir   "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Done! Annotated images saved to: $OUTPUT_DIR"
