#!/bin/bash
# ============================================================
# Inference script — YOLOv5 PyTorch / ONNX
#
# Usage:
#   bash scripts/infer.sh [MODEL] [SOURCE] [OUTPUT_DIR] [CONF] [IOU]
#
# Examples:
#   # ONNX inference
#   bash scripts/infer.sh yolov5s.onnx data/images/ results/inference
#
#   # PyTorch inference
#   bash scripts/infer.sh yolov5s.pt  data/images/ results/inference
#
#   # Single image
#   bash scripts/infer.sh yolov5s.onnx data/sample/dog.jpg results/inference
# ============================================================

set -e

MODEL=${1:-"runs/train/exp/weights/best.onnx"}
SOURCE=${2:-"data/coco128/images/train2017/"}
OUTPUT_DIR=${3:-"results/inference"}
CONF=${4:-0.25}
IOU=${5:-0.45}

echo "=================================================="
echo " YOLOv5 Inference"
echo "=================================================="
echo "  Model      : $MODEL"
echo "  Source     : $SOURCE"
echo "  Output dir : $OUTPUT_DIR"
echo "  Confidence : $CONF"
echo "  IoU        : $IOU"
echo "=================================================="

python src/inference/infer.py \
    --model "$MODEL" \
    --source "$SOURCE" \
    --output-dir "$OUTPUT_DIR" \
    --conf "$CONF" \
    --iou "$IOU" \
    --benchmark \
    --save-json "$OUTPUT_DIR/detections.json"

echo ""
echo "Done. Results saved to: $OUTPUT_DIR"
