#!/bin/bash
# ============================================================
# MSC Comparison — PyTorch vs ONNX on same input
# Runs both backends, draws BBoxes, reports accuracy baseline
#
# Usage:
#   bash scripts/compare.sh [PT_MODEL] [ONNX_MODEL] [SOURCE] [OUTPUT_DIR]
#
# Example:
#   bash scripts/compare.sh \
#       yolov5s.pt \
#       artifacts/exports/yolov5s.onnx \
#       data/images/ \
#       results/compare
# ============================================================

set -e

PT_MODEL=${1:-"yolov5s.pt"}
ONNX_MODEL=${2:-"artifacts/exports/yolov5s.onnx"}
SOURCE=${3:-"data/coco128/images/train2017/"}
OUTPUT_DIR=${4:-"results/compare"}
CONF=${5:-0.25}
IOU=${6:-0.45}

echo "=================================================="
echo " YOLOv5 MSC Comparison: PyTorch vs ONNX"
echo "=================================================="
echo "  PyTorch model : $PT_MODEL"
echo "  ONNX model    : $ONNX_MODEL"
echo "  Source        : $SOURCE"
echo "  Output dir    : $OUTPUT_DIR"
echo "=================================================="

python src/inference/infer.py \
    --pt-model "$PT_MODEL" \
    --onnx-model "$ONNX_MODEL" \
    --source "$SOURCE" \
    --output-dir "$OUTPUT_DIR" \
    --conf "$CONF" \
    --iou "$IOU" \
    --compare \
    --benchmark \
    --save-json "$OUTPUT_DIR/msc_report.json"

echo ""
echo "MSC comparison complete. Report saved to: $OUTPUT_DIR/msc_report.json"
