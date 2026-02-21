#!/bin/bash
# ============================================================
# Download COCO val2017 dataset for evaluation
# Downloads ~1GB images + annotations to data/coco/
# ============================================================

set -e

COCO_DIR=${1:-"data/coco"}
mkdir -p "$COCO_DIR/images"
mkdir -p "$COCO_DIR/annotations"

echo "Downloading COCO val2017 images (~1 GB)..."
wget -q --show-progress \
    http://images.cocodataset.org/zips/val2017.zip \
    -O "$COCO_DIR/val2017.zip"
unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR/images/"
rm "$COCO_DIR/val2017.zip"

echo "Downloading COCO annotations (~241 MB)..."
wget -q --show-progress \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
    -O "$COCO_DIR/annotations.zip"
unzip -q "$COCO_DIR/annotations.zip" -d "$COCO_DIR/"
rm "$COCO_DIR/annotations.zip"

echo ""
echo "COCO val2017 downloaded to: $COCO_DIR"
echo "  Images      : $COCO_DIR/images/val2017/"
echo "  Annotations : $COCO_DIR/annotations/instances_val2017.json"
