#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Download COCO val2017 images + annotations for full evaluation
# Total size: ~1 GB images + ~240 MB annotations
# ─────────────────────────────────────────────────────────────────────────────
set -e

DEST="${1:-data/coco}"
mkdir -p "$DEST/images" "$DEST/annotations"

echo "Downloading COCO val2017 images (~1 GB)..."
wget -q --show-progress -O "$DEST/val2017.zip" \
    http://images.cocodataset.org/zips/val2017.zip
unzip -q "$DEST/val2017.zip" -d "$DEST/images/"
rm "$DEST/val2017.zip"

echo "Downloading COCO 2017 annotations (~240 MB)..."
wget -q --show-progress -O "$DEST/annotations_trainval2017.zip" \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q "$DEST/annotations_trainval2017.zip" -d "$DEST/"
rm "$DEST/annotations_trainval2017.zip"

echo ""
echo "COCO val2017 ready at: $DEST"
echo "  images : $DEST/images/val2017/"
echo "  annots : $DEST/annotations/instances_val2017.json"
