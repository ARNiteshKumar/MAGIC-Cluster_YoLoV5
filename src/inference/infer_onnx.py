#!/usr/bin/env python3
"""
ONNX Inference Script for YOLOv5
Performs inference using exported ONNX model with:
  - Bounding box post-processing (NMS)
  - Class label annotations drawn on output image
  - Output image saved to results folder
  - Latency benchmarking
"""

import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse
from pathlib import Path

# Shared helpers (NMS, drawing, COCO classes, etc.)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from infer import (
    COCO_CLASSES, CLASS_COLORS,
    preprocess_image, postprocess, draw_detections,
)


class YOLOv5ONNXInference:
    """YOLOv5 ONNX Inference Handler"""

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name : {self.input_name}")
        print(f"Output name: {self.output_name}")

    def preprocess(self, image_path):
        """Preprocess image for inference; returns (blob, orig_bgr)."""
        blob, orig = preprocess_image(str(image_path),
                                      img_size=self.input_shape[2])
        return blob, orig

    def warmup(self, num_iterations=5):
        """Warmup the model with dummy inputs."""
        print(f"Warming up model with {num_iterations} iterations...")
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(num_iterations):
            self.session.run([self.output_name],
                             {self.input_name: dummy_input})
        print("Warmup complete")

    def infer(self, image_path, output_dir=None, measure_latency=True):
        """
        Run inference on an image.

        Returns:
            detections : list of dicts with x1,y1,x2,y2,conf,class_id,class_name
            latency_ms : float or None
            output_path: Path where annotated image was saved (or None)
        """
        blob, orig = self.preprocess(image_path)
        h, w = orig.shape[:2]

        if measure_latency:
            start_time = time.perf_counter()

        raw = self.session.run([self.output_name],
                               {self.input_name: blob})[0]

        if measure_latency:
            latency_ms = (time.perf_counter() - start_time) * 1000
        else:
            latency_ms = None

        detections = postprocess(raw, (h, w),
                                 self.conf_threshold, self.iou_threshold)

        # ── Save annotated image ──────────────────────────────────────────────
        output_path = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            img_name = Path(image_path).stem
            out_path = out_dir / f"{img_name}_result.jpg"
            vis = draw_detections(orig, detections)
            cv2.imwrite(str(out_path), vis)
            output_path = out_path

        return detections, latency_ms, output_path

    def benchmark(self, image_path, num_runs=100):
        """Benchmark inference latency (no image saving)."""
        print(f"\nBenchmarking with {num_runs} runs...")
        blob, _ = self.preprocess(image_path)
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: blob})
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        stats = {
            "mean": float(latencies.mean()),
            "std": float(latencies.std()),
            "min": float(latencies.min()),
            "max": float(latencies.max()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }

        print(f"\n{'='*60}")
        print(f"Benchmark Results ({num_runs} runs)")
        print(f"{'='*60}")
        for k, v in stats.items():
            print(f"  {k.upper():<6}: {v:.2f} ms")
        print(f"{'='*60}\n")
        return stats


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 ONNX Inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output-dir", type=str,
                        default="results/inference/onnx",
                        help="Directory to save annotated output images")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IOU threshold for NMS")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save annotated output image")

    args = parser.parse_args()

    inferencer = YOLOv5ONNXInference(
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou)
    inferencer.warmup()

    if args.benchmark:
        inferencer.benchmark(args.image, num_runs=args.runs)
    else:
        output_dir = None if args.no_save else args.output_dir
        detections, latency, out_path = inferencer.infer(
            args.image, output_dir=output_dir)

        print(f"\n{'='*60}")
        print(f"Inference Complete")
        print(f"{'='*60}")
        print(f"Model   : {args.model}")
        print(f"Image   : {args.image}")
        print(f"Latency : {latency:.2f} ms")
        print(f"Detections: {len(detections)}")
        if detections:
            print(f"\n  {'Class':<22} {'Conf':>6}  BBox (x1,y1,x2,y2)")
            print(f"  {'-'*54}")
            for d in sorted(detections, key=lambda x: -x["conf"]):
                bbox = f"({d['x1']},{d['y1']},{d['x2']},{d['y2']})"
                print(f"  {d['class_name']:<22} {d['conf']:>6.3f}  {bbox}")
        if out_path:
            print(f"\nAnnotated image saved to: {out_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
