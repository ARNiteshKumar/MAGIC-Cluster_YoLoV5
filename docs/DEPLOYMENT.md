# Deployment Guide

## Overview

This guide provides instructions for deploying YOLOv5 ONNX models in various production environments.

## Prerequisites

### Minimum Requirements
- **CPU:** 2+ cores
- **RAM:** 4GB+
- **Storage:** 1GB+
- **OS:** Linux, macOS, Windows

### Software Requirements
- Python 3.8+
- ONNX Runtime 1.15+
- OpenCV 4.5+
- NumPy 1.21+

## Quick Start

### Local Deployment
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/yolov5-model-export.git
cd yolov5-model-export

# Setup environment
bash scripts/setup.sh

# Run pipeline
bash scripts/run_pipeline.sh
```

## Deployment Options

### 1. CPU Deployment (Universal)

**Pros:** Works anywhere, no special hardware  
**Cons:** Slower inference (~266ms)  
**Use Cases:** Development, low-traffic apps
```bash
python src/inference/infer_onnx.py \
    --model model.onnx \
    --image test.jpg
```

### 2. GPU Deployment (Recommended)

**Pros:** Fast inference (~10-20ms), high throughput  
**Cons:** Requires NVIDIA GPU  
**Use Cases:** Production APIs, real-time apps

Install GPU-enabled ONNX Runtime:
```bash
pip install onnxruntime-gpu
```

### 3. Docker Deployment
```bash
# Build image
docker build -t yolov5-export:latest .

# Run container
docker run -it --rm \
    --gpus all \
    -v $(pwd)/data:/workspace/data \
    yolov5-export:latest \
    bash scripts/run_pipeline.sh
```

## Cloud Deployment

### AWS EC2
1. Launch EC2 instance (g4dn.xlarge for GPU or c5.2xlarge for CPU)
2. Install dependencies: `bash scripts/setup.sh`
3. Run inference: `python src/inference/infer_onnx.py --model model.onnx --image test.jpg`

### Google Cloud Platform
Similar to AWS, use Compute Engine with appropriate instance type.

### Azure
Use Virtual Machine with appropriate configuration.

## API Deployment

### FastAPI Example
```python
from fastapi import FastAPI, File, UploadFile
from src.inference.infer_onnx import YOLOv5ONNXInference

app = FastAPI()
model = YOLOv5ONNXInference("model.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Process image and run inference
    output, latency = model.infer(image_path)
    return {"latency_ms": latency, "detections": output.tolist()}
```

Run server:
```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Performance Optimization

### 1. Model Quantization (INT8)
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QUInt8
)
```
**Expected speedup:** 2-4x on CPU

### 2. Batch Processing
Process multiple images in parallel for better throughput.

### 3. TensorRT (NVIDIA GPUs)
```bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```
**Expected speedup:** 5-10x on GPU

## Monitoring

### Basic Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Inference latency: {latency:.2f}ms")
```

### Health Checks
```python
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
```

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce batch size
- Use INT8 quantization

**Slow Inference:**
- Enable all optimizations
- Use GPU if available
- Profile with `cProfile`

**Model Not Loading:**
```bash
# Verify ONNX file
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

## Best Practices

### Security
1. Validate input images
2. Implement rate limiting
3. Add authentication

### Scalability
1. Use load balancers
2. Configure auto-scaling
3. Implement caching

### Maintenance
1. Track model versions
2. Set up monitoring alerts
3. Plan rollback strategy

## Support

For issues or questions:
- 📖 Check documentation
- 🐛 Open GitHub issue
- 📧 Contact support

---

**Last Updated:** 2026 
**Version:** 1.0
```

3. **Save:** Ctrl+S

---

### FILE 14: `artifacts/.gitkeep`

This creates the artifacts folder structure.

1. **New file:** `artifacts/.gitkeep`
2. **Paste:** (leave empty or type one line)
```
# Placeholder for artifacts directory
```
3. **Save:** Ctrl+S

---


1. **Click the Source Control icon** (left sidebar - 3rd icon, looks like branches)
2. You'll see **20 changes** listed
3. **At the top**, in the message box, type:
```

   Initial commit: Complete YOLOv5 model export pipeline with training, inference, and documentation
