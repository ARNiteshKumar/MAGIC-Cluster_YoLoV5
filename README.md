# YOLOv5 Model Export Pipeline

A complete, production-ready pipeline for training, exporting, and deploying YOLOv5 object detection models with ONNX support.

## 🎯 Overview

This repository provides a structured workflow for:
- Training YOLOv5 models on custom datasets
- Exporting models to ONNX format (opset 17+)
- Running optimized inference with latency benchmarking
- Deploying models in production environments

## 📁 Repository Structure

```
yolov5-model-export/
├── src/
│   ├── data/              # Data processing utilities
│   ├── models/            # Model export scripts
│   ├── training/          # Training scripts
│   ├── inference/         # Inference scripts
│   └── utils/             # Helper utilities
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   └── train_config.yaml  # Training configuration
├── scripts/               # Automation scripts
│   ├── setup.sh          # Environment setup
│   ├── train.sh          # Training script
│   ├── export.sh         # Model export
│   ├── infer.sh          # Inference script
│   ├── benchmark.sh      # Benchmarking
│   └── run_pipeline.sh   # Complete pipeline
├── docs/                  # Documentation
│   ├── DATA_CARD.md      # Dataset documentation
│   ├── EVALUATION.md     # Evaluation report
│   └── DEPLOYMENT.md     # Deployment guide
├── artifacts/
│   ├── models/           # Trained model weights
│   └── exports/          # Exported ONNX models
├── data/
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── sample/           # Sample data
├── tests/                # Unit tests
├── Dockerfile            # Docker container definition
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── README.md            # This file
```

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the entire pipeline with a single command:

```bash
bash scripts/run_pipeline.sh
```

This will:
1. Set up the environment
2. Train the model
3. Export to ONNX
4. Run inference

### Option 2: Step-by-Step

#### 1. Setup Environment

```bash
# Using pip
pip install -r requirements.txt
bash scripts/setup.sh

# Using conda
conda env create -f environment.yml
conda activate yolov5-export
bash scripts/setup.sh
```

#### 2. Train Model

```bash
bash scripts/train.sh
```

Training artifacts will be saved to `runs/train/exp/`.

#### 3. Export to ONNX

```bash
bash scripts/export.sh runs/train/exp/weights/best.pt
```

The ONNX model will be saved alongside the weights file.

#### 4. Run Inference

```bash
bash scripts/infer.sh \
    runs/train/exp/weights/best.onnx \
    data/coco128/images/train2017/000000000009.jpg
```

#### 5. Benchmark Performance

```bash
bash scripts/benchmark.sh \
    runs/train/exp/weights/best.onnx \
    data/coco128/images/train2017/000000000009.jpg \
    100
```

## 🐳 Docker Usage

### Build Image

```bash
docker build -t yolov5-export:latest .
```

### Run Container

```bash
docker run -it --rm \
    --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/artifacts:/workspace/artifacts \
    yolov5-export:latest \
    bash scripts/run_pipeline.sh
```

## 📊 Model Performance

### Training Results
- Dataset: COCO128
- Model: YOLOv5s
- Epochs: 3
- Batch Size: 16
- Image Size: 640x640

### Inference Latency (GPU)
- Mean: ~266 ms
- P95: ~280 ms
- P99: ~290 ms

See [EVALUATION.md](docs/EVALUATION.md) for detailed metrics.

## 📝 Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Export settings
- Inference parameters

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

## 📚 Documentation

- [Data Card](docs/DATA_CARD.md) - Dataset documentation
- [Evaluation Report](docs/EVALUATION.md) - Model performance metrics
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - Base model implementation
- [COCO Dataset](https://cocodataset.org/) - Training dataset

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [www.linkedin.com/in/arniteshkumar].

## E-Mail: arniteshkumar@gmail.com

- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
