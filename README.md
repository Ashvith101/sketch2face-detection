# Sketch-to-Face Generation using GAN in PyTorch

A PyTorch implementation of a Generative Adversarial Network (GAN) for converting facial sketches to photorealistic faces using the FS2K dataset.

## Overview
This project implements a sketch-to-face generation system using a conditional GAN architecture. It utilizes the FS2K dataset which contains 2,104 high-quality sketch-photo pairs drawn by professional artists.

## Features
- Robust dataset loader for FS2K dataset
- UNet-based generator with residual blocks
- PatchGAN discriminator for high-quality generation
- Support for multiple image formats
- Real-time training visualization
- Model checkpointing

## Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU support)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sketch-to-face-gan.git
cd sketch-to-face-gan

# Create and activate conda environment
conda create -n pytorch_env python=3.8
conda activate pytorch_env

# Install required packages
pip install torch torchvision pillow matplotlib tqdm
```

## Dataset
Download the FS2K dataset and organize it as follows:
```
FS2K/
├── photo/
│   ├── photo1/
│   ├── photo2/
│   └── photo3/
├── sketch/
│   ├── sketch1/
│   ├── sketch2/
│   └── sketch3/
├── anno_train.json
└── anno_test.json
```

## Training
```bash
python gan.py
```

## Model Architecture

### Generator
- UNet architecture with skip connections
- Input: 3-channel sketch (256×256)
- Output: 3-channel photo (256×256)
- Features:
  - 6 residual blocks
  - Instance normalization
  - LeakyReLU activation

### Discriminator
- PatchGAN architecture (15×15 output)
- Input: 6-channel concatenated sketch+photo
- 5 convolutional layers with instance norm

## Training Parameters
- Batch size: 8
- Learning rate: 0.0002
- Epochs: 50
- Loss: Combined adversarial + L1
- Optimizer: Adam (β1=0.5, β2=0.999)

## Results
Generated images are saved in the `results/` directory during training. Model checkpoints are saved every 10 epochs in the `models/` directory.

## Acknowledgments
- [FS2K Dataset](https://arxiv.org/abs/2203.15712)
- Based on Pix2Pix GAN architecture

## License
MIT License
