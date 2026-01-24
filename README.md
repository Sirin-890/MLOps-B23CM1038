# MLOps Assignment 1 -



## Overview

This repository contains experiments comparing CNN architectures (ResNet18 and ResNet50) against classical machine learning approaches (SVM) on MNIST and FashionMNIST datasets. The experiments also include CPU vs GPU performance comparisons.

**Full Implementation:** [Colab Notebook](https://colab.research.google.com/drive/1wKxNy3JGgYKm7Uc3U3Mo9Bk7kH_GPgQF?usp=sharing)

---

## Experiment 1: Model Performance Comparison

### 1.1 MNIST Dataset Results

#### CNN Results (3 Epochs)

| Model | Optimizer | Learning Rate | Batch Size | Test Accuracy (%) | Time (ms) |
|-------|-----------|---------------|------------|-------------------|-----------|
| ResNet18 | SGD | 0.001 | 16 | 99.34 | 68,412 |
| ResNet50 | SGD | 0.001 | 16 | 99.08 | 131,114 |
| ResNet18 | SGD | 0.0001 | 16 | 97.91 | 71,290 |
| ResNet50 | SGD | 0.0001 | 16 | 96.74 | 130,402 |
| ResNet18 | Adam | 0.001 | 16 | 99.53 | 72,631 |
| ResNet50 | Adam | 0.001 | 16 | 99.29 | 137,044 |
| ResNet18 | Adam | 0.0001 | 32 | 99.61 | 60,388 |
| ResNet50 | Adam | 0.0001 | 32 | 99.22 | 123,011 |

#### CNN Results (5 Epochs)

| Model | Optimizer | Learning Rate | Batch Size | Test Accuracy (%) | Time (ms) |
|-------|-----------|---------------|------------|-------------------|-----------|
| ResNet18 | SGD | 0.001 | 16 | 99.41 | 113,402 |
| ResNet50 | SGD | 0.001 | 16 | 99.26 | 219,384 |
| ResNet18 | Adam | 0.001 | 16 | 99.71 | 119,844 |
| ResNet50 | Adam | 0.001 | 16 | 99.48 | 226,312 |
| ResNet18 | Adam | 0.0001 | 32 | 99.76 | 102,884 |
| ResNet50 | Adam | 0.0001 | 32 | 99.43 | 211,446 |

### 1.2 FashionMNIST Dataset Results

#### CNN Results (3 Epochs)

| Model | Optimizer | Learning Rate | Batch Size | Test Accuracy (%) | Time (ms) |
|-------|-----------|---------------|------------|-------------------|-----------|
| ResNet18 | SGD | 0.001 | 16 | 85.28 | 84,211 |
| ResNet50 | SGD | 0.001 | 16 | 79.84 | 133,611 |
| ResNet18 | SGD | 0.0001 | 16 | 77.12 | 72,884 |
| ResNet50 | SGD | 0.0001 | 16 | 62.14 | 131,774 |
| ResNet18 | Adam | 0.001 | 16 | 86.48 | 79,816 |
| ResNet50 | Adam | 0.001 | 16 | 83.24 | 150,388 |
| ResNet18 | Adam | 0.0001 | 32 | 86.22 | 61,128 |
| ResNet50 | Adam | 0.0001 | 32 | 82.56 | 122,006 |

#### CNN Results (5 Epochs)

| Model | Optimizer | Learning Rate | Batch Size | Test Accuracy (%) | Time (ms) |
|-------|-----------|---------------|------------|-------------------|-----------|
| ResNet18 | SGD | 0.001 | 16 | 87.42 | 102,314 |
| ResNet50 | SGD | 0.001 | 16 | 82.48 | 159,432 |
| ResNet18 | Adam | 0.001 | 16 | 89.96 | 106,428 |
| ResNet50 | Adam | 0.001 | 16 | 87.12 | 175,412 |
| ResNet18 | Adam | 0.0001 | 32 | 89.38 | 81,245 |
| ResNet50 | Adam | 0.0001 | 32 | 86.44 | 146,782 |

### 1.3 Pin Memory Configuration

**Experiment:** ResNet50 with Adam optimizer, batch size 32, learning rate 0.0001

- **pin_memory=false:** Training time changed to 80,388.21 ms with nearly same accuracy

### 1.4 SVM Baseline Experiments

| Dataset | Kernel | C | Accuracy (%) | Time (ms) |
|---------|--------|---|--------------|-----------|
| MNIST | Poly | 0.1 | 93.84 | 24,620 |
| MNIST | Poly | 1.0 | 91.98 | 13,214 |
| MNIST | Poly | 10 | 91.62 | 11,430 |
| MNIST | RBF | 0.1 | 94.72 | 20,114 |
| MNIST | RBF | 1.0 | 94.14 | 13,492 |
| MNIST | RBF | 10 | 94.96 | 12,103 |
| FashionMNIST | Poly | 0.1 | 79.48 | 18,120 |
| FashionMNIST | Poly | 1.0 | 84.92 | 12,236 |
| FashionMNIST | Poly | 10 | 86.84 | 11,084 |
| FashionMNIST | RBF | 0.1 | 83.14 | 19,486 |
| FashionMNIST | RBF | 1.0 | 89.12 | 11,830 |
| FashionMNIST | RBF | 10 | 90.64 | 10,618 |

---

## Experiment 2: CPU vs GPU Performance Comparison

| Device | Model | Optimizer | Batch Size | Test Acc (%) | Training Time (ms) | FLOPs |
|--------|-------|-----------|------------|--------------|-------------------|-------|
| CPU | ResNet18 | SGD | 16 | 82.10 | 8,593,745.54 | 1,823,526,912 |
| CPU | ResNet50 | SGD | 16 | 81.01 | 26,735,498.56 | 4,133,970,944 |
| CPU | ResNet18 | Adam | 16 | 87.20 | 9,342,563.67 | 1,823,526,912 |
| CPU | ResNet50 | Adam | 16 | 88.10 | 27,645,175.89 | 4,133,970,944 |
| GPU | ResNet18 | SGD | 16 | 84.58 | 332,970.71 | 1,823,526,912 |
| GPU | ResNet50 | SGD | 16 | 79.55 | 1,005,069.34 | 4,133,970,944 |
| GPU | ResNet18 | Adam | 16 | 90.50 | 346,891.90 | 1,823,526,912 |
| GPU | ResNet50 | Adam | 16 | 90.60 | 1,038,062.73 | 4,133,970,944 |

---

## Key Findings

### CNN vs SVM Performance
- **CNN-based models significantly outperform classical SVM classifiers** on both MNIST and FashionMNIST, highlighting their superior ability to learn discriminative image representations.

### Model Architecture
- **ResNet18 provides an excellent balance** between accuracy and computational efficiency, often matching or exceeding the performance of ResNet50 while requiring fewer resources, making it preferable for small to medium-scale datasets.

### Optimizer Comparison
- **The Adam optimizer consistently achieves faster convergence and higher accuracy** than SGD, with performance improving further when increasing training epochs from 3 to 5, particularly for the more complex FashionMNIST dataset.

### Hardware Performance
- **GPU training provides massive speedup** (approximately 25-26x faster) compared to CPU training while maintaining comparable or better accuracy.

---

## References

[1] Sirin Changulani, *MNIST and FashionMNIST Experiments*, Colab Notebook, 2026.  
[https://colab.research.google.com/drive/1wKxNy3JGgYKm7Uc3U3Mo9Bk7kH_GPgQF?usp=sharing](https://colab.research.google.com/drive/1wKxNy3JGgYKm7Uc3U3Mo9Bk7kH_GPgQF?usp=sharing)
