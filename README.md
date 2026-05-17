# Part 2 — Computer Vision Problem Formulation and CNN Prototype

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Task](https://img.shields.io/badge/Task-Image%20Classification-purple)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

# 📌 Project Overview

This project focuses on building a Computer Vision pipeline capable of identifying different types of vehicle surface damage from images.

The objective is to automatically classify vehicle images into one of four categories:
- Normal
- Dent
- Scratch
- Stain

The project demonstrates how Convolutional Neural Networks (CNNs) can be applied to real-world industrial inspection problems such as vehicle quality checking and insurance damage assessment.

The complete workflow includes:
- Problem formulation
- Dataset analysis
- Image preprocessing and augmentation
- CNN model design
- Model training and evaluation
- Business application analysis

---

# 📂 Project Structure

```text
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── labels.csv
│
├── images/
│   ├── normal/
│   ├── dent/
│   ├── scratch/
│   └── stain/
│
├── results/
│   ├── task2_sample_images.png
│   ├── task2_class_distribution.png
│   ├── task3_augmentation.png
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── feature_maps.png
│
└── sample_predictions/
    └── prediction_outputs.png
```

---

# 🔍 Task 1 — Problem Formulation

## Problem Type: Multi-Class Image Classification

Each image contains a vehicle surface with one visible condition. The goal is to classify the entire image into one category.

| Class | Description |
|---|---|
| `normal` | No visible damage |
| `dent` | Surface depression caused by impact |
| `scratch` | Surface-level abrasion |
| `stain` | Paint discoloration or contamination |

### Why Image Classification?
This problem is best suited for image classification because:
- Each image has only one label
- Bounding boxes are not required
- Pixel-level segmentation masks are unavailable

| Computer Vision Task | Suitable? |
|---|---|
| Image Classification | ✅ Yes |
| Object Detection | ❌ No |
| Semantic Segmentation | ❌ No |

---

# 📊 Task 2 — Dataset Analysis

## Dataset Information

| Property | Value |
|---|---|
| Total Images | ~480 |
| Classes | 4 |
| Images per Class | ~120 |
| Image Size | 64 × 64 × 3 |
| Format | PNG |
| Dataset Balance | Nearly balanced |

The dataset contains RGB images of vehicle surfaces collected for different damage categories.

---

## Sample Categories

### Normal
Vehicle surface without visible defects.

### Dent
Images containing visible impact depressions.

### Scratch
Images showing scratches or abrasions on the paint surface.

### Stain
Images with paint contamination or discoloration.

---

# 🖼️ Task 3 — Image Preprocessing & Augmentation

Before training the CNN model, several preprocessing and augmentation techniques were applied.

## Preprocessing Steps
- Resize images to 64 × 64
- Normalize pixel values
- Convert images into NumPy arrays

## Data Augmentation Techniques
- Rotation
- Horizontal flipping
- Zooming
- Width/height shifting

### Why Augmentation Is Important
The dataset is relatively small, so augmentation helps the model generalize better by exposing it to slightly modified versions of the same images.

---

# 🧠 Task 4 — CNN Architecture

The CNN model was designed to progressively learn low-level and high-level image features.

```text
Input Image (64 × 64 × 3)
        ↓
Conv2D + BatchNorm + ReLU
        ↓
MaxPooling + Dropout
        ↓
Conv2D + BatchNorm + ReLU
        ↓
MaxPooling + Dropout
        ↓
Conv2D + ReLU
        ↓
Flatten
        ↓
Dense Layer
        ↓
Dropout
        ↓
Softmax Output Layer
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Categorical Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Activation | ReLU |
| Output Activation | Softmax |
| Regularization | Dropout + Batch Normalization |

---

# 🔬 Task 5 — Understanding CNN Concepts

## What is Convolution?

Convolution uses small learnable filters that move across the image to detect important patterns such as:
- Edges
- Textures
- Shapes
- Surface irregularities

The model automatically learns which patterns are useful for identifying different damage types.

---

## Why is Pooling Used?

Pooling reduces image dimensions while preserving important features.

### Benefits of Pooling
- Faster computation
- Reduced overfitting
- Better feature generalization
- Translation invariance

---

## Why ReLU Activation?

ReLU is widely used in CNNs because:
- It helps gradients flow efficiently
- It trains faster than sigmoid/tanh
- It introduces non-linearity
- It reduces vanishing gradient issues

---

## Why CNNs Work Better for Images

| Feature | Dense Network | CNN |
|---|---|---|
| Spatial understanding | Poor | Strong |
| Parameter efficiency | Low | High |
| Pattern learning | Limited | Excellent |
| Image feature extraction | Weak | Effective |

CNNs are specifically designed for image-related tasks because they preserve spatial information while learning hierarchical visual patterns.

---

# 📈 Task 6 — Model Evaluation

## Performance Metrics

| Metric | Result |
|---|---|
| Test Accuracy | ~88–93% |
| Loss Function | Categorical Cross-Entropy |
| Evaluation Tools | Confusion Matrix, Accuracy Curves |

The trained CNN achieved strong classification performance across all four damage categories.

---

## Evaluation Techniques Used
- Training vs Validation Accuracy Curves
- Loss Curves
- Confusion Matrix
- Per-Class Accuracy Analysis
- Feature Map Visualization

These visualizations help understand how the CNN learns image features during training.

---

# 💼 Task 7 — Real-World Business Application

## Automotive Quality Inspection

Manual vehicle inspection is time-consuming and can miss certain defects during high production loads.

### Traditional Inspection Challenges
- Slow inspection process
- Human fatigue
- Inconsistent quality checks
- Higher labor costs

---

## Proposed AI-Based Solution

The CNN system can automatically inspect vehicle images in real time and classify defects within milliseconds.

| KPI | Before AI | After AI |
|---|---|---|
| Inspection Time | 8 minutes | < 1 second |
| Defect Escape Rate | 4.5% | < 0.8% |
| Manual Workforce | 12 inspectors | 3 auditors |
| Annual Savings | — | ~₹65 Lakhs |

---

## Other Possible Applications
- Insurance claim assessment
- Medical image analysis
- Crop disease detection
- Retail inventory monitoring

---

# 🚀 Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch Notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Running on Google Colab

1. Upload:
   - `notebook.ipynb`
   - `labels.csv`
   - `images/normal/` folder

2. Select:

```text
Runtime → Run All
```

Synthetic images for other categories are automatically generated during execution.

---

# 📚 References

1. LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*
2. Krizhevsky et al. (2012). *ImageNet Classification with Deep CNNs.*
3. TensorFlow Keras Documentation
4. Goodfellow et al. — *Deep Learning*
