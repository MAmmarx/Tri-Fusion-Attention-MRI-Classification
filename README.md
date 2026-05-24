# Tri-Fusion CNN-Based Models with Attention for MRI Brain Tumor Classification

This repository contains the official PyTorch implementation and image preprocessing assets for the paper **"Tri-fusion CNN-based Models with Attention for MRI Brain Tumor Classification"**, accepted for publication at the **IEEE IMSA 2026 Conference**.

This project introduces a sophisticated deep learning framework that adaptively integrates feature representations from multiple pre-trained convolutional neural networks. By replacing traditional, rigid linear concatenation strategies with a learned dynamic weighting mechanism, the system minimizes spatial feature fatigue and eliminates computational data noise across heterogeneous diagnostic imaging.

**Academic Research & Implementation | Accepted to IEEE (IMSA 2026)**

> This framework and the accompanying research paper were developed and published during my **3rd year of undergraduate studies**, representing a contribution to the fields of Computer Vision, Deep Learning, and Automated Clinical Decision Support Systems.

## Overview
Manual interpretation of Magnetic Resonance Imaging (MRI) is inherently time-consuming, subjective, and prone to inter-observer variability. While separate deep learning backbones show promise, single-stream networks are bounded by architecture-specific biases that cannot comprehensively isolate localized tumor boundaries, clear tissue density anomalies, and multi-scale textural variations simultaneously.

To resolve these constraints, this project implements a **Tri-Fusion+Attention** network. The pipeline runs three parallel multi-scale streams—**VGG16**, **DenseNet121**, and **EfficientNetV2-S**—to extract independent, complementary deep features. A learned Attention Gate then automatically computes dynamic channel-wise importance weights per image scan, amplifying highly discriminative pathological regions while suppressing background artifacts or healthy tissue structures.

Tested on a completely balanced benchmark of 7,200 human brain MRI images across four classes (Glioma, Meningioma, Pituitary Tumor, and No Tumor), our Tri-Fusion architecture pushes diagnostic boundaries to a peak test accuracy of **95.06%**, outperforming standard baseline feature streams.

## Features
- **Multi-Backbone Extraction:** Extracts robust spatial textures via VGG16, ensures dense feature reuse via DenseNet121, and leverages compound global scaling via EfficientNetV2-S.
- **Dynamic Attention Gating:** Integrates a learned two-layer Multi-Layer Perceptron (MLP) Softmax gate to adaptively score feature maps instead of treating all channels uniformly.
- **Dimensional Projection Layers:** Maps diverse non-aligned backbone outputs ($4096$-dim, $1024$-dim, and $1280$-dim forces) into a normalized $512$-dimensional vector space before weighting operations.
- **Advanced Preprocessing (OpenCV Contour Pipeline):** Employs an automated, non-destructive brain contour cropping mask that isolates regions of interest, strips away background edge margins, and applies elastic augmentations safely.
- **Differential Learning Rates:** Utilizes isolated step-optimization configurations ($1 \times 10^{-5}$ for frozen feature extractors vs. $1 \times 10^{-4}$ for attention head metrics) to safeguard foundation weights.

## How It Works
- **Brain Contour Masking:** Grayscale conversions and Gaussian blurring filter the raw scan, finding absolute spatial coordinate extremes ($Top$, $Bottom$, $Left$, $Right$) to crop out empty background spaces.
- **Stream Projection:** Deep feature extractors pass pure numeric tensors into fully connected normalization layers to standardize vector dimensions safely.
- **Global Context Integration:** Fused vectors are concatenated into a $1536$-dimensional global context matrix, which serves as direct empirical feedback for the gating layer.
- **Dynamic Importance Allocation:** The Attention Gate derives precise scalars ($w_1$, $w_2$, $w_3$) where $\sum w_i = 1$, dynamically dampening or amplifying entire model outputs.
- **Harmonic Prediction:** Weighted features feed into a localized classification layer running Cross-Entropy Loss to output accurate diagnostic assignments.

## Performance Profiles

The framework demonstrates exceptional cross-class calibration, matching high macro precision with deep sensitivity:

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Harmonic F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| EfficientNetV2-S Baseline | 77.50% | 0.7810 | 0.7750 | 0.7668 |
| DenseNet121 Baseline | 82.94% | 0.8497 | 0.8294 | 0.8258 |
| VGG16 Baseline | 92.00% | 0.9234 | 0.9200 | 0.9179 |
| **Proposed Tri-Fusion** | 94.44% | 0.9486 | 0.9444 | 0.9431 |
| **Proposed Tri-Fusion + Attention** | **95.06%** | **0.9540** | **0.9506** | **0.9498** |

## Repository Architecture
```text
├── dataset/
│   ├── Training/             # 5,600 balanced image arrays (1,400 per class)
│   └── Testing/              # 1,600 balanced validation arrays (400 per class)
├── src/
│   ├── preprocessing.py      # Automated OpenCV brain contour cropping module
│   ├── fusion_models.py      # Linear Tri-Fusion and AttentionTriFusion network definitions
│   ├── evaluate.py           # Evaluation pipeline compiling Precision, Recall, F1, and AUC-ROC
│   └── train_fusion.py       # Differential learning loop utilizing step learning rate schedulers
└── Research_Project_2026.ipynb # Complete, self-contained Google Colab reproducibility notebook
