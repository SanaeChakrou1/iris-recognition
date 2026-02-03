# 👁️ Iris Recognition System — CASIA Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> **End-to-end biometric iris recognition pipeline** — from raw eye images to identity classification using the CASIA-Iris dataset and a fine-tuned ResNet50.

---

## Overview

This project implements a complete **iris recognition system** built on the [CASIA Iris Dataset](http://ignored-for-readme). The system identifies individuals by analysing the unique texture patterns of the human iris — a biometric trait that remains stable throughout a person's lifetime and is unique even between identical twins.

The pipeline covers every stage: eye validation → iris & pupil detection → rubber-sheet normalisation → deep-learning classification across **250 identities**.

---

## Pipeline

```
Raw Eye Image
      │
      ▼
┌─────────────┐
│  Eye        │  Validate image dimensions & convert to grayscale
│  Detection  │
└──────┬──────┘
       ▼
┌─────────────┐
│  Iris &     │  Hybrid: Hough Circles + Integro-Differential (Daugman)
│  Pupil      │  → detect pupil boundary (inner circle)
│  Detection  │  → detect iris boundary (outer circle)
└──────┬──────┘
       ▼
┌─────────────┐
│  Iris       │  Rubber-Sheet Model (GPU-accelerated via PyTorch)
│  Normal-    │  Unwrap annular iris region → fixed 64×512 rectangle
│  isation    │  + CLAHE contrast enhancement
└──────┬──────┘
       ▼
┌─────────────┐
│  Classifica-│  ResNet50 (ImageNet pre-trained)
│  tion       │  Phase 1: frozen backbone → train head
│             │  Phase 2: fine-tune last 40 layers
└──────┬──────┘
       ▼
  Identity Prediction  (Top-1 / Top-5)
```

---


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SanaeChakrou1/iris-recognition.git
cd iris-recognition
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


---

## Dataset

| Property | Value |
|---|---|
| Name | CASIA Iris Dataset |
| Total images | 20 000 |
| Subjects | 2 categories (Left / Right) |
| Format | PNG / JPG, near-infrared (NIR) |
| Classes used for training | 250 (top-N by sample count) |
| Train / Val split | 80 % / 20 % (stratified) |

The dataset images are **pre-cropped** to the eye region, so no face detection step is needed.

---

## Results

| Metric | Value |
|---|---|
| Top-1 Accuracy (sample) | 100 % |
| Top-5 Accuracy | tracked via `SparseTopKCategoricalAccuracy` |
| Detection Success Rate | reported per batch (visualised in notebook) |
| Classes | 250 |

Training curves (Phase 1 & Phase 2) and a full confusion matrix are generated automatically inside the notebook.

---

## Technologies

| Tool | Role |
|---|---|
| **OpenCV** | Image I/O, Hough Circles, filtering, CLAHE |
| **NumPy** | Array math, gradient computation |
| **PyTorch** | GPU-accelerated iris normalisation (`grid_sample`) |
| **TensorFlow / Keras** | ResNet50 model, training, evaluation |
| **scikit-learn** | Train/val split, classification report |
| **Matplotlib / Seaborn** | Visualisation & plotting |
| **tqdm** | Progress bars |

---


---

*Projet de reconnaissance biométrique par l'iris — Dataset CASIA*
