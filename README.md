# AGA-Net

# Adaptive Geometric-Attention Network for Two-Stage Lung Nodule Segmentation and Malignancy Classification in Federated Healthcare IoT Systems

## Description
This repository provides the official implementation of the **Adaptive Geometric-Attention Network (AGA-Net)** for automated lung nodule analysis from chest CT images. The proposed framework performs **two-stage learning**, where lung nodules are first segmented and subsequently classified as benign or malignant. The system is designed to be compatible with **federated healthcare IoT environments**, enabling privacy-preserving and distributed model training.

## Dataset Information
Experiments are conducted using the **LIDC-IDRI lung CT dataset** in slice-based format. The dataset is organized at patient, nodule, and slice levels, where each nodule includes multiple CT image slices and corresponding expert-annotated segmentation masks. Multi-annotator masks are supported to capture annotation variability.

Dataset exploration, verification, and preprocessing are handled using:
- `file-1.py` – Dataset loading and preprocessing pipeline  
- `file-2.py` – Exploratory data analysis and statistical inspection  

## Code Structure
- `aganet_implementation.py`  
  Implements the complete AGA-Net framework, including the geometric-constrained attention modules, segmentation network, classification network, training procedure, and evaluation metrics.

- `file-1.py`  
  Handles dataset loading, preprocessing operations, and data organization.

- `file-2.py`  
  Performs dataset analysis, visualization, and descriptive statistics for images and masks.

## Installation
1. Install Python (version ≥ 3.8).
2. Create and activate a virtual environment.
3. Install required dependencies:
   ```bash
   pip install torch numpy pandas matplotlib seaborn opencv-python pillow scikit-learn
## Step 1: Dataset Preparation

Download the LIDC-IDRI dataset and organize it according to the expected directory structure used in file-1.py and file-2.py.

## Step 2: Data Loading and Preprocessing
Run the dataset preprocessing and exploration scripts to verify data integrity:
```bash
python file-1.py
python file-2.py

# Methodology

The proposed pipeline includes:

CT image loading and slice extraction

Intensity normalization and spatial resizing

Geometric-aware attention-based segmentation

Feature-guided malignancy classification

Performance evaluation using segmentation and classification metrics

The design supports future integration into federated learning setups.

# Data Preprocessing

CT slices are normalized to ensure intensity consistency and resized to a fixed spatial resolution. Corresponding segmentation masks are aligned with input images, and invalid or empty slices are removed. The dataset is split into training, validation, and testing subsets to avoid data leakage.

## Requirements

Python 3.8+

PyTorch

NumPy

Pandas

OpenCV

Pillow

Matplotlib

Seaborn

scikit-learn (for evaluation metrics)

# Citation

If you use this code in your research, please cite:

The associated PeerJ Computer Science article.

The original LIDC-IDRI dataset publication.

# License

This repository is provided for academic and research use only. Please refer to the LICENSE file for detailed terms.

# Contribution Guidelines

Contributions are welcome. Please ensure reproducibility, clear documentation, and adherence to coding standards when submitting pull requests.
