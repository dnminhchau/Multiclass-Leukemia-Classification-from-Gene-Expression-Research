# 🩸 Blood Cancer Prediction and Classification using Deep Learning
### Dự đoán và Phân loại Ung thư Máu dựa vào Mô hình Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In--Progress-yellow.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![UIT](https://img.shields.io/badge/University-UIT-blue.svg)](https://www.uit.edu.vn/)

> ⚠️ **Important Disclaimer:** This project is conducted for academic research purposes only. The output results are not intended to replace professional clinical diagnoses.

---

## 📖 Introduction

This research focuses on developing **Deep Learning** architectures to classify various types of blood cancer (Leukemia) using Gene Expression Profiling (GEP) data. The project addresses the challenges of high-dimensional data and aims to identify the most effective models by comparing traditional approaches with state-of-the-art architectures.

### Targeted Pathologies:

| Category | Full Name | Dataset Subtypes |
|------|-----------|--------------------------|
| **ALL** | Acute Lymphoblastic Leukemia | c-ALL/Pre-B-ALL without t(9;22), c-ALL/Pre-B-ALL with t(9;22), T-ALL, ALL with t(12;21), Pro-B-ALL with t(11q23)/MLL, ALL with hyperdiploid karyotype, ALL with t(1;19), mature B-ALL with t(8;14) |
| **AML** | Acute Myeloid Leukemia | AML with normal karyotype + other abnormalities, AML complex aberrant karyotype, AML with t(15;17), AML with t(8;21), AML with t(11q23)/MLL, AML with inv(16)/t(16;16) |
| **CLL** | Chronic Lymphocytic Leukemia | CLL |
| **CML** | Chronic Myeloid Leukemia | CML |
| **MDS** | Myelodysplastic Syndrome | MDS |
| **Other** | Non-leukemia / Healthy | Non-leukemia and healthy bone marrow |

---

## 🏗️ Research Architectures

The project evaluates and implements the following Deep Learning architectures:

1. **Baseline:** MLP (Multi-Layer Perceptron)
2. **Feature Extraction:** 1D-CNN (1D Convolutional Neural Network)
3. **Dimensionality Reduction:** Autoencoder (For data compression and denoising)
4. **Tabular Data SOTA:** TabTransformer & Hybrid CNN-Transformer

---

## 📊 Dataset

We utilize the normalized **GSE13204** dataset from the NCBI GEO system.

- **Link Dataset:** [NCBI GEO GSE13204](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13204)
- **Preprocessing Pipeline:** Data Cleaning → Log2 Transformation → Normalization → Feature Selection (LASSO/RF)

---

## 📁 Directory Structure

```bash
leukemia-gene-expression-dl/
├── data/
│   ├── raw/            # Original data from NCBI GEO
│   └── processed/      # Data after Feature Engineering
├── notebooks/          # Jupyter notebooks for model experimentation
├── src/                # Main source code (Models, Data Loaders, Trainers)
├── outputs/
│   ├── figures/        # Loss/Accuracy plots, Confusion Matrices
│   ├── metrics/        # Evaluation results (Precision, Recall, F1)
│   └── models/         # Model Checkpoints (.pth / .h5)
└── README.md
```

---

## 👥 Research Team

| No. | Full Name | Student ID | Email |
|:---:|-----------|:----:|-------|
| 1 | Doan Ngoc Minh Chau | 23520168 | 23520168@gm.uit.edu.vn |
| 2 | Do Hong Ngan | 23520994 | 23520994@gm.uit.edu.vn |

**Supervisor:** Dr. Phan Xuan Thien
**Institution:** University of Information Technology (UIT) — VNU-HCM
**Semester:** Semester 2 — Academic Year 2025–2026

---

## 💬 Feedback & Contact

If you have any suggestions or feedback, please feel free to reach out to the authors via the emails listed above or by opening an **Issue** directly in this repository.
