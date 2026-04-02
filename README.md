# 🩸 Blood Cancer Prediction and Classification using Deep Learning
### Dự đoán và Phân loại Ung thư Máu dựa vào Mô hình Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In--Progress-yellow.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![UIT](https://img.shields.io/badge/University-UIT-blue.svg)](https://www.uit.edu.vn/)

> ⚠️ **Lưu ý Quan trọng:** Đây là đồ án phục vụ mục đích nghiên cứu học thuật. Các kết quả đầu ra không được sử dụng để thay thế cho các chẩn đoán lâm sàng chuyên sâu.

---

## 📖 Giới thiệu

Đề tài tập trung xây dựng các kiến trúc **Deep Learning** để phân loại các thể ung thư máu (Leukemia) từ dữ liệu biểu hiện gen (Gene Expression Profiling). Dự án giải quyết bài toán dữ liệu có số chiều cực lớn (High-dimensional) và tìm ra mô hình tối ưu nhất giữa các phương pháp truyền thống và hiện đại.

### Các loại bệnh lý tập trung:

| Loại | Tên đầy đủ | Các Subtype trong dataset |
|------|-----------|--------------------------|
| **ALL** | Acute Lymphoblastic Leukemia | c-ALL/Pre-B-ALL without t(9;22), c-ALL/Pre-B-ALL with t(9;22), T-ALL, ALL with t(12;21), Pro-B-ALL with t(11q23)/MLL, ALL with hyperdiploid karyotype, ALL with t(1;19), mature B-ALL with t(8;14) |
| **AML** | Acute Myeloid Leukemia | AML with normal karyotype + other abnormalities, AML complex aberrant karyotype, AML with t(15;17), AML with t(8;21), AML with t(11q23)/MLL, AML with inv(16)/t(16;16) |
| **CLL** | Chronic Lymphocytic Leukemia | CLL |
| **CML** | Chronic Myeloid Leukemia | CML |
| **MDS** | Myelodysplastic Syndrome | MDS |
| **Other** | Non-leukemia / Healthy | Non-leukemia and healthy bone marrow |

---

## 🏗️ Kiến trúc Mô hình nghiên cứu

Nhóm thực hiện so sánh và thực nghiệm trên các kiến trúc:

1. **Baseline:** MLP (Multi-Layer Perceptron)
2. **Feature Extraction:** 1D-CNN (Mạng tích chập 1 chiều)
3. **Dimensionality Reduction:** Autoencoder (Nén và khử nhiễu dữ liệu)
4. **Tabular Data SOTA:** TabTransformer & Hybrid CNN-Transformer

---

## 📊 Dữ liệu (Dataset)

Sử dụng bộ dữ liệu chuẩn hóa **GSE13204** từ hệ thống NCBI GEO.

- **Link Dataset:** [NCBI GEO GSE13204](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13204)
- **Tiền xử lý:** Data Cleaning → Log2 Transformation → Normalization → Feature Selection (LASSO/RF)

---

## 📁 Cấu trúc thư mục

```bash
leukemia-gene-expression-dl/
├── data/
│   ├── raw/            # Dữ liệu gốc từ NCBI GEO
│   └── processed/      # Dữ liệu sau khi xử lý Feature Engineering
├── notebooks/          # Jupyter notebooks thử nghiệm từng mô hình
├── src/                # Source code chính (Model, Data Loader, Trainer)
├── outputs/
│   ├── figures/        # Biểu đồ Loss, Accuracy, Confusion Matrix
│   ├── metrics/        # Kết quả Evaluation (Precision, Recall, F1)
│   └── models/         # Model Checkpoints (.pth / .h5)
└── README.md
```

---

## 👥 Nhóm thực hiện

| STT | Họ và tên | MSSV | Email |
|:---:|-----------|:----:|-------|
| 1 | Đoàn Ngọc Minh Châu | 23520168 | 23520168@gm.uit.edu.vn |
| 2 | Đỗ Hồng Ngân | 23520994 | 23520994@gm.uit.edu.vn |

**Giảng viên hướng dẫn:** TS. Phan Xuân Thiện  
**Trường:** Đại học Công nghệ Thông tin — ĐHQG TP. HCM  
**Học kỳ:** 2 — Năm học 2025–2026

---

## 💬 Góp ý & Liên hệ

Nếu bạn có bất kỳ ý kiến đóng góp hoặc phản hồi nào, vui lòng liên hệ nhóm tác giả qua email ở bảng trên hoặc mở Issue trực tiếp trên repo này.
