# Leukemia Gene Expression Classification

## Mục tiêu
Huấn luyện mô hình Deep Learning (MLP) để phân loại các nhóm leukemia từ dữ liệu gene expression.

## Dữ liệu
Đặt file vào:
- data/raw/GSE13204.csv
- data/raw/GSE13204_label_summary.csv

## Cách chạy
1. Tạo môi trường ảo
2. Cài thư viện:
   pip install -r requirements.txt
3. Mở Jupyter Notebook
4. Chạy file:
   notebooks/01_train_mlp_gse13204.ipynb

## Output
- outputs/figures: biểu đồ
- outputs/metrics: file csv kết quả
- outputs/models: model và object tiền xử lý