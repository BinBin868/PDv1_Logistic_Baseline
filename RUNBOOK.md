# RUNBOOK — PDv1 Logistic Baseline
**SYNC:** PDV1_SYNC_2025-11-07_v1  
**Goal:** Baseline PD (rank-order + calibration) — chạy end-to-end nhanh, không cần đường dẫn Drive cá nhân.

---

## 1) Chuẩn bị
- **Môi trường:** Conda (hoặc Mamba), Python 3.11.
- **Dữ liệu vào:** 
  - GitHub **Releases** (khuyến nghị): `train.csv`, `test.csv`, hoặc
  - Google Drive **FILE_ID** (public) của file Excel có 2 sheet: `train`, `test`.
- **Bảo mật:** Không commit dữ liệu thô/PII vào repo. ID sẽ được hash khi xuất kết quả.

---

## 2) Các bước chạy (copy từng lệnh)
### Bước 2.1 — Tạo môi trường
```bash
conda env create -f env/environment.yml
conda activate pdv1
