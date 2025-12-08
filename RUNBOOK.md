SYNC: PDV1_SYNC_2025-11-07_v2
# [1.1] Mở Codespaces & chuẩn bị phiên làm việc
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p artifacts
# [1.2] Nâng pip và cài libs tối thiểu
python -m pip install --upgrade pip
pip install "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" joblib tabulate
# (tabulate bắt buộc cho bước tạo report)
2) Lấy dữ liệu (GitHub Releases v1) 
   # [2.1] Lấy train.csv & test.csv từ Releases tag v1
python scripts/00_fetch_data.py \
  --source github \
  --owner BinBin868 \
  --repo PDv1_Logistic_Baseline \
  --release-tag v1

# [2.2] Kiểm tra nhanh
ls -l data/
3) Vệ sinh dữ liệu bắt buộc (tránh lỗi ép kiểu ở cột target)
( Cột target default_90d phải 0/1 và không có NA.
Sinh ra file sạch train_clean.csv, test_clean.csv.) 
python - <<'PY'
import pandas as pd, numpy as np

for split in ["train","test"]:
    df = pd.read_csv(f"data/{split}.csv", sep=None, engine="python")
    # ép kiểu target về {0,1}, loại NA
    df["default_90d"] = pd.to_numeric(df["default_90d"], errors="coerce").fillna(0).astype(int)
    # (tuỳ chọn) loại hàng NA ở target: df = df.dropna(subset=["default_90d"])
    # có thể xử lý thêm các cột số khác nếu cần:
    # for c in ["app_id","loan_amt",...]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(f"data/{split}_clean.csv", index=False)

print("[ok] wrote data/train_clean.csv & data/test_clean.csv")
PY
4) Train + Calibrate (Platt)
( Ví dụ hiệu chỉnh theo cal_months = 2024-11, 2024-12.
Có thể đặt trước s, b nếu muốn tái hiện kết quả (giữ AUC/KS do phép biến đổi đơn điệu).
# [4.1] Train + Platt
python scripts/10_train.py \
  --train data/train_clean.csv \
  --cal-months 2024-11 2024-12 \
  --platt-s 0.5 \
  --platt-b -0.996 \
  --mean-matching on

# Artifacts kỳ vọng:
#   artifacts/model.joblib
#   artifacts/pdv1_metadata.json  (ghi s, b, cal_months,...)
5) Score (TEST)
 ( SALT dùng để hash/ẩn danh id nếu cần (đặt tạm “demo_salt_123”).
Có thể bật “mean-matching” ở bước report (phần dưới).
export SALT="demo_salt_123"

python scripts/20_score.py \
  --model artifacts/model.joblib \
  --input data/test_clean.csv \
  --id-col app_id \
  --out artifacts/pdv1_test_scores.csv

ls -l artifacts/pdv1_test_scores.csv
6) Report (TEST) — AUC/KS/Brier + decile
python scripts/30_report.py \
  --scores artifacts/pdv1_test_scores.csv \
  --y-col default_90d \
  --out artifacts/report_test.md

# Mở xem trong Codespaces:
# Explorer (panel trái) → artifacts/ → mở report_test.md
7) (Tuỳ chọn) Score & Report HOLDOUT (Q4/2024)
( Thêm file data/holdout.csv (ví dụ gộp tháng 2024-11, 2024-12; ODR_current = bad rate thực tế)
python - <<'PY'
import pandas as pd
months = ["2024-11","2024-12"]
df = pd.read_csv("data/train_clean.csv", sep=None, engine="python")
mask = df["ym"].isin(months)
if "approved" in df.columns:
    mask = mask & (df["approved"]==1)
# Ép target 0/1 và loại NA ở target
df["default_90d"] = pd.to_numeric(df["default_90d"], errors="coerce")
df = df[df["default_90d"].isin([0,1]) & mask].copy()
print("Holdout shape:", df.shape, "| ODR (bad rate):", df["default_90d"].mean())
df.to_csv("data/holdout_clean.csv", index=False)
PY
# Sau đó: Score HOLDOUT bằng model đã train (+ Platt)
python scripts/20_score.py \
  --model artifacts/model.joblib \
  --input data/holdout_clean.csv \
  --id-col app_id \
  --out artifacts/pdv1_holdout_scores.csv
# Report HOLDOUT (AUC/KS/Brier + decile, p̄/ODR/wMAE)
python scripts/30_report.py \
  --scores artifacts/pdv1_holdout_scores.csv \
  --y-col default_90d \
  --out artifacts/report_holdout.md
8) Gợi ý “linh hoạt”
Đổi cửa sổ calibration: thay --cal-months ... ở Bước 4.
Override s,b: cập nhật --platt-s, --platt-b ở Bước 4; AUC/KS giữ nguyên (mapping đơn điệu).
Mean-matching: bật --mean-matching on (train hoặc report) để dịch logit hằng số sao cho p̄ ≈ ODR, giúp wMAE_decile giảm nhưng AUC/KS không đổi.
Tạo 2 báo cáo: report_test.md và report_holdout.md để so sánh AUC/KS/Brier, p̄ vs ODR, KS max (at decile k), wMAE_decile.
9) Deliverables kiểm tra
artifacts/model.joblib
artifacts/pdv1_metadata.json
artifacts/pdv1_test_scores.csv (+ tuỳ chọn pdv1_holdout_scores.csv)
artifacts/report_test.md

   
