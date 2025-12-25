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
**7) Calibration sanity (TEST) — ODR, p̄, wMAE_decile, KS@decile**  
  SCORES="artifacts/pdv1_test_scores.csv" REPORT="artifacts/report_test.md" python - <<'PY'
import os, math
import pandas as pd
import numpy as np

scores_path = os.environ["SCORES"]
report_path = os.environ["REPORT"]

df = pd.read_csv(scores_path, sep=None, engine="python")

# --- find columns robustly ---
y_candidates = ["default_90d", "y", "target"]
p_candidates = ["p_mean","p_platt","p_cal","p_score","pd","prob","proba","score"]

y_col = next((c for c in y_candidates if c in df.columns), None)
if y_col is None:
    # fallback: find a binary-ish column
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            u = set(pd.Series(s.dropna().unique()).astype(float).tolist())
            if u.issubset({0.0,1.0}) and len(u) >= 2:
                y_col = c
                break
if y_col is None:
    raise SystemExit(f"[ERR] Cannot find target column. Expected one of {y_candidates} or a 0/1 column.")

p_col = next((c for c in p_candidates if c in df.columns), None)
if p_col is None:
    # fallback: choose numeric column in [0,1] excluding y
    num_cols = [c for c in df.columns if c != y_col and pd.api.types.is_numeric_dtype(df[c])]
    best = None
    for c in num_cols:
        s = df[c].dropna()
        if len(s)==0: 
            continue
        if (s.min() >= 0) and (s.max() <= 1):
            best = c
            break
    p_col = best
if p_col is None:
    raise SystemExit(f"[ERR] Cannot find probability column. Expected one of {p_candidates} or any numeric in [0,1].")

# --- clean ---
df = df[[y_col, p_col]].copy()
df = df.dropna(subset=[y_col, p_col])
df[y_col] = df[y_col].astype(int)

# --- overall metrics ---
odr = float(df[y_col].mean())
pbar = float(df[p_col].mean())
abs_gap = float(abs(pbar - odr))

# --- deciles (1 = highest risk) ---
df = df.sort_values(p_col, ascending=False).reset_index(drop=True)
df["decile"] = pd.qcut(df.index + 1, 10, labels=False) + 1

g = df.groupby("decile", as_index=False).agg(
    n=(y_col,"size"),
    odr=(y_col,"mean"),
    pbar=(p_col,"mean"),
)
g["abs_gap"] = (g["pbar"] - g["odr"]).abs()
wmae_decile = float((g["n"] * g["abs_gap"]).sum() / g["n"].sum())

# --- KS max by decile (decile of score-sorted bins) ---
tot_bad = df[y_col].sum()
tot_good = len(df) - tot_bad
if tot_bad == 0 or tot_good == 0:
    ks_max = float("nan"); ks_decile = None
else:
    df["bad"] = df[y_col]
    df["good"] = 1 - df[y_col]
    by = df.groupby("decile", as_index=False).agg(bad=("bad","sum"), good=("good","sum"))
    by["cum_bad"] = by["bad"].cumsum() / tot_bad
    by["cum_good"] = by["good"].cumsum() / tot_good
    by["ks"] = (by["cum_bad"] - by["cum_good"]).abs()
    idx = int(by["ks"].idxmax())
    ks_max = float(by.loc[idx, "ks"])
    ks_decile = int(by.loc[idx, "decile"])

# --- write markdown block ---
lines = []
lines.append("\n\n## Calibration sanity (p̄ vs ODR, wMAE_decile, KS@decile)\n")
lines.append(f"- Target column: `{y_col}` | Score column: `{p_col}`\n")
lines.append(f"- ODR (mean(y)): **{odr:.6f}**\n")
lines.append(f"- p̄ (mean(pred)): **{pbar:.6f}**\n")
lines.append(f"- |p̄ − ODR|: **{abs_gap:.6f}**\n")
lines.append(f"- wMAE_decile (weighted |p̄_dec − ODR_dec|): **{wmae_decile:.6f}**\n")
if ks_decile is None:
    lines.append(f"- KS max @ decile: **N/A** (all y are same)\n")
else:
    lines.append(f"- KS max @ decile **{ks_decile}**: **{ks_max:.6f}**\n")

lines.append("\nDecile table (1 = highest risk):\n\n")
# avoid relying on tabulate: format manually
lines.append("| decile | n | ODR | p̄ | |p̄-ODR| |\n")
lines.append("|---:|---:|---:|---:|---:|\n")
for _, r in g.iterrows():
    lines.append(f"| {int(r['decile'])} | {int(r['n'])} | {r['odr']:.6f} | {r['pbar']:.6f} | {r['abs_gap']:.6f} |\n")

block = "".join(lines)

print(block)  # show in terminal
with open(report_path, "a", encoding="utf-8") as f:
    f.write(block)
print(f"[ok] appended calibration sanity to {report_path}")
PY****
**9) Create data/holdout_clean.csv**
(ví dụ gộp tháng 2024-11, 2024-12; ODR_current = bad rate thực tế)
python - <<'PY'
import pandas as pd
months = ["2024-11","2024-12"]
df = pd.read_csv("data/train_clean.csv", sep=None, engine="python")
mask = df["ym"].isin(months)
if "approved" in df.columns:
    mask = mask & (df["approved"]==1)
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





**8) Calibration sanity (HOLDOUT) — ODR, p̄, wMAE_decile, KS@decile****
SCORES="artifacts/pdv1_holdout_scores.csv" REPORT="artifacts/report_holdout.md" python - <<'PY'
import os, math
import pandas as pd
import numpy as np

scores_path = os.environ["SCORES"]
report_path = os.environ["REPORT"]

df = pd.read_csv(scores_path, sep=None, engine="python")

# --- find columns robustly ---
y_candidates = ["default_90d", "y", "target"]
p_candidates = ["p_mean","p_platt","p_cal","p_score","pd","prob","proba","score"]

y_col = next((c for c in y_candidates if c in df.columns), None)
if y_col is None:
    # fallback: find a binary-ish column
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            u = set(pd.Series(s.dropna().unique()).astype(float).tolist())
            if u.issubset({0.0,1.0}) and len(u) >= 2:
                y_col = c
                break
if y_col is None:
    raise SystemExit(f"[ERR] Cannot find target column. Expected one of {y_candidates} or a 0/1 column.")

p_col = next((c for c in p_candidates if c in df.columns), None)
if p_col is None:
    # fallback: choose numeric column in [0,1] excluding y
    num_cols = [c for c in df.columns if c != y_col and pd.api.types.is_numeric_dtype(df[c])]
    best = None
    for c in num_cols:
        s = df[c].dropna()
        if len(s)==0: 
            continue
        if (s.min() >= 0) and (s.max() <= 1):
            best = c
            break
    p_col = best
if p_col is None:
    raise SystemExit(f"[ERR] Cannot find probability column. Expected one of {p_candidates} or any numeric in [0,1].")

# --- clean ---
df = df[[y_col, p_col]].copy()
df = df.dropna(subset=[y_col, p_col])
df[y_col] = df[y_col].astype(int)

# --- overall metrics ---
odr = float(df[y_col].mean())
pbar = float(df[p_col].mean())
abs_gap = float(abs(pbar - odr))

# --- deciles (1 = highest risk) ---
df = df.sort_values(p_col, ascending=False).reset_index(drop=True)
df["decile"] = pd.qcut(df.index + 1, 10, labels=False) + 1

g = df.groupby("decile", as_index=False).agg(
    n=(y_col,"size"),
    odr=(y_col,"mean"),
    pbar=(p_col,"mean"),
)
g["abs_gap"] = (g["pbar"] - g["odr"]).abs()
wmae_decile = float((g["n"] * g["abs_gap"]).sum() / g["n"].sum())

# --- KS max by decile (decile of score-sorted bins) ---
tot_bad = df[y_col].sum()
tot_good = len(df) - tot_bad
if tot_bad == 0 or tot_good == 0:
    ks_max = float("nan"); ks_decile = None
else:
    df["bad"] = df[y_col]
    df["good"] = 1 - df[y_col]
    by = df.groupby("decile", as_index=False).agg(bad=("bad","sum"), good=("good","sum"))
    by["cum_bad"] = by["bad"].cumsum() / tot_bad
    by["cum_good"] = by["good"].cumsum() / tot_good
    by["ks"] = (by["cum_bad"] - by["cum_good"]).abs()
    idx = int(by["ks"].idxmax())
    ks_max = float(by.loc[idx, "ks"])
    ks_decile = int(by.loc[idx, "decile"])

# --- write markdown block ---
lines = []
lines.append("\n\n## Calibration sanity (p̄ vs ODR, wMAE_decile, KS@decile)\n")
lines.append(f"- Target column: `{y_col}` | Score column: `{p_col}`\n")
lines.append(f"- ODR (mean(y)): **{odr:.6f}**\n")
lines.append(f"- p̄ (mean(pred)): **{pbar:.6f}**\n")
lines.append(f"- |p̄ − ODR|: **{abs_gap:.6f}**\n")
lines.append(f"- wMAE_decile (weighted |p̄_dec − ODR_dec|): **{wmae_decile:.6f}**\n")
if ks_decile is None:
    lines.append(f"- KS max @ decile: **N/A** (all y are same)\n")
else:
    lines.append(f"- KS max @ decile **{ks_decile}**: **{ks_max:.6f}**\n")

lines.append("\nDecile table (1 = highest risk):\n\n")
# avoid relying on tabulate: format manually
lines.append("| decile | n | ODR | p̄ | |p̄-ODR| |\n")
lines.append("|---:|---:|---:|---:|---:|\n")
for _, r in g.iterrows():
    lines.append(f"| {int(r['decile'])} | {int(r['n'])} | {r['odr']:.6f} | {r['pbar']:.6f} | {r['abs_gap']:.6f} |\n")

block = "".join(lines)

print(block)  # show in terminal
with open(report_path, "a", encoding="utf-8") as f:
    f.write(block)

print(f"[ok] appended calibration sanity to {report_path}")
PY

**10) Gợi ý “linh hoạt”**
Đổi cửa sổ calibration: thay --cal-months ... ở Bước 4.
Override s,b: cập nhật --platt-s, --platt-b ở Bước 4; AUC/KS giữ nguyên (mapping đơn điệu).
Mean-matching: bật --mean-matching on (train hoặc report) để dịch logit hằng số sao cho p̄ ≈ ODR, giúp wMAE_decile giảm nhưng AUC/KS không đổi.
Tạo 2 báo cáo: report_test.md và report_holdout.md để so sánh AUC/KS/Brier, p̄ vs ODR, KS max (at decile k), wMAE_decile.
11) Deliverables kiểm tra
artifacts/model.joblib
artifacts/pdv1_metadata.json
artifacts/pdv1_test_scores.csv (+ tuỳ chọn pdv1_holdout_scores.csv)
artifacts/report_test.md

   
