# PDv1 — Probability of Default (PD) Model Card (v1)

**Repository:** PDv1_Logistic_Baseline  
**Owner:** BinBin868  
**Build/Run guide:** `RUNBOOK.md`  
**Artifacts output folder:** `artifacts/`  
**Last updated (sync):** PDV1_SYNC_2025-11-07_v2  

---

## 1) Model overview (Mô hình này là gì?)

**PD (Probability of Default)** là xác suất một khoản vay/khách hàng sẽ **default** trong một “cửa sổ thời gian” đã định nghĩa.  
Trong repo này, mô hình dự báo **PD cho default_90d** (default trong 90 ngày).

- **Model family:** Logistic Regression (baseline, dễ giải thích, dễ kiểm soát)
- **Calibration:** Platt (sigmoid) với tham số `s` và `b`
- **Optional alignment:** Mean-matching (dịch mức PD để mean(PD) gần bằng ODR thực tế)

**Mục tiêu nghiệp vụ:**  
- Xếp hạng rủi ro (risk ranking) để hỗ trợ quyết định tín dụng/giám sát danh mục
- Tạo **PD score** nhất quán, có kiểm soát, có báo cáo (report) phục vụ review như “production-like”

---

## 2) Intended use (Dùng để làm gì?)

### Dùng đúng (Recommended)
- **Pre-screen / scorecard hỗ trợ quyết định cấp tín dụng**: kết hợp PD với policy (income, DTI, fraud rules, KYC…)
- **Risk-based pricing** (nếu tổ chức cho phép): khách có PD cao → pricing/limit khác
- **Portfolio monitoring**: theo dõi shift rủi ro theo thời gian/kênh (kết hợp monitoring KPI)

### Không dùng / dùng thận trọng
- Không dùng như “quyết định duy nhất” (single point of truth). PD chỉ là 1 thành phần.
- Không dùng ngoài “population” mô hình được fit (ví dụ sản phẩm khác, kênh khác, luật chơi khác) khi chưa đánh giá drift.

---

## 3) Target definition (Định nghĩa biến mục tiêu)

- **Target column:** `default_90d`
- **Meaning:** 1 = default trong 90 ngày, 0 = không default trong 90 ngày
- **Data quality rule (trong RUNBOOK):** ép kiểu về 0/1, xử lý NA để tránh lỗi pipeline

---

## 4) Population & splits (Đối tượng áp dụng & chia tập)

### Population
- Áp dụng cho các record trong dữ liệu scoring có schema tương thích `train_clean.csv / test_clean.csv`.
- Nếu có cột `approved`, RUNBOOK có ví dụ tạo HOLDOUT với điều kiện `approved == 1` (tùy dữ liệu).

### Splits trong repo
- **TRAIN:** `data/train_clean.csv`
- **TEST:** `data/test_clean.csv`
- **HOLDOUT (optional, demo production-check):** `data/holdout_clean.csv`  
  - HOLDOUT được tạo từ TRAIN theo `CAL_MONTHS` (ví dụ `2024-11`, `2024-12`) để kiểm tra “ổn định theo thời gian” / “out-of-time-like”.

> Ghi chú: HOLDOUT trong repo hiện được tạo “từ TRAIN theo tháng” để demo luồng & kiểm tra. Nếu bạn có dữ liệu out-of-time thật thì thay file này bằng dữ liệu thật sẽ giá trị hơn.

---

## 5) Model inputs (Biến đầu vào)

- **ID column:** `app_id` (dùng để định danh record khi scoring)
- **Time column (nếu có):** `ym` (dùng để chọn calibration window / holdout months)
- **Features:** lấy từ các cột còn lại trong dataset (trừ id/target), được xử lý trong pipeline của `scripts/10_train.py`.

> Repo tập trung vào “baseline production flow”: data → train → score → report.  
> Với các job risk thực tế, bạn sẽ bổ sung thêm: feature list rõ ràng, missing policy, outlier policy, leakage checks theo business.

---

## 6) Training & calibration logic (Logic huấn luyện & hiệu chỉnh)

### 6.1 Logistic Regression (baseline)
- Mục tiêu: tạo **risk ranking** ổn định, dễ giải thích.

### 6.2 Platt calibration (sigmoid)
- Dùng để “hiệu chỉnh xác suất” sao cho PD gần hơn với bad rate thực tế ở một calibration window.
- Tham số được lưu trong: `artifacts/pdv1_metadata.json`
  - `platt_s`, `platt_b`, `cal_months`, trạng thái mean-matching…

### 6.3 Mean-matching (optional)
- Mục tiêu: làm **mean(PD)** gần **ODR** (Observed Default Rate) của tập đánh giá.
- Ý nghĩa nghiệp vụ: giảm lệch mức PD (level bias) mà vẫn giữ **ranking** (AUC/KS thường không đổi nhiều vì biến đổi đơn điệu/shift).

---

## 7) Outputs & artifacts (Sản phẩm đầu ra)

### Model + metadata
- `artifacts/model.joblib` — model pipeline đã train
- `artifacts/pdv1_metadata.json` — metadata (cal months, s/b, cấu hình)

### Scores
- `artifacts/pdv1_test_scores.csv` — score trên TEST
- `artifacts/pdv1_holdout_scores.csv` — score trên HOLDOUT (nếu chạy)

### Reports (dành cho HR/manager review)
- `artifacts/report_test.md`
- `artifacts/report_holdout.md`

Mỗi report gồm:
- **AUC / KS / Brier**
- **Decile table** (high → low risk) với `p_mean`/PD theo decile
- **Calibration sanity block** (append thêm sau report):
  - ODR (mean(y))
  - p̄ = mean(predicted PD)
  - |p̄ − ODR|
  - wMAE_decile (weighted |p̄_dec − ODR_dec|)
  - KS max @ decile k (điểm tách lớp mạnh nhất)

---

## 8) How to use in practice (Cách dùng trong nghiệp vụ)

**Cách đọc PD để ra quyết định:**
- PD **không** phải “duyệt/không duyệt” trực tiếp, mà dùng để:
  - Xếp hạng rủi ro → gắn policy
  - Xác định cut-off theo “risk appetite”
  - Theo dõi danh mục: khi PD phân phối dịch chuyển, cần kiểm tra drift & strategy

**Ví dụ workflow trong bank/fintech:**
1) Score PD cho hồ sơ mới
2) Apply policy: verify, income, fraud rules
3) Dựa PD bin/decile để set:
   - hạn mức / pricing / yêu cầu bổ sung chứng từ
4) Monitoring: theo tháng/kênh, kiểm tra PSI + ODR drift

---

## 9) Validation & controls (Kiểm soát & kiểm định)

Repo thể hiện các kiểm soát quan trọng dạng “production-like”:

- **Reproducibility:** chạy lệnh theo RUNBOOK là ra artifacts nhất quán
- **Calibration sanity:** p̄ vs ODR, wMAE_decile
- **Ranking quality:** AUC/KS + KS max decile
- **Out-of-time-like check (HOLDOUT):** kiểm tra hiệu năng khi tách theo tháng (CAL_MONTHS)

---

## 10) Limitations (Giới hạn)

- Dataset hiện là “project dataset”; để production cần:
  - Data dictionary + lineage (nguồn dữ liệu, định nghĩa, logic join)
  - Leakage checks theo nghiệp vụ
  - Stability checks theo channel/product/region
  - Policy về missing/outlier & reject inference (nếu có)

- HOLDOUT trong repo là “holdout tạo từ train theo tháng” (demo).  
  Production nên dùng **out-of-time thật** hoặc snapshot theo thời gian.

---

## 11) Monitoring recommendations (Gợi ý monitoring tối thiểu)

Nếu deploy/monitor:
- Theo tháng (ym):
  - ODR thực tế vs p̄ dự báo
  - PSI / drift theo các feature chính hoặc theo score distribution
  - KS/AUC theo cohort (nếu có outcome đủ trễ)
- Trigger review khi:
  - PSI vượt ngưỡng (ví dụ 0.1/0.2 tùy chuẩn)
  - |p̄ − ODR| tăng mạnh, wMAE_decile xấu đi
  - KS giảm đáng kể hoặc KS-max decile “trôi” bất thường

---

## 12) Where to find results quickly (HR/Manager quick view)

Sau khi chạy RUNBOOK:
- **Open:** `artifacts/report_test.md`  
- **Optional:** `artifacts/report_holdout.md`  
Hai file này là “1-page-ish review” giúp đọc nhanh chất lượng + calibration sanity.

---

## 13) Contact / ownership
- Owner: BinBin868
- Purpose: Portfolio Risk / Model Risk / Monitoring-ready baseline project
