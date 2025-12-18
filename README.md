# PDv1_Logistic_Baseline
This repository demonstrates a **production-style Probability of Default (PD) modeling workflow** for credit risk roles, covering the full lifecycle:
**train → calibrate → score → report → governance**.

The project is designed to reflect how a baseline PD model is built, validated, and documented in real-world banking/fintech environments, with emphasis on **calibration sanity, reproducibility, and model controls**, not just predictive performance.
## Who this project is for
- Credit Risk Analyst / Portfolio Risk Analyst (entry–mid level)
- Model Risk / Model Validation (junior/associate)
- Risk Analytics roles in banks, fintech lenders, or consulting
- Hiring managers looking for candidates with **production-aware risk modeling**, not just ML notebooks
## What makes this project different

Unlike typical ML assignments, this project focuses on:
- **Probability calibration** (Platt scaling, mean-matching) and sanity checks (p̄ vs ODR, wMAE_decile)
- **Out-of-time-like HOLDOUT evaluation** to mimic production review
- **Reproducible artifacts** (model, metadata, scores, reports)
- **Model governance and controls** (data quality, leakage, monitoring, overrides)
- Clear linkage between **PD outputs and credit policy usage**
  
## How to review this repository

1. Read `MODEL_CARD.md` to understand the **business intent, scope, assumptions, and controls** of the PD model.
2. Follow `RUNBOOK.md` to **reproduce the full workflow** and generate model artifacts.
3. Review outputs in `artifacts/`:
   - `report_test.md`
   - `report_holdout.md`
These reports summarize performance, calibration sanity, and decile-level behavior.

