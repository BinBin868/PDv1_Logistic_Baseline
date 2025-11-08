#!/usr/bin/env python
import argparse, pandas as pd
from pdv1.eval import metrics_and_deciles

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--scores",required=True)
    ap.add_argument("--y-col",default="default_90d")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    df=pd.read_csv(a.scores, sep=None, engine="python")
    if a.y_col not in df.columns: raise SystemExit(f"y '{a.y_col}' not in {a.scores}")
    res=metrics_and_deciles(df[a.y_col], df["pd_cal"], 10)
    with open(a.out,"w") as f:
        f.write("# PDv1 — Baseline Report\n\n")
        f.write(f"- AUC: **{res['auc']:.4f}**\n- KS: **{res['ks']:.4f}**\n- Brier: **{res['brier']:.6f}**\n\n")
        f.write("## Decile table (high → low)\n\n")
        f.write(res["deciles"].to_markdown(index=False))
    print(f"[ok] wrote {a.out}")
