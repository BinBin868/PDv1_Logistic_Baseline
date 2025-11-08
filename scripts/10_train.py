#!/usr/bin/env python
import argparse, json, hashlib, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from pdv1.model import make_baseline_pipeline, infer_columns
from pdv1.calibration import platt_calibrate, mean_match
from pdv1.eval import metrics_and_deciles

ART=Path("artifacts"); ART.mkdir(exist_ok=True)

def sha_path(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--target",default="default_90d")
    ap.add_argument("--platt-s",type=float,default=0.5)
    ap.add_argument("--platt-b",type=float,default=-0.996)
    ap.add_argument("--mean-matching",choices=["on","off"],default="on")
    ap.add_argument("--mean-target",type=float)
    ap.add_argument("--cal-months",nargs="*",default=["2024-11","2024-12"])
    ap.add_argument("--month-col",default="ym")  # cột tháng nếu có
    a=ap.parse_args()

    df=pd.read_csv(a.train, sep=None, engine="python")  # auto detect , or ;
    y=df[a.target].astype(int).values
    num,cat=infer_columns(df,a.target)
    pipe=make_baseline_pipeline(num,cat).fit(df.drop(columns=[a.target]),y)

    p_raw=pipe.predict_proba(df.drop(columns=[a.target]))[:,1]
    p_cal=platt_calibrate(p_raw,a.platt_s,a.platt_b)

    if a.mean_matching=="on":
        if a.mean_target is not None: target=a.mean_target
        elif a.month_col in df.columns:
            mask=df[a.month_col].astype(str).isin(a.cal_months)
            target=float(df.loc[mask,a.target].mean()) if mask.any() else float(df[a.target].mean())
        else:
            target=float(df[a.target].mean())
        p_cal=mean_match(p_cal,target)

    res=metrics_and_deciles(y,p_cal,10); meta={k:v for k,v in res.items() if k!="deciles"}
    joblib.dump(pipe, ART/"model.joblib")
    with open(ART/"pdv1_metadata.json","w") as f:
        json.dump({
            "sync":"PDV1_SYNC_2025-11-07_v1",
            "calibration":{"type":"Platt","s":a.platt_s,"b":a.platt_b,"mean_matching":a.mean_matching=="on"},
            "cal_months":a.cal_months,
            "metrics":meta,
            "data_sha":sha_path(Path(a.train)),
            "created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        },f,indent=2)
    res["deciles"].to_csv(ART/"deciles.csv",index=False)
    print("[ok] saved artifacts/model.joblib & pdv1_metadata.json")
