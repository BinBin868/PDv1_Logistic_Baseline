#!/usr/bin/env python
import argparse, os, hashlib
from pathlib import Path
import joblib, pandas as pd
from pdv1.calibration import platt_calibrate, mean_match

def hash_id(x,salt): return hashlib.sha256((str(x)+salt).encode()).hexdigest()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--input",required=True)
    ap.add_argument("--id-col",default=None)
    ap.add_argument("--out",required=True)
    ap.add_argument("--platt-s",type=float,default=0.5)
    ap.add_argument("--platt-b",type=float,default=-0.996)
    ap.add_argument("--mean-matching",choices=["on","off"],default="on")
    ap.add_argument("--mean-target",type=float)
    a=ap.parse_args()

    salt=os.getenv("SALT","")
    clf=joblib.load(a.model)
    df=pd.read_csv(a.input, sep=None, engine="python")
    X=df.drop(columns=[c for c in ("default_90d",) if c in df.columns])
    p_raw=clf.predict_proba(X)[:,1]
    p_cal=platt_calibrate(p_raw,a.platt_s,a.platt_b)
    if a.mean_matching=="on" and a.mean_target is not None:
        p_cal=mean_match(p_cal,a.mean_target)

    ids=(df[a.id_col].astype(str) if a.id_col and a.id_col in df.columns else pd.Series(range(len(df))).astype(str))
    if salt: ids=ids.map(lambda x: hash_id(x,salt))

    out=pd.DataFrame({"id_hash":ids,"score_raw":p_raw,"pd_cal":p_cal})
    if "default_90d" in df.columns: out["default_90d"]=df["default_90d"].astype(int)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print(f"[ok] wrote {a.out}")
