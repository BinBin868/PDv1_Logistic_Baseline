import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def ks_stat(y, p):
    df=pd.DataFrame({"y":y,"p":p}).sort_values("p")
    bad=(df.y==1).cumsum()/(df.y==1).sum()
    good=(df.y==0).cumsum()/(df.y==0).sum()
    return float(np.max(np.abs(bad-good)))

def metrics_and_deciles(y, p, bins=10):
    auc=float(roc_auc_score(y,p)); ks=ks_stat(y,p); brier=float(brier_score_loss(y,p))
    dec=(pd.DataFrame({"y":y,"p":p})
         .assign(decile=pd.qcut(p,bins,labels=False,duplicates="drop"))
         .groupby("decile").agg(n=("y","size"),bad=("y","sum"),rate=("y","mean"),p_mean=("p","mean"))
         .reset_index().sort_values("decile",ascending=False))
    return {"auc":auc,"ks":ks,"brier":brier,"deciles":dec}
