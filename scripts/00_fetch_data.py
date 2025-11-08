#!/usr/bin/env python
import argparse, json, urllib.request
from pathlib import Path
import pandas as pd

OUT=Path("data"); OUT.mkdir(exist_ok=True)

def dl(url, out): 
    with urllib.request.urlopen(url) as r, open(out,"wb") as f: f.write(r.read())

def from_github(owner, repo, tag):
    api=f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(api) as r: rel=json.load(r)
    assets={a["name"]:a["browser_download_url"] for a in rel.get("assets",[])}
    for name in ("train.csv","test.csv"):
        url=assets.get(name); 
        if url: dl(url, OUT/name); print(f"[ok] {name} downloaded")
        else: print(f"[!] missing {name} in release {tag}")

def from_gdrive(file_id):
    url=f"https://drive.google.com/uc?export=download&id={file_id}"
    tmp=OUT/"dataset.xlsx"; dl(url,tmp)
    xl=pd.ExcelFile(tmp); xl.parse("train").to_csv(OUT/"train.csv",index=False)
    xl.parse("test").to_csv(OUT/"test.csv",index=False); tmp.unlink(missing_ok=True)
    print("[ok] extracted train.csv & test.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",choices=["github","gdrive"],required=True)
    ap.add_argument("--owner"); ap.add_argument("--repo"); ap.add_argument("--release-tag")
    ap.add_argument("--file-id")
    a=ap.parse_args()
    if a.source=="github": from_github(a.owner,a.repo,a.release_tag)
    else: from_gdrive(a.file_id)
