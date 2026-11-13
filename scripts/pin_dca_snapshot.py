#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结定投实验行情快照（F1）：2800.HK（含息）与 ^HSI（价格）。"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, tempfile
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "dca_snapshot")
MAN = os.path.join(DIR, "manifest.json")
PREREG = os.path.join(ROOT, "docs", "DCA_LONGTERM_PREREG.md")
HIBOR = os.path.join(ROOT, "data", "cnh_hibor", "hibor_hkd.csv")
TICKERS = {"etf2800": "2800.HK", "hsi": "^HSI"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp): os.remove(tmp)
        raise


def do_pin():
    import yfinance as yf
    man = {"pulled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "series": {}}
    for name, tk in TICKERS.items():
        h = yf.Ticker(tk).history(period="max", interval="1d", auto_adjust=True, timeout=30)
        if h.empty:
            print(f"❌ {tk} 取数失败"); return 1
        s = h["Close"].copy(); s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert(None)
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna(); s.name = "Close"
        p = os.path.join(DIR, f"{name}.pkl"); atomic_pickle(s, p)
        man["series"][name] = {"ticker": tk, "rows": int(len(s)),
                               "first": str(s.index.min().date()), "last": str(s.index.max().date()),
                               "sha256": sha(p)}
    man["hibor"] = {"path": os.path.relpath(HIBOR, ROOT), "sha256": sha(HIBOR)}
    man["prereg_sha256"] = sha(PREREG) if os.path.exists(PREREG) else None
    json.dump(man, open(MAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k, v in man["series"].items():
        print(f"✅ {k}: {v['rows']} 行 {v['first']}→{v['last']}")
    return 0


def do_check():
    if not os.path.exists(MAN):
        print("❌ 快照不存在"); return 1
    man = json.load(open(MAN, encoding="utf-8"))
    for name, v in man["series"].items():
        p = os.path.join(DIR, f"{name}.pkl")
        if not os.path.exists(p) or sha(p) != v["sha256"]:
            print(f"❌ {name} 快照缺失/变更"); return 1
    if sha(HIBOR) != man["hibor"]["sha256"]:
        print("❌ HIBOR 文件在冻结后变更"); return 1
    print("✅ 定投快照完整"); return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); sys.exit(do_check() if a.check else do_pin())
