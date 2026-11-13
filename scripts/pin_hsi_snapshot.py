#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结恒指日线快照（F1），供季节性择时回测**只读**使用。

背景：指数级日历规则的回测若每次实时取数，数据源日内更新会让净值/指标漂移
（同 lessons 三.29/三.50 的冻结原则，只是标的换成指数）。

产物
  data/hsi_snapshot/hsi_daily.pkl           # Series(Close, DatetimeIndex, tz-naive)
  data/hsi_snapshot/hsi_daily_manifest.json # 拉取时间/行数/首末/sha256 + prereg sha256

用法
  python3 scripts/pin_hsi_snapshot.py          # 建快照
  python3 scripts/pin_hsi_snapshot.py --check  # 校验
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.path.join(ROOT, "data", "hsi_snapshot")
PKL = os.path.join(SNAP_DIR, "hsi_daily.pkl")
MAN = os.path.join(SNAP_DIR, "hsi_daily_manifest.json")
PREREG = os.path.join(ROOT, "docs", "SEASONAL_TIMING_PREREG.md")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_pickle(obj, path: str) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def do_pin() -> int:
    import yfinance as yf
    h = yf.Ticker("^HSI").history(period="max", interval="1d", auto_adjust=True, timeout=30)
    if h.empty:
        print("❌ ^HSI 取数失败，未写快照")
        return 1
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert(None)
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s.name = "Close"
    atomic_pickle(s, PKL)
    man = {
        "pulled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "yfinance ^HSI auto_adjust=True",
        "rows": int(len(s)),
        "first": str(s.index.min().date()),
        "last": str(s.index.max().date()),
        "sha256_pkl": sha256_file(PKL),
        "prereg": os.path.relpath(PREREG, ROOT),
        "sha256_prereg": sha256_file(PREREG) if os.path.exists(PREREG) else None,
    }
    with open(MAN, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"✅ 快照写入 {os.path.relpath(PKL, ROOT)}（{len(s)} 行，{man['first']} → {man['last']}）")
    print(f"   manifest sha256_pkl={man['sha256_pkl'][:16]}…")
    return 0


def do_check() -> int:
    if not (os.path.exists(PKL) and os.path.exists(MAN)):
        print("❌ 快照不存在 —— 先运行 python3 scripts/pin_hsi_snapshot.py")
        return 1
    man = json.load(open(MAN, encoding="utf-8"))
    cur = sha256_file(PKL)
    if cur != man.get("sha256_pkl"):
        print(f"❌ 快照 pkl sha256 与 manifest 不符：{cur[:16]}… vs {man.get('sha256_pkl', '')[:16]}…")
        return 1
    s = pickle.load(open(PKL, "rb"))
    print(f"✅ 快照完整：{len(s)} 行，{s.index.min().date()} → {s.index.max().date()}，"
          f"sha256={cur[:16]}…")
    if os.path.exists(PREREG) and man.get("sha256_prereg"):
        if sha256_file(PREREG) != man["sha256_prereg"]:
            print("⚠️ 预注册文档在冻结后已被修改（sha256 变了）——预注册应保持写死")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    return do_check() if a.check else do_pin()


if __name__ == "__main__":
    sys.exit(main())
