#!/usr/bin/env python3
"""A-H 溢价数据集抓取

数据源（与 HIBOR 线零重叠）：
  A 腿: akshare.stock_zh_a_daily(adjust="qfq")   新浪
  H 腿: akshare.stock_hk_daily(adjust="qfq")     新浪
  FX : yfinance HKDCNY=X (HKD/CNY)

约定：每源独立超时 + 重试；原子写；manifest 记录行数/区间/来源。
落盘 data/ah_premium/，供 M1/M1b 实验消费。
"""
import json
import os
import signal
import time
import warnings

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
OUT = os.path.join(REPO, "data", "ah_premium")
os.makedirs(OUT, exist_ok=True)

# 港股 57 池内的 A+H 双重上市：A股代码, 名称, H股5位
PAIRS = [
    ("601398", "工商银行", "01398"),
    ("600036", "招商银行", "03968"),
    ("601939", "建设银行", "00939"),
    ("601318", "中国平安", "02318"),
    ("601628", "中国人寿", "00966"),
    ("601088", "中国神华", "01088"),
    ("601601", "中国太保", "02601"),
    ("601288", "农业银行", "01288"),
    ("601633", "长城汽车", "02333"),
    ("002594", "比亚迪", "01211"),
    ("601728", "中国电信", "00728"),
    ("600941", "中国移动", "00941"),
    ("600938", "中国海洋石油", "00883"),
]

FX_SYM = "HKDCNY=X"


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def guarded(fn, *a, timeout=60, retries=3, pause=4, label=""):
    """单源硬超时 + 重试（lessons 三.21：依赖链内部请求也会挂死）"""
    signal.signal(signal.SIGALRM, _alarm)
    last = None
    for i in range(retries):
        try:
            signal.alarm(timeout)
            out = fn(*a)
            signal.alarm(0)
            if out is None or len(out) == 0:
                raise ValueError("empty")
            return out
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            last = f"{type(e).__name__}: {e}"
            print(f"  [{label}] attempt{i + 1}/{retries} fail: {last}")
            time.sleep(pause)
    raise RuntimeError(f"{label} all {retries} attempts failed -> {last}")


def atomic_write(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp)
    os.replace(tmp, path)


def fetch_fx():
    import yfinance as yf

    d = guarded(lambda: yf.Ticker(FX_SYM).history(period="max"),
                timeout=60, label="fx")
    s = d["Close"].dropna()
    out = s.copy()
    out.index = [str(i.date()) for i in out.index]
    out.index.name = "date"
    return out.rename("fx_hkd_cny").to_frame()


def main():
    import akshare as ak
    import pandas as pd

    fx = fetch_fx()
    atomic_write(fx, os.path.join(OUT, "fx_hkdcny.csv"))
    print(f"FX {FX_SYM}: rows={len(fx)} {fx.index.min()}..{fx.index.max()}")

    manifest = {"fx": {"rows": int(len(fx)), "start": fx.index.min(),
                       "end": fx.index.max(), "source": "yfinance " + FX_SYM},
                "pairs": {}}
    for acode, name, hk5 in PAIRS:
        mkt = ("sh" if acode[0] == "6" else "sz") + acode
        try:
            a = guarded(lambda: ak.stock_zh_a_daily(symbol=mkt, adjust="qfq"),
                        timeout=60, label=f"A{acode}")
            time.sleep(1)
            h = guarded(lambda: ak.stock_hk_daily(symbol=hk5, adjust="qfq"),
                        timeout=60, label=f"H{hk5}")
            a = a.assign(date=a["date"].astype(str)).set_index("date")
            h = h.assign(date=h["date"].astype(str)).set_index("date")
            df = pd.concat([a["close"].rename("A_cny"),
                            h["close"].rename("H_hkd")], axis=1, join="inner").dropna()
            df["fx"] = fx["fx_hkd_cny"].reindex(df.index)
            df = df.dropna()
            df = df.join(fx["fx_hkd_cny"].rename("fx2")).drop(columns=["fx2"])
            df["premium"] = 1.0 / ((df["H_hkd"] / df["A_cny"]) * df["fx"])
            path = os.path.join(OUT, f"ah_{acode}_{hk5}.csv")
            atomic_write(df, path)
            manifest["pairs"][acode] = {
                "name": name, "a_code": acode, "h_code": hk5,
                "rows": int(len(df)), "start": df.index.min(), "end": df.index.max(),
                "file": os.path.basename(path),
            }
            print(f"{name:8s} {acode}/{hk5}: rows={len(df)} "
                  f"{df.index.min()}..{df.index.max()} "
                  f"premium_last={df['premium'].iloc[-1]:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name} {acode}/{hk5}: FETCH FAIL {e}")
            manifest["pairs"][acode] = {"name": name, "error": str(e)}

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ok = sum(1 for v in manifest["pairs"].values() if "error" not in v)
    print(f"\nDONE ok={ok}/{len(PAIRS)} -> {OUT}")


if __name__ == "__main__":
    main()