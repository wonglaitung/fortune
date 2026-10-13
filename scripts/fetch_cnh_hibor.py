#!/usr/bin/env python3
"""CNH/HIBOR 市场级利率数据采集（2026-10-11 起本地落盘，纯数据制备不进生产模型）。

- HIBOR：东财 datacenter 经 akshare rate_interbank（香港银行同业拆借市场 / Hibor港币），
  7 个期限；1月及以上 1996-07 起、隔夜/1周/2周 2006-01 起，全历史免费可回拉
- CNH 离岸：yfinance CNH=X 已死（仅 1 行）→ 回落在岸 USDCNY=X（2001 起，6000+ 行）；
  CNH 若复活（>100 行）自动优先
- PIT：均为 T 日定盘/收盘，特征接入须用 T-1（未接入，仅数据层）
"""
import json
import os
import sys
import time

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cnh_hibor")
OUT_HIBOR = os.path.join(OUT_DIR, "hibor_hkd.csv")
OUT_FX = os.path.join(OUT_DIR, "usdcny.csv")
OUT_MANIFEST = os.path.join(OUT_DIR, "manifest.json")

TENORS = ["隔夜", "1周", "2周", "1月", "3月", "6月", "1年"]
TENOR_COL = {"隔夜": "hibor_on", "1周": "hibor_1w", "2周": "hibor_2w",
             "1月": "hibor_1m", "3月": "hibor_3m", "6月": "hibor_6m", "1年": "hibor_1y"}


def fetch_hibor():
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")
    cols = {}
    for tenor in TENORS:
        df = ak.rate_interbank(market="香港银行同业拆借市场", symbol="Hibor港币", indicator=tenor)
        if df is None or df.empty:
            raise RuntimeError(f"HIBOR {tenor} empty")
        d = pd.to_datetime(df["报告日"]).dt.normalize()
        r = pd.to_numeric(df["利率"], errors="coerce")
        s = pd.Series(r.to_numpy(), index=d)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cols[TENOR_COL[tenor]] = s
        print(f"  [hibor] {tenor} rows={len(s)} {s.index.min().date()}->{s.index.max().date()}", flush=True)
        time.sleep(1.5)
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out.reset_index()


def fetch_fx():
    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf
    try:
        cnh = yf.download("CNH=X", start="2000-01-01", progress=False, auto_adjust=False)
        if cnh is not None and len(cnh) > 100:
            close = cnh["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            print(f"  [fx] CNH=X 复活 rows={len(close)}", flush=True)
            s = pd.Series(close.to_numpy(), index=pd.to_datetime(close.index).tz_localize(None).normalize())
            df = pd.DataFrame({"usdcny": s})
            df.index.name = "date"
            df = df.reset_index()
            df.attrs["source"] = "yfinance CNH=X (offshore)"
            return df
    except Exception as e:
        print(f"  [fx] CNH=X {type(e).__name__}（已知死源，继续回落）", flush=True)
    usd = yf.download("USDCNY=X", start="2000-01-01", progress=False, auto_adjust=False)
    if usd is None or usd.empty:
        raise RuntimeError("USDCNY=X empty")
    close = usd["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = pd.Series(close.to_numpy(), index=pd.to_datetime(close.index).tz_localize(None).normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    print(f"  [fx] USDCNY=X 回落 rows={len(s)} {s.index.min().date()}->{s.index.max().date()}", flush=True)
    df = pd.DataFrame({"usdcny": s})
    df.index.name = "date"
    df = df.reset_index()
    df.attrs["source"] = "yfinance USDCNY=X (inshore fallback)"
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hibor = fetch_hibor()
    fx = fetch_fx()
    hibor.to_csv(OUT_HIBOR + ".tmp", index=False)
    os.replace(OUT_HIBOR + ".tmp", OUT_HIBOR)
    fx.to_csv(OUT_FX + ".tmp", index=False)
    os.replace(OUT_FX + ".tmp", OUT_FX)
    manifest = {
        "hibor_rows": len(hibor),
        "hibor_span": [str(hibor["date"].min().date()), str(hibor["date"].max().date())],
        "fx_rows": len(fx),
        "fx_span": [str(fx["date"].min().date()), str(fx["date"].max().date())],
        "fx_source": fx.attrs.get("source", "unknown"),
        "source_hibor": "eastmoney datacenter via akshare rate_interbank (market=香港银行同业拆借市场, symbol=Hibor港币)",
        "pit_note": "T-day fixing; feature integration must use T-1; NOT wired into production model",
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
