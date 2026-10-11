#!/usr/bin/env python3
"""南向持仓逐股采集器（F：自采积累管线）

来源: 东财 RPT_MUTUAL_STOCK_HOLDRANKS（经 akshare stock_hsgt_individual_em）
  - 免费侧服务端保留墙: 2024-10-14 起（约 471 交易日），旧日期返回空
  - 每次调用即返回该股全部保留窗历史 → 增量=按股票整段重取合并

产物: data/southbound_holdings/southbound_long.csv（原子写）+ manifest.json
数据文件不入库；manifest 记录行数/股票数/失败清单。

用法:
  python3 scripts/fetch_southbound_holdings.py          # 常规：整段重取全池
  python3 scripts/fetch_southbound_holdings.py --dry    # 只打印计划不落盘
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "southbound_holdings",
)
OUT_CSV = os.path.join(OUT_DIR, "southbound_long.csv")
OUT_MANIFEST = os.path.join(OUT_DIR, "manifest.json")

COL_MAP = {
    "持股日期": "date",
    "当日收盘价": "close",
    "持股数量": "hold_shares",
    "持股市值": "hold_mkt_cap",
    "持股数量占A股百分比": "hold_ratio",
}


def _fetch_one(symbol5: str):
    import akshare as ak
    df = ak.stock_hsgt_individual_em(symbol=symbol5)
    if df is None or df.empty:
        return pd.DataFrame()
    missing = [c for c in COL_MAP if c not in df.columns]
    if missing:
        raise RuntimeError(f"{symbol5} 缺列 {missing}（源结构变更）")
    out = df[list(COL_MAP)].rename(columns=COL_MAP)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for c in ("close", "hold_shares", "hold_mkt_cap", "hold_ratio"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    import config
    frames, failed = [], []
    pool = list(config.TRAINING_STOCKS.keys())
    for i, pool_code in enumerate(pool, 1):
        sym = pool_code.split(".")[0].zfill(5)
        try:
            df = _fetch_one(sym)
        except Exception as e:  # 网络/源变更：跳过该股并记录，不中断整批
            failed.append({"code": pool_code, "error": f"{type(e).__name__}: {str(e)[:80]}"})
            df = pd.DataFrame()
        if df.empty:
            failed.append({"code": pool_code, "error": "empty"})
        else:
            df["pool_code"] = pool_code
            frames.append(df)
            print(f"[{i:2d}/{len(pool)}] {pool_code} rows={len(df)}", flush=True)
        time.sleep(args.sleep)

    if not frames:
        print("全部失败，不写盘", file=sys.stderr)
        return 2

    long_df = pd.concat(frames, ignore_index=True).sort_values(["date", "pool_code"])
    manifest = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rows": int(len(long_df)),
        "stocks": int(long_df["pool_code"].nunique()),
        "date_min": str(long_df["date"].min().date()),
        "date_max": str(long_df["date"].max().date()),
        "failed": failed,
        "source": "eastmoney RPT_MUTUAL_STOCK_HOLDRANKS via akshare stock_hsgt_individual_em",
        "retention_note": "free tier starts 2024-10-14; older dates empty server-side",
    }
    if args.dry:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT_CSV + ".tmp"
    long_df.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
