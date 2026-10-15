#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取恒指 PIT 成分表（**免费路径**，D34）。

来源：https://yfiua.github.io/index-constituents/ （该源标注 Bloomberg 等）
覆盖：**HSI 仅自 2023-07 起**（按月快照；2020-2023 无 HSI）——由 `docs/2020/*` 无 HSI 文件证实。
产物：`data/hsi_constituents_pit.csv`（长表：month,symbol,name）+ `manifest.json`。

用途：为「恒指成分池」提供 PIT 成分。⚠️ 覆盖不足 36 个月训练窗 ⇒ 不足以跑标准 WF（见 D34 调研判决）。
"""

import json
import os
import time
import urllib.request
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://raw.githubusercontent.com/yfiua/index-constituents/main/docs'
START = (2023, 7)
OUT_CSV = os.path.join(ROOT, 'data', 'hsi_constituents_pit.csv')
OUT_MANIFEST = os.path.join(ROOT, 'data', 'hsi_constituents_pit_manifest.json')


def months():
    y, m = START
    end = date.today()
    out = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch(y, m):
    url = f"{BASE}/{y:04d}/{m:02d}/constituents-hsi.csv"
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return pd.read_csv(r)
    except Exception:
        return None


def main():
    rows, got = [], []
    for (y, m) in months():
        df = fetch(y, m)
        if df is None or 'Symbol' not in df.columns:
            continue
        mm = f"{y:04d}-{m:02d}"
        for _, r in df.iterrows():
            rows.append((mm, str(r['Symbol']).strip(), str(r['Name']).strip()))
        got.append((mm, len(df)))
        time.sleep(0.15)

    out = pd.DataFrame(rows, columns=['month', 'symbol', 'name'])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    manifest = {
        'source': 'yfiua/index-constituents (Bloomberg-based)',
        'url': BASE,
        'coverage_first': got[0][0] if got else None,
        'coverage_last': got[-1][0] if got else None,
        'n_months': len(got),
        'n_rows': int(len(out)),
        'n_unique_symbols': int(out['symbol'].nunique()),
        'fetched_at': date.today().isoformat(),
        'note': 'HSI PIT 仅自 2023-07 起；不足 36 个月训练窗，不能直接跑标准 WF（D34）',
    }
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"覆盖月数: {len(got)}  {got[0][0] if got else '—'} ~ {got[-1][0] if got else '—'}")
    print(f"去重股票数: {out['symbol'].nunique()}")
    print("每月成分数:", ", ".join(f"{mm}:{n}" for mm, n in got))
    print(f"已写: {OUT_CSV}")
    print(f"已写: {OUT_MANIFEST}")


if __name__ == '__main__':
    main()
