#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建港股宽基候选池 + PIT 流动性成分（B′ 方案，2026-10-22）。

用途：把 56 只自选池扩到"客观规则池"，用于「池宽度」A/B 实验。
- 候选：HKEX 主板股本证券（ListOfSecurities.xlsx，剔除 Trading Only / DR）
- 行情：yfinance 全历史（auto_adjust=True，与 qfq 同为复权口径）
- 成分：每月末按过去 20 日成交额（Close×Volume）取前 N，**只用当日及以前数据**

产物（均不入库，见 .gitignore: data/hk_broad/）：
- panel_{field}.parquet   宽表 OHLCV（float32）
- membership.csv          长表 month,code,rank,adv
- meta.json               抓取时间 / 覆盖 / 失败清单

用法：
    python3 scripts/build_hk_broad_universe.py --topn 300
    python3 scripts/build_hk_broad_universe.py --topn 300 --refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'hk_broad')
LIST_XLSX = os.path.join(OUT, 'ListOfSecurities.xlsx')
LIST_URL = ('https://www.hkex.com.hk/eng/services/trading/securities/'
            'securitieslists/ListOfSecurities.xlsx')
EXCLUDE_SUB = {'Trading Only Securities', 'Depositary Receipts'}
FIELDS = ['Open', 'High', 'Low', 'Close', 'Volume']


def hkex_list(refresh: bool) -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    if refresh or not os.path.exists(LIST_XLSX):
        import urllib.request
        req = urllib.request.Request(LIST_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r, open(LIST_XLSX, 'wb') as f:
            f.write(r.read())
    d = pd.read_excel(LIST_XLSX, header=2)
    d.columns = [str(c).strip().replace('\n', ' ') for c in d.columns]
    q = d[(d['Category'] == 'Equity')
          & (d['Sub-Category'].astype(str).str.contains('Main Board'))
          & (~d['Sub-Category'].astype(str).isin(EXCLUDE_SUB))].copy()
    q['code'] = q['Stock Code'].astype(str).str.zfill(5)
    q['name'] = q['Name of Securities'].astype(str)
    return q[['code', 'name']].drop_duplicates('code').reset_index(drop=True)


def _tk(code: str) -> str:
    return f"{int(code):04d}.HK"


def fetch_panel(codes: list[str], batch: int = 80) -> tuple[dict, list[str]]:
    import yfinance as yf
    panels = {k: [] for k in FIELDS}
    failed = []
    tk2code = {_tk(c): c for c in codes}
    tks = list(tk2code)
    for i in range(0, len(tks), batch):
        chunk = tks[i:i + batch]
        try:
            df = yf.download(chunk, start='2006-01-01', auto_adjust=True,
                             actions=False, group_by='ticker', threads=True,
                             progress=False)
        except Exception as e:  # noqa: BLE001
            print(f'  [batch {i//batch}] 失败 {str(e)[:60]}')
            failed.extend(chunk)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for tk in chunk:
                if tk not in df.columns.get_level_values(0):
                    failed.append(tk)
                    continue
                sub = df[tk].dropna(how='all')
                if len(sub) < 60:
                    failed.append(tk)
                    continue
                for k in FIELDS:
                    s = pd.to_numeric(sub[k], errors='coerce')
                    s.name = tk2code[tk]
                    panels[k].append(s)
        else:
            failed.extend(chunk)
        print(f'  batch {i//batch + 1}/{(len(tks) + batch - 1)//batch} '
              f'ok={len(codes) - len(failed)} failed={len(failed)}', flush=True)
        time.sleep(0.3)
    out = {}
    for k in FIELDS:
        if panels[k]:
            out[k] = pd.concat(panels[k], axis=1).sort_index().astype('float32')
    return out, failed


def monthly_membership(close: pd.DataFrame, volume: pd.DataFrame,
                       topn: int) -> pd.DataFrame:
    """每月末按 20 日均成交额排名（PIT：只用当日及以前）。"""
    adv = (close * volume).rolling(20, min_periods=15).mean()
    # 每月最后一个有数据的交易日
    me = pd.Series(adv.index, index=adv.index).groupby(
        [adv.index.year, adv.index.month]).last().values
    rows = []
    for d in me:
        d = pd.Timestamp(d)
        s = adv.loc[d].dropna()
        if len(s) < topn:
            continue
        s = s.sort_values(ascending=False).head(topn)
        for rank, (code, a) in enumerate(s.items(), 1):
            rows.append((d.strftime('%Y-%m-%d'), code, rank, float(a)))
    return pd.DataFrame(rows, columns=['month', 'code', 'rank', 'adv'])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--topn', type=int, default=300)
    ap.add_argument('--refresh', action='store_true', help='重新下载 HKEX 名单与行情')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    lst = hkex_list(args.refresh)
    print(f'候选（HKEX 主板股本）：{len(lst)} 只')

    reuse = (not args.refresh) and all(
        os.path.exists(os.path.join(OUT, f'panel_{k.lower()}.parquet')) for k in FIELDS)
    if reuse:
        panels = {k: pd.read_parquet(os.path.join(OUT, f'panel_{k.lower()}.parquet'))
                  for k in FIELDS}
        failed = []
        print('复用已落盘行情')
    else:
        panels, failed = fetch_panel(lst['code'].tolist())
        for k, df in panels.items():
            df.to_parquet(os.path.join(OUT, f'panel_{k.lower()}.parquet'))
        print(f'行情落盘：{len(panels.get("Close", []))} 只，失败 {len(failed)}')

    close, volume = panels['Close'], panels['Volume']
    mem = monthly_membership(close, volume, args.topn)
    mem.to_csv(os.path.join(OUT, 'membership.csv'), index=False)

    union = sorted(mem['code'].unique())
    per_month = mem.groupby('month').size()
    meta = {
        'built_at': date.today().isoformat(),
        'topn': args.topn,
        'n_candidates': int(len(lst)),
        'n_with_data': int(close.shape[1]),
        'n_failed': len(failed),
        'failed_sample': failed[:20],
        'months': [str(mem['month'].min()), str(mem['month'].max())],
        'n_months': int(per_month.size),
        'union_size': len(union),
        'date_range': [str(close.index.min().date()), str(close.index.max().date())],
    }
    with open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    try:
        import config
        base = {c.replace('.HK', '').zfill(5) for c in config.TRAINING_STOCKS}
        overlap = len(base & set(union))
        print(f'与现有 56 池重叠：{overlap}/{len(base)}')
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
