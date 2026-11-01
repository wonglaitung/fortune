#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HKEX Stock Connect 北向合资格证券（SSE/SZSE）。

输出：data/stock_connect_northbound.csv
"""
import os
import re

import pandas as pd
import requests

URLS = {
    'SSE': 'https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities/SSE_Securities.csv',
    'SZSE': 'https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities/SZSE_Securities.csv',
}
OUT = 'data/stock_connect_northbound.csv'


def fetch(market, url):
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    r.raise_for_status()
    p = f'/tmp/{market}_securities.csv'
    open(p, 'wb').write(r.content)
    df = pd.read_csv(p, encoding='utf-16', sep='\t', header=None, dtype=str)
    # 表头行含 Stock Code
    hi = None
    for i in range(min(12, len(df))):
        row = ' '.join(str(x) for x in df.iloc[i].tolist() if pd.notna(x))
        if 'Stock Code' in row:
            hi = i
            break
    if hi is None:
        return pd.DataFrame()
    data = df.iloc[hi + 1:, :6].copy()
    data.columns = ['no', 'code', 'ccass_code', 'name', 'face_value', 'instrument_type']
    data = data[data.code.notna()]
    data['code'] = data.code.str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)
    data['ccass_code'] = data.ccass_code.str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(5)
    data['market'] = market
    updated = ''
    for i in range(min(5, len(df))):
        m = re.search(r'Updated:\s*(\d{1,2} \w+ \d{4})', str(df.iloc[i, 0]))
        if m:
            updated = m.group(1)
    data['updated'] = updated
    return data[['market', 'code', 'ccass_code', 'name', 'face_value', 'instrument_type', 'updated']]


def main():
    rows = [fetch(k, v) for k, v in URLS.items()]
    df = pd.concat([x for x in rows if not x.empty], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f'{len(df)} 条 → {OUT}')


if __name__ == '__main__':
    main()
