#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HKEX 除牌证券名单（用于幸存者检查）。

来源：https://di.hkex.com.hk/di/NSDelistedStockList.aspx?lang=EN
输出：data/hkex_delisted.csv（code, name）
注意：该页不含除牌日期/原因，只能识别“已除牌代码”。
"""
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = 'https://di.hkex.com.hk/di/NSDelistedStockList.aspx?lang=EN'
OUT = 'data/hkex_delisted.csv'


def main():
    r = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'lxml')
    rows = []
    for tr in soup.find_all('tr'):
        tds = [x.get_text(' ', strip=True) for x in tr.find_all('td')]
        if len(tds) >= 2 and tds[0].isdigit():
            rows.append({'code': tds[0].zfill(5), 'name': tds[1]})
    df = pd.DataFrame(rows).drop_duplicates(subset=['code', 'name'])
    df.to_csv(OUT, index=False)
    print(f'{len(df)} 条 → {OUT}')


if __name__ == '__main__':
    main()
