#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取最新 FF301 月度报表的已发行股数（310 只）。

输出：data/hkex_monthly_shares.csv（code, month_end, issued, treasury, total,
filing_date, url）
"""
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'data/hkex_monthly_shares.csv'
PDF_DIR = 'data/hkex_monthly_pdfs'


def universe():
    d = json.load(open('data/hk_broad/universe_310.json'))
    return sorted({c.split('.')[0].zfill(5) for c in d})


def latest_notice(code5):
    best = None
    for f in glob.glob(f'data/hkex_announcements/{code5}_202*.json'):
        for x in json.load(open(f)):
            if 'monthly return' not in x.get('TITLE', '').lower():
                continue
            try:
                dt = datetime.strptime(x['DATE_TIME'].split()[0], '%d/%m/%Y')
            except Exception:
                continue
            if best is None or dt > best[0]:
                best = (dt, x)
    return best


def download(code5, x):
    os.makedirs(PDF_DIR, exist_ok=True)
    p = f'{PDF_DIR}/{code5}_{x["NEWS_ID"]}.pdf'
    if not (os.path.exists(p) and os.path.getsize(p) > 10000):
        r = requests.get('https://www1.hkexnews.hk' + x['FILE_LINK'], timeout=60,
                         headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        open(p, 'wb').write(r.content)
    return p


def parse(path):
    import fitz
    t = ' '.join(' '.join(p.get_text().split()) for p in fitz.open(path))
    # Section II: Balance at close of the month -> issued, treasury, total
    m = re.search(r'Number of issued shares.*?Balance at close of the month\s*([\d,]+)\s*([\d,]+)\s*([\d,]+)', t)
    if not m:
        return None
    vals = [int(x.replace(',', '')) for x in m.groups()]
    me = re.search(r'For the month ended:\s*(\d{1,2} [A-Za-z]+ \d{4})', t)
    return {'issued': vals[0], 'treasury': vals[1], 'total': vals[2],
            'month_end': pd.to_datetime(me.group(1)).date().isoformat() if me else ''}


def fallback_shares(code5):
    p = 'data/shares_history.csv'
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype={'code': str})
    d['code'] = d.code.str.zfill(5)
    x = d[d.code == code5].sort_values('date')
    if not x.empty:
        r = x.iloc[-1]
        v = int(r.shares)
        return {'issued': v, 'treasury': 0, 'total': v,
                'month_end': str(r.date), 'filing_date': str(r.date),
                'url': '', 'source': 'yfinance_fallback'}
    try:
        import yfinance as yf
        sym = code5.lstrip('0').zfill(4) + '.HK'
        sh = yf.Ticker(sym).get_shares_full(start='2024-01-01')
        if sh is not None and len(sh):
            d = pd.to_datetime(sh.index).tz_localize(None).normalize()
            v = int(sh.iloc[-1])
            return {'issued': v, 'treasury': 0, 'total': v,
                    'month_end': str(d[-1].date()), 'filing_date': str(d[-1].date()),
                    'url': '', 'source': 'yfinance_fallback'}
    except Exception:
        pass
    return None


def job(code5):
    hit = latest_notice(code5)
    if not hit:
        fb = fallback_shares(code5)
        if fb:
            fb['code'] = code5
            return fb
        return {'code': code5, 'error': 'no monthly return'}
    dt, x = hit
    try:
        p = download(code5, x)
        r = parse(p)
        if not r:
            return {'code': code5, 'error': 'parse failed', 'url': 'https://www1.hkexnews.hk' + x['FILE_LINK']}
        r.update({'code': code5, 'filing_date': dt.date().isoformat(),
                  'url': 'https://www1.hkexnews.hk' + x['FILE_LINK'],
                  'source': 'HKEX_FF301'})
        return r
    except Exception as e:
        return {'code': code5, 'error': str(e)[:80]}


def main():
    rows, errs = [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(job, c) for c in universe()]
        for f in as_completed(futs):
            r = f.result()
            (errs if r.get('error') else rows).append(r)
    df = pd.DataFrame(rows).sort_values('code')
    df.to_csv(OUT, index=False)
    print(f'ok={len(df)} err={len(errs)}')
    for e in errs[:8]:
        print('ERR', e)


if __name__ == '__main__':
    main()
