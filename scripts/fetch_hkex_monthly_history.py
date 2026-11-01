#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FF301 历史月度股本（自选池，2016+）。

输出：data/hkex_monthly_shares_history.csv
"""
import argparse
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
from config import WATCHLIST

OUT = 'data/hkex_monthly_shares_history.csv'
PDF_DIR = 'data/hkex_monthly_pdfs'


def notices(code5, start, end):
    out = []
    for f in glob.glob(f'data/hkex_announcements/{code5}_*.json'):
        for x in json.load(open(f)):
            if 'monthly return' not in x.get('TITLE', '').lower():
                continue
            try:
                dt = datetime.strptime(x['DATE_TIME'].split()[0], '%d/%m/%Y')
            except Exception:
                continue
            if start <= dt.year <= end:
                out.append((dt, x))
    return sorted(out, key=lambda z: z[0])


def parse(path):
    import fitz
    t = ' '.join(' '.join(p.get_text().split()) for p in fitz.open(path))
    m = re.search(r'(?:Number of issued shares|Total number of issued shares).*?Balance at close of the month\s*([\d,]+)\s*([\d,]+)\s*([\d,]+)', t)
    if m:
        vals = [int(x.replace(',', '')) for x in m.groups()]
        issued, treasury, total = vals
    else:
        # 旧模板：II. Movements in Issued Share Capital -> Balance at close of the month
        m2 = re.search(r'Movements in Issued Share(?: Capital)?s?.*?Balance at close of the month\s*([\d,]+)', t)
        if not m2:
            return None
        issued = int(m2.group(1).replace(',', ''))
        treasury, total = 0, issued
    me = re.search(r'For the month ended[^0-9]*(\d{1,2} [A-Za-z]+ \d{4})', t)
    if not me:
        me = re.search(r'For the month ended[^0-9]*(\d{2}/\d{2}/\d{4})', t)
    if not me:
        return None
    month_end = pd.to_datetime(me.group(1), dayfirst=True).date().isoformat()
    return {'issued': issued, 'treasury': treasury, 'total': total, 'month_end': month_end}


def job(code5, dt, x):
    p = f'{PDF_DIR}/{code5}_{x["NEWS_ID"]}.pdf'
    try:
        if not (os.path.exists(p) and os.path.getsize(p) > 10000):
            r = requests.get('https://www1.hkexnews.hk' + x['FILE_LINK'], timeout=60,
                             headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            open(p, 'wb').write(r.content)
        v = parse(p)
        if v:
            v.update({'code': code5, 'filing_date': dt.date().isoformat(),
                      'url': 'https://www1.hkexnews.hk' + x['FILE_LINK']})
        return v
    except Exception as e:
        return {'code': code5, 'error': str(e)[:60], 'filing_date': dt.date().isoformat()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=None)
    ap.add_argument('--start', type=int, default=2016)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-merge', action='store_true')
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(',')] if args.codes else list(WATCHLIST)
    tasks = []
    for cfg in codes:
        c5 = cfg.split('.')[0].zfill(5)
        for dt, x in notices(c5, args.start, args.end):
            tasks.append((c5, dt, x))
    print(f'tasks={len(tasks)}', flush=True)
    rows, errs = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            (errs if r and r.get('error') else rows if r else errs).append(r)
            if i % 500 == 0:
                print(f'done {i}', flush=True)
    df = pd.DataFrame([r for r in rows if r])
    if not args.no_merge and os.path.exists(OUT) and len(df):
        try:
            old = pd.read_csv(OUT, dtype={'code': str})
            old['code'] = old.code.str.zfill(5)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    df = df.drop_duplicates(subset=['code', 'month_end'], keep='last').sort_values(['code', 'month_end'])
    df.to_csv(OUT, index=False)
    print(f'rows={len(df)} err={len(errs)} codes={df.code.nunique() if len(df) else 0}', flush=True)
    print('err sample', errs[:3], flush=True)


if __name__ == '__main__':
    main()
