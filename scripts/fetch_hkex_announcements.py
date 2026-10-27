#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HKEXnews 公告抓取（按股票 × 年份，2010+）。

- 端点：titleSearchServlet.do（JSON）+ prefix.do（code→stockId）
- 缓存：data/hkex_announcements/{code}_{year}.json（本地，不入库）
- 输出：data/hkex_announcements/all.csv（date_time, code, title, file_link, news_id）
- 单股单年 >1000 条时按半年拆分
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = 'data/hkex_announcements'
BASE = 'https://www1.hkexnews.hk/search'
UA = {'User-Agent': 'Mozilla/5.0 (personal research)'}


def _jsonp(text):
    return json.loads(text[text.index('(') + 1:text.rindex(')')])


def stock_id(sess, code5):
    p = os.path.join(ROOT, '_ids.json')
    cache = json.load(open(p)) if os.path.exists(p) else {}
    if code5 in cache:
        return cache[code5]
    r = sess.get(f'{BASE}/prefix.do', params={
        'callback': 'cb', 'lang': 'EN', 'type': 'A', 'name': code5, 'market': 'SEHK'}, timeout=30)
    info = _jsonp(r.text).get('stockInfo') or []
    hit = next((x for x in info if x.get('code') == code5), None)
    if hit:
        cache[code5] = int(hit['stockId'])
        os.makedirs(ROOT, exist_ok=True)
        json.dump(cache, open(p, 'w'), indent=0, sort_keys=True)
    return cache.get(code5)


def search(sess, sid, f, t):
    r = sess.get(f'{BASE}/titleSearchServlet.do', params={
        'sortDir': 0, 'sortByOptions': 'DateTime', 'category': 0, 'market': 'SEHK',
        'stockId': sid, 'documentType': -1, 'fromDate': f, 'toDate': t, 'title': '',
        'searchType': 1, 't1code': -2, 't2Gcode': -2, 't2code': -2,
        'rowRange': 1000, 'lang': 'en'}, timeout=60)
    d = r.json()
    rows = json.loads(d['result']) if d.get('result') else []
    return rows


def fetch_year(sess, code5, sid, year, sleep_s):
    p = os.path.join(ROOT, f'{code5}_{year}.json')
    if os.path.exists(p):
        return json.load(open(p))
    rows = search(sess, sid, f'{year}0101', f'{year}1231')
    if len(rows) >= 1000:  # 可能截断，按半年再取
        h1 = search(sess, sid, f'{year}0101', f'{year}0630')
        h2 = search(sess, sid, f'{year}0701', f'{year}1231')
        seen, merged = set(), []
        for x in h1 + h2:
            if x['NEWS_ID'] not in seen:
                seen.add(x['NEWS_ID'])
                merged.append(x)
        rows = merged
    os.makedirs(ROOT, exist_ok=True)
    json.dump(rows, open(p, 'w'))
    time.sleep(sleep_s)
    return rows


def worker(args):
    code5, years, sleep_s = args
    sess = requests.Session(); sess.headers.update(UA)
    sid = stock_id(sess, code5)
    if not sid:
        return code5, []
    out = []
    for y in years:
        try:
            out.extend(fetch_year(sess, code5, sid, y, sleep_s))
        except Exception as e:
            print(f'  ! {code5} {y}: {e}', flush=True)
    return code5, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=None, help='逗号分隔；默认 310 池')
    ap.add_argument('--start', type=int, default=2010)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--sleep', type=float, default=0.25)
    args = ap.parse_args()

    if args.codes:
        codes = [c.strip().split('.')[0].zfill(5) for c in args.codes.split(',')]
    else:
        uni = json.load(open('data/hk_broad/universe_310.json'))
        codes = sorted({c.split('.')[0].zfill(5) for c in uni})

    years = list(range(args.start, args.end + 1))
    os.makedirs(ROOT, exist_ok=True)
    tasks = [(c, years, args.sleep) for c in codes]
    frames = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, rows) in enumerate(ex.map(worker, tasks), 1):
            if rows:
                df = pd.DataFrame(rows)
                df['code'] = code
                frames.append(df)
            if i % 20 == 0:
                print(f'[{i}/{len(tasks)}] {code}: 累计 {sum(len(x) for x in frames)} 条', flush=True)

    if frames:
        allrows = pd.concat(frames, ignore_index=True)
        keep = ['DATE_TIME', 'code', 'TITLE', 'FILE_LINK', 'NEWS_ID', 'STOCK_NAME']
        keep = [c for c in keep if c in allrows.columns]
        out = allrows[keep].rename(columns={'DATE_TIME': 'date_time', 'TITLE': 'title',
                                            'FILE_LINK': 'file_link', 'NEWS_ID': 'news_id',
                                            'STOCK_NAME': 'stock_name'})
        out = out.drop_duplicates(subset=['news_id']).sort_values('date_time')
        out.to_csv(os.path.join(ROOT, 'all.csv'), index=False)
        print(f'完成：{len(out)} 条 → {ROOT}/all.csv')


if __name__ == '__main__':
    main()
