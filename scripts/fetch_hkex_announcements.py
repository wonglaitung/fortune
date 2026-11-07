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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = 'data/hkex_announcements'
BASE = 'https://www1.hkexnews.hk/search'
UA = {'User-Agent': 'Mozilla/5.0 (personal research)'}
_IDS_LOCK = threading.Lock()


def _session():
    sess = requests.Session()
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(['GET']), raise_on_status=False)
    sess.mount('https://', HTTPAdapter(max_retries=retry))
    sess.headers.update(UA)
    return sess


def _jsonp(text):
    return json.loads(text[text.index('(') + 1:text.rindex(')')])


def _read_ids(p):
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_ids(p, cache):
    os.makedirs(ROOT, exist_ok=True)
    tmp = f'{p}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    os.replace(tmp, p)


def stock_id(sess, code5):
    p = os.path.join(ROOT, '_ids.json')
    with _IDS_LOCK:
        cache = _read_ids(p)
        if code5 in cache:
            return cache[code5]
    r = sess.get(f'{BASE}/prefix.do', params={
        'callback': 'cb', 'lang': 'EN', 'type': 'A', 'name': code5, 'market': 'SEHK'}, timeout=30)
    r.raise_for_status()
    info = _jsonp(r.text).get('stockInfo') or []
    hit = next((x for x in info if x.get('code') == code5), None)
    if hit:
        with _IDS_LOCK:
            cache = _read_ids(p)
            cache[code5] = int(hit['stockId'])
            _write_ids(p, cache)
        return cache[code5]
    return None


def search(sess, sid, f, t):
    r = sess.get(f'{BASE}/titleSearchServlet.do', params={
        'sortDir': 0, 'sortByOptions': 'DateTime', 'category': 0, 'market': 'SEHK',
        'stockId': sid, 'documentType': -1, 'fromDate': f, 'toDate': t, 'title': '',
        'searchType': 1, 't1code': -2, 't2Gcode': -2, 't2code': -2,
        'rowRange': 1000, 'lang': 'en'}, timeout=60)
    r.raise_for_status()
    d = r.json()
    rows = json.loads(d['result']) if d.get('result') else []
    return rows


def fetch_year(sess, code5, sid, year, sleep_s):
    p = os.path.join(ROOT, f'{code5}_{year}.json')
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
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
    tmp = f'{p}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(rows, f)
    os.replace(tmp, p)
    time.sleep(sleep_s)
    return rows


def worker(args):
    code5, years, sleep_s = args
    try:
        sess = _session()
        sid = stock_id(sess, code5)
        if not sid:
            print(f'  ! {code5} id: 未找到 stockId', flush=True)
            return code5, [], len(years)
        out = []
        failed = 0
        for y in years:
            try:
                out.extend(fetch_year(sess, code5, sid, y, sleep_s))
            except Exception as e:
                failed += 1
                print(f'  ! {code5} {y}: {e}', flush=True)
        return code5, out, failed
    except Exception as e:
        print(f'  ! {code5} id: {e}', flush=True)
        return code5, [], len(years)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=None, help='逗号分隔；默认 310 池')
    ap.add_argument('--start', type=int, default=2010)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--sleep', type=float, default=0.25)
    ap.add_argument('--max-fail-ratio', type=float, default=0.02,
                    help='失败代码年占比超过该值即报错（默认 2%）')
    args = ap.parse_args()
    if not 0 <= args.max_fail_ratio <= 1:
        raise SystemExit('--max-fail-ratio 必须在 0–1 之间')

    if args.codes:
        codes = [c.strip().split('.')[0].zfill(5) for c in args.codes.split(',')]
    else:
        uni = json.load(open('data/hk_broad/universe_310.json'))
        codes = sorted({c.split('.')[0].zfill(5) for c in uni})

    years = list(range(args.start, args.end + 1))
    os.makedirs(ROOT, exist_ok=True)
    tasks = [(c, years, args.sleep) for c in codes]
    frames = []
    failed_codes = set()
    failed_years = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, rows, failed) in enumerate(ex.map(worker, tasks), 1):
            if rows:
                df = pd.DataFrame(rows)
                df['code'] = code
                frames.append(df)
            if failed:
                failed_codes.add(code)
                failed_years += failed
            if i % 20 == 0:
                print(f'[{i}/{len(tasks)}] {code}: 累计 {sum(len(x) for x in frames)} 条', flush=True)

    total_years = len(codes) * len(years)
    print(f'失败代码 {len(failed_codes)}/{len(codes)}；失败代码年 {failed_years}/{total_years}', flush=True)
    if failed_codes:
        print(f'失败代码示例：{", ".join(sorted(failed_codes)[:20])}', flush=True)
    if total_years and failed_years / total_years > args.max_fail_ratio:
        raise SystemExit(f'公告抓取失败率过高：{failed_years}/{total_years} > {args.max_fail_ratio:.1%}')
    if not frames:
        raise SystemExit('未获取到任何公告数据')

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
