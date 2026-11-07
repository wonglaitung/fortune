#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取 HKEX 权益披露（DI）通知（Part XV，DION 2017+；旧库另行处理）。

输出：data/di/di_notices.csv（不入库）+ manifest.json
用法：
  python3 scripts/fetch_di.py --universe watchlist --start 2016 --end 2026
  python3 scripts/fetch_di.py --codes 0700.HK --start 2026 --end 2026
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WATCHLIST

BASE = 'https://di.hkex.com.hk/di/'
OUT = 'data/di'
CACHE = 'data/di/cache'
UA = {'User-Agent': 'Mozilla/5.0 (personal research)'}


def preflight(sess):
    try:
        r = sess.get(BASE + 'NSSrchCorp.aspx?g_lang=en&lang=EN&src=MAIN', timeout=15)
        if r.status_code != 200:
            print(f'DI 预检失败：HTTP {r.status_code}', flush=True)
            return False
        return True
    except Exception as e:
        print(f'DI 预检失败：{e}', flush=True)
        return False


def _val(soup, name):
    t = soup.find('input', {'name': name})
    return t.get('value', '') if t else ''


def _result_links(soup):
    out = {}
    for a in soup.find_all('a'):
        txt = a.get_text(' ', strip=True)
        href = a.get('href', '')
        for key, label in [('all', 'List of all notices'),
                           ('ss', 'List of notices filed by substantial shareholders'),
                           ('dir', 'List of notices filed by directors')]:
            if label.lower() in txt.lower():
                out[key] = BASE + href if href.startswith('NS') else href
    return out


def search(sess, code5, start, end):
    url = BASE + 'NSSrchCorp.aspx?g_lang=en&lang=EN&src=MAIN'
    r = sess.get(url, timeout=30); r.raise_for_status()
    soup = BeautifulSoup(r.text, 'lxml')
    data = {
        '__EVENTTARGET': '', '__EVENTARGUMENT': '',
        '__VIEWSTATE': _val(soup, '__VIEWSTATE'),
        '__VIEWSTATEGENERATOR': _val(soup, '__VIEWSTATEGENERATOR'),
        '__EVENTVALIDATION': _val(soup, '__EVENTVALIDATION'),
        'txtStockCode': code5, 'txtCorpName': '',
        'ddlStartDateDD': f'{start.day:02d}', 'ddlStartDateMM': f'{start.month:02d}', 'ddlStartDateYYYY': str(start.year),
        'ddlEndDateDD': f'{end.day:02d}', 'ddlEndDateMM': f'{end.month:02d}', 'ddlEndDateYYYY': str(end.year),
        'cmdSearch': 'Search',
    }
    r2 = sess.post(url, data=data, timeout=60); r2.raise_for_status()
    return BeautifulSoup(r2.text, 'lxml')


def _num(s):
    m = re.search(r'([\d,]+(?:\.\d+)?)', str(s))
    return float(m.group(1).replace(',', '')) if m else None


def _pair(s):
    """从 '58,494(L) 0(S)' 取 (long, short)。"""
    s = str(s)
    lm = re.search(r'([\d,]+(?:\.\d+)?)\s*\(L\)', s)
    sm = re.search(r'([\d,]+(?:\.\d+)?)\s*\(S\)', s)
    return (_num(lm.group(1)) if lm else None, _num(sm.group(1)) if sm else None)


def parse_notice_table(soup):
    rows = []
    for table in soup.find_all('table'):
        for tr in table.find_all('tr'):
            tds = [td.get_text(' ', strip=True) for td in tr.find_all('td')]
            if len(tds) < 8:
                continue
            # 表头/空行过滤：Form Serial 形如 DA2026... / IS2026...
            if not re.match(r'^(DA|IS|CS|MP)\d{8}', tds[0] or ''):
                continue
            link = ''
            a = tr.find('a', href=True)
            if a:
                href = a['href']
                link = BASE + href if href.startswith('NS') else href
            sh_l, sh_s = _pair(tds[3])
            int_l, int_s = _pair(tds[5])
            pct_l, pct_s = _pair(tds[6])
            rows.append({
                'form_serial': tds[0], 'name': tds[1], 'reason': tds[2],
                'shares_raw': tds[3], 'shares_long': sh_l, 'shares_short': sh_s,
                'avg_price': _num(tds[4]), 'interested_raw': tds[5],
                'interested_long': int_l, 'interested_short': int_s,
                'pct_raw': tds[6], 'pct_long': pct_l, 'pct_short': pct_s,
                'event_date': tds[7] if len(tds) > 7 else '', 'link': link,
            })
    return rows


def parse_pages(sess, url):
    all_rows = []
    seen_pages = set()
    page = 1
    while True:
        r = sess.get(url if page == 1 else f'{url}&pg={page}', timeout=60)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, 'lxml')
        rows = parse_notice_table(soup)
        if page > 1 and not rows:
            break
        all_rows.extend(rows)
        # 下一页链接
        pgs = set()
        for a in soup.find_all('a', href=True):
            m = re.search(r'[?&]pg=(\d+)', a['href'])
            if m:
                pgs.add(int(m.group(1)))
        nxt = max(pgs) if pgs else None
        if nxt is None or nxt <= page or nxt in seen_pages:
            break
        seen_pages.add(nxt)
        page = nxt
        time.sleep(0.2)
    return all_rows


def _cache_path(code5, year):
    return os.path.join(CACHE, f'{code5}_{year}.json')


def _load_cache(code5, year, refresh=False):
    p = _cache_path(code5, year)
    if not refresh and os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _save_cache(code5, year, rows):
    os.makedirs(CACHE, exist_ok=True)
    json.dump(rows, open(_cache_path(code5, year), 'w'), ensure_ascii=False)


def code_list(args):
    if args.codes:
        return [c.strip() for c in args.codes.split(',') if c.strip()]
    if args.universe == 'watchlist':
        return list(WATCHLIST)
    import json
    return list(json.load(open('data/hk_broad/universe_310.json')))


def process_code(cfg, args):
    code5 = cfg.split('.')[0].zfill(5)
    sess = requests.Session(); sess.headers.update(UA)
    out = []
    for y in range(args.start, args.end + 1):
        rows = _load_cache(code5, y, refresh=args.refresh)
        if rows is None:
            try:
                soup = search(sess, code5, datetime(y, 1, 1), min(datetime(y, 12, 31), datetime.now()))
                links = _result_links(soup)
                target = 'all' if args.include_directors else 'ss'
                rows = parse_pages(sess, links[target]) if target in links else []
                _save_cache(code5, y, rows)
            except Exception as e:
                print(f'{code5} {y}: ERR {e}', flush=True)
                rows = []
        for r in rows:
            r['code'] = code5
            r['name_cfg'] = WATCHLIST.get(cfg, '')
        out.extend(rows)
        print(f'{code5} {y}: {len(rows)}', flush=True)
        time.sleep(args.sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=None)
    ap.add_argument('--universe', choices=['watchlist', '310'], default='watchlist')
    ap.add_argument('--start', type=int, default=2016)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--sleep', type=float, default=0.3)
    ap.add_argument('--include-directors', action='store_true', help='同时抓董事通知（量大）')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--refresh', action='store_true', help='忽略缓存重新抓取')
    ap.add_argument('--no-merge', action='store_true', help='不合并已有 CSV')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sess = requests.Session(); sess.headers.update(UA)
    if not preflight(sess):
        raise SystemExit(2)
    all_rows = []
    codes = code_list(args)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_code, c, args) for c in codes]
        for f in as_completed(futs):
            all_rows.extend(f.result())
    df = pd.DataFrame(all_rows)
    p = os.path.join(args.out, 'di_notices.csv')
    if not args.no_merge and os.path.exists(p) and len(df):
        try:
            old = pd.read_csv(p)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    if df.empty:
        print('未获取到任何 DI 记录；保留现有 CSV/manifest', flush=True)
        raise SystemExit(2)
    df = df.drop_duplicates(subset=['form_serial'], keep='last')
    df['filing_date'] = pd.to_datetime(df.form_serial.str.extract(r'^[A-Z]+(\d{8})')[0], format='%Y%m%d', errors='coerce')
    df['event_date'] = pd.to_datetime(df.event_date, format='mixed', dayfirst=True, errors='coerce')
    df = df.sort_values(['filing_date', 'code'], na_position='last')
    df.to_csv(p, index=False)
    manifest = {'generated_at': datetime.now().isoformat(timespec='seconds'),
                'rows': len(df), 'codes': df.code.nunique() if len(df) else 0,
                'start': args.start, 'end': args.end}
    import json
    json.dump(manifest, open(os.path.join(args.out, 'manifest.json'), 'w'), ensure_ascii=False, indent=1)
    print(manifest)


if __name__ == '__main__':
    main()
