#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCASS 参与者持股日归档（港交所 SDW 查询）。

⚠️ 许可：HKEX 明确「仅供个人非商业使用，不得再分发/汇编」。因此：
- 原始与派生数据只落本地 `data/ccass/`（已 gitignore），**不入库**
- 仓库只保留本脚本与 manifest 结构说明
- HKEX 仅保留最近约 12 个月 → 必须每日归档，历史无法补回

用法：
  python3 scripts/fetch_ccass.py                       # 抓最近一个交易日（310 池）
  python3 scripts/fetch_ccass.py --date 2026/10/26 --stocks 00700,00005
  python3 scripts/fetch_ccass.py --backfill-weeks 4    # 回填最近 4 个周五
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = 'https://www3.hkexnews.hk/sdw/search/searchsdw.aspx?lang=en'
ROOT = 'data/ccass'


def _session():
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; personal research)'})
    r = s.get(URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    def hidden(name):
        e = soup.find('input', {'name': name})
        return e['value'] if e and e.has_attr('value') else ''

    s._form = {
        '__EVENTTARGET': 'btnSearch', '__EVENTARGUMENT': '',
        '__VIEWSTATE': hidden('__VIEWSTATE'),
        '__VIEWSTATEGENERATOR': hidden('__VIEWSTATEGENERATOR'),
        'today': hidden('today'),
        'sortBy': 'shareholding', 'sortDirection': 'desc',
        'txtStockName': '', 'txtParticipantID': '', 'txtParticipantName': '',
        'txtSelPartID': '',
    }
    return s


def parse_rows(html):
    soup = BeautifulSoup(html, 'html.parser')
    tables = [t for t in soup.find_all('table') if len(t.find_all('tr')) > 10]
    if not tables:
        return pd.DataFrame()
    rows = []
    for tr in tables[-1].find_all('tr')[1:]:
        cells = [td.get_text(' ', strip=True) for td in tr.find_all('td')]
        if len(cells) < 4:
            continue
        rec = {}
        for c in cells:
            if ':' not in c:
                continue
            k, v = c.split(':', 1)
            k, v = k.strip(), v.strip()
            if k.startswith('Participant ID'):
                rec['participant_id'] = v
            elif k.startswith('Name of CCASS Participant'):
                rec['name'] = v
            elif k.startswith('Address'):
                rec['address'] = v
            elif k.startswith('Shareholding'):
                rec['shares'] = int(v.replace(',', '')) if v.replace(',', '').isdigit() else None
            elif k.startswith('% of the total'):
                rec['pct'] = float(v.rstrip('%')) if re.fullmatch(r'[\d.]+%?', v) else None
        if rec.get('participant_id'):
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df.reindex(columns=['participant_id', 'name', 'address', 'shares', 'pct'])


def fetch_one(session, date_str, code):
    data = dict(session._form)
    data['txtShareholdingDate'] = date_str
    data['txtStockCode'] = code
    r = session.post(URL, data=data, timeout=60)
    r.raise_for_status()
    return parse_rows(r.text)


def trading_days(n=1):
    cal = pd.read_pickle('data/hk_broad/snapshot/00700.pkl').index
    cal = pd.to_datetime(cal).tz_convert(None).normalize()
    cal = cal[cal < pd.Timestamp.today().normalize()]
    return list(cal[-n:])


def fridays(n):
    cal = pd.read_pickle('data/hk_broad/snapshot/00700.pkl').index
    cal = pd.to_datetime(cal).tz_convert(None).normalize()
    fri = [d for d in cal if d.weekday() == 4 and d < pd.Timestamp.today().normalize()]
    # HKEX 只保留约 12 个月
    fri = [d for d in fri if d >= pd.Timestamp.today().normalize() - pd.Timedelta(days=330)]
    return fri[-n:]


def save_derived(date_dir, code, df):
    """派生：估算总发行股数（用 pct≥0.5 的参与者中位数）。仅本地。"""
    if df.empty or 'pct' not in df.columns or 'shares' not in df.columns:
        return None
    sub = df[(df['pct'].notna()) & (df['pct'] >= 0.5) & (df['shares'].notna())]
    if sub.empty:
        return None
    est = (sub['shares'] / (sub['pct'] / 100.0)).median()
    return int(round(est)) if est > 0 else None


def run_date(session, date_str, codes, out_root, sleep_s=0.15):
    d = pd.Timestamp(date_str.replace('/', '-')).strftime('%Y%m%d')
    date_dir = os.path.join(out_root, d)
    os.makedirs(date_dir, exist_ok=True)
    stats = {'ok': 0, 'skip': 0, 'fail': 0, 'empty': 0}
    issued = {}
    for code in codes:
        path = os.path.join(date_dir, f'{code}.csv')
        failed = os.path.join(date_dir, f'{code}.failed')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            stats['skip'] += 1
            continue
        if os.path.exists(failed):
            stats['fail'] += 1
            continue
        try:
            df = fetch_one(session, date_str, code)
            if df.empty:
                open(failed, 'w').write('empty')
                stats['empty'] += 1
            else:
                df.to_csv(path, index=False)
                est = save_derived(date_dir, code, df)
                if est:
                    issued[code] = est
                stats['ok'] += 1
        except Exception as e:
            open(failed, 'w').write(str(e)[:200])
            stats['fail'] += 1
        time.sleep(sleep_s)

    return d, stats, issued


def update_manifest(d, stats, root):
    p = os.path.join(root, 'manifest.json')
    m = {}
    if os.path.exists(p):
        try:
            m = json.load(open(p))
        except Exception:
            m = {}
    m.setdefault('dates', {})[d] = stats
    m['last_run'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(m, f, indent=2, sort_keys=True)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='YYYY/MM/DD；默认最近交易日')
    ap.add_argument('--backfill-weeks', type=int, default=0)
    ap.add_argument('--stocks', default=None, help='逗号分隔 5 位代码；默认 310 池')
    ap.add_argument('--out', default=ROOT)
    ap.add_argument('--sleep', type=float, default=0.15)
    ap.add_argument('--workers', type=int, default=1,
                    help='>1 时按分片并发（每片独立 session）')
    ap.add_argument('--retry-failed', action='store_true',
                    help='重试已有 .failed 哨兵的股票（用于补齐当日档案）')
    ap.add_argument('--rebuild-issued', action='store_true',
                    help='不抓取，仅从已落盘 CSV 重建 _issued_shares.csv 与 manifest')
    args = ap.parse_args()

    if args.stocks:
        codes = [c.strip().split('.')[0].zfill(5) for c in args.stocks.split(',')]
    else:
        uni = json.load(open('data/hk_broad/universe_310.json'))
        codes = sorted({c.split('.')[0].zfill(5) for c in uni})

    if args.backfill_weeks:
        dates = [d.strftime('%Y/%m/%d') for d in fridays(args.backfill_weeks)]
    elif args.date:
        dates = [args.date]
    else:
        dates = [d.strftime('%Y/%m/%d') for d in trading_days(1)]

    os.makedirs(args.out, exist_ok=True)
    if args.rebuild_issued:
        for date_str in dates:
            d = pd.Timestamp(date_str.replace('/', '-')).strftime('%Y%m%d')
            date_dir = os.path.join(args.out, d)
            if not os.path.isdir(date_dir):
                continue
            issued, ok = {}, 0
            for f in glob.glob(os.path.join(date_dir, '*.csv')):
                code = os.path.basename(f)[:-4]
                if code.startswith('_'):
                    continue
                df = pd.read_csv(f, dtype={'participant_id': str})
                est = save_derived(date_dir, code, df)
                if est:
                    issued[code] = est
                ok += 1
            if issued:
                pd.Series(issued, name='issued_shares_est').rename_axis('code').to_csv(
                    os.path.join(date_dir, '_issued_shares.csv'), header=True,
                    quoting=csv.QUOTE_NONNUMERIC)
            stats = {'ok': ok, 'skip': 0, 'fail': 0, 'empty': 0}
            update_manifest(d, stats, args.out)
            print(f"{d}: 重建 issued_shares {len(issued)} 只（{ok} 个 CSV）")
        return
    if args.retry_failed:
        for date_str in dates:
            d = pd.Timestamp(date_str.replace('/', '-')).strftime('%Y%m%d')
            for f in glob.glob(os.path.join(args.out, d, '*.failed')):
                os.remove(f)
    print(f"日期: {dates} | 股票: {len(codes)} | 输出: {args.out}")
    for date_str in dates:
        t0 = time.time()
        if args.workers > 1:
            chunks = [codes[i::args.workers] for i in range(args.workers)]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_worker_chunk, date_str, ch, args.out, args.sleep)
                        for ch in chunks]
                parts = [f.result() for f in futs]
            stats = {k: sum(p[1][k] for p in parts) for k in ('ok', 'skip', 'fail', 'empty')}
            issued = {}
            for p in parts:
                issued.update(p[2])
            d = parts[0][0]
        else:
            d, stats, issued = run_date(_session(), date_str, codes, args.out, args.sleep)
        if issued:
            pd.Series(issued, name='issued_shares_est').rename_axis('code').to_csv(
                os.path.join(args.out, d, '_issued_shares.csv'), header=True,
                quoting=csv.QUOTE_NONNUMERIC)
        update_manifest(d, stats, args.out)
        print(f"{date_str} → {d}: {stats}（{time.time()-t0:.0f}s）")


def _worker_chunk(date_str, codes, out_root, sleep_s):
    return run_date(_session(), date_str, codes, out_root, sleep_s)


if __name__ == '__main__':
    main()
