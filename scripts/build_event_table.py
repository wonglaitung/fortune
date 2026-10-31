#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HKEXnews 公告标题构建 PIT 事件表（P0，高精度优先）。

输出：data/events/events.csv + manifest.json（数据目录 gitignore，可重生成）
事件：盈警/盈喜、配售、供股、公开招股、可转债、私有化、协议安排、锁定
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WATCHLIST

OUT = 'data/events'
YEARS = range(2016, 2027)


def load_universe(kind):
    if kind == 'watchlist':
        return dict(WATCHLIST)
    d = json.load(open('data/hk_broad/universe_310.json'))
    return {c: (v.get('name') or c) for c, v in d.items()}

PATTERNS = [
    ('PROFIT_WARNING', re.compile(r'\bPROFIT WARNING\b|盈警', re.I), 'high', '-'),
    ('POSITIVE_PROFIT_ALERT', re.compile(r'\bPOSITIVE PROFIT ALERT\b|盈喜', re.I), 'high', '+'),
    ('RIGHTS_ISSUE', re.compile(r'\bRIGHTS ISSUE\b|供股', re.I), 'review', '-'),
    ('OPEN_OFFER', re.compile(r'\bOPEN OFFER\b', re.I), 'high', '-'),
    ('PLACEMENT_COMPANY', re.compile(r'\bPLACING OF NEW SHARES\b|\bPLACING AND ISSUE OF SHARES\b|\bISSUE OF (?:NEW )?SHARES UNDER GENERAL MANDATE\b|\bSUBSCRIPTION OF NEW SHARES\b', re.I), 'high', '-'),
    ('PLACEMENT_SHAREHOLDER', re.compile(r'\bPLACING OF (?:EXISTING )?SHARES BY\b', re.I), 'review', '-'),
    ('CB_ISSUANCE', re.compile(r'\b(?:PROPOSED )?ISSUE OF .{0,40}CONVERTIBLE (?:BONDS?|NOTES?)\b', re.I), 'high', '-'),
    ('CB_CONVERSION', re.compile(r'\bCONVERSION OF CONVERTIBLE\b|\bADJUSTMENT TO CONVERSION PRICE\b', re.I), 'review', '-'),
    ('PRIVATISATION', re.compile(r'\bPRIVATI[ZS]ATION\b', re.I), 'high', '0'),
    ('SCHEME', re.compile(r'\bSCHEME OF ARRANGEMENT\b', re.I), 'high', '0'),
    ('LOCKUP', re.compile(r'\bLOCK-?UP (?:UNDERTAKING|EXTENSION)\b', re.I), 'review', '+'),
]
EXCLUDE = re.compile(r'SHARE AWARD|SHARE OPTION|SCRIP|DIVIDEND|CBE\b|TRINKAUS|BANK OF COMMUNICATIONS', re.I)


def classify(title):
    if EXCLUDE.search(title):
        return None
    for typ, pat, conf, direction in PATTERNS:
        if pat.search(title):
            return typ, conf, direction
    return None

def stage(title):
    t = title.upper()
    if 'COMPLETION' in t:
        return 'completion'
    if 'MONTHLY UPDATE' in t or 'UPDATE ON' in t:
        return 'update'
    return 'announcement'


def parse_dt(s):
    try:
        return datetime.strptime(s, '%d/%m/%Y %H:%M')
    except ValueError:
        return datetime.strptime(s.split()[0], '%d/%m/%Y')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', choices=['watchlist', '310'], default='310')
    ap.add_argument('--start-year', type=int, default=2016)
    ap.add_argument('--end-year', type=int, default=2026)
    args = ap.parse_args()
    universe = load_universe(args.universe)
    years = range(args.start_year, args.end_year + 1)
    suffix = '' if args.universe == 'watchlist' else '_310'

    os.makedirs(OUT, exist_ok=True)
    rows, seen = [], set()
    for code, name in universe.items():
        c5 = code.split('.')[0].zfill(5)
        for y in years:
            for fn in glob.glob(f'data/hkex_announcements/{c5}_{y}.json'):
                for x in json.load(open(fn)):
                    hit = classify(x.get('TITLE', ''))
                    if not hit:
                        continue
                    typ, conf, direction = hit
                    dt = parse_dt(x['DATE_TIME'])
                    key = (c5, typ, dt.date().isoformat(), x.get('TITLE', '')[:60])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        'code': c5, 'name': name, 'event_type': typ,
                        'announce_ts': dt.isoformat(sep=' '), 'confidence': conf, 'direction': direction,
                        'stage': stage(x.get('TITLE', '')),
                        'title': x.get('TITLE', ''), 'url': 'https://www1.hkexnews.hk' + x.get('FILE_LINK', ''),
                        'source': 'HKEXnews',
                    })
    rows.sort(key=lambda r: (r['announce_ts'], r['code']))
    with open(f'{OUT}/events{suffix}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['code', 'name', 'event_type', 'announce_ts', 'confidence', 'title', 'url', 'source'])
        w.writeheader(); w.writerows(rows)

    manifest = {'generated_at': datetime.now().isoformat(timespec='seconds'),
                'n_events': len(rows), 'by_type': {}, 'by_year': {}, 'by_confidence': {}}
    for r in rows:
        manifest['by_type'][r['event_type']] = manifest['by_type'].get(r['event_type'], 0) + 1
        y = r['announce_ts'][:4]
        manifest['by_year'][y] = manifest['by_year'].get(y, 0) + 1
        manifest['by_confidence'][r['confidence']] = manifest['by_confidence'].get(r['confidence'], 0) + 1
    manifest['universe'] = args.universe
    json.dump(manifest, open(f'{OUT}/manifest{suffix}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
