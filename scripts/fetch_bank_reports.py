#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载港股银行年报（英文版，用于 NPL/拨备/CET1/地产敞口抽取）。

数据源：HKEXnews 公告索引（项目已抓取的 data/hkex_announcements/*.json）
输出：data/bank_reports/{code}_{fy}.pdf + manifest.json（数据目录已 gitignore）
"""
import glob
import json
import os
import re
import sys

import requests

BASE = 'https://www1.hkexnews.hk'
OUT = 'data/bank_reports'
BANKS = {
    '00939': '建设银行', '01398': '工商银行', '01288': '农业银行',
    '03988': '中国银行', '03968': '招商银行', '03328': '交通银行',
    '02388': '中银香港', '00005': '汇丰控股',
}
FY = [2021, 2022, 2023, 2024]


def _mb(s):
    m = re.match(r'([\d.]+)\s*(MB|KB)', s or '')
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1024 if m.group(2) == 'KB' else v


def pick_report(code, pub_year):
    fs = glob.glob(f'data/hkex_announcements/{code}_{pub_year}.json')
    if not fs:
        return None
    items = []
    for x in json.load(open(fs[0])):
        t = (x.get('TITLE') or '').lower()
        if 'annual report' not in t:
            continue
        if any(k in t for k in ('notification', 'letter', 'form 20-f', 'supplementary',
                                'interim', 'summary', 'esg', 'sustainability',
                                'circular', 'notice')):
            continue
        if x.get('FILE_TYPE') != 'PDF':
            continue
        items.append(x)
    if not items:
        return None
    # 优先无括号附注的正式年报，再取最大文件，再取最新日期
    items.sort(key=lambda x: ('(' in x['TITLE'], _mb(x.get('FILE_INFO')), x.get('DATE_TIME', '')), reverse=False)
    items.sort(key=lambda x: (_mb(x.get('FILE_INFO')), x.get('DATE_TIME', '')), reverse=True)
    items.sort(key=lambda x: ('(' in x['TITLE'], -_mb(x.get('FILE_INFO'))))
    items[0] = max((x for x in items if '(' not in x['TITLE']), key=lambda x: _mb(x['FILE_INFO']), default=items[0])
    x = items[0]
    m = re.search(r'(20\d{2})', x['TITLE'])
    fy = int(m.group(1)) if m else pub_year - 1
    return {'fy': fy, 'url': BASE + x['FILE_LINK'], 'title': x['TITLE'],
            'date': x['DATE_TIME'], 'size': x['FILE_INFO']}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    mf = os.path.join(OUT, 'manifest.json')
    if os.path.exists(mf):
        manifest = json.load(open(mf))
    sess = requests.Session()
    sess.headers['User-Agent'] = 'Mozilla/5.0'
    for code, name in BANKS.items():
        for fy in FY:
            pub = fy + 1
            meta = pick_report(code, pub)
            if not meta or meta['fy'] != fy:
                print(f'[skip] {code} FY{fy}: 未找到年报')
                continue
            dst = os.path.join(OUT, f'{code}_{fy}.pdf')
            key = f'{code}_{fy}'
            manifest[key] = {'code': code, 'name': name, 'fy': fy, **meta}
            if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
                print(f'[have] {key} {os.path.getsize(dst)/1e6:.1f}MB')
                continue
            try:
                r = sess.get(meta['url'], timeout=180, stream=True)
                r.raise_for_status()
                tmp = dst + '.part'
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dst)
                print(f'[ok] {key} {os.path.getsize(dst)/1e6:.1f}MB')
            except Exception as e:
                print(f'[fail] {key}: {e}')
                if os.path.exists(dst + '.part'):
                    os.remove(dst + '.part')
    json.dump(manifest, open(mf, 'w'), ensure_ascii=False, indent=1)
    print(f'manifest: {len(manifest)} 条')


if __name__ == '__main__':
    main()
