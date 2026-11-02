#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从港股银行英文年报抽取 NPL / 拨备覆盖率 / CET1 / 资本充足率 / 地产敞口。

用法：python3 scripts/extract_bank_metrics.py
输出：data/bank_reports/bank_metrics.csv + output/_logs/bank_quality.md
"""
import csv
import glob
import json
import os
import re

import fitz

OUT = 'data/bank_reports'
NUM = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')
LABELS = {
    'npl': [r'non-performing loan \(npl\) ratio', r'\bnpl ratio\b',
            r'non-performing loans? to total loans', r'non-performing loan ratio'],
    'coverage': [r'allowances? to npls?', r'allowance coverage ratio',
                 r'allowance for loan impairment losses to non-performing loans'],
    'cet1': [r'common equity tier 1 ratio', r'\bcet1 ratio\b',
             r'common equity tier 1 capital adequacy ratio',
             r'core tier 1 capital adequacy ratio', r'core tier 1 ratio'],
    'car': [r'total capital ratio', r'total capital adequacy ratio'],
    'roe': [r'return on average equity', r'return on average shareholders.? equity'],
    'nim': [r'net interest margin'],
    'cir': [r'cost-to-income ratio', r'cost income ratio'],
}
RANGE = {'npl': (0, 30), 'coverage': (10, 2000), 'cet1': (3, 50), 'car': (5, 60),
         'roe': (-50, 60), 'nim': (-5, 15), 'cir': (5, 120)}


def _nums(line):
    return [x.replace(',', '') for x in NUM.findall(line)]


def _plausible(key, v):
    lo, hi = RANGE.get(key, (-1e9, 1e9))
    return v is not None and lo <= v <= hi


def _match(line, pats):
    for pat in pats:
        m = re.search(pat, line, re.I)
        if m:
            return m
    return None


def _value(lines, i, label):
    line = lines[i]
    m = re.search(re.escape(label), line, re.I)
    rest = line[m.end():] if m else ''
    rest = re.sub(r'^\s*\d{1,2}\s*(?=$|[A-Za-z(])', '', rest)
    rest = re.sub(r'^\s*\(%\)', '', rest)
    for tok in _nums(rest):
        if '.' in tok or ',' in tok or len(tok) >= 3:
            return float(tok)
    for j in range(i + 1, min(i + 4, len(lines))):
        toks = _nums(lines[j])
        if toks:
            return float(toks[0])
    return None


def _scan_page(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    found = {}
    for key, pats in LABELS.items():
        for i, line in enumerate(lines):
            m = _match(line, pats)
            if not m:
                continue
            v = _value(lines, i, m.group(0))
            if _plausible(key, v):
                found[key] = v
                break
    return found


def extract(path):
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc]
    best, best_score = None, 0
    for idx, t in enumerate(pages[:80]):
        f = _scan_page(t)
        score = len(f)
        if score > best_score:
            best, best_score = f, score
    out = dict(best or {})
    # 缺失项再全文档搜一次
    for key in LABELS:
        if key in out:
            continue
        for t in pages:
            f = _scan_page(t)
            if key in f:
                out[key] = f[key]
                break
    out.update(re_loans=None, re_share=None, re_npl=None, re_npl_ratio=None)
    for t in pages:
        low = t.lower()
        if 'by industry' not in low or 'real estate' not in low:
            continue
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            if line.lower().startswith('real estate'):
                toks = _nums(' '.join(lines[i:i + 4]))
                vals = [float(x) for x in toks]
                # 取前 4 个：贷款额, 占比, NPL额, NPL率
                if len(vals) >= 4:
                    out.update(re_loans=vals[0], re_share=vals[1],
                               re_npl=vals[2], re_npl_ratio=vals[3])
                break
        break
    out['pages'] = len(doc)
    return {k: out.get(k) for k in list(LABELS) + ['re_loans', 're_share', 're_npl', 're_npl_ratio', 'pages']}


def main():
    rows = []
    manifest = {}
    mf = os.path.join(OUT, 'manifest.json')
    if os.path.exists(mf):
        manifest = json.load(open(mf))
    for path in sorted(glob.glob(os.path.join(OUT, '[0-9][0-9][0-9][0-9][0-9]_*.pdf'))):
        key = os.path.basename(path)[:-4]
        code, fy = key.split('_')
        meta = manifest.get(key, {})
        try:
            v = extract(path)
        except Exception as e:
            print(f'[fail] {key}: {e}')
            continue
        rows.append({'code': code, 'name': meta.get('name', ''), 'fy': int(fy), **v,
                     'file': os.path.basename(path)})
        print(key, {k: v.get(k) for k in ('npl', 'coverage', 'cet1', 'car', 're_npl_ratio')})

    rows.sort(key=lambda r: (r['code'], r['fy']))
    if rows:
        with open(os.path.join(OUT, 'bank_metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    lines = ['# 港股银行资产质量（年报抽取）', '',
             '| 代码 | 名称 | 财年 | NPL | 拨备覆盖 | CET1 | 总资本 | ROE | 地产贷款占比 | 地产NPL |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        def p(x):
            return f'{x:.2f}%' if isinstance(x, (int, float)) else '—'
        lines.append(f"| {r['code']} | {r['name']} | {r['fy']} | {p(r['npl'])} | {p(r['coverage'])} | "
                     f"{p(r['cet1'])} | {p(r['car'])} | {p(r['roe'])} | {p(r['re_share'])} | {p(r['re_npl_ratio'])} |")
    os.makedirs('output/_logs', exist_ok=True)
    open('output/_logs/bank_quality.md', 'w').write('\n'.join(lines) + '\n')
    print(f'\n{len(rows)} 份年报；CSV: {OUT}/bank_metrics.csv；报告: output/_logs/bank_quality.md')


if __name__ == '__main__':
    main()
