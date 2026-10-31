#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从事件公告 PDF 抽取规模字段（P2a，30 只；缺失标 low，不臆造）。

输出：data/events/event_magnitudes.csv
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'data/event_pdfs'
NUM = r'[\d,]+(?:\.\d+)?'
FX = {'HKD': 1.0, 'USD': 7.8, 'GBP': 10.0, 'CNY': 1.08, 'RMB': 1.08}


def norm(t):
    return re.sub(r'\s+', ' ', t)


def num(s):
    return float(s.replace(',', '')) if s else None


def amount(tok, unit):
    v = num(tok)
    if v is None:
        return None
    unit = (unit or '').lower()
    return v * (1e9 if unit.startswith('b') else 1e6 if unit.startswith('m') else 1)


def first(pat, text, flags=re.I):
    m = re.search(pat, text, flags)
    return m.group(1) if m else None


def first2(pat, text, flags=re.I):
    m = re.search(pat, text, flags)
    return (m.group(1), m.group(2)) if m else (None, None)


def extract_placement(t):
    price = first(r'(?:placing\s+price|issue\s+price|at a price of)\s*(?:of\s*)?(?:HK\$)?\s*(' + NUM + r')', t)
    shares = first(r'(' + NUM + r')\s+Placing Shares', t)
    disc = first(r'discount of approximately\s*(' + NUM + r')\s*%', t)
    proceeds = first2(r'gross proceeds.*?(?:HK\$|US\$)\s*(' + NUM + r')\s*(billion|million)', t)
    return {'issue_price': num(price), 'new_shares': num(shares),
            'discount_pct': num(disc) / 100 if disc else None,
            'proceeds': amount(*proceeds) if proceeds else None}


def extract_cb(t):
    principal = first2(r'aggregate principal amount of\s*(?:U\.S\.\$|US\$|HK\$)?\s*(' + NUM + r')\s*(billion|million)?', t)
    cp = first(r'Conversion Price of\s*(?:HK\$|US\$)?\s*(' + NUM + r')', t)
    cs = first(r'(' + NUM + r')\s+Conversion Shares', t)
    dil = first(r'representing approximately\s*(' + NUM + r')\s*%\s+of the (?:number of )?existing issued', t)
    return {'conversion_price': num(cp), 'conversion_shares': num(cs),
            'cb_principal': amount(*principal) if principal else None,
            'cb_dilution_pct': num(dil) / 100 if dil else None}


def extract_privatisation(t):
    offer = first(r'Scheme Consideration of\s*(?:HK\$|US\$)\s*(' + NUM + r')', t)
    prem = first(r'premium of approximately\s*(' + NUM + r')\s*%', t)
    return {'offer_price': num(offer), 'offer_premium_pct': num(prem) / 100 if prem else None}


def extract_lockup(t):
    days = first(r'(\d+)\s+days?', t)
    return {'lockup_days': int(days) if days else None}


def text_of(url, key):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{key}.pdf')
    if not os.path.exists(p) or os.path.getsize(p) < 10000:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
        r.raise_for_status()
        open(p, 'wb').write(r.content)
    import fitz
    return norm(' '.join(pg.get_text() for pg in fitz.open(p)))


def load_ref(code):
    c5 = str(code).zfill(5)
    p = f'data/hk_broad/snapshot/{c5}.pkl'
    pre = None
    if os.path.exists(p):
        s = pd.read_pickle(p)['Close'].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        pre = s
    fund = c5.lstrip('0').zfill(4) + '.HK'
    shares = None
    try:
        isd = pd.read_pickle(f'data/fund_cache/{fund}_利润表_年度.pkl')
        net = (isd['股东应占溢利'] if '股东应占溢利' in isd else isd['除税后溢利']).dropna()
        eps = isd['每股基本盈利'].dropna()
        v = (net / eps).replace([float('inf'), -float('inf')], pd.NA).dropna()
        if len(v):
            shares = float(v.iloc[-1])
    except Exception:
        pass
    return pre, shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='data/events/events.csv')
    ap.add_argument('--out', default='data/events/event_magnitudes.csv')
    args = ap.parse_args()
    ev = pd.read_csv(args.events)
    ev = ev[ev.stage == 'announcement'].copy()
    rows = []
    for _, r in ev.iterrows():
        if r.event_type not in ('PLACEMENT_COMPANY', 'CB_ISSUANCE', 'PRIVATISATION', 'LOCKUP'):
            continue
        key = f"{str(r.code).zfill(5)}_{r.announce_ts[:10]}"
        try:
            t = text_of(r.url, key)
        except Exception as e:
            rows.append({'code': r.code, 'event_type': r.event_type, 'announce_ts': r.announce_ts,
                         'magnitude_conf': 'missing', 'error': str(e)[:80]})
            continue
        m = {}
        if r.event_type == 'PLACEMENT_COMPANY':
            m = extract_placement(t)
        elif r.event_type == 'CB_ISSUANCE':
            m = extract_cb(t)
        elif r.event_type == 'PRIVATISATION':
            m = extract_privatisation(t)
        elif r.event_type == 'LOCKUP':
            m = extract_lockup(t)

        pre, shares = load_ref(r.code)
        dt = pd.Timestamp(r.announce_ts)
        pre_close = None
        if pre is not None and len(pre):
            s = pre[pre.index < dt.normalize()]
            if len(s):
                pre_close = float(s.iloc[-1])
        new_shares = m.get('new_shares') or m.get('conversion_shares')
        dilution = None
        if new_shares and shares:
            dilution = new_shares / (shares + new_shares) if r.event_type == 'PLACEMENT_COMPANY' else new_shares / shares
        size_pct = None
        proceeds = m.get('proceeds') or (m.get('cb_principal') * FX['USD'] if m.get('cb_principal') else None)
        if proceeds and pre_close and shares:
            size_pct = proceeds / (pre_close * shares)
        discount = m.get('discount_pct')
        if discount is None and m.get('issue_price') and pre_close:
            discount = m['issue_price'] / pre_close - 1
        premium = m.get('offer_premium_pct')
        if premium is None and m.get('offer_price') and pre_close:
            premium = m['offer_price'] / pre_close - 1
        if r.event_type == 'PLACEMENT_COMPANY' and dilution is not None and not (0.005 <= dilution <= 0.5):
            dilution = None  # 解析到的是零股/错数，不猜
        if premium is not None and not (-0.5 <= premium <= 2.0):
            premium = None
        conf = 'high' if (dilution is not None or premium is not None or m.get('lockup_days')) else 'low'
        rows.append({'code': r.code, 'event_type': r.event_type, 'announce_ts': r.announce_ts,
                     'pre_close': pre_close, 'issue_price': m.get('issue_price'),
                     'discount_pct': discount, 'new_shares': new_shares,
                     'dilution_pct': dilution, 'proceeds_hkd': proceeds,
                     'size_pct_mcap': size_pct, 'conversion_price': m.get('conversion_price'),
                     'conversion_shares': m.get('conversion_shares'),
                     'cb_principal_hkd': (m.get('cb_principal') * FX['USD']) if m.get('cb_principal') else None,
                     'offer_price': m.get('offer_price'), 'offer_premium_pct': premium,
                     'lockup_days': m.get('lockup_days'), 'magnitude_conf': conf})
    df = pd.DataFrame(rows)
    df['announce_ts'] = pd.to_datetime(df.announce_ts)
    df = df.sort_values(['code', 'event_type', 'announce_ts']).reset_index(drop=True)
    df['is_followup'] = 0
    for _, idx in df.groupby(['code', 'event_type']).groups.items():
        last = None
        for i in idx:
            t = df.loc[i, 'announce_ts']
            if last is not None and (t - last).days <= 120:
                df.loc[i, 'is_followup'] = 1
            else:
                last = t
    df.to_csv(args.out, index=False)
    print(f'{len(df)} 条；high={sum(df.magnitude_conf=="high")} low={sum(df.magnitude_conf=="low")} missing={sum(df.magnitude_conf=="missing")}')
    print(df[['code', 'event_type', 'dilution_pct', 'discount_pct', 'offer_premium_pct', 'lockup_days', 'magnitude_conf']].to_string(index=False))


if __name__ == '__main__':
    main()
