#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用股票基本面横截面（业界口径，D103）。

通用：PE/PB/ROE/ROA/毛利率/净利率/营收与净利增速/资产负债率/OCF÷净利/股息率/派息率。
金融股（--bank）：额外 NIM/成本收入/信用成本/贷存比/合理 PB。
缺：分部数据、一致预期、NPL/拨备/CET1（需年报）。

用法：
  python3 scripts/stock_fundamentals.py --codes 0700.HK,1810.HK,0883.HK
  python3 scripts/stock_fundamentals.py --codes 0939.HK,1398.HK --bank
  python3 scripts/stock_fundamentals.py --codes ... --ccy HKD --trend --out output/_logs/equity.md
"""
import argparse
import os

import numpy as np
import pandas as pd

FUND = 'data/fund_cache'
SNAP = 'data/hk_broad/snapshot'
FX = {'CNY': 1.08, 'HKD': 1.0, 'USD': 7.8}


def justified_pb(roe, payout, coe=0.11, g_cap=0.05):
    """剩余收益简化：PB* = (ROE - g)/(COE - g)，g=min(ROE*(1-payout), cap)。"""
    g = min(roe * (1 - payout), g_cap)
    if roe <= g or coe <= g:
        return np.nan
    return (roe - g) / (coe - g)


def pick(df, date, names):
    for n in names:
        if date in df.index and n in df.columns and pd.notna(df.loc[date, n]):
            return float(df.loc[date, n])
    return np.nan


def load_stock(code, ccy='CNY', bank=False):
    def L(kind):
        p = os.path.join(FUND, f'{code}_{kind}.pkl')
        return pd.read_pickle(p) if os.path.exists(p) else None

    bs, is_, cf = L('资产负债表_年度'), L('利润表_年度'), L('现金流量表_年度')
    if bs is None or is_ is None or is_.empty:
        return None
    dates = sorted(is_.index)
    d, d0 = dates[-1], dates[-2]

    net = pick(is_, d, ['股东应占溢利', '除税后溢利'])
    eps = pick(is_, d, ['每股基本盈利'])
    rev = pick(is_, d, ['营业额', '营业收入', '收入'])
    rev0 = pick(is_, d0, ['营业额', '营业收入', '收入'])
    gross = pick(is_, d, ['毛利'])
    op = pick(is_, d, ['经营溢利'])
    equity = pick(bs, d, ['归属于母公司股东权益', '总权益'])
    assets = pick(bs, d, ['总资产', '总权益及总负债'])
    assets0 = pick(bs, d0, ['总资产', '总权益及总负债'])
    liab = pick(bs, d, ['总负债'])
    ocf = pick(cf, d, ['经营业务现金净额', '经营产生现金']) if cf is not None else np.nan

    if not (net and eps and equity and assets):
        return None
    p = os.path.join(SNAP, code.split('.')[0].zfill(5) + '.pkl')
    if not os.path.exists(p):
        return None
    px = float(pd.read_pickle(p)['Close'].iloc[-1])

    fx = FX.get(ccy, 1.08)
    shares = net / eps
    bvps = equity / shares * fx
    pe = px / (eps * fx)
    pb = px / bvps
    roe = net / equity
    roa = net / ((assets + assets0) / 2)
    gm = gross / rev if rev and gross == gross else np.nan
    nm = net / rev if rev else np.nan
    rg = rev / rev0 - 1 if rev0 else np.nan
    npg = net / pick(is_, d0, ['股东应占溢利', '除税后溢利']) - 1
    dar = liab / assets if liab == liab else np.nan
    cash_conv = ocf / net if ocf == ocf and net else np.nan
    div = np.nan
    if cf is not None and not cf.empty:
        cols = [c for c in cf.columns if '已付股息' in str(c)]
        if cols:
            div = abs(float(cf.loc[cf.index.max(), cols[0]]))
    payout = div / net if div == div else np.nan
    dy = (div / shares * fx) / px if div == div else np.nan

    row = {'code': code, 'ccy': ccy, 'PB': pb, 'PE': pe, 'ROE': roe, 'ROA': roa,
           '毛利率': gm, '净利率': nm, '营收增速': rg, '净利增速': npg,
           '资产负债率': dar, 'OCF÷净利': cash_conv, '股息率': dy, '派息率': payout}

    if bank:
        nii = pick(is_, d, '净利息收入')
        opi = pick(is_, d, ['经营收入总额', '营业额'])
        opex = pick(is_, d, ['经营支出总额', '营运支出'])
        imp = pick(is_, d, ['减值及拨备'])
        loans = pick(bs, d, ['客户贷款及其他款项'])
        loans0 = pick(bs, d0, ['客户贷款及其他款项'])
        dep = pick(bs, d, ['客户存款'])
        row.update({
            'NIM': nii / ((assets + assets0) / 2) if nii == nii else np.nan,
            '成本收入': opex / opi if opex == opex and opi else np.nan,
            '信用成本': imp / ((loans + loans0) / 2) if imp == imp else np.nan,
            '贷存比': loans / dep if loans == loans and dep else np.nan,
        })
        row['合理PB'] = justified_pb(roe, payout if payout == payout else 0.4)
        row['空间'] = row['合理PB'] / pb - 1 if row['合理PB'] == row['合理PB'] else np.nan
    return row


def to_md(rows, bank=False):
    t = pd.DataFrame(rows)
    cols = ['code', 'ccy', 'PB', 'PE', 'ROE', 'ROA', '毛利率', '净利率',
            '营收增速', '净利增速', '资产负债率', 'OCF÷净利', '股息率', '派息率']
    if bank:
        cols += ['NIM', '成本收入', '信用成本', '贷存比', '合理PB', '空间']
    fmts = {'PB': '{:.2f}', 'PE': '{:.1f}', 'ROE': '{:.1%}', 'ROA': '{:.2%}',
            '毛利率': '{:.1%}', '净利率': '{:.1%}', '营收增速': '{:.1%}', '净利增速': '{:.1%}',
            '资产负债率': '{:.0%}', 'OCF÷净利': '{:.2f}', '股息率': '{:.1%}', '派息率': '{:.0%}',
            'NIM': '{:.2%}', '成本收入': '{:.0%}', '信用成本': '{:.2%}', '贷存比': '{:.0%}',
            '合理PB': '{:.2f}', '空间': '{:+.0%}'}
    out = t[cols].copy()
    for c in cols[2:]:
        out[c] = out[c].map(lambda x: fmts[c].format(x) if pd.notna(x) else '-')
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in out.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', required=True)
    ap.add_argument('--ccy', default='CNY', choices=['CNY', 'HKD', 'USD'],
                    help='报表币种（内地企业多为 CNY；香港/英国多为 HKD/USD）')
    ap.add_argument('--bank', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(',') if c.strip()]
    rows = [r for r in (load_stock(c, ccy=args.ccy, bank=args.bank) for c in codes) if r]
    if not rows:
        raise SystemExit('无可分析股票（检查 fund_cache / 快照 / 币种）')
    text = ('# 股票基本面（业界口径）\n\n'
            f'- 股票：{len(rows)}；报表币种假设：**{args.ccy}**（价格 HKD；错误币种会让 PE/PB/股息率失真）\n'
            '- 数据：akshare 三表 + 快照价格；缺分部、一致预期；金融股缺 NPL/拨备/CET1\n\n'
            + to_md(rows, bank=args.bank) + '\n\n'
            '> 读法：先看增速/毛利率/ROE 趋势，再看现金流质量与负债，最后看估值；\n'
            '> 单一倍数不构成结论，情景不是预测。\n')
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, 'w').write(text)
        print(f'报告: {args.out}')


if __name__ == '__main__':
    main()
