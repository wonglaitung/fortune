#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长期持有候选筛选（310 池）：质量 + 价值 + 股东回报 + 摩擦成本口径。

用途：为"底仓（长期持有）"层提供候选与指标，**不是买卖信号**。
口径见 docs/LONG_TERM_CORE_SCREEN.md。

数据源（全部本地）：
- data/fund_cache/{code}_{利润表|资产负债表|现金流量表}_年度.pkl（akshare 港股年报）
- data/hk_broad/snapshot/{5位}.pkl（前复权行情）
- data/hkex_monthly_shares.csv（FF301 官方股本）

币种口径（重要）：
- akshare 港股财报对**内地企业**保留 CNY，对**港资/英美企业已折算成 HKD**。
- ⇒ 内地企业用 FX['CNY']=1.08 折成 HKD；港资/英美企业**必须按 HKD 直接用**，
  再按 USD 折算会差 7.8 倍（已实测：汇丰 PE 会变成 2.1，实际 16.5）。
- 分类错了 PE/PB/股息率只差约 8%（CNY↔HKD），但 USD 折算错会差 7.8 倍。

用法：
  python3 scripts/screen_longterm_core.py
  python3 scripts/screen_longterm_core.py --out output/_logs/longterm_core_screen.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from stock_fundamentals import load_stock, pick  # noqa: E402

FUND = os.path.join(ROOT, 'data', 'fund_cache')
SNAP = os.path.join(ROOT, 'data', 'hk_broad', 'snapshot')
UNIV = os.path.join(ROOT, 'data', 'hk_broad', 'universe_310.json')
FF301 = os.path.join(ROOT, 'data', 'hkex_monthly_shares.csv')

FX = {'CNY': 1.08, 'HKD': 1.0}

# 港资（本地币 HKD）
HK_LOCAL = {
    '0388', '0001', '0027', '0016', '1997', '1113', '0003', '0002', '0066', '0004', '0880',
    '0012', '1038', '0006', '0017', '0083', '0242', '0014', '0101', '0270', '0288', '0303',
    '0341', '2638', '0683', '0152', '0392', '0371', '1472', '3383', '1622', '6823', '0551',
    '0267', '0023', '1988', '1109', '2388', '0425', '0135', '1193', '0688', '1099',
}
# 英美企业（akshare 已折成 HKD，禁止再按 USD 折算）
FOREIGN = {'0005', '1299', '2888', '0019', '1910', '1913'}

FIN_SECTORS = ('bank', 'insurance', 'financial')


HK_LOCAL = {c.zfill(5) for c in HK_LOCAL}
FOREIGN = {c.zfill(5) for c in FOREIGN}


def ccy_of(code5):
    if code5 in FOREIGN or code5 in HK_LOCAL:
        return 'HKD'
    return 'CNY'


def annual(code, kind):
    p = os.path.join(FUND, f'{code}_{kind}.pkl')
    return pd.read_pickle(p) if os.path.exists(p) else None


def series(df, names, n=8):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out = {}
    for dt in sorted(df.index)[-n:]:
        v = pick(df, dt, names)
        out[dt] = v
    return pd.Series(out, dtype=float)


def latest_ff301():
    if not os.path.exists(FF301):
        return {}
    d = pd.read_csv(FF301, dtype={'code': str})
    d['code'] = d.code.str.zfill(5)
    d = d.sort_values('month_end').groupby('code').tail(1)
    return d.set_index('code')['issued'].to_dict()


def price_stats(code5):
    p = os.path.join(SNAP, code5 + '.pkl')
    if not os.path.exists(p):
        return {}
    d = pd.read_pickle(p).copy()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    c = d['Close'].dropna()
    if len(c) < 300:
        return {}
    def cagr(n):
        if len(c) <= n * 240:
            return np.nan
        return (c.iloc[-1] / c.iloc[-n * 240]) ** (1 / n) - 1
    turnover = (d['Close'] * d['Volume']).dropna()
    s5 = c.iloc[-1200:] if len(c) > 1200 else c
    return {'px_snap': float(c.iloc[-1]), 'snap_date': str(c.index[-1].date()),
            'cagr3': cagr(3), 'cagr5': cagr(5), 'cagr10': cagr(10),
            'mdd5': float((s5 / s5.cummax() - 1).min()),
            'ret1y': float(c.iloc[-1] / c.iloc[-242] - 1) if len(c) > 242 else np.nan,
            'adv': float(turnover.tail(120).median())}


def build():
    uni = json.load(open(UNIV))
    ff = latest_ff301()
    rows = []
    for code, meta in uni.items():
        c5 = code.split('.')[0].zfill(5)
        is_ = annual(code, '利润表_年度')
        bs = annual(code, '资产负债表_年度')
        cf = annual(code, '现金流量表_年度')
        if is_ is None or is_.empty:
            continue
        ccy = ccy_of(c5)
        sec = meta.get('sector', '')
        financial = sec in FIN_SECTORS
        rev = series(is_, ['营业额', '营业收入', '收入'], 8)
        net = series(is_, ['股东应占溢利', '除税后溢利'], 8)
        gross = series(is_, ['毛利'], 8)
        eq = series(bs, ['归属于母公司股东权益', '股东权益', '总权益'], 8)
        assets = series(bs, ['总资产', '总权益及总负债'], 8)
        liab = series(bs, ['总负债'], 8)
        ocf = series(cf, ['经营业务现金净额', '经营产生现金'], 8)
        r = {'code': code, 'name': meta.get('name', code), 'sector': sec, 'ccy': ccy,
             'financial': financial, 'fy': str(rev.index[-1].year) if len(rev) else None,
             'yrs': len(net), 'neg_years': int((net < 0).sum())}
        r['profit_cagr3'] = ((net.iloc[-1] / net.iloc[-4]) ** (1 / 3) - 1
                             if len(net) >= 4 and net.iloc[-4] > 0 and net.iloc[-1] > 0 else np.nan)
        r['rev_cagr3'] = ((rev.iloc[-1] / rev.iloc[-4]) ** (1 / 3) - 1
                          if len(rev) >= 4 and rev.iloc[-4] > 0 and rev.iloc[-1] > 0 else np.nan)
        yoy = net.pct_change()
        r['worst_yoy'] = float(yoy.min()) if len(yoy.dropna()) else np.nan
        roes = []
        for i in range(1, 4):
            if len(net) >= i and len(eq) >= i and eq.iloc[-i] and eq.iloc[-i] > 0:
                roes.append(net.iloc[-i] / eq.iloc[-i])
        r['roe3'] = float(np.mean(roes)) if roes else np.nan
        r['nm'] = net.iloc[-1] / rev.iloc[-1] if len(net) and len(rev) and rev.iloc[-1] else np.nan
        r['gm'] = (gross.iloc[-1] / rev.iloc[-1]
                   if len(gross) and len(rev) and rev.iloc[-1] and gross.iloc[-1] == gross.iloc[-1] else np.nan)
        r['dar'] = (liab.iloc[-1] / assets.iloc[-1]
                    if len(liab) and len(assets) and assets.iloc[-1] and liab.iloc[-1] == liab.iloc[-1] else np.nan)
        if not financial:
            cc = (ocf / net).replace([np.inf, -np.inf], np.nan).dropna()
            r['cash_conv3'] = float(cc.tail(3).mean()) if len(cc) else np.nan
        else:
            r['cash_conv3'] = np.nan
        # 股本一致性：隐含股数（净利/EPS）vs FF301 官方股本
        implied = None
        try:
            if len(net) and net.iloc[-1] and net.iloc[-1] == net.iloc[-1]:
                eps = pick(is_, sorted(is_.index)[-1], ['每股基本盈利'])
                if eps:
                    implied = net.iloc[-1] / eps
        except Exception:
            implied = None
        iss = ff.get(c5)
        ratio = (float(iss) / implied) if (implied and iss) else np.nan
        r['shares_implied'] = implied
        r['ff301_issued'] = iss
        r['share_ratio'] = ratio
        # 注：A+H 公司的 FF301 常只报 H 股 ⇒ 比率 <0.8 属已知口径差异，不等于数据错误；
        #     需回年报核对的是无 A 股却比率显著偏离的公司。
        r['share_flag'] = '' if not ratio == ratio else ('ok' if 0.8 <= ratio <= 1.2 else 'FF301差异大')
        try:
            fs = load_stock(code, ccy=ccy, bank=financial)
        except Exception:
            fs = None
        if fs:
            for k in ('PE', 'PB', '股息率', '派息率', '净回购收益率', '总股东回报', '合理PB', '空间'):
                if k in fs:
                    r[k] = fs[k]
        r.update(price_stats(c5))
        rows.append(r)
    return pd.DataFrame(rows)


def screen(df, min_roe=0.10, min_cc=0.9, max_pe=25, max_pb=1.5, min_tsr=0.02,
           min_adv=5e7, bank_roe=0.08, bank_pe=8, bank_pb=0.95, bank_dy=0.03,
           max_dar=0.70, max_fail_year=-0.30):
    """预注册筛选口径（非金融 / 金融分开）。返回 (通过代码集合, 每股失败原因)。"""
    fin = df[df.financial == True]      # noqa: E712
    non = df[df.financial != True]      # noqa: E712
    reasons = {}

    def why(code, ok, msgs):
        if not ok:
            reasons[code] = '、'.join(msgs)

    for _, r in non.iterrows():
        m = []
        if not (r.yrs >= 5):
            m.append('年数<5')
        if r.neg_years > 0:
            m.append('有亏损年')
        if not (r.roe3 >= min_roe):
            m.append(f'ROE3<{min_roe:.0%}')
        if not (r.cash_conv3 >= min_cc):
            m.append(f'OCF/净利<{min_cc}')
        if not (r.rev_cagr3 >= 0):
            m.append('营收3yCAGR<0')
        if not (r.profit_cagr3 >= 0):
            m.append('净利3yCAGR<0')
        if not (r.worst_yoy > max_fail_year):
            m.append(f'最差年份<{max_fail_year:.0%}')
        if not (r.dar <= max_dar):
            m.append(f'负债率>{max_dar:.0%}')
        if not (((r.PE > 0) & (r.PE <= max_pe)) | ((r.PB > 0) & (r.PB <= max_pb))):
            m.append('估值超限')
        if not (r.总股东回报 >= min_tsr):
            m.append(f'总股东回报<{min_tsr:.0%}')
        if not (r.adv >= min_adv):
            m.append('流动性不足')
        why(r.code, not m, m)

    for _, r in fin.iterrows():
        m = []
        if not (r.yrs >= 5):
            m.append('年数<5')
        if not (r.roe3 >= bank_roe):
            m.append(f'ROE3<{bank_roe:.0%}')
        if not (r.profit_cagr3 >= 0):
            m.append('净利3yCAGR<0')
        if not (r.worst_yoy > -0.20):
            m.append('最差年份<-20%')
        if not (((r.PE > 0) & (r.PE <= bank_pe)) | ((r.PB > 0) & (r.PB <= bank_pb))):
            m.append('估值超限')
        if not (r.股息率 >= bank_dy):
            m.append(f'股息率<{bank_dy:.0%}')
        if not (r.adv >= min_adv):
            m.append('流动性不足')
        why(r.code, not m, m)
    return set(df.code) - set(reasons), reasons


def refresh_prices(df):
    """用腾讯行情刷新最新价，并按比例重算 PE/PB/股息率/净回购收益率。

    快照价常滞后数周（data/hk_broad/snapshot 为周更缓存），估值必须用当日价重算。
    """
    from data_services.tencent_finance import get_hk_stock_data_tencent
    out = []
    for _, r in df.iterrows():
        c5 = r.code.split('.')[0].zfill(5)
        try:
            d = get_hk_stock_data_tencent(c5, 30)
            px_new = float(d['Close'].iloc[-1]) if d is not None and len(d) else np.nan
            dt_new = str(pd.to_datetime(d.index[-1]).tz_localize(None).date()) if d is not None and len(d) else ''
        except Exception:
            px_new, dt_new = np.nan, ''
        k = px_new / r.px_snap if (px_new == px_new and r.px_snap) else np.nan
        r['px'] = px_new
        r['px_date'] = dt_new
        if k == k:
            for k_ in ('PE', 'PB'):
                if k_ in r and r[k_] == r[k_]:
                    r[k_] = r[k_] * k
            for k_ in ('股息率', '净回购收益率', '总股东回报'):
                if k_ in r and r[k_] == r[k_]:
                    r[k_] = r[k_] / k
        out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


def md_table(df, cols, fmts):
    out = df[cols].copy()
    for c in cols:
        if c in fmts:
            out[c] = out[c].map(lambda x: fmts[c].format(x) if pd.notna(x) else '-')
    head = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols)
    body = '\n'.join('| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for _, r in out.iterrows())
    return head + '\n' + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', '_logs', 'longterm_core_screen.csv'))
    ap.add_argument('--min-adv', type=float, default=5e7, help='最低中位日成交额（HKD）')
    ap.add_argument('--refresh-prices', action='store_true', help='用腾讯行情刷新最新价并重算估值')
    args = ap.parse_args()
    df = build()
    if args.refresh_prices:
        df = refresh_prices(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    passed, reasons = screen(df, min_adv=args.min_adv)
    print(f'样本 {len(df)} 只；通过筛选 {len(passed)} 只；明细 → {args.out}')
    cols = ['code', 'name', 'sector', 'ccy', 'fy', 'roe3', 'rev_cagr3', 'profit_cagr3',
            'cash_conv3', 'dar', 'PE', 'PB', '股息率', '总股东回报', 'adv', 'cagr5', 'mdd5']
    if 'px' in df.columns:
        cols = ['code', 'name', 'sector', 'ccy', 'px'] + cols[4:]
    fmts = {'roe3': '{:.1%}', 'rev_cagr3': '{:.1%}', 'profit_cagr3': '{:.1%}',
            'cash_conv3': '{:.2f}', 'dar': '{:.0%}', 'PE': '{:.1f}', 'PB': '{:.2f}',
            '股息率': '{:.1%}', '总股东回报': '{:.1%}', 'adv': '{:,.0f}',
            'cagr5': '{:.1%}', 'mdd5': '{:.0%}', 'px': '{:.2f}'}
    p = df[df.code.isin(passed)].sort_values(['financial', 'roe3'], ascending=[False, False])
    print(md_table(p, cols, fmts))
    print('\n未通过原因（示例 40 条）：')
    for c, why in list(reasons.items())[:40]:
        print(f'  {c}: {why}')


if __name__ == '__main__':
    main()
