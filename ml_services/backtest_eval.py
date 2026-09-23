#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测评估：对 Walk-forward 的 prediction_analysis.csv 做严谨的合并层面评估

用法：
  python3 ml_services/backtest_eval.py --input output/<dir>/prediction_analysis.csv
  python3 ml_services/backtest_eval.py --input <csv> --horizon 20 --output output/backtest_eval.md

输出：
  - 合并准确率 + 95%CI + 有效样本 + 可靠性 + 显著性
  - 买入胜率 + 基准胜率 + 超额 lift（剔除市场 beta 水分）
  - 板块表现（方向技能 + 胜率 + lift + 可靠性）
  - 逐 fold 准确率
  - 概率校准分桶
  - 逐股票表现（名称 + 方向技能 + 胜率 + lift + 跨月一致性）
  - 逐月表现（准确率/胜率/lift）
  - 月份×股票最佳组合（含 n_eff 与多重比较警告）

  方向技能 = 准确率 − "永远看涨"基准（该组上涨占比），>0 才说明方向判断超越趋势。

跨周期一致性（可选）：
  --compare <另一个周期的 prediction_analysis.csv>
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.eval_stats import summarize, wilson_ci, effective_n

# 与 walk_forward_validation.py 的 TOTAL_COST 保持一致（双边佣金+滑点+印花税 ≈ 0.5%）
DEFAULT_TRANSACTION_COST = 0.005
# 无市场情绪过滤信息时的默认买入阈值
DEFAULT_PROB_THRESHOLD = 0.5
# 无法映射板块时的分组名
SECTOR_UNKNOWN = 'unknown'

SECTOR_NAME_ZH = {
    'bank': '银行', 'tech': '科技', 'semiconductor': '半导体', 'ai': 'AI',
    'new_energy': '新能源', 'environmental': '环保', 'energy': '能源',
    'shipping': '航运', 'gas': '燃气', 'exchange': '交易所',
    'utility': '公用事业', 'insurance': '保险', 'biotech': '生物医药',
    'index': '指数基金', 'real_estate': '地产', 'auto': '汽车', 'consumer': '消费',
}


def _sector_mapping():
    """加载股票 → 板块映射（HK config）；失败时返回空映射"""
    try:
        from config import STOCK_SECTOR_MAPPING
        return STOCK_SECTOR_MAPPING
    except Exception:
        return {}


def _map_sectors(codes, mapping=None):
    """股票代码 → 板块名（未知归为 unknown）"""
    mapping = mapping if mapping is not None else _sector_mapping()

    def _one(code):
        info = mapping.get(str(code)) or {}
        return info.get('sector') or SECTOR_UNKNOWN

    return codes.map(_one)


def _sector_label(sector):
    if sector == SECTOR_UNKNOWN:
        return SECTOR_UNKNOWN
    return f"{SECTOR_NAME_ZH.get(sector, sector)}({sector})"


def _stock_mapping():
    try:
        from config import STOCK_SECTOR_MAPPING
        return STOCK_SECTOR_MAPPING
    except Exception:
        return {}


def _stock_label(code, mapping=None):
    """股票代码 → '代码 名称'（无名称时只显示代码）"""
    mapping = mapping if mapping is not None else _stock_mapping()
    name = (mapping.get(str(code)) or {}).get('name')
    return f"{code} {name}" if name else str(code)


# 月份×股票组合的最小样本门槛（过滤噪声）
MIN_COMBO_N = 15
MIN_COMBO_TRADES = 5


def _normalize_columns(df):
    """兼容 HK / A股 两种列命名"""
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ('stock_code', 'code'):
            rename[c] = 'code'
        elif lc == 'is_correct':
            rename[c] = 'is_correct'
        elif lc in ('predict_prob', 'predict_proba', 'probability'):
            rename[c] = 'prob'
        elif lc == 'actual_return':
            rename[c] = 'actual_return'
        elif lc == 'dynamic_threshold':
            rename[c] = 'dynamic_threshold'
        elif lc == 'market_layer':
            rename[c] = 'market_layer'
        elif lc == 'fold':
            rename[c] = 'fold'
        elif lc == 'date':
            rename[c] = 'date'
    return df.rename(columns=rename)


def _to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def _resolve_trade_mask(df, default_threshold=DEFAULT_PROB_THRESHOLD):
    """还原买入信号：市场情绪过滤后的 filtered_signal

    规则与 market_regime.MarketSentimentFilter.apply_filter 一致：
        signal = (prob >= 动态阈值) and (方向为 UP)
    动态阈值 >= 0.5，因此等价于 prob >= 动态阈值。
    缺失或无过滤列时回退到 default_threshold（0.5）。
    """
    prob = pd.to_numeric(df['prob'], errors='coerce')
    if 'dynamic_threshold' in df.columns:
        thr = pd.to_numeric(df['dynamic_threshold'], errors='coerce')
        thr = thr.where(thr.notna(), default_threshold)
    else:
        thr = default_threshold
    return prob >= thr


def _prepare_win_df(df, cost=DEFAULT_TRANSACTION_COST,
                    default_threshold=DEFAULT_PROB_THRESHOLD):
    """准备买入信号/胜负列：_trade（过滤后买入）、_win（净收益>0）、_sector

    返回 None 表示 CSV 缺少 actual_return / prob 列。
    """
    if 'actual_return' not in df.columns or 'prob' not in df.columns:
        return None
    ret = pd.to_numeric(df['actual_return'], errors='coerce')
    prob = pd.to_numeric(df['prob'], errors='coerce')
    valid = ret.notna() & prob.notna()
    d = df.loc[valid].copy()
    if d.empty:
        return None
    d['ret'] = ret[valid]
    d['prob'] = prob[valid]
    d['_trade'] = _resolve_trade_mask(d, default_threshold)
    d['_win'] = d['ret'] > cost
    if 'code' in d.columns:
        d['_sector'] = _map_sectors(d['code'])
    return d


def _group_full_metrics(group, key, horizon, reliability_threshold):
    """按分组键聚合：样本 / 准确率 / 交易数 / 胜率 / lift / 可靠性

    返回 list[dict]，key 为分组值。
    """
    rows = []
    for gkey, g in group.groupby(key, dropna=True):
        nb = int(len(g))
        n_eff = (nb / horizon) if (horizon and horizon > 0) else float(nb)
        base = float(g['_win'].mean())
        n_trade = int(g['_trade'].sum())
        k_win = int((g['_trade'] & g['_win']).sum())
        wr = (k_win / n_trade) if n_trade > 0 else None
        acc = float(g['is_correct'].mean()) if 'is_correct' in g else None
        # 方向技能的公平基准：该组"永远看涨"的准确率 = 实际上涨占比
        # （Actual_Direction 与 Actual_Return>0 等价，已校验一致）
        naive_up = float((g['ret'] > 0).mean()) if 'ret' in g else None
        acc_skill = (acc - naive_up) if (acc is not None and naive_up is not None) else None
        rows.append({
            'key': gkey,
            'n': nb,
            'n_eff': n_eff,
            'accuracy': acc,
            'naive_up': naive_up,
            'acc_skill': acc_skill,
            'baseline': base,
            'trades': n_trade,
            'win_rate': wr,
            'lift': (wr - base) if wr is not None else None,
            'reliability': 'reliable' if n_eff >= reliability_threshold else 'insufficient',
        })
    return rows


def evaluate_win_rate(df, horizon, cost=DEFAULT_TRANSACTION_COST,
                      reliability_threshold=30, default_threshold=DEFAULT_PROB_THRESHOLD):
    """买入胜率 + 基准胜率 + 超额 lift（含板块 / 个股拆分）

    胜率：通过市场情绪过滤的买入信号中，扣除 cost 双边成本后净收益>0 的比例。
    基准：同一样本集内"无条件买入"净收益>0 的比例（含未交易样本），代表市场 beta。
    lift = 胜率 − 基准，才是剔除行情水分后的选股能力。

    返回 None 表示 CSV 缺少 actual_return / prob 列。
    """
    d = _prepare_win_df(df, cost=cost, default_threshold=default_threshold)
    if d is None:
        return None

    n = int(len(d))
    baseline = float(d['_win'].mean())
    n_trade = int(d['_trade'].sum())
    k_win = int((d['_trade'] & d['_win']).sum())
    win_rate = (k_win / n_trade) if n_trade > 0 else None
    s = summarize(k_win, n_trade, horizon=horizon, p0=baseline,
                  reliability_threshold=reliability_threshold)
    lift = (win_rate - baseline) if win_rate is not None else None

    def _group_rows(group, key):
        rows = []
        for gkey, g in group.groupby(key, dropna=True):
            nb = int(len(g))
            bb = float(g['_win'].mean()) if nb else None
            nt = int(g['_trade'].sum())
            kw = int((g['_trade'] & g['_win']).sum())
            w = (kw / nt) if nt > 0 else None
            rows.append((gkey, nb, bb, nt, w, (w - bb) if w is not None else None))
        return rows

    fold_rows = []
    if 'fold' in d.columns:
        fold_rows = _group_rows(d, 'fold')

    year_rows = []
    if 'date' in d.columns:
        d['_year'] = pd.to_datetime(d['date'], errors='coerce').dt.year
        year_rows = _group_rows(d, '_year')

    sector_rows = []
    if '_sector' in d.columns:
        sector_rows = _group_full_metrics(d, '_sector', horizon, reliability_threshold)
        sector_rows.sort(key=lambda r: -(r['lift'] if r['lift'] is not None else -9e9))

    stock_rows = []
    if 'code' in d.columns:
        stock_rows = _group_full_metrics(d, 'code', horizon, reliability_threshold)
        stock_rows.sort(key=lambda r: -(r['lift'] if r['lift'] is not None else -9e9))

    # 逐月（按自然年月）
    month_rows = []
    if 'date' in d.columns:
        _dt = pd.to_datetime(d['date'], errors='coerce')
        d['_ym'] = _dt.dt.to_period('M').astype(str)
        month_rows = _group_full_metrics(d, '_ym', horizon, reliability_threshold)
        month_rows = [r for r in month_rows if r['key'] != 'NaT']
        month_rows.sort(key=lambda r: str(r['key']))

    # 月份×股票组合 + 个股跨月一致性
    month_stock_rows = []
    if 'code' in d.columns and '_ym' in d.columns:
        for (ym, code), g in d.groupby(['_ym', 'code'], dropna=True):
            nb = int(len(g))
            if nb < MIN_COMBO_N:
                continue
            nt = int(g['_trade'].sum())
            if nt < MIN_COMBO_TRADES:
                continue
            kw = int((g['_trade'] & g['_win']).sum())
            base = float(g['_win'].mean())
            acc = float(g['is_correct'].mean()) if 'is_correct' in g else None
            wr = kw / nt
            month_stock_rows.append({
                'ym': str(ym), 'code': str(code), 'n': nb,
                'n_eff': (nb / horizon) if (horizon and horizon > 0) else float(nb),
                'accuracy': acc, 'baseline': base, 'trades': nt,
                'win_rate': wr, 'lift': wr - base,
            })
        # 跨月一致性：每只股票在多少个月 acc>50% / lift>0
        consistency = {}
        for code, g in d.groupby('code'):
            acc_win = lift_pos = months = 0
            for _ym, gg in g.groupby('_ym'):
                if len(gg) == 0:
                    continue
                months += 1
                if 'is_correct' in gg and gg['is_correct'].mean() > 0.5:
                    acc_win += 1
                t = gg[gg['_trade']]
                if len(t) > 0 and t['_win'].mean() > gg['_win'].mean():
                    lift_pos += 1
            consistency[str(code)] = (months, acc_win, lift_pos)
        for r in stock_rows:
            m = consistency.get(str(r['key']))
            if m:
                r['months'], r['acc_win_months'], r['lift_pos_months'] = m

    return {
        'n': n,
        'trades': n_trade,
        'trade_ratio': (n_trade / n) if n else None,
        'baseline': baseline,
        'wins': k_win,
        'win_rate': win_rate,
        'win_rate_ci_low': s['accuracy_ci_low'],
        'win_rate_ci_high': s['accuracy_ci_high'],
        'lift': lift,
        'p_value_vs_baseline': s['p_value_vs_random'],
        'vs_baseline': s['vs_random'],
        'reliability': s['reliability'],
        'n_effective': s['n_effective'],
        'avg_return_trade': float(d.loc[d['_trade'], 'ret'].mean()) if n_trade > 0 else None,
        'avg_return_all': float(d['ret'].mean()),
        'cost': cost,
        'folds': fold_rows,
        'years': year_rows,
        'sectors': sector_rows,
        'stocks': stock_rows,
        'months': month_rows,
        'month_stocks': month_stock_rows,
    }


def evaluate(df, horizon, reliability_threshold=30, cost=DEFAULT_TRANSACTION_COST):
    df = _normalize_columns(df.copy())
    if 'is_correct' not in df.columns:
        raise ValueError("CSV 缺少 Is_Correct 列")

    df['is_correct'] = _to_bool(df['is_correct'])
    n = len(df)
    k = int(df['is_correct'].sum())

    pooled = summarize(k, n, horizon=horizon, reliability_threshold=reliability_threshold)

    # 逐 fold
    fold_rows = []
    if 'fold' in df.columns:
        for f, g in df.groupby('fold'):
            nf = len(g)
            kf = int(g['is_correct'].sum())
            s = summarize(kf, nf, horizon=horizon, reliability_threshold=reliability_threshold)
            fold_rows.append((f, s))

    # 逐股票
    stock_rows = []
    if 'code' in df.columns:
        for c, g in df.groupby('code'):
            ns = len(g)
            ks = int(g['is_correct'].sum())
            s = summarize(ks, ns, horizon=horizon, reliability_threshold=reliability_threshold)
            stock_rows.append((str(c), s))

    # 概率校准
    calib_rows = []
    if 'prob' in df.columns:
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
        for lo, hi in bins:
            g = df[(df['prob'] >= lo) & (df['prob'] < hi)]
            if len(g) == 0:
                continue
            ng = len(g)
            kg = int(g['is_correct'].sum())
            s = summarize(kg, ng, horizon=horizon, reliability_threshold=reliability_threshold)
            calib_rows.append((lo, hi, s))

    win_rate = evaluate_win_rate(
        df, horizon, cost=cost, reliability_threshold=reliability_threshold)

    return {
        'n': n, 'k': k, 'pooled': pooled,
        'folds': fold_rows, 'stocks': stock_rows, 'calib': calib_rows,
        'win_rate': win_rate,
    }


def _fmt_pct(x):
    return 'N/A' if x is None else f'{x*100:.1f}%'


def _reliability_mark(r):
    return '✅' if r == 'reliable' else '⚠️'


def _fmt_lift(x):
    return 'N/A' if x is None else f'{x*100:+.1f}pp'


def _render_win_rate(wr):
    """买入胜率 + 基准 lift 章节"""
    if wr is None:
        return []
    L = []
    L.append("## 二、买入胜率与基准 lift（超额胜率）\n")
    L.append(f"> 胜率 = 通过市场情绪过滤的买入信号中，扣除 {wr['cost']*100:.1f}% 双边成本后净收益>0 的比例。")
    L.append("> 基准 = 同一样本集内\"无条件买入\"净收益>0 的比例（含未交易样本），代表市场 beta 水分。")
    L.append("> **lift = 胜率 − 基准**，才是剔除行情后的选股能力。\n")
    L.append(f"- 已验证样本: {wr['n']}　交易信号: {wr['trades']} ({_fmt_pct(wr['trade_ratio'])})")
    L.append(f"- 基准胜率: {_fmt_pct(wr['baseline'])}")
    L.append(f"- 信号胜率: **{_fmt_pct(wr['win_rate'])}** "
             f"[{_fmt_pct(wr['win_rate_ci_low'])}, {_fmt_pct(wr['win_rate_ci_high'])}] (95%CI)")
    L.append(f"- **超额 lift: {_fmt_lift(wr['lift'])}** "
             f"(vs 基准: {wr['vs_baseline']}, p={wr['p_value_vs_baseline']:.4f})")
    L.append(f"- 交易信号平均收益: {_fmt_pct(wr['avg_return_trade'])}　|　"
             f"全样本平均收益: {_fmt_pct(wr['avg_return_all'])}\n")

    if wr['years']:
        L.append("### 逐年（揭示牛市水分）\n")
        L.append("| 年份 | 样本 | 基准胜率 | 交易数 | 信号胜率 | 超额 lift |")
        L.append("|------|------|---------|--------|---------|----------|")
        for y, nb, bb, nt, w, lf in sorted(wr['years'], key=lambda r: r[0]):
            L.append(f"| {int(y)} | {nb} | {_fmt_pct(bb)} | {nt} | "
                     f"{_fmt_pct(w)} | {_fmt_lift(lf)} |")
        L.append("")

    if wr['folds']:
        lifts = [lf for _, _, _, _, _, lf in wr['folds'] if lf is not None]
        if lifts:
            L.append("### 逐 Fold\n")
            L.append(f"- lift 标准差: {np.std(lifts)*100:.1f}pp（越小越稳定）\n")
        L.append("| Fold | 交易数 | 信号胜率 | 超额 lift |")
        L.append("|------|--------|---------|----------|")
        for f, _, _, nt, w, lf in wr['folds']:
            L.append(f"| {f} | {nt} | {_fmt_pct(w)} | {_fmt_lift(lf)} |")
        L.append("")

    return L


def _lift_map(rows):
    """分组行 → {str(key): lift}"""
    if not rows:
        return {}
    return {str(r['key']): r['lift'] for r in rows if r.get('lift') is not None}


def _consistency_mark(lift, ref_lift):
    """跨周期一致性：两个周期的 lift 同为正/同为负"""
    if lift is None or ref_lift is None:
        return '—'
    if lift > 0 and ref_lift > 0:
        return '✅ 一致正'
    if lift < 0 and ref_lift < 0:
        return '❌ 一致负'
    return '↔ 混合'


def _render_sectors(wr, compare=None):
    """板块表现章节（逐股聚合后的板块层信号更稳健）"""
    rows = wr.get('sectors') if wr else None
    if not rows:
        return []
    ref = _lift_map(compare.get('sectors')) if compare else None
    L = []
    L.append("## 三、板块表现（方向技能 + 买入 lift）\n")
    L.append("> 按 lift 降序。板块层样本足够（n_eff 较个股高一个量级），信号比个股稳健。")
    L.append("> 「永远看涨」= 该板块上涨占比，即方向准确率的公平基准；")
    L.append("> 「方向技能」= 准确率 − 永远看涨（>0 才说明方向判断超越趋势）。")
    if ref is not None:
        L.append("> 「一致性」= 与对比周期 lift 同号情况。")
    L.append("")
    if ref is not None:
        L.append("| 板块 | 样本 | 永远看涨 | 准确率 | 方向技能 | 信号胜率 | 超额 lift | 可靠性 | 一致性 |")
        L.append("|------|------|---------|--------|---------|---------|----------|--------|--------|")
    else:
        L.append("| 板块 | 样本 | 永远看涨 | 准确率 | 方向技能 | 信号胜率 | 超额 lift | 可靠性 |")
        L.append("|------|------|---------|--------|---------|---------|----------|--------|")
    for r in rows:
        line = (f"| {_sector_label(r['key'])} | {r['n']} | {_fmt_pct(r.get('naive_up'))} | "
                f"{_fmt_pct(r['accuracy'])} | {_fmt_lift(r.get('acc_skill'))} | "
                f"{_fmt_pct(r['win_rate'])} | {_fmt_lift(r['lift'])} | "
                f"{_reliability_mark(r['reliability'])} |")
        if ref is not None:
            line += f" {_consistency_mark(r['lift'], ref.get(str(r['key'])))} |"
        L.append(line)
    L.append("")
    return L


def _fmt_consistency(r):
    """跨月一致性：'acc胜月/总月数 · lift正月'"""
    if not r.get('months'):
        return '—'
    return f"{r.get('acc_win_months', 0)}/{r['months']} · {r.get('lift_pos_months', 0)}"


def _render_stocks(wr, res, compare=None):
    """逐股票表现章节：名称 + 方向技能 + 胜率 lift + 跨月一致性"""
    if wr and wr.get('stocks'):
        rows = [dict(r, key=str(r['key'])) for r in wr['stocks']]
        has_win = True
    else:
        rows = [{'key': str(c), 'n': s['total'], 'n_eff': s['n_effective'],
                 'accuracy': s['accuracy'], 'reliability': s['reliability'],
                 'win_rate': None, 'lift': None} for c, s in res['stocks']]
        has_win = False
    rows.sort(key=lambda r: -(r['lift'] if r.get('lift') is not None else
                              (r['accuracy'] if r.get('accuracy') is not None else -9e9)))
    ref = _lift_map(wr.get('stocks')) if (compare and wr and wr.get('stocks')) else None

    L = []
    L.append("## 六、逐股票表现\n")
    L.append("> ⚠️ 个股有效样本低（20d 约 n/20），排名噪声大；仅「两期一致 + 跨月一致」者相对可信。")
    L.append("> 「永远看涨」= 该股上涨占比（方向准确率基准）；「方向技能」= 准确率 − 永远看涨。")
    L.append("> 「跨月」= 在多少个月 acc>50% / lift>0。")
    if ref is None and has_win:
        L.append("> 传入 `--compare` 可显示跨周期一致性。")
    L.append("")
    if has_win and ref is not None:
        L.append("| 股票 | 样本 | 永远看涨 | 准确率 | 方向技能 | 信号胜率 | 超额 lift | 有效样本 | 跨月(acc胜·lift正) | 可靠性 | 一致性 |")
        L.append("|------|------|---------|--------|---------|---------|----------|---------|------------------|--------|--------|")
    elif has_win:
        L.append("| 股票 | 样本 | 永远看涨 | 准确率 | 方向技能 | 信号胜率 | 超额 lift | 有效样本 | 跨月(acc胜·lift正) | 可靠性 |")
        L.append("|------|------|---------|--------|---------|---------|----------|---------|------------------|--------|")
    else:
        L.append("| 股票 | 样本 | 准确率 | 有效样本 | 可靠性 |")
        L.append("|------|------|--------|---------|--------|")
    for r in rows:
        if has_win:
            line = (f"| {_stock_label(r['key'])} | {r['n']} | {_fmt_pct(r.get('naive_up'))} | "
                    f"{_fmt_pct(r.get('accuracy'))} | {_fmt_lift(r.get('acc_skill'))} | "
                    f"{_fmt_pct(r.get('win_rate'))} | {_fmt_lift(r.get('lift'))} | "
                    f"{r['n_eff']:.1f} | {_fmt_consistency(r)} | {_reliability_mark(r['reliability'])} |")
            if ref is not None:
                line += f" {_consistency_mark(r.get('lift'), ref.get(r['key']))} |"
        else:
            line = (f"| {_stock_label(r['key'])} | {r['n']} | {_fmt_pct(r.get('accuracy'))} | "
                    f"{r['n_eff']:.1f} | {_reliability_mark(r['reliability'])} |")
        L.append(line)
    L.append("")
    return L


def _render_months(wr):
    """逐月表现（按自然年月）"""
    rows = wr.get('months') if wr else None
    if not rows:
        return []
    L = []
    L.append("## 七、逐月表现（按预测月份）\n")
    L.append("> 准确率/胜率月度波动极大；绝对胜率受行情主导，应看超额 lift。")
    L.append("")
    L.append("| 月份 | 样本 | 永远看涨 | 准确率 | 信号胜率 | 基准胜率 | 超额 lift | 交易数 |")
    L.append("|------|------|---------|--------|---------|---------|----------|--------|")
    for r in rows:
        L.append(f"| {r['key']} | {r['n']} | {_fmt_pct(r.get('naive_up'))} | "
                 f"{_fmt_pct(r.get('accuracy'))} | {_fmt_pct(r.get('win_rate'))} | "
                 f"{_fmt_pct(r.get('baseline'))} | {_fmt_lift(r.get('lift'))} | {r.get('trades')} |")
    L.append("")
    return L


def _render_month_stocks(wr):
    """月份×股票最佳组合（含多重比较警告）"""
    rows = wr.get('month_stocks') if wr else None
    if not rows:
        return []
    L = []
    L.append("## 八、月份×股票（最佳组合）⚠️\n")
    L.append("> ⚠️ **多重比较警告**：约 (月数 × 股票数) 个组合，每格 5d 有效样本 `n_eff≈4`（20d≈1）。")
    L.append("> 下表的 90–100% 是多重比较的**必然极值**，不代表可预测；仅供识别，**禁用于择时/选股**。")
    L.append(f"> 过滤门槛：样本 ≥{MIN_COMBO_N}、交易 ≥{MIN_COMBO_TRADES}。\n")

    top_acc = sorted(rows, key=lambda r: -(r['accuracy'] or 0))[:10]
    top_lift = sorted(rows, key=lambda r: -(r['lift'] if r['lift'] is not None else -9e9))[:10]

    def _table(title, data, by):
        T = [f"### {title}\n",
             "| 月份 | 股票 | 样本 | n_eff | 准确率 | 信号胜率 | 超额 lift |",
             "|------|------|------|-------|--------|---------|----------|"]
        for r in data:
            T.append(f"| {r['ym']} | {_stock_label(r['code'])} | {r['n']} | {r['n_eff']:.1f} | "
                     f"{_fmt_pct(r['accuracy'])} | {_fmt_pct(r['win_rate'])} | {_fmt_lift(r['lift'])} |")
        T.append("")
        return T

    L.extend(_table("准确率 TOP10", top_acc, 'accuracy'))
    L.extend(_table("超额 lift TOP10", top_lift, 'lift'))
    return L


def render_markdown(res, horizon, source, compare=None):
    p = res['pooled']
    L = []
    L.append("# 回测评估报告（合并层面）\n")
    L.append(f"- 数据源: `{source}`")
    L.append(f"- 预测周期: {horizon} 天")
    L.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    L.append("## 一、合并准确率\n")
    L.append(f"- 已验证样本: {res['n']}")
    L.append(f"- 正确: {res['k']}")
    L.append(f"- 准确率: **{_fmt_pct(p['accuracy'])}** "
             f"[{_fmt_pct(p['accuracy_ci_low'])}, {_fmt_pct(p['accuracy_ci_high'])}] (95%CI)")
    L.append(f"- 有效独立样本: {p['n_effective']:.1f}")
    L.append(f"- 可靠性: {_reliability_mark(p['reliability'])} {p['reliability']}")
    L.append(f"- vs 随机(50%): **{p['vs_random']}** (p={p['p_value_vs_random']:.4f})\n")

    L.extend(_render_win_rate(res.get('win_rate')))
    L.extend(_render_sectors(res.get('win_rate'), compare))

    L.append("## 四、逐 Fold 准确率\n")
    L.append("| Fold | 样本 | 准确率 | 95%CI | 可靠性 |")
    L.append("|------|------|--------|-------|--------|")
    for f, s in res['folds']:
        L.append(f"| {f} | {s['total']} | {_fmt_pct(s['accuracy'])} | "
                 f"[{_fmt_pct(s['accuracy_ci_low'])}, {_fmt_pct(s['accuracy_ci_high'])}] | "
                 f"{_reliability_mark(s['reliability'])} |")
    L.append("")

    L.append("## 五、概率校准分桶\n")
    L.append("| 概率区间 | 样本 | 准确率 | 95%CI |")
    L.append("|---------|------|--------|-------|")
    for lo, hi, s in res['calib']:
        L.append(f"| [{lo:.0%}-{hi:.0%}) | {s['total']} | {_fmt_pct(s['accuracy'])} | "
                 f"[{_fmt_pct(s['accuracy_ci_low'])}, {_fmt_pct(s['accuracy_ci_high'])}] |")
    L.append("")

    L.extend(_render_stocks(res.get('win_rate'), res, compare))
    L.extend(_render_months(res.get('win_rate')))
    L.extend(_render_month_stocks(res.get('win_rate')))

    return '\n'.join(L)


def main():
    parser = argparse.ArgumentParser(description='回测评估（合并层面）')
    parser.add_argument('--input', type=str, required=True, help='prediction_analysis.csv 路径')
    parser.add_argument('--horizon', type=int, default=20, help='预测周期（默认20）')
    parser.add_argument('--output', type=str, default=None, help='输出 markdown 路径')
    parser.add_argument('--reliability-threshold', type=int, default=30,
                        help='可靠性有效样本阈值（默认30）')
    parser.add_argument('--cost', type=float, default=DEFAULT_TRANSACTION_COST,
                        help='双边交易成本（默认0.005，与 walk-forward 一致）')
    parser.add_argument('--compare', type=str, default=None,
                        help='另一个周期的 prediction_analysis.csv，用于跨周期一致性标注')
    parser.add_argument('--compare-horizon', type=int, default=None,
                        help='对比周期的预测周期（默认与 --horizon 相同）')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    res = evaluate(df, args.horizon, reliability_threshold=args.reliability_threshold,
                   cost=args.cost)

    compare = None
    if args.compare:
        compare_horizon = args.compare_horizon or args.horizon
        compare = evaluate(pd.read_csv(args.compare), compare_horizon,
                           reliability_threshold=args.reliability_threshold, cost=args.cost)
        compare = compare.get('win_rate')

    md = render_markdown(res, args.horizon, args.input, compare=compare)

    print(md)
    out = args.output or os.path.join(
        'output', f"backtest_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\n✅ 报告已保存: {out}")


if __name__ == '__main__':
    main()
