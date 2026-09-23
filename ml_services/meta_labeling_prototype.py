#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三重障碍 + 元标签（Triple Barrier + Meta-labeling）原型

思路（de Prado）：
  1) 主模型（已有 CatBoost）负责方向，产出买入信号（prob >= Dynamic_Threshold）。
  2) 用三重障碍（波动率缩放的止盈/止损 + 时间障碍）为每笔信号生成"是否获利"标签。
  3) 训练元模型预测 P(主模型这笔会赢)，用 purged/embargo walk-forward 过滤低置信交易。
  4) 报告过滤前后：precision / recall / F1 / 保留率 / 净收益 / 净 IR。

用法：
  python3 ml_services/meta_labeling_prototype.py --horizon 5
  python3 ml_services/meta_labeling_prototype.py --horizon 20

依赖：prediction_analysis.csv（主模型 Walk-forward 输出）+ data/feature_cache/*_shift.pkl（OHLC）
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST = 0.005            # 双边成本（与 walk_forward TOTAL_COST 一致）
PT = 1.0                # 上障碍倍数（× 波动率）
SL = 1.0                # 下障碍倍数
VOL_SPAN = 20           # 日波动率 EWMA 窗口
MIN_TRAIN = 800         # 元模型最小训练样本
THRESH_GRID = np.arange(0.30, 0.80, 0.05)

# 元模型使用的原始特征（来自特征缓存；主模型预测/市场层级另外拼接）
META_COLS = [
    'RSI', 'RSI_ROC', 'Vol_Ratio', 'MA_Ratio_5d', 'MA_Ratio_60d', 'MA_Ratio_120d',
    'ATR_Ratio', 'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d',
    'Turnover_Z_Score', 'VIX_Level', 'VIX_Ratio_MA20', 'Market_Regime_Encoded',
]


# ----------------------------- 数据加载 -----------------------------

def _latest_cache_map(cache_dir='data/feature_cache'):
    """{code: 最新 _shift.pkl 路径}"""
    m = {}
    for p in glob.glob(os.path.join(cache_dir, '*_shift.pkl')):
        code = os.path.basename(p).split('_')[0]
        if code not in m or p > m[code]:
            m[code] = p
    return m


def load_prices_and_features(codes, cache_dir='data/feature_cache'):
    """加载每只股票的 OHLC + 元特征（只取需要的列，控制内存）"""
    cmap = _latest_cache_map(cache_dir)
    meta_cols = META_COLS
    out = {}
    for code in codes:
        p = cmap.get(code)
        if not p:
            continue
        try:
            df = pd.read_pickle(p)['data']['stock_df']
        except Exception:
            continue
        keep = ['Open', 'High', 'Low', 'Close'] + [c for c in meta_cols if c in df.columns]
        d = df[keep].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        out[code] = d
    return out


# ----------------------------- 三重障碍 -----------------------------

def triple_barrier(px, t, horizon, pt=PT, sl=SL, vol_span=VOL_SPAN):
    """对 px 在日期 t 生成三重障碍结果

    返回 (tb_bin, tb_return)：
      tb_bin ∈ {+1, -1, 0}（触上 / 触下 / 到期）
      tb_return = 障碍路径收益（+width / -width / 到期收益）
    """
    idx = px.index
    if t not in idx:
        return None
    i = idx.get_loc(t)
    if i + horizon >= len(px):
        return None
    close = px['Close']
    c0 = close.iloc[i]
    if not np.isfinite(c0) or c0 <= 0:
        return None
    rets = close.pct_change()
    vol = rets.ewm(span=vol_span, min_periods=10).std().iloc[i]
    if not np.isfinite(vol) or vol <= 0:
        return None
    unit = vol * np.sqrt(horizon)          # 1 倍波动率的障碍宽度
    up_w = pt * unit
    dn_w = sl * unit
    upper = c0 * (1 + up_w)
    lower = c0 * (1 - dn_w)

    highs = px['High'].iloc[i + 1:i + 1 + horizon].values
    lows = px['Low'].iloc[i + 1:i + 1 + horizon].values
    for j in range(horizon):
        hi, lo = highs[j], lows[j]
        hit_up = np.isfinite(hi) and hi >= upper
        hit_dn = np.isfinite(lo) and lo <= lower
        if hit_up and hit_dn:
            return -1, -dn_w                     # 同日双触：保守按止损
        if hit_up:
            return 1, up_w
        if hit_dn:
            return -1, -dn_w
    final = close.iloc[i + horizon] / c0 - 1
    if not np.isfinite(final):
        return None
    return 0, final


# ----------------------------- 元特征 -----------------------------

def build_dataset(pred_csv, cache_dir='data/feature_cache'):
    pred = pd.read_csv(pred_csv).rename(columns={'Fold': 'fold'})
    pred['date'] = pd.to_datetime(pred['Date'], errors='coerce')
    pred['code'] = pred['Stock_Code'].astype(str).str.replace('.HK', '', regex=False).str.zfill(4)
    for c in ('Predict_Prob', 'Actual_Return', 'Dynamic_Threshold', 'Market_Up_Ratio'):
        if c in pred.columns:
            pred[c] = pd.to_numeric(pred[c], errors='coerce')
    pred = pred.dropna(subset=['date', 'Predict_Prob', 'Dynamic_Threshold'])

    prices = load_prices_and_features(sorted(pred['code'].unique()), cache_dir)
    h = None  # filled by caller via closure param

    return pred, prices


def attach_labels(pred, prices, horizon):
    """为每条信号计算三重障碍结果与元标签"""
    tb_bin, tb_ret, daily_vol = [], [], []
    for _, row in pred.iterrows():
        px = prices.get(row['code'])
        res = triple_barrier(px, row['date'], horizon) if px is not None else None
        if res is None:
            tb_bin.append(np.nan); tb_ret.append(np.nan); daily_vol.append(np.nan)
        else:
            b, r = res
            tb_bin.append(b); tb_ret.append(r)
            close = px['Close']
            v = close.pct_change().ewm(span=VOL_SPAN, min_periods=10).std().get(row['date'], np.nan)
            daily_vol.append(v)
    pred = pred.copy()
    pred['tb_bin'] = tb_bin
    pred['tb_ret'] = tb_ret
    pred['daily_vol'] = daily_vol
    # 元标签：主模型做多，这笔是否获利（未扣成本）
    pred['meta_label'] = ((pred['tb_bin'] > 0) |
                          ((pred['tb_bin'] == 0) & (pred['tb_ret'] > 0))).astype(int)
    # 净赢：扣成本后为正
    pred['net_win'] = (pred['tb_ret'] > COST).astype(int)
    return pred


# ----------------------------- 元特征矩阵 -----------------------------

def meta_feature_frame(pred, prices):
    rows = []
    for _, r in pred.iterrows():
        px = prices.get(r['code'])
        f = {}
        if px is not None and r['date'] in px.index:
            fr = px.loc[r['date']]
            for c in META_COLS:
                if c in px.columns:
                    f[c] = fr.get(c, np.nan)
        rows.append(f)
    fx = pd.DataFrame(rows, index=pred.index)
    fx['prob'] = pred['Predict_Prob'].values
    fx['margin'] = pred['Predict_Prob'].values - 0.5
    fx['dyn_thresh'] = pred['Dynamic_Threshold'].values
    fx['mkt_up_ratio'] = pred['Market_Up_Ratio'].values if 'Market_Up_Ratio' in pred.columns else np.nan
    fx['daily_vol'] = pred['daily_vol'].values
    # market_layer one-hot
    if 'Market_Layer' in pred.columns:
        for lv in ('normal', 'weak', 'bear', 'extreme_bear', 'unknown'):
            fx[f'layer_{lv}'] = (pred['Market_Layer'].astype(str) == lv).astype(int)
    return fx


# ----------------------------- purged walk-forward -----------------------------

def purged_walk_forward(pred, X, horizon, return_calibrated=False):
    """按 Fold 顺序：训练 folds < f，embargo=horizon 天，预测 fold f

    return_calibrated=True 时额外返回逐折 Isotonic 校准后的概率。
    """
    folds = sorted(pred['fold'].unique()) if 'fold' in pred.columns else []
    if not folds:
        empty = pd.Series(np.nan, index=pred.index)
        return (empty, empty, {}) if return_calibrated else (empty, {})
    pred = pred.copy()
    meta_prob = pd.Series(np.nan, index=pred.index)
    cal_prob = pd.Series(np.nan, index=pred.index) if return_calibrated else None
    thresholds = {}
    for f in folds:
        test_mask = pred['fold'] == f
        test = pred[test_mask]
        if len(test) == 0:
            continue
        test_start = test['date'].min()
        train_mask = pred['fold'] < f
        # embargo：训练集去掉测试开始前 horizon 天的样本
        embargo_cut = test_start - pd.Timedelta(days=horizon)
        train_mask &= pred['date'] < embargo_cut
        tr = pred[train_mask]
        if len(tr) < MIN_TRAIN or tr['meta_label'].nunique() < 2:
            continue
        model = _fit_meta(X.loc[tr.index], tr['meta_label'].values)
        if model is None:
            continue
        # 阈值在训练集上按 F1 选
        p_tr = model.predict_proba(X.loc[tr.index])[:, 1]
        thr = _pick_threshold(p_tr, tr['meta_label'].values)
        thresholds[f] = thr
        p_te = model.predict_proba(X.loc[test.index])[:, 1]
        if return_calibrated:
            try:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(p_tr, tr['meta_label'].values)
                cal_prob.loc[test.index] = iso.predict(p_te)
            except Exception:
                cal_prob.loc[test.index] = p_te
        meta_prob.loc[test.index] = p_te
    if return_calibrated:
        return meta_prob, cal_prob, thresholds
    return meta_prob, thresholds


def _fit_meta(X, y):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                           learning_rate=0.05, l2_regularization=1.0,
                                           random_state=42)
        m.fit(X.values, y)
        return m
    except Exception:
        return None


def _pick_threshold(p, y):
    best_t, best_f1 = 0.5, -1
    for t in THRESH_GRID:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ----------------------------- 评估 -----------------------------

def _metrics(net_win, net_ret, horizon, total_wins):
    n = len(net_win)
    if n == 0:
        return dict(n=0, precision=np.nan, recall=np.nan, f1=np.nan, retention=np.nan,
                    win_rate=np.nan, mean_ret=np.nan, ir=np.nan)
    tp = int(net_win.sum())
    precision = tp / n
    recall = tp / total_wins if total_wins > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (recall and precision + recall > 0) else np.nan
    mean_r = float(np.mean(net_ret))
    std_r = float(np.std(net_ret, ddof=1)) if n > 1 else 0.0
    ann = 252.0 / horizon
    ir = (mean_r / std_r * np.sqrt(ann)) if std_r > 0 else np.nan
    return dict(n=n, precision=precision, recall=recall, f1=f1, retention=np.nan,
                win_rate=precision, mean_ret=mean_r, ir=ir)


def evaluate(pred, meta_prob, horizon, threshold_default=0.5):
    """比较 baseline（全部信号）vs meta 过滤后"""
    d = pred.dropna(subset=['tb_ret']).copy()
    d['meta_prob'] = meta_prob.reindex(d.index)
    d['net_ret'] = d['tb_ret'] - COST
    d['net_win'] = (d['net_ret'] > 0).astype(int)
    # 仅在有元模型预测的区间做公平对比
    valid = d['meta_prob'].notna()
    dv = d[valid]
    total_wins = int(dv['net_win'].sum())

    base = _metrics(dv['net_win'].values, dv['net_ret'].values, horizon, total_wins)
    base['retention'] = 1.0
    # meta 阈值：用每折训练集选出的阈值（存在 pred['_thr']）；缺失用默认
    thr = d.get('_thr', pd.Series(threshold_default, index=d.index))
    kept = dv['meta_prob'] >= thr.reindex(dv.index).fillna(threshold_default)
    meta = _metrics(dv.loc[kept, 'net_win'].values, dv.loc[kept, 'net_ret'].values,
                    horizon, total_wins)
    meta['retention'] = len(dv[kept]) / len(dv) if len(dv) else np.nan
    return base, meta, len(dv)


# ----------------------------- 主流程 -----------------------------

def run(horizon, pred_csv, cache_dir, out_md):
    print(f"\n{'='*70}\n三重障碍 + 元标签原型  |  horizon={horizon}\n{'='*70}")
    pred, prices = build_dataset(pred_csv, cache_dir)
    # 仅保留买入信号（主模型做多）
    sig = pred[pred['Predict_Prob'] >= pred['Dynamic_Threshold']].copy()
    print(f"主模型买入信号: {len(sig)} / {len(pred)}")
    sig = attach_labels(sig, prices, horizon)
    sig = sig.dropna(subset=['tb_ret']).reset_index(drop=True)
    print(f"成功生成三重障碍标签: {len(sig)}")

    X = meta_feature_frame(sig, prices).reset_index(drop=True)
    sig = sig.reset_index(drop=True)

    meta_prob, thresholds = purged_walk_forward(sig, X, horizon)
    sig['_thr'] = sig['fold'].map(thresholds) if 'fold' in sig.columns else 0.5
    print(f"元模型覆盖样本: {meta_prob.notna().sum()}  (阈值示例: {list(thresholds.items())[:3]})")

    base, meta, n_eval = evaluate(sig, meta_prob, horizon)
    base['retention'] = 1.0

    # 逐 Fold（仅统计有元模型预测的样本）
    d = sig.dropna(subset=['tb_ret']).copy()
    d['meta_prob'] = meta_prob.reindex(d.index)
    d['net_ret'] = d['tb_ret'] - COST
    d['net_win'] = (d['net_ret'] > 0).astype(int)
    d = d[d['meta_prob'].notna()]
    fold_rows = []
    if 'fold' in d.columns:
        for f, g in d.groupby('fold'):
            thr = float(g['_thr'].iloc[0]) if g['_thr'].notna().any() else 0.5
            kept = g[g['meta_prob'] >= thr]
            bp = g['net_win'].mean()
            mp = kept['net_win'].mean() if len(kept) else np.nan
            fold_rows.append((int(f), len(g), bp, len(kept), len(kept) / len(g), mp))

    def _row(name, m):
        return (f"| {name} | {m['n']} | {_p(m['precision'])} | {_p(m['recall'])} | "
                f"{_f(m['f1'])} | {_p(m['retention'])} | {_p(m['mean_ret'])} | {_f(m['ir'])} |")

    lines = []
    lines.append(f"# 三重障碍 + 元标签原型（{horizon}d）\n")
    lines.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 主模型信号数: {len(sig)}　三重障碍标签成功: {len(sig)}")
    lines.append(f"- 元模型评估样本: {n_eval}（有 purged walk-forward 预测的区间）")
    lines.append(f"- 参数: PT={PT} SL={SL} 波动率窗口={VOL_SPAN} 成本={COST:.3f}\n")
    lines.append("## 结果对比\n")
    lines.append("| 方案 | 交易数 | Precision | Recall | F1 | 保留率 | 净均收益 | 净IR |")
    lines.append("|------|--------|-----------|--------|----|--------|---------|------|")
    lines.append(_row("Baseline（全部信号）", base))
    lines.append(_row("Meta（过滤后）", meta))
    lines.append("")
    lines.append("> Precision = 保留交易中净收益>0 的比例；Recall = 保留的盈利交易 / 全部盈利交易；")
    lines.append("> 保留率 = 过滤后交易数 / 全部；净IR = mean/std × √(252/h)。\n")

    if fold_rows:
        lines.append("## 逐 Fold\n")
        lines.append("| Fold | 交易数 | 基准 Precision | 保留数 | 保留率 | Meta Precision |")
        lines.append("|------|--------|----------------|--------|--------|----------------|")
        for f, n, bp, nk, ret, mp in fold_rows:
            lines.append(f"| {f} | {n} | {_p(bp)} | {nk} | {ret*100:.0f}% | {_p(mp)} |")
        lines.append("")

    # 结论
    prec_lift = meta['precision'] - base['precision']
    ir_lift = (meta['ir'] - base['ir']) if not (np.isnan(meta['ir']) or np.isnan(base['ir'])) else np.nan
    lines.append("## 结论\n")
    if prec_lift > 0.02 and meta['ir'] > base['ir']:
        verdict = "✅ 元标签有效"
    elif prec_lift > 0:
        verdict = "⚠️ 元标签提升微弱（<2pp）"
    else:
        verdict = "❌ 元标签未提升（甚至变差）"
    lines.append(f"- Precision 变化: **{prec_lift*100:+.1f}pp**（保留率 {meta['retention']*100:.0f}%）")
    if not np.isnan(ir_lift):
        lines.append(f"- 净 IR 变化: **{ir_lift:+.2f}**（{base['ir']:.2f} → {meta['ir']:.2f}）")
    lines.append(f"- 判定: {verdict}")
    lines.append(f"- 说明: 元标签的收益依赖主模型本身有稳定 edge；若主模型 lift≈0，")
    lines.append(f"  元模型学不到可用于过滤的信号，精度提升会远低于业界示例（55%→75%）。")
    lines.append("")

    md = "\n".join(lines)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")
    return base, meta


def _p(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"


def _f(x):
    return 'N/A' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description='三重障碍 + 元标签原型')
    ap.add_argument('--horizon', type=int, required=True, choices=[5, 20])
    ap.add_argument('--pred', type=str, default=None, help='prediction_analysis.csv')
    ap.add_argument('--cache-dir', type=str, default='data/feature_cache')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    if args.pred is None:
        dirs = {5: 'output/20260922_212530_catboost_5d',
                20: 'output/20260922_162806_catboost_20d'}
        args.pred = os.path.join(dirs[args.horizon], 'prediction_analysis.csv')
    out = args.output or f"output/meta_labeling_{args.horizon}d.md"
    run(args.horizon, args.pred, args.cache_dir, out)


if __name__ == '__main__':
    main()
