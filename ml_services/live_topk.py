#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘 20d 行业中性 TopK 选股器

把日频模型预测（概率）转成**可执行的调仓指令**：
  1. 读最新预测（code/probability）
  2. 行业内 z-score（行业中性）
  3. 排序取 TopK（k = min(topk, n//4)，与回测一致）
  4. 与当前持仓对比 → 生成买入/卖出清单 + 换手 + 成本估算
  5. --commit 时更新持仓状态、记录交易并提示下次调仓日

用法：
  # 预演（不改持仓）
  python3 ml_services/live_topk.py --pred data/ml_trading_model_catboost_predictions_20d.csv --topk 10
  # 调仓日确认执行（更新状态）
  python3 ml_services/live_topk.py --pred data/...csv --topk 10 --horizon 20 --commit
  # 查看当前持仓
  python3 ml_services/live_topk.py --show
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST = 0.005  # 双边成本（与 walk_forward / portfolio_backtest 一致）
STATE_FILE = 'data/live_portfolio/topk_state.json'


def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'holdings': [], 'last_rebalance_date': None, 'rebalance_count': 0,
            'cost_basis': {}, 'log': []}


def _save_state(st):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def _sector_neutral(g):
    s = g.groupby('sector')['prob']
    z = (g['prob'] - s.transform('mean')) / s.transform('std').replace(0, np.nan)
    return z.fillna(0.0)


def select(pred_csv, topk, sector_neutral=True):
    """读预测并返回目标持仓（含 score）"""
    df = pd.read_csv(pred_csv)
    df = df.rename(columns={'Stock_Code': 'code', 'code': 'code'})
    if 'prob' not in df.columns:
        for cand in ('probability', 'Predict_Prob', 'predict_prob'):
            if cand in df.columns:
                df['prob'] = pd.to_numeric(df[cand], errors='coerce')
                break
    df['code'] = df['code'].astype(str)
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    df = df.dropna(subset=['code', 'prob'])
    try:
        from config import STOCK_SECTOR_MAPPING
        df['sector'] = df['code'].map(lambda c: (STOCK_SECTOR_MAPPING.get(c) or {}).get('sector', 'unknown'))
    except Exception:
        df['sector'] = 'unknown'

    df['score'] = _sector_neutral(df) if sector_neutral else df['prob']
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    k = min(topk, len(df) // 4)
    target = df.head(k).copy()
    return target, df, k


def run(pred_csv, topk, horizon, sector_neutral=True, commit=False):
    target, ranked, k = select(pred_csv, topk, sector_neutral)
    st = _load_state()
    cur = set(st.get('holdings', []))
    tgt = list(target['code'])
    tgt_set = set(tgt)

    buys = [c for c in tgt if c not in cur]
    sells = [c for c in cur if c not in tgt_set]
    turnover = (len(buys) / k) if k > 0 else 0.0
    est_cost = turnover * COST

    data_date = 'unknown'
    try:
        raw = pd.read_csv(pred_csv)
        if 'data_date' in raw.columns:
            data_date = str(raw['data_date'].iloc[0])
    except Exception:
        pass

    print("=" * 72)
    print(f"实盘 20d 行业中性 TopK   target_date={data_date}   K={k}（预算 topk={topk}）")
    print("=" * 72)
    print(f"预测池: {len(ranked)} 只　当前持仓: {len(cur)} 只　上次调仓: {st.get('last_rebalance_date')}")
    print(f"\n目标持仓（行业中性 TopK）：")
    for _, r in target.iterrows():
        print(f"  {r['code']:<10} {str(r.get('name','')):<8} score={r['score']:+.3f} prob={r['prob']:.3f} [{r.get('sector')}]")
    print(f"\n买入 {len(buys)}: {buys}")
    print(f"卖出 {len(sells)}: {sells}")
    print(f"换手率(单边): {turnover*100:.0f}%　预计成本: {est_cost*100:.2f}%")
    print(f"\n执行建议：次日开盘等权买入目标持仓；卖出手续费+滑点按 {COST*100:.1f}% 双边计")

    if commit:
        if data_date != 'unknown' and st.get('last_rebalance_date') == data_date:
            print(f"\n⚠️ 已在本期({data_date})调仓过，跳过 commit")
        else:
            st['holdings'] = tgt
            st['last_rebalance_date'] = data_date
            st['rebalance_count'] = st.get('rebalance_count', 0) + 1
            st.setdefault('log', []).append({
                'date': data_date, 'buys': buys, 'sells': sells,
                'turnover': turnover, 'est_cost': est_cost,
                'holdings': tgt, 'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            _save_state(st)
            print(f"\n✅ 已记录调仓（第 {st['rebalance_count']} 次），状态保存到 {STATE_FILE}")

    print(f"\n下次调仓：约 {horizon} 个交易日后（或距上次调仓满 {horizon} 交易日）")


def main():
    ap = argparse.ArgumentParser(description='实盘 20d 行业中性 TopK 选股器')
    ap.add_argument('--pred', type=str,
                    default='data/ml_trading_model_catboost_predictions_20d.csv',
                    help='预测 CSV（code, probability, ...）')
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--raw', action='store_true', help='不做行业中性（对照）')
    ap.add_argument('--commit', action='store_true', help='确认调仓并更新持仓状态')
    ap.add_argument('--show', action='store_true', help='仅显示当前持仓')
    args = ap.parse_args()

    if args.show:
        st = _load_state()
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return

    run(args.pred, args.topk, args.horizon,
        sector_neutral=not args.raw, commit=args.commit)


if __name__ == '__main__':
    main()
