#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 超参数调优（rank 标签 / 部署口径 IC）

设计要点（与生产/WF 同口径）：
1. 标签 = 当日横截面百分位（label_mode='rank'，D59），学习器 = LGBMRegressor
2. 只用**训练期**数据（默认截止 2023-05-31），测试窗不参与调参
3. 时序扩展窗 CV + purge/embargo（gap = horizon + 1），避免重叠标签泄漏
4. 选择指标 = 逐日横截面 Spearman IC（预测分数 vs Future_Return），
   按折取均值并惩罚折间波动（G3：不看绝对收益/准确率）
5. 特征集合固定为生产模型的 feature_columns（本实验只调超参，不调特征）

用法：
  python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 20 --stocks 120
  python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 20 --stocks 120 --write
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.logger_config import get_logger
from ml_services.ml_trading_model import TradingModel
from ml_services.network_ids import resolve_community_ids
from config import TRAINING_STOCKS

logger = get_logger('hyperparameter_tuner')

SEARCH_SPACE = {
    'n_estimators': [300, 400, 600, 800],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 6, 8, 10],
    'learning_rate': [0.03, 0.04, 0.06, 0.08],
    'min_child_samples': [30, 50, 100],
    'reg_lambda': [1.0, 2.0, 5.0, 10.0],
    'subsample': [0.7, 0.75, 0.85, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# 当前生产参数（_train_lightgbm 默认值）
BASELINE_PARAMS = {
    'n_estimators': 400,
    'num_leaves': 64,
    'max_depth': 8,
    'learning_rate': 0.06,
    'min_child_samples': 50,
    'reg_lambda': 2.0,
    'subsample': 0.75,
    'colsample_bytree': 0.8,
}


def _daily_rank_ic(dates, pred, actual):
    """逐日横截面 Spearman IC 的均值（预测分数 vs 实际未来收益）。"""
    frame = pd.DataFrame({'d': dates, 'p': pred, 'a': actual}).dropna()
    if frame.empty:
        return np.nan, 0
    ics = []
    for _, g in frame.groupby('d'):
        if len(g) < 5 or g['p'].nunique() < 2 or g['a'].nunique() < 2:
            continue
        ic = g['p'].corr(g['a'], method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return np.nan, 0
    return float(np.mean(ics)), len(ics)


class HyperparameterTuner:
    def __init__(self, horizon=20, n_folds=4, stocks=120, train_end='2023-05-31',
                 train_start='2018-06-01', top_k=None):
        self.horizon = horizon
        self.n_folds = n_folds
        self.n_stocks = stocks
        self.train_end = train_end
        self.train_start = train_start
        self.top_k = top_k
        self.df = None
        self.feature_cols = None
        self.cat_cols = []

    def load_data(self):
        codes = list(TRAINING_STOCKS.keys())
        rng = random.Random(42)
        selected = rng.sample(codes, min(self.n_stocks, len(codes)))
        logger.info(f"调参股票数: {len(selected)} / {len(codes)}；区间 {self.train_start} → {self.train_end}")

        model = TradingModel()
        community_ids = resolve_community_ids(include_pit=True, logger=logger)
        df = model.prepare_data(
            selected, start_date=self.train_start, end_date=self.train_end,
            horizon=self.horizon, for_backtest=False,
            community_ids=community_ids, mode='production', label_mode='rank',
        )
        self.df = df.sort_index()
        self.model_ref = model

        feats = None
        pkl = f'data/ml_trading_model_lgbm_{self.horizon}d.pkl'
        if os.path.exists(pkl):
            try:
                import pickle
                with open(pkl, 'rb') as f:
                    payload = pickle.load(f)
                feats = payload.get('feature_columns')
                self.cat_cols = list(payload.get('categorical_encoders', {}).keys())
            except Exception as e:
                logger.warning(f"读取生产特征列失败（改用相关性筛选）: {e}")
        if not feats:
            numeric = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
            sample = self.df.sample(min(50000, len(self.df)), random_state=42)
            corr = sample[numeric + ['Label']].corr(method='spearman')['Label'].drop('Label')
            feats = list(corr.abs().sort_values(ascending=False).head(self.top_k or 500).index)
        feats = [c for c in feats if c in self.df.columns]
        if self.top_k:
            feats = feats[:self.top_k]
        self.feature_cols = feats
        self.cat_cols = [c for c in self.cat_cols if c in self.feature_cols]
        logger.info(f"特征列: {len(self.feature_cols)}（其中分类 {len(self.cat_cols)}）")
        return len(self.df)

    def _encode(self, train_df, val_df):
        X_tr = train_df[self.feature_cols].copy()
        X_va = val_df[self.feature_cols].copy()
        for col in self.cat_cols:
            tr = X_tr[col].astype('object')
            codes, uniques = pd.factorize(tr, sort=False)
            X_tr[col] = codes.astype(np.int32)
            mapping = {v: i for i, v in enumerate(uniques)}
            X_va[col] = X_va[col].map(mapping).fillna(-1).astype(np.int32)
        return X_tr, X_va

    def evaluate_params(self, params):
        df = self.df
        dates = df.index.normalize()
        n = len(df)
        # 前 60% 只做训练起点，后 40% 切 n_folds 个扩展窗验证
        start = int(n * 0.6)
        span = (n - start) // self.n_folds
        gap = self.horizon + 1

        fold_ics = []
        for k in range(self.n_folds):
            va_lo = start + k * span
            va_hi = min(va_lo + span, n)
            tr_hi = max(0, va_lo - gap)
            if va_hi - va_lo < 100 or tr_hi < 500:
                continue
            train_df = df.iloc[:tr_hi]
            val_df = df.iloc[va_lo:va_hi]
            X_tr, X_va = self._encode(train_df, val_df)
            y_tr = train_df['Label'].values

            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                objective='regression', subsample_freq=1, random_state=42,
                n_jobs=int(os.environ.get('LGBM_N_JOBS', '-1')), verbose=-1,
                deterministic=True, force_row_wise=True, **params,
            )
            model.fit(X_tr, y_tr)
            pred = np.clip(model.predict(X_va), 0.0, 1.0)
            ic, ndays = _daily_rank_ic(dates[va_lo:va_hi], pred,
                                       val_df['Future_Return'].values)
            if np.isfinite(ic):
                fold_ics.append(ic)

        if not fold_ics:
            return {'score': -999, 'ic': np.nan, 'ic_std': np.nan,
                    'folds': 0, 'error': '无有效 fold'}
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        return {
            'score': ic_mean - 0.5 * ic_std,
            'ic': ic_mean,
            'ic_std': ic_std,
            'fold_ics': [round(float(x), 4) for x in fold_ics],
            'folds': len(fold_ics),
            'positive_folds': int(sum(1 for x in fold_ics if x > 0)),
        }

    def random_search(self, n_iter=20, seed=42):
        rng = random.Random(seed)
        results = []
        best = None
        t0 = time.time()
        for i in range(n_iter):
            params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
            print(f"\n[{i + 1}/{n_iter}] {params}")
            res = self.evaluate_params(params)
            print(f"    IC={res.get('ic'):.4f} std={res.get('ic_std'):.4f} "
                  f"folds={res.get('folds')} pos={res.get('positive_folds')}")
            results.append({'params': params, **res})
            if res.get('folds') and (best is None or res['score'] > best['score']):
                best = results[-1]
        return {'best': best, 'results': results,
                'elapsed_sec': round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--stocks', type=int, default=120)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--train-start', default='2018-06-01')
    p.add_argument('--train-end', default='2023-05-31')
    p.add_argument('--top-k', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--write', action='store_true',
                   help='把最优参数写入 data/lgbm_best_params.json（默认只报告）')
    p.add_argument('--baseline', action='store_true',
                   help='先用当前生产默认参数跑同一 CV，作为相对提升的分母')
    args = p.parse_args()

    tuner = HyperparameterTuner(args.horizon, args.folds, args.stocks,
                                args.train_end, args.train_start, args.top_k)
    rows = tuner.load_data()
    if rows < 2000:
        print(f"❌ 样本不足: {rows}")
        return 1

    base_res = None
    if args.baseline:
        print("\n[baseline] 当前生产默认参数")
        base_res = tuner.evaluate_params(BASELINE_PARAMS)
        print(f"    IC={base_res.get('ic'):.4f} std={base_res.get('ic_std'):.4f} "
              f"folds={base_res.get('folds')} pos={base_res.get('positive_folds')}")

    out = tuner.random_search(args.n_iter, args.seed)
    out['baseline'] = base_res
    best = out['best']
    if base_res and base_res.get('ic'):
        lift = best['ic'] / base_res['ic'] - 1
        out['relative_improvement'] = round(float(lift), 4)
        print(f"\n相对默认参数 IC 提升: {lift:+.2%}")
    print("\n" + "=" * 80)
    print(f"最优: IC={best['ic']:.4f} std={best['ic_std']:.4f} "
          f"pos={best['positive_folds']}/{best['folds']}")
    print(best['params'])
    print("=" * 80)

    os.makedirs('output/_logs', exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'output/_logs/lgbm_tune_{args.horizon}d_{stamp}.json'
    with open(path, 'w') as f:
        json.dump({'horizon': args.horizon, 'n_iter': args.n_iter,
                   'stocks': args.stocks, 'train_start': args.train_start,
                   'train_end': args.train_end, **out}, f, indent=2, default=str)
    print(f"结果已保存: {path}")

    if args.write and best:
        cfg_path = 'data/lgbm_best_params.json'
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        cfg[str(args.horizon)] = best['params']
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"已写入 {cfg_path}（horizon={args.horizon}）")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
