#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日预测优化：概率校准 + 置信度（P(主模型方向判对)）

- 校准概率：用 prediction_history 拟合 Isotonic，把模型 probability 映射成"真概率"
  （高置信≠高胜率，校准后阈值/分层才可信）。
- 置信度：拟合 (prob → 方向是否判对) 的 Isotonic，得到"这次预测靠谱程度"（=元模型置信的最简可靠版）。

产出/缓存：data/calibrators/{prob_cal_{h},conf_cal_{h}}.pkl
用法：
  命令行拟合: python3 ml_services/daily_confidence.py
  代码接入:   from ml_services.daily_confidence import DailyConfidence
              dc=DailyConfidence(); dc.apply_to_results(three_horizon_results)
"""

import os
import pickle
import sys
import glob
import json
from datetime import datetime

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PREDICTION_HORIZONS

HISTORY_FILE = 'data/prediction_history.json'
CAL_DIR = 'data/calibrators'
HORIZONS = list(PREDICTION_HORIZONS)   # D59：5/10/20（单一来源 config.PREDICTION_HORIZONS）
MIN_SAMPLES = 200
# D26 退化守卫：某周期 confidence（=P(方向判对)）在当日横截面上的极差 < 该值
# ⇒ 校准器对今天所有股票给出同一个数（如 20d：p<0.741 全平台 0.4933），
# 此时 confidence 不携带任何个股信息 → 撤销并告警，**不回退 probability**（已是原始 P(涨)）。
CONF_DEGENERATE_SPREAD = 0.02
# A股（P3.1 决策点2：用 walk-forward OOF 预测拟合，不依赖生产 history）
A_STOCK_OOF_GLOB = 'output/*_a_stock_catboost_{h}d/prediction_analysis.csv'


class DailyConfidence:
    def __init__(self, history_file=HISTORY_FILE, cal_prefix='', oof_glob=None,
                 min_samples=MIN_SAMPLES, cal_dir=CAL_DIR, label_mode=None):
        """概率校准 + 置信度拟合。

        Args:
            history_file: 生产预测历史 JSON（默认数据源）
            cal_prefix: 校准器文件名前缀（A股用 'a_stock_'，与港股校准器隔离）
            oof_glob: 若提供则从 walk-forward OOF CSV 拟合（含 {h} 占位符），
                优先于 history_file —— PIT 口径、样本足（决策点2：OOF 立即拟合）
            min_samples: 最小样本数（不足则不拟合，calibrate 透传原值）
            label_mode: 'absolute'（默认，probability=P(上涨)）或 'rank'。
                **D26 补充：rank 下 probability 是横截面分位（score），无概率语义**
                ⇒ 不得拟合 P(上涨) 校准器；本类在 rank 下直接停用（返回 0 项改动）。
        """
        self.history_file = history_file
        self.cal_prefix = cal_prefix or ''
        self.oof_glob = oof_glob
        self.min_samples = min_samples
        self.cal_dir = cal_dir
        self.label_mode = label_mode or os.environ.get('LABEL_MODE', 'absolute')
        self.prob_cal = {}
        self.conf_cal = {}
        self.meta = {}   # {horizon: {source, n, fitted_at, file}}
        if self.label_mode == 'rank':
            logger.info('rank 口径：probability=横截面分位（score），跳过概率/置信度校准')
            return
        self._load_or_fit()

    # ---------- 拟合 ----------
    def _history(self, horizon):
        if not os.path.exists(self.history_file):
            return pd.DataFrame()
        h = json.load(open(self.history_file, encoding='utf-8'))
        preds = [p for p in h.get('predictions', [])
                 if p.get('horizon') == horizon
                 and p.get('outcome') is not None
                 and p.get('prediction_probability') is not None]
        if not preds:
            return pd.DataFrame()
        df = pd.DataFrame(preds)
        df['prob'] = pd.to_numeric(df['prediction_probability'], errors='coerce')
        if 'actual_direction' in df:
            df['up'] = df['actual_direction'].astype(str).str.lower().isin(['up', '1', 'true']).astype(int)
        else:
            df['up'] = pd.to_numeric(df['actual_return'], errors='coerce').gt(0).astype(int)
        df['correct'] = (df['outcome'].astype(str).str.lower() == 'correct').astype(int)
        return df.dropna(subset=['prob', 'correct'])

    def _oof(self, horizon):
        """从最新一份 walk-forward OOF CSV 读 (prob, up, correct)。"""
        if not self.oof_glob:
            return pd.DataFrame()
        files = sorted(glob.glob(self.oof_glob.format(h=horizon)),
                       key=os.path.getmtime)
        if not files:
            return pd.DataFrame()
        try:
            df = pd.read_csv(files[-1])
        except Exception:
            return pd.DataFrame()
        prob_col = next((c for c in ('Predict_Prob', 'Predicted_Prob', 'probability')
                         if c in df.columns), None)
        if prob_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out['prob'] = pd.to_numeric(df[prob_col], errors='coerce')
        if 'Actual_Direction' in df.columns:
            out['up'] = (df['Actual_Direction'].astype(str).str.upper()
                         .isin(['UP', '1', 'TRUE'])).astype(int)
        elif 'Actual_Return' in df.columns:
            out['up'] = pd.to_numeric(df['Actual_Return'], errors='coerce').gt(0).astype(int)
        else:
            return pd.DataFrame()
        if 'Is_Correct' in df.columns:
            out['correct'] = (df['Is_Correct'].astype(str).str.lower()
                              .isin(['true', '1'])).astype(int)
        else:
            out['correct'] = (out['up'] == (out['prob'] >= 0.5).astype(int)).astype(int)
        out = out.dropna(subset=['prob'])
        out.attrs['source_file'] = files[-1]
        return out

    def _fit_one(self, horizon, col, path):
        if self.oof_glob:
            df = self._oof(horizon)
            source = df.attrs.get('source_file', '')
        else:
            df = self._history(horizon)
            source = self.history_file
        if len(df) < self.min_samples:
            return None
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(df['prob'].values, df[col].values)
        os.makedirs(self.cal_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(iso, f)
        # 快照元数据：写明数据源与拟合时间（OOF 校准 = 该次 walk-forward 的分布，
        # 重跑 walk-forward 后须重新拟合）
        if col == 'up':
            self.meta[horizon] = {
                'source_file': source, 'n_samples': int(len(df)),
                'fitted_at': datetime.now().isoformat(timespec='seconds'),
                'prob_min': float(df['prob'].min()),
                'prob_max': float(df['prob'].max()),
            }
            meta_path = os.path.join(self.cal_dir, f'{self.cal_prefix}cal_meta_{horizon}.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta[horizon], f, ensure_ascii=False, indent=2)
        return iso

    def _load_or_fit(self):
        for h in HORIZONS:
            pp = os.path.join(self.cal_dir, f'{self.cal_prefix}prob_cal_{h}.pkl')
            cp = os.path.join(self.cal_dir, f'{self.cal_prefix}conf_cal_{h}.pkl')
            self.prob_cal[h] = self._load(pp) or self._fit_one(h, 'up', pp)
            self.conf_cal[h] = self._load(cp) or self._fit_one(h, 'correct', cp)

    def _load(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    # ---------- 应用 ----------
    def calibrate(self, prob, horizon):
        iso = self.prob_cal.get(horizon)
        if iso is None or prob is None:
            return prob
        return float(iso.predict([prob])[0])

    def confidence(self, prob, horizon):
        iso = self.conf_cal.get(horizon)
        if iso is None or prob is None:
            return None
        return float(iso.predict([prob])[0])

    def apply_to_results(self, three_horizon_results):
        """后处理：**probability 保持原始 P(上涨)**，校准值只写入 confidence。

        D26（2026-10-13，理由见 DECISIONS D26 / lessons 三.61）：
        旧行为把 `probability`（P(该股上涨)）覆写为 P(该方向判对)，并按校准值 ≥0.5 重判方向。
        后果：① 20d 校准器在 p<0.741 区间是平台（≈0.4933）→ 32 只邮件全显示「上涨概率 0.49」，
        原始 0.23~0.74 的排序信息消失，三重确认 0.55/0.60 门槛**结构性不可达**；
        ② 原始 0.74 的强涨股会被判成「↓」；③ A 闸跑未校准概率、生产阈值跑校准后概率（同一性缺口）。

        现在：probability / prediction / direction 一律用模型原始输出（与 A 闸、Walk-forward 同口径），
        confidence = P(该方向判对) 单独承载校准信息；若某周期 confidence 当日无区分度
        （横截面极差 < CONF_DEGENERATE_SPREAD），撤销该周期的 confidence 并告警。
        """
        changed = 0
        if getattr(self, 'label_mode', 'absolute') == 'rank':
            # D26 补充：rank 的 score 是横截面分位，不是 P(上涨)；
            # 任何等距变换（Isotonic）都会给它强加不存在的概率语义 → 不改动。
            return 0
        conf_by_h = {h: [] for h in HORIZONS}
        conf_locs = {h: [] for h in HORIZONS}
        for code, res in three_horizon_results.items():
            # 兼容两种结构：res['predictions'][h] 或 res[h]
            preds = res.get('predictions', res) if isinstance(res, dict) else {}
            if not isinstance(preds, dict):
                continue
            for h in HORIZONS:
                if h in preds and isinstance(preds[h], dict) and 'probability' in preds[h]:
                    p = preds[h]['probability']
                    if p is None:
                        continue
                    conf = self.confidence(p, h)
                    if conf is not None:
                        preds[h]['confidence'] = conf
                        conf_by_h[h].append(conf)
                        conf_locs[h].append(preds[h])
                    changed += 1
        # 退化守卫：校准值当日无区分度 → 该周期的 confidence 不携带个股信息，撤销 + 告警
        for h in HORIZONS:
            vals = conf_by_h[h]
            if len(vals) < 5:
                continue
            spread = max(vals) - min(vals)
            if spread < CONF_DEGENERATE_SPREAD:
                for d in conf_locs[h]:
                    d.pop('confidence', None)
                logger.warning(
                    '⚠️ 校准退化守卫：%dd confidence 当日横截面极差 %.4f < %.2f'
                    '（唯一值 %d/%d）→ 已撤销该周期 confidence，'
                    'probability 仍为模型原始 P(上涨)',
                    h, spread, CONF_DEGENERATE_SPREAD, len(set(round(v, 4) for v in vals)), len(vals))
        return changed


def main():
    import argparse
    ap = argparse.ArgumentParser(description='概率校准拟合（港股=生产history / A股=walk-forward OOF）')
    ap.add_argument('--market', choices=['hk', 'a'], default='hk',
                    help='hk=港股 prediction_history；a=A股 OOF CSV（P3.1 决策点2）')
    ap.add_argument('--refit', action='store_true', help='忽略已有校准器强制重拟合')
    args = ap.parse_args()

    if args.market == 'a':
        dc_kwargs = dict(cal_prefix='a_stock_', oof_glob=A_STOCK_OOF_GLOB)
    else:
        dc_kwargs = dict()

    if args.refit:
        for h in HORIZONS:
            for name in ('prob_cal', 'conf_cal'):
                p = os.path.join(CAL_DIR, f"{dc_kwargs.get('cal_prefix', '')}{name}_{h}.pkl")
                if os.path.exists(p):
                    os.remove(p)

    dc = DailyConfidence(**dc_kwargs)
    for h in HORIZONS:
        prob_ok = dc.prob_cal.get(h) is not None
        conf_ok = dc.conf_cal.get(h) is not None
        df = dc._oof(h) if args.market == 'a' else dc._history(h)
        n = len(df)
        src = df.attrs.get('source_file', dc.history_file) if n else '-'
        print(f"{h}d: 校准{'✓' if prob_ok else '✗'}  置信{'✓' if conf_ok else '✗'}  样本={n}  源={src}")
        if prob_ok:
            for q in (0.3, 0.5, 0.65, 0.8):
                print(f"   P={q} -> 校准概率 {dc.calibrate(q, h):.3f}, 置信 {dc.confidence(q, h):.3f}")


if __name__ == '__main__':
    main()