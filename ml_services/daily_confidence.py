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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = 'data/prediction_history.json'
CAL_DIR = 'data/calibrators'
HORIZONS = [1, 5, 20]
MIN_SAMPLES = 200


class DailyConfidence:
    def __init__(self, history_file=HISTORY_FILE):
        self.history_file = history_file
        self.prob_cal = {}
        self.conf_cal = {}
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

    def _fit_one(self, horizon, col, path):
        df = self._history(horizon)
        if len(df) < MIN_SAMPLES:
            return None
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(df['prob'].values, df[col].values)
        os.makedirs(CAL_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(iso, f)
        return iso

    def _load_or_fit(self):
        for h in HORIZONS:
            pp = os.path.join(CAL_DIR, f'prob_cal_{h}.pkl')
            cp = os.path.join(CAL_DIR, f'conf_cal_{h}.pkl')
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
        """后处理：把 1/5/20d 的 probability 替换为校准值，并加 confidence 字段"""
        changed = 0
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
                    preds[h]['probability'] = self.calibrate(p, h)
                    conf = self.confidence(p, h)
                    if conf is not None:
                        preds[h]['confidence'] = conf
                    changed += 1
        return changed


def main():
    dc = DailyConfidence()
    for h in HORIZONS:
        pp = os.path.join(CAL_DIR, f'prob_cal_{h}.pkl')
        cp = os.path.join(CAL_DIR, f'conf_cal_{h}.pkl')
        prob_ok = dc.prob_cal.get(h) is not None
        conf_ok = dc.conf_cal.get(h) is not None
        n = len(dc._history(h))
        print(f"{h}d: 校准{'✓' if prob_ok else '✗'}  置信{'✓' if conf_ok else '✗'}  样本={n}")
        if prob_ok and n >= MIN_SAMPLES:
            df = dc._history(h)
            # 抽几个概率点看校准效果
            for q in (0.3, 0.5, 0.65, 0.8):
                print(f"   P={q} -> 校准概率 {dc.calibrate(q,h):.3f}, 置信 {dc.confidence(q,h):.3f}")


if __name__ == '__main__':
    main()