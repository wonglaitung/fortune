#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""选股打分的**唯一实现**（lessons 三.73 / 三.74）。

背景：行业中性打分此前有**两份逐字相同**的实现——
  - 生产侧：`ml_services/live_topk.py::_sector_neutral`（`comprehensive_analysis.py` 调用）
  - 回测侧：`ml_services/portfolio_backtest.py::_sector_neutral_score`
两份并存 = 谁"优化"一份就会静默分叉（正是 D25/D30 修过的那类口径漂移）。
本模块把它们收敛成一份，两边都从这里导入。
"""

import numpy as np


def sector_neutral_score(g):
    """行业内 z-score（对 prob）。样本内 z 均值 = 0；单只行业 z 记 0。

    ⚠️ 该 z 对**逐日常数缩放不变**：`λ·z` 是单调变换 ⇒ 不改排序、不改 TopK。
    因此"把分数往行业均值收缩"这类改法是空操作；要动只能动**基准估计**
    （行业均值/方差的 partial pooling），见三.74。
    """
    s = g.groupby('sector')['prob']
    z = (g['prob'] - s.transform('mean')) / s.transform('std').replace(0, np.nan)
    return z.fillna(0.0)


def selection_score(g, use_sector_neutral):
    """单日横截面的选股打分。`backtest` 与能力度量必须共用本函数。"""
    if use_sector_neutral:
        return sector_neutral_score(g)
    return g['prob']
