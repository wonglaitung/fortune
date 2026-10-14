# -*- coding: utf-8 -*-
"""选股打分「单一实现」契约（lessons 三.73 / 三.74）

背景：行业中性打分此前有**两份逐字相同**的实现——
  生产侧 `ml_services/live_topk.py::_sector_neutral`（`comprehensive_analysis.py` 用）
  回测侧 `ml_services/portfolio_backtest.py::_sector_neutral_score`（护栏/回测用）

两份并存 = 谁"优化"一份就静默分叉（D25/D30 修过的那类口径漂移）。
本文件锁死：**打分实现只有一处，两边是同一函数对象**。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import ml_services.live_topk as lt
import ml_services.portfolio_backtest as pb
from ml_services import selection

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / 'ml_services'


def test_single_function_object_across_paths():
    """生产与回测必须绑定**同一个函数对象**（不是拷贝、不是包装）。"""
    assert pb._sector_neutral_score is selection.sector_neutral_score
    assert pb.selection_score is selection.selection_score
    assert lt._sector_neutral is selection.sector_neutral_score


def test_groupby_sector_zscore_defined_in_one_file_only():
    """源码级护栏：`groupby('sector')` 只许出现在 selection.py。

    这是「加回第二份实现」的绊线——一旦有人复制粘贴，本测试转红。
    """
    pat = re.compile(r"groupby\(\s*['\"]sector['\"]\s*\)")
    hits = sorted(p.name for p in ML.glob('*.py')
                  if pat.search(p.read_text(encoding='utf-8')))
    assert hits == ['selection.py'], f"行业内 z-score 出现多份实现：{hits}"


def test_production_and_backtest_pick_same_names():
    """行为等价：生产选股器（select_from_df）与共享打分函数取到相同名单。

    ⚠️ 必须用 `config` 里的**真实代码**：`select_from_df` 会用
    `STOCK_SECTOR_MAPPING` 重新映射 sector，合成代码会被打成 'unknown'。
    """
    from config import STOCK_SECTOR_MAPPING
    codes = list(STOCK_SECTOR_MAPPING.keys())[:20]
    day = pd.DataFrame({'code': codes, 'prob': np.linspace(0.31, 0.69, len(codes))})
    day['sector'] = day['code'].map(
        lambda c: (STOCK_SECTOR_MAPPING.get(c) or {}).get('sector', 'unknown'))

    k = min(4, len(day) // 4)
    prod = lt.select_from_df(day.copy(), topk=4, sector_neutral=True)[0]
    expect = set(day.assign(score=selection.selection_score(day, True))
                    .nlargest(k, 'score')['code'])
    assert set(prod['code']) == expect, "生产侧选股与 selection_score 不一致"


def test_scores_are_identical_between_paths():
    """逐值一致：两条路径对同一输入给出完全相同的分数。"""
    df = pd.DataFrame({
        'code': ['a1', 'a2', 'a3', 'b1', 'b2', 'b3'],
        'sector': ['a', 'a', 'a', 'b', 'b', 'b'],
        'prob': [0.6, 0.5, 0.4, 0.7, 0.55, 0.45],
    })
    s1 = selection.selection_score(df, True)
    s2 = pb.selection_score(df, True)
    s3 = lt._sector_neutral(df)
    assert np.allclose(s1.values, s2.values)
    assert np.allclose(s1.values, s3.values)
