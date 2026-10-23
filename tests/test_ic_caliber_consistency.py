# -*- coding: utf-8 -*-
"""能力指标的口径契约（⓪ 同一性闸第⑤问 / lessons 三.73）

背景：2026-10-14 发现同一份 D30 数据上「IC」有五个口径，同一天同一个模型：

| 口径 | 打分变换 | 聚合 | 值 |
|---|---|---|---|
| A 闸 H4 | 原始 prob | 全表跨日池化 | +0.0322 |
| WF 报告 `avg_rank_ic` | 原始 prob | 每折内池化再平均 | +0.0442 |
| 全池逐日 | 原始 prob | 逐日再平均 | +0.0136 |
| Top10 内 | 原始 prob | 逐日、仅选中 | +0.0430 |
| **部署口径（2026-10-23 前）** | 行业内中性分 | 逐日 | −0.0026 |

部署规则 2026-10-23 起改为「raw prob 后取 TopK」（D73），故部署口径 = 第三行的逐日口径。
教训不变：**被评估的量必须是生产用的量**——这正是 ⓪ 闸要拦的错。

本文件的测试必须能**变红**（变异测试思路，lessons 三.72）：
若有人重新引入中性/raw 开关或绕开 `selection_score`，`test_no_neutral_switch_remains`
与 `test_backtest_ranks_by_selection_score` 会失败。
"""

import ast
from pathlib import Path

ML = Path(__file__).resolve().parents[1] / 'ml_services'

import numpy as np
import pandas as pd
import pytest

import ml_services.portfolio_backtest as pb

ROOT = Path(__file__).resolve().parents[1]
GUARDRAIL_SRC = ROOT / 'ml_services' / 'monthly_guardrail.py'


def _synthetic_panel(n_days=30, per_sector=20, seed=0, sec_gap=0.05, day_common=0.0):
    """构造：板块间有收益差（raw prob 与之同向），板块内 prob 与收益无关。

    ⇒ 原始 prob 的 IC 明显为正，行业内中性分的 IC ≈ 0。
    这正是「用错口径会把 0 能力读成正 IC」的最小复现。

    `day_common>0` 时另加一层**日间共同因子**（同日 prob+收益同向移动）：
    逐日口径把它扣掉，池化口径把它算进去 ⇒ 复现「池化天然偏高」（三.68）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=d)
        dc = rng.normal(0, 1.0) if day_common else 0.0
        for sec, base in [('A', -sec_gap), ('B', +sec_gap)]:
            for i in range(per_sector):
                rows.append({
                    'date': date, 'code': f'{sec}{i:02d}', 'sector': sec,
                    'prob': 0.5 + base + day_common * dc * 0.01 + rng.normal(0, 0.01),
                    'ret': base / 0.1 + day_common * dc * 0.5 + rng.normal(0, 1.0),
                })
    return pd.DataFrame(rows)


def test_pooled_ic_is_inflated_by_day_level_component():
    """对照③：全表跨日池化含日间共同层 ⇒ 天然高于逐日口径（三.68）。

    构造：板块内无信号（sec_gap=0）、只有日间共同因子 ⇒ 逐日 IC≈0，池化 IC>0。
    """
    df = _synthetic_panel(sec_gap=0.0, day_common=1.0, n_days=40)
    r = pb.selection_ic(df)
    assert abs(r['ic']) < 0.05, f"逐日口径（去掉日间层）应 ≈0，实际 {r['ic']}"
    assert r['ic_pooled'] > 0.1, f"池化口径应明显偏高，实际 {r['ic_pooled']}"


def test_backtest_ranks_by_selection_score(monkeypatch):
    """契约：backtest 的排序打分必须来自 selection_score（唯一实现）。"""
    calls = []
    orig = pb.selection_score

    def spy(g):
        calls.append(1)
        return orig(g)

    monkeypatch.setattr(pb, 'selection_score', spy)
    df = _synthetic_panel()
    pb.backtest(df, horizon=5, topk=5)
    assert calls, "backtest 未经过 selection_score ⇒ 打分口径已绕开契约"


def test_no_neutral_switch_remains():
    """D73 之后不得再有任何中性/raw 开关（用户要求：不做向后兼容）。"""
    for name in ('monthly_guardrail.py', 'portfolio_backtest.py', 'selection.py', 'live_topk.py'):
        src = (ML / name).read_text(encoding='utf-8')
        assert 'use_sector_neutral' not in src, f"{name} 仍保留中性开关"
        assert 'SECTOR_NEUTRAL' not in src, f"{name} 仍保留 SECTOR_NEUTRAL"
