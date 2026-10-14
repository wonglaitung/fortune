# -*- coding: utf-8 -*-
"""能力指标的口径契约（⓪ 同一性闸第⑤问 / lessons 三.73）

背景：2026-10-14 发现同一份 D30 数据上「IC」有五个口径，同一天同一个模型：

| 口径 | 打分变换 | 聚合 | 值 |
|---|---|---|---|
| A 闸 H4 | 原始 prob | 全表跨日池化 | +0.0322 |
| WF 报告 `avg_rank_ic` | 原始 prob | 每折内池化再平均 | +0.0442 |
| 全池逐日 | 原始 prob | 逐日再平均 | +0.0136 |
| Top10 内 | 原始 prob | 逐日、仅选中 | +0.0430 |
| **部署口径** | **行业内中性分** | **逐日** | **−0.0026** |

部署规则是「行业内 z 后取 TopK」，所以**只有最后一行是「被评估的量 = 生产用的量」**。
把第三行（0.0136）当能力，等于量了别的量——这正是 ⓪ 闸要拦的错。

本文件的测试必须能**变红**（变异测试思路，lessons 三.72）：
若有人把护栏的能力指标换回原始 prob，`test_deployed_caliber_ic_ignores_raw_prob` 会失败。
"""

import ast
from pathlib import Path

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


def test_deployed_caliber_ic_ignores_raw_prob():
    """★变异检测★：部署口径（中性）≈0，原始 prob 口径显著>0。

    若把 selection_score 的 neutral 分支换成 g['prob']，本测试转红。
    """
    df = _synthetic_panel()
    neu = pb.selection_ic(df, use_sector_neutral=True)
    raw = pb.selection_ic(df, use_sector_neutral=False)
    assert abs(neu['ic']) < 0.05, f"中性口径应 ≈0，实际 {neu['ic']}"
    assert raw['ic'] > 0.20, f"原始口径应明显>0，实际 {raw['ic']}"
    assert abs(raw['ic'] - neu['ic']) > 0.15, "两个口径必须可区分"


def test_pooled_ic_is_inflated_by_day_level_component():
    """对照③：全表跨日池化含日间共同层 ⇒ 天然高于逐日口径（三.68）。

    构造：板块内无信号（sec_gap=0）、只有日间共同因子 ⇒ 逐日 IC≈0，池化 IC>0。
    """
    df = _synthetic_panel(sec_gap=0.0, day_common=1.0, n_days=40)
    r = pb.selection_ic(df, use_sector_neutral=False)
    assert abs(r['ic']) < 0.05, f"逐日口径（去掉日间层）应 ≈0，实际 {r['ic']}"
    assert r['ic_pooled'] > 0.1, f"池化口径应明显偏高，实际 {r['ic_pooled']}"


def test_backtest_ranks_by_selection_score(monkeypatch):
    """契约：backtest 的排序打分必须来自 selection_score（唯一实现）。"""
    calls = []
    orig = pb.selection_score

    def spy(g, use_sector_neutral):
        calls.append(use_sector_neutral)
        return orig(g, use_sector_neutral)

    monkeypatch.setattr(pb, 'selection_score', spy)
    df = _synthetic_panel()
    pb.backtest(df, horizon=5, topk=5, use_sector_neutral=True)
    assert calls, "backtest 未经过 selection_score ⇒ 打分口径已绕开契约"
    assert all(c is True for c in calls)


def test_guardrail_uses_single_neutral_flag_for_judgement_and_capability():
    """护栏的「判定」与「能力指标」必须读同一个常量，禁止调用点各写一遍。"""
    src = GUARDRAIL_SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'backtest' and len(node.args) >= 4:
                found['backtest'] = ast.dump(node.args[3])
            if node.func.id == 'selection_ic':
                for kw in node.keywords:
                    if kw.arg == 'use_sector_neutral':
                        found['selection_ic'] = ast.dump(kw.value)
    assert set(found) == {'backtest', 'selection_ic'}, f"未找到两处调用：{found.keys()}"
    assert found['backtest'] == found['selection_ic'], \
        f"护栏判定与能力指标口径不一致：{found}"
    assert 'SECTOR_NEUTRAL' in src, "口径须来自单一常量 SECTOR_NEUTRAL"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
