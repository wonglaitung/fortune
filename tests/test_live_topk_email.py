"""D2 行业中性 TopK（live_topk.select_from_df）单元测试。

背景（2026-09-27）：综合分析邮件此前只按 raw 20d 概率排序，与 DECISIONS D2
保留的"20d 行业中性 TopK"策略口径割裂；现把行业内 z-score 排名抽成
`select_from_df` 供邮件表格内联 🎯 标记复用（单一真相源）。本测试锁定其行为：
行业内 z-score、K 上限 = min(topk, n//4)、raw 模式、概率列别名、缺失报错。
"""
import pytest

import numpy as np
import pandas as pd

from config import STOCK_SECTOR_MAPPING
from ml_services.live_topk import select, select_from_df


def _real_codes(n=None):
    codes = list(STOCK_SECTOR_MAPPING.keys())
    return codes if n is None else codes[:n]


def _make_df(codes, probs):
    return pd.DataFrame({
        'code': codes,
        'probability': probs,
        'name': [f"name-{c}" for c in codes],
    })


def test_sector_neutral_scores_mean_zero_per_sector():
    codes = _real_codes(31)
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.3, 0.7, len(codes))
    target, ranked, k = select_from_df(_make_df(codes, probs), topk=10, sector_neutral=True)

    for sector, g in ranked.groupby('sector'):
        assert g['score'].mean() == pytest.approx(0.0, abs=1e-9), f"{sector} 行业内 z 均值应≈0"
    assert k == min(10, len(codes) // 4)


def test_k_cap_matches_backtest_rule():
    codes = _real_codes(31)
    probs = np.linspace(0.3, 0.7, len(codes))
    _, _, k = select_from_df(_make_df(codes, probs), topk=10, sector_neutral=True)
    assert k == len(codes) // 4  # 31 只池 → K=7，与 portfolio_backtest 的 k=min(topk,n//4) 一致


def test_target_is_top_k_by_score():
    codes = _real_codes(59)
    rng = np.random.RandomState(7)
    probs = rng.uniform(0.3, 0.8, len(codes))
    target, ranked, k = select_from_df(_make_df(codes, probs), topk=10, sector_neutral=True)

    assert len(target) == k
    top_codes = set(ranked.sort_values('score', ascending=False).head(k)['code'])
    assert set(target['code']) == top_codes


def test_raw_mode_score_equals_prob():
    codes = _real_codes(20)
    probs = np.linspace(0.3, 0.8, len(codes))
    target, ranked, _ = select_from_df(_make_df(codes, probs), topk=5, sector_neutral=False)
    assert ranked['score'].equals(ranked['prob'])


def test_sector_neutral_differs_from_raw():
    """同 prob 的两只股票，所处板块均值越低，行业中性分越高（跨板块可比性）。"""
    from collections import defaultdict
    sector_codes = defaultdict(list)
    for code, meta in STOCK_SECTOR_MAPPING.items():
        sector_codes[meta.get('sector')].append(code)
    sectors = [s for s in sector_codes if len(sector_codes[s]) >= 3][:2]
    a, b = sectors
    a_codes, b_codes = sector_codes[a][:3], sector_codes[b][:3]
    # a 板块整体更高；两只同 0.70 的票分属两板块（a 高均值 / b 低均值）
    df = _make_df(a_codes + b_codes, [0.70, 0.68, 0.66, 0.70, 0.48, 0.46])

    _, raw_ranked, _ = select_from_df(df, topk=2, sector_neutral=False)
    s_raw = raw_ranked.set_index('code')['score']
    assert s_raw[a_codes[0]] == pytest.approx(s_raw[b_codes[0]], abs=1e-9)  # raw 口径下两者平手

    _, neutral_ranked, _ = select_from_df(df, topk=2, sector_neutral=True)
    s = neutral_ranked.set_index('code')['score']
    assert s[b_codes[0]] > s[a_codes[0]]  # 行业中性：低均值板块内的 0.70 反而更高


def test_prob_column_aliases():
    codes = _real_codes(16)
    probs = np.linspace(0.4, 0.7, len(codes))
    df = _make_df(codes, probs).rename(columns={'probability': 'Predict_Prob'})
    target, ranked, _ = select_from_df(df, topk=4, sector_neutral=False)
    assert 'prob' in ranked.columns
    assert len(target) == 4  # 16 只池 → k = min(4, 16//4) = 4


def test_missing_prob_column_raises():
    df = pd.DataFrame({'code': _real_codes(5)})
    with pytest.raises(ValueError):
        select_from_df(df, topk=2)


def test_select_reads_csv(tmp_path):
    codes = _real_codes(12)
    probs = np.linspace(0.3, 0.8, len(codes))
    csv = tmp_path / 'pred.csv'
    _make_df(codes, probs).to_csv(csv, index=False)
    target, ranked, k = select(csv, topk=5)
    assert k == min(5, len(codes) // 4)
    assert len(target) == k