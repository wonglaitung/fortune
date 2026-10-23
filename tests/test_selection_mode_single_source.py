# -*- coding: utf-8 -*-
"""打分口径「唯一实现 = raw」契约（D73）

2026-10-23 起生产口径固定为 raw prob（保留板块间信息），行业中性已**整体删除**，
不保留开关、不保留向后兼容分支。本文件锁死三件事：

  1. 生产选股（select_from_df）打分 == raw prob；
  2. 生产调用点不再传任何口径参数；
  3. 代码里不再存在 `use_sector_neutral` / `SECTOR_NEUTRAL` 这类开关符号。
"""

import re
from pathlib import Path

import pandas as pd

import ml_services.live_topk as lt

ROOT = Path(__file__).resolve().parents[1]


def _df():
    # 真实代码：select_from_df 会用 config 的板块映射覆盖 sector 列
    return pd.DataFrame({
        'code': ['0005.HK', '0939.HK', '0700.HK', '9988.HK', '1810.HK', '0941.HK'],
        'prob': [0.90, 0.80, 0.60, 0.55, 0.50, 0.40],
    })


def test_score_is_raw_prob():
    _, ranked, _ = lt.select_from_df(_df(), topk=4)
    assert ranked['score'].equals(ranked['prob'])


def test_top_is_global_max_not_sector_adjusted():
    target, _, _ = lt.select_from_df(_df(), topk=1)
    assert target['code'].tolist() == ['0005.HK']


def test_production_call_site_has_no_caliber_argument():
    src = (ROOT / 'comprehensive_analysis.py').read_text(encoding='utf-8')
    m = re.search(r'select_from_df\(df_catboost[^\n]*\)', src)
    assert m, '未找到生产 select_from_df 调用'
    assert 'sector_neutral' not in m.group(0), '生产调用点又传了口径参数'


def test_no_switch_symbols_anywhere():
    """D73：不保留开关。任何 py 文件出现这两个符号即失败。"""
    bad = []
    files = list((ROOT / 'ml_services').rglob('*.py')) + list((ROOT / 'scripts').rglob('*.py')) \
        + list(ROOT.glob('*.py'))
    for p in files:
        if '__pycache__' in p.parts:
            continue
        src = p.read_text(encoding='utf-8', errors='ignore')
        if 'use_sector_neutral' in src or 'SECTOR_NEUTRAL' in src:
            bad.append(str(p.relative_to(ROOT)))
    assert bad == [], f'仍存在中性/raw 开关：{bad}'
