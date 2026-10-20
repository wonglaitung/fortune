# -*- coding: utf-8 -*-
"""特征缓存文件定位 —— O4/D27 缓存键重构后的**单一入口**。

缓存文件命名史：
- 旧：`{code}_{YYYYMMDD}_shift.pkl`
- 新（2026-10-13 O4/D27，`ml_trading_model._get_feature_cache_key`）：
  `{code}_{YYYYMMDD}_{tminus1|tclose}_{f1|base}_{manifest_fp}.pkl`

**为何需要本模块**：多个研究脚本只 `glob('*_shift.pkl')`。重构后该模式**永远取到空集**，
脚本**静默产出空/旧结果而不报错**（本项目已实际发生——异常检测 10d/20d 与 D58 一组
研究脚本；见 DECISIONS D59 §十八）。本模块统一兼容两代命名，避免各处再写一遍。
"""
import glob
import os

CACHE_GLOBS = ('*_shift.pkl', '*_tclose_*.pkl', '*_tminus1_*.pkl')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'data', 'feature_cache')


def latest_feature_cache_files(cache_dir=None):
    """返回 {code: 最新缓存文件路径}（按 mtime 取每股最新）。

    Args:
        cache_dir: 缓存目录；None ⇒ 仓库内 `data/feature_cache`（绝对路径）。
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    best = {}
    for pat in CACHE_GLOBS:
        for p in glob.glob(os.path.join(cache_dir, pat)):
            code = os.path.basename(p).split('_')[0]
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            prev = best.get(code)
            if prev is None or mtime > prev[0]:
                best[code] = (mtime, p)
    return {c: v[1] for c, v in best.items()}
