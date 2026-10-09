"""
F1 特征模块（D20 预注册）：SFC 卖空 + 基本面三表，均为 PIT 正确、可单测的独立组件。
- SFC 卖空：周频，PIT 取 week_end <= t-7（规避 SFC ~1 周发布滞后），前向填充到日。
- 基本面三表：stock_financial_hk_report_em 历史多期，固定滞后 PIT（年报+120d/中报+90d），派生比率。
本模块只产出"时点正确"的特征序列；跨股票截面 z 标准化由集成层处理（或 cross_section_z 辅助函数）。
"""
import os
import numpy as np
import pandas as pd

try:
    import config
except Exception:
    config = None

SFC_CSV = 'data/sfc_short/sfc_short_long.csv'


def _sfc_to_config_code(sfc_code: str) -> str:
    """SFC 无前导零代码 -> config 代码（如 '700' -> '0700.HK'）。"""
    if config is None:
        return sfc_code
    for c in config.TRAINING_STOCKS:
        if c.split('.')[0].lstrip('0') == sfc_code.lstrip('0'):
            return c
    return sfc_code


_SFC_CACHE = None


def load_sfc_long():
    global _SFC_CACHE
    if _SFC_CACHE is not None:
        return _SFC_CACHE
    sfc = pd.read_csv(SFC_CSV, dtype={'stock_code': str})
    sfc['date'] = pd.to_datetime(sfc['date'], format='%Y%m%d')
    sfc['short_shares'] = pd.to_numeric(sfc['short_shares'], errors='coerce')
    sfc['short_hk'] = pd.to_numeric(
        sfc['short_hk$'].astype(str).str.replace(',', ''), errors='coerce')
    sfc['cfg'] = sfc['stock_code'].map(_sfc_to_config_code)
    sfc = sfc.dropna(subset=['cfg'])
    _SFC_CACHE = sfc
    return sfc


def _weekly_z(series: pd.Series, win: int = 52, min_n: int = 26) -> pd.Series:
    """对单股周度 short_shares 序列算 trailing z（截至当前周之前的历史）。"""
    out = pd.Series(index=series.index, dtype=float)
    vals = series.values
    for k in range(len(vals)):
        hist = vals[max(0, k - win):k]
        if len(hist) >= min_n and hist.std() > 0:
            out.iloc[k] = (vals[k] - hist.mean()) / hist.std()
    return out


def sfc_feature_frame(cfg_code: str, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """返回某股在 trade_dates 上的 SFC 特征（PIT t-7 前向填充）。

    列: log_short_shares, log_short_hk, short_z
    PIT 保证: 任意 t 使用的 SFC week_end <= t - 7 天。
    """
    sfc = load_sfc_long()
    sub = sfc[sfc['cfg'] == cfg_code].sort_values('date')
    if sub.empty:
        return pd.DataFrame(index=trade_dates,
                           columns=['log_short_shares', 'log_short_hk', 'short_z'],
                           dtype=float)
    weekly = sub.set_index('date')[['short_shares', 'short_hk']].sort_index()
    weekly['short_z'] = _weekly_z(weekly['short_shares'])
    weekly = weekly.reset_index().rename(columns={'date': 'week_end'})
    weekly['week_end'] = pd.to_datetime(weekly['week_end'])
    if weekly['week_end'].dt.tz is None:
        weekly['week_end'] = weekly['week_end'].dt.tz_localize('UTC')
    weekly['log_short_shares'] = np.log1p(weekly['short_shares'].clip(lower=0))
    weekly['log_short_hk'] = np.log1p(weekly['short_hk'].clip(lower=0))
    # merge_asof: 用 t-7 作为查找键，direction=backward -> 最新 week_end <= t-7
    # 时区对齐：交易索引可能为 UTC，特征日期为 naive，merge_asof 要求同类型
    td = pd.DataFrame({'t': pd.to_datetime(trade_dates)})
    if td['t'].dt.tz is None:
        td['t'] = td['t'].dt.tz_localize('UTC')
    td['key'] = td['t'] - pd.Timedelta(days=7)
    weekly_sorted = weekly.sort_values('week_end')
    merged = pd.merge_asof(
        td.sort_values('key'), weekly_sorted, left_on='key',
        right_on='week_end', direction='backward')
    merged = merged.sort_values('t').set_index('t')
    merged.index.name = None
    feats = merged[['log_short_shares', 'log_short_hk', 'short_z']]
    feats = feats.reindex(trade_dates)
    return feats
