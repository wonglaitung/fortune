"""
F1 基本面特征模块（D20 预注册）：stock_financial_hk_report_em 历史多期三表。
- PIT 固定滞后：年报 period_end+120d / 中报+90d = 可得日（港股披露法定期限，保守防漏）
- 派生比率：ROE/ROA/净利率/营收同比/负债权益/现金流质量 等
- 跨股票截面 z 由集成层处理（提供 cross_section_z 辅助）
按股+报表缓存到 FUND_CACHE（/tmp，不入库），可断点续传。
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/data/fortune')
try:
    import config
except Exception:
    config = None

FUND_CACHE = '/tmp/opencode/fund_cache'
STATEMENTS = [('资产负债表', 'BS'), ('利润表', 'IS'), ('现金流量表', 'CF')]
INDICATORS = [('年度', 120), ('中期', 90)]  # (indicator, 滞后天数)


def _code5(cfg_code: str) -> str:
    return cfg_code.split('.')[0].zfill(5)


def _cache_file(cfg_code, symbol, indicator):
    os.makedirs(FUND_CACHE, exist_ok=True)
    return os.path.join(FUND_CACHE, f"{cfg_code}_{symbol}_{indicator}.pkl")


def fetch_one_statement(cfg_code, symbol, indicator):
    """抓取单股单报表单频率，返回透视表(period_end x item_name)。带缓存。"""
    cf = _cache_file(cfg_code, symbol, indicator)
    if os.path.exists(cf):
        return pd.read_pickle(cf)
    import akshare as ak
    try:
        df = ak.stock_financial_hk_report_em(stock=_code5(cfg_code), symbol=symbol, indicator=indicator)
    except Exception as e:
        df = pd.DataFrame()
    if df is None or df.empty:
        pd.to_pickle(pd.DataFrame(), cf)
        return pd.DataFrame()
    # 透视: index=REPORT_DATE, columns=STD_ITEM_NAME, values=AMOUNT
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    wide = df.pivot_table(index='REPORT_DATE', columns='STD_ITEM_NAME', values='AMOUNT', aggfunc='last')
    pd.to_pickle(wide, cf)
    return wide


def _find_item(wide, *names):
    cols = wide.columns
    for n in names:
        if n in cols:
            return wide[n]
    # 模糊匹配（含任一关键词）
    for n in names:
        for c in cols:
            if n in str(c):
                return wide[c]
    return None


def derive_ratios(wide_bs, wide_is, wide_cf, period_end, lag_days):
    """由三表透视表派生比率，返回 (DataFrame[index=period_end, cols=ratios], availability_date)。"""
    # 对齐三表到共同 period_end 集合（各表 period_end 可能略有差异，取交集）
    idx = wide_bs.index
    if wide_is is not None and not wide_is.empty:
        idx = idx.intersection(wide_is.index)
    if wide_cf is not None and not wide_cf.empty:
        idx = idx.intersection(wide_cf.index)
    if len(idx) == 0:
        return pd.DataFrame(), pd.Series(dtype='datetime64[ns]')
    rev = _find_item(wide_is, '营业收入', '营业额', '收入') if wide_is is not None else None
    np_ = _find_item(wide_is, '净利润', '除税后溢利') if wide_is is not None else None
    eq = _find_item(wide_bs, '总权益', '股东权益', '净资产', '权益合计') if wide_bs is not None else None
    liab = _find_item(wide_bs, '总负债', '负债合计', '负债总额') if wide_bs is not None else None
    ta = _find_item(wide_bs, '总资产', '资产总计') if wide_bs is not None else None
    ocf = _find_item(wide_cf, '经营业务现金净额', '经营产生现金', '经营活动产生的现金流量净额') if wide_cf is not None else None
    rows = {}
    for pe in idx:
        r = {}
        r_rev = None if rev is None else rev.get(pe)
        r_np = None if np_ is None else np_.get(pe)
        r_eq = None if eq is None else eq.get(pe)
        r_liab = None if liab is None else liab.get(pe)
        r_ta = None if ta is None else ta.get(pe)
        r_ocf = None if ocf is None else ocf.get(pe)
        if r_np is not None and r_rev not in (None, 0):
            r['Net_Margin'] = r_np / r_rev
        if r_np is not None and r_eq not in (None, 0):
            r['ROE'] = r_np / r_eq
        if r_np is not None and r_ta not in (None, 0):
            r['ROA'] = r_np / r_ta
        if r_liab is not None and r_eq not in (None, 0):
            r['Debt_Equity'] = r_liab / r_eq
        if r_ocf is not None and r_np not in (None, 0):
            r['OCF_Quality'] = r_ocf / r_np
        if r_rev is not None and r_ta not in (None, 0):
            r['Asset_Turnover'] = r_rev / r_ta
        if r_rev not in (None, 0):
            r['log_Revenue'] = np.log1p(abs(r_rev)) * np.sign(r_rev)
        if r_np is not None:
            r['log_NetProfit'] = np.log1p(abs(r_np)) * np.sign(r_np)
        rows[pe] = r
    rat = pd.DataFrame(rows).T
    rat.index = pd.to_datetime(rat.index)
    avail = pd.Series(rat.index + pd.Timedelta(days=lag_days), index=rat.index, name='avail')
    return rat, avail


def build_fundamental_features(pool_codes, indicators=INDICATORS, statements=STATEMENTS):
    """对池内每只股构建基本面比率时间序列（含可得日）。返回 dict: cfg_code -> (ratios_df, avail_series)。"""
    out = {}
    for cfg in pool_codes:
        # 先集齐该股全部 6 张表（3 报表 × 2 频率）
        tables = {}
        for sym, abbr in statements:
            for ind, lag in indicators:
                tables[(sym, ind)] = fetch_one_statement(cfg, sym, ind)
        # 按频率(年度/中报)集齐三表后统一派生比率
        combined = {}
        combined_avail = {}
        for ind, lag in indicators:
            wbs = tables.get(('资产负债表', ind))
            wis = tables.get(('利润表', ind))
            wcf = tables.get(('现金流量表', ind))
            rat, av = derive_ratios(wbs, wis, wcf, None, lag)
            if not rat.empty:
                combined[ind] = rat
                combined_avail[ind] = av
        if not combined:
            out[cfg] = (pd.DataFrame(), pd.Series(dtype='datetime64[ns]'))
            continue
        # 合并年度与中报（中报频率更高，优先；年度补充缺失期）
        if '中期' in combined and '年度' in combined:
            rat = combined['中期'].combine_first(combined['年度'])
            av = combined_avail['中期'].combine_first(combined_avail['年度'])
        else:
            k = next(iter(combined))
            rat, av = combined[k], combined_avail[k]
        rat = rat.sort_index()
        av = av.reindex(rat.index)
        out[cfg] = (rat, av)
    return out


def fundamental_feature_frame(cfg_code, trade_dates, fund_data):
    """返回某股在 trade_dates 上的基本面比率特征（PIT: 取 avail <= t 最新期，前向填充）。"""
    rat, av = fund_data.get(cfg_code, (pd.DataFrame(), pd.Series(dtype='datetime64[ns]')))
    if rat is None or rat.empty:
        return pd.DataFrame(index=trade_dates, columns=[], dtype=float)
    # 每个 t: 最新 avail <= t
    td = pd.DataFrame({'t': trade_dates})
    td['key'] = td['t']
    avf = av.reset_index(); avf.columns = ['period_end', 'avail']
    merged = pd.merge_asof(td.sort_values('key'), avf.sort_values('avail'),
                          left_on='key', right_on='avail', direction='backward')
    merged = merged.sort_values('t').set_index('t'); merged.index.name = None
    # 映射 period_end -> ratio 行
    rat_idx = rat.index
    out = pd.DataFrame(index=trade_dates, columns=rat.columns, dtype=float)
    for t in trade_dates:
        row = merged.loc[t]
        pe = row['period_end'] if 'period_end' in row and not pd.isna(row.get('period_end')) else None
        if pe is not None:
            pe = pd.Timestamp(pe)
            if pe in rat.index:
                out.loc[t] = rat.loc[pe].values
    return out


def cross_section_z(pool_frames: dict, trade_dates):
    """对池内所有股在每交易日做截面 z 标准化（AGENTS: 绝对值/规模特征跨股须标准化）。"""
    cols = list(next(iter(pool_frames.values())).columns)
    out = {}
    for t in trade_dates:
        snap = pd.DataFrame({c: fr.loc[t] for c, fr in pool_frames.items()}).T
        z = (snap - snap.mean()) / (snap.std() + 1e-9)
        out[t] = z
    return out
