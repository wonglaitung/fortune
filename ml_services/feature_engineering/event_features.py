"""公司行动事件特征（P2b）。

输入：PIT 事件表（announce_ts）+ 可选规模表（仅 high 置信）。
输出：每只股票按交易日的特征矩阵；事件在公告日 T+1 才可用。
"""
import numpy as np
import pandas as pd

DECAY_DAYS = 20
MAX_DAYS = 90  # 超过 90 天视为无事件，避免旧事件长期占据排序

TYPE_FEATURE = {
    'PROFIT_WARNING': 'ev_warn_decay',
    'POSITIVE_PROFIT_ALERT': 'ev_alert_decay',
    'PLACEMENT_COMPANY': 'ev_place_decay',
    'PLACEMENT_SHAREHOLDER': 'ev_holder_place_decay',
    'CB_ISSUANCE': 'ev_cb_decay',
    'CB_CONVERSION': 'ev_cb_conv_decay',
    'PRIVATISATION': 'ev_privatise_decay',
    'SCHEME': 'ev_scheme_decay',
    'LOCKUP': 'ev_lockup_decay',
    'RIGHTS_ISSUE': 'ev_rights_decay',
    'OPEN_OFFER': 'ev_open_offer_decay',
}

MAG_FEATURES = {
    'PLACEMENT_COMPANY': [('dilution_pct', 'place_dilution_decay', 1.0),
                          ('discount_pct', 'place_discount_decay', -1.0),
                          ('size_pct_mcap', 'place_size_decay', 1.0)],
    'CB_ISSUANCE': [('dilution_pct', 'cb_dilution_decay', 1.0),
                    ('cb_principal_hkd', 'cb_principal_decay', 1.0)],
    'PRIVATISATION': [('offer_premium_pct', 'offer_premium_decay', 1.0)],
    'LOCKUP': [('lockup_days', 'lockup_days_decay', 1.0)],
}

FEATURES = sorted(set(TYPE_FEATURE.values()) |
                  {f for v in MAG_FEATURES.values() for _, f, _ in v} |
                  {'ev_active', 'ev_days_since', 'ev_magnitude_missing'})


def _code5(x):
    return str(x).split('.')[0].zfill(5)


def build_event_features(code, dates, events, magnitudes=None):
    """返回 DataFrame(index=dates)，列见 FEATURES。"""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    out = pd.DataFrame(0.0, index=dates, columns=FEATURES)
    if events is None or events.empty:
        return out
    ev = events.copy()
    ev['code'] = ev.code.map(_code5)
    ev = ev[ev.code == _code5(code)]
    ev = ev[ev.stage == 'announcement']
    ev['announce_ts'] = pd.to_datetime(ev.announce_ts).dt.tz_localize(None)
    ev = ev.sort_values('announce_ts')
    if ev.empty:
        return out

    for _, e in ev.iterrows():
        t = e.announce_ts.normalize()
        # T+1：严格大于公告日
        future = dates > t
        if not future.any():
            continue
        days = (dates[future] - t).days.values
        keep = days <= MAX_DAYS
        if not keep.any():
            continue
        idx = np.where(future)[0][keep]
        days = days[keep]
        decay = np.exp(-days / DECAY_DAYS)
        col = TYPE_FEATURE.get(e.event_type)
        if col:
            out.iloc[idx, out.columns.get_loc(col)] = np.maximum(out.iloc[idx, out.columns.get_loc(col)].values, decay)
        out.iloc[idx, out.columns.get_loc('ev_active')] = 1.0
        ds = out.iloc[idx, out.columns.get_loc('ev_days_since')].values
        out.iloc[idx, out.columns.get_loc('ev_days_since')] = np.where(ds == 0, days, np.minimum(ds, days))

    if magnitudes is not None and not magnitudes.empty:
        mg = magnitudes.copy()
        mg['code'] = mg.code.map(_code5)
        mg = mg[(mg.code == _code5(code)) & (mg.magnitude_conf == 'high')]
        mg['announce_ts'] = pd.to_datetime(mg.announce_ts).dt.tz_localize(None)
        for _, e in mg.sort_values('announce_ts').iterrows():
            future = dates > e.announce_ts.normalize()
            if not future.any():
                continue
            days = (dates[future] - e.announce_ts.normalize()).days.values
            keep = days <= MAX_DAYS
            if not keep.any():
                continue
            idx = np.where(future)[0][keep]
            decay = np.exp(-days[keep] / DECAY_DAYS)
            for src, dst, sign in MAG_FEATURES.get(e.event_type, []):
                v = e.get(src)
                if pd.isna(v):
                    continue
                if src == 'lockup_days':
                    v = float(v) / 365.0
                new = sign * float(v) * decay
                cur = out.iloc[idx, out.columns.get_loc(dst)].values
                out.iloc[idx, out.columns.get_loc(dst)] = np.minimum(cur, new) if sign < 0 else np.maximum(cur, new)
    # 有事件但规模缺失：active 且 magnitude 全 0
    mag_cols = [f for v in MAG_FEATURES.values() for _, f, _ in v]
    out['ev_magnitude_missing'] = ((out['ev_active'] > 0) &
                                   (out[mag_cols].abs().sum(axis=1) == 0)).astype(float)
    return out
