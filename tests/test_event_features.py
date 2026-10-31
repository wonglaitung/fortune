import pandas as pd

from ml_services.feature_engineering.event_features import build_event_features


def test_pit_and_decay():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    ev = pd.DataFrame([{'code': '00700', 'event_type': 'PROFIT_WARNING',
                        'announce_ts': '2020-01-03 18:00', 'stage': 'announcement'}])
    out = build_event_features('0700.HK', dates, ev)
    assert out.loc['2020-01-03', 'ev_warn_decay'] == 0
    assert out.loc['2020-01-04', 'ev_warn_decay'] > 0
    assert out.loc['2020-01-05', 'ev_warn_decay'] < out.loc['2020-01-04', 'ev_warn_decay']


def test_magnitude_high_only():
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    ev = pd.DataFrame([{'code': '00700', 'event_type': 'PLACEMENT_COMPANY',
                        'announce_ts': '2020-01-02 08:00', 'stage': 'announcement'}])
    mg = pd.DataFrame([{'code': '00700', 'event_type': 'PLACEMENT_COMPANY',
                        'announce_ts': '2020-01-02 08:00', 'dilution_pct': 0.03,
                        'discount_pct': 0.05, 'size_pct_mcap': 0.02,
                        'magnitude_conf': 'high'}])
    out = build_event_features('00700', dates, ev, mg)
    assert out.loc['2020-01-03', 'place_dilution_decay'] > 0
    assert out.loc['2020-01-03', 'place_discount_decay'] < 0
