# -*- coding: utf-8 -*-
"""get_last_trading_day 港股/A股日期锚定测试

背景：2026-09-25 中秋 A 股休市但港股开市，用 A 股日历会让港股报告日期错位。
"""
import sys

import pandas as pd
import pytest

from data_services.calendar_features import (
    get_last_trading_day,
    _get_hk_last_trade_date,
)


class _FakeAk:
    """伪 akshare：A 股日历缺 2026-09-25（中秋休市），含 09-28"""

    @staticmethod
    def tool_trade_date_hist_sina():
        return pd.DataFrame({
            'trade_date': [
                '2026-09-21', '2026-09-22', '2026-09-23',
                '2026-09-24', '2026-09-28',
            ],
        })


def _hk_quote_df():
    """腾讯 hkfqkline 真实格式：Date 为 tz-aware 索引，09-25 港股有交易"""
    return pd.DataFrame(
        {'Close': [441.0, 438.4, 436.6]},
        index=pd.to_datetime(['2026-09-23', '2026-09-24', '2026-09-25'], utc=True),
    )


@pytest.fixture
def fake_ak(monkeypatch):
    monkeypatch.setitem(sys.modules, 'akshare', _FakeAk)


def test_hk_uses_quote_last_date(monkeypatch):
    monkeypatch.setattr(
        'data_services.tencent_finance.get_hsi_data_tencent',
        lambda period_days=90: _hk_quote_df(),
    )
    assert get_last_trading_day(market='HK') == '2026-09-25'


def test_hk_date_ref_not_exceed_reference(monkeypatch):
    monkeypatch.setattr(
        'data_services.tencent_finance.get_hsi_data_tencent',
        lambda period_days=90: _hk_quote_df(),
    )
    assert get_last_trading_day('2026-09-24', market='HK') == '2026-09-24'
    assert get_last_trading_day('2026-09-26', market='HK') == '2026-09-25'


def test_hk_quote_failure_falls_back_to_a_calendar(monkeypatch, fake_ak):
    monkeypatch.setattr(
        'data_services.tencent_finance.get_hsi_data_tencent',
        lambda period_days=90: None,
    )
    assert get_last_trading_day(market='HK') == '2026-09-24'


def test_hk_quote_exception_falls_back_to_a_calendar(monkeypatch, fake_ak):
    def _boom(period_days=90):
        raise RuntimeError('network down')
    monkeypatch.setattr(
        'data_services.tencent_finance.get_hsi_data_tencent', _boom)
    assert get_last_trading_day(market='HK') == '2026-09-24'


def test_a_market_default_unchanged(fake_ak):
    assert get_last_trading_day() == '2026-09-24'
    assert get_last_trading_day(market='A') == '2026-09-24'


def test_a_market_before_open_rolls_back_one_day(fake_ak):
    # 09-26 08:00（开市前）→ 回退到 09-25 → A 股日历无此日 → 09-24
    from datetime import datetime
    assert get_last_trading_day(datetime(2026, 9, 26, 8, 0)) == '2026-09-24'


def test_hk_last_date_direct(monkeypatch):
    monkeypatch.setattr(
        'data_services.tencent_finance.get_hsi_data_tencent',
        lambda period_days=90: _hk_quote_df(),
    )
    assert _get_hk_last_trade_date() == '2026-09-25'
    assert _get_hk_last_trade_date('2026-09-24') == '2026-09-24'
    assert _get_hk_last_trade_date('2026-09-20') is None  # 覆盖范围外
