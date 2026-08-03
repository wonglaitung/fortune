"""
主力资金数据服务测试

覆盖：
1. fetch_history 重试机制（瞬时失败后恢复、3次全败返回 None、退避间隔 5s/10s）
2. get_main_fund_trend 的 available 标志
3. generate_comprehensive_report 在数据不可用时输出显式警告
"""

import pytest
import pandas as pd
import requests

from data_services import main_fund_flow
from data_services.main_fund_flow import MainFundFlowService


# 15字段格式的 kline 样本（日期,主力净,小单,中单,大单,超大单,5个占比,上证收盘/涨跌,深证收盘/涨跌）
SAMPLE_KLINE = (
    "2026-07-31,-1234567890,-500000000,-300000000,-200000000,-1034567890,"
    "-0.62,-0.25,-0.15,-0.10,-0.52,3500.50,1.20,11000.25,-0.50"
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _ok_payload():
    return {"rc": 0, "data": {"klines": [SAMPLE_KLINE]}}


@pytest.fixture
def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(main_fund_flow.time, "sleep", lambda s: calls.append(s))
    return calls


@pytest.fixture
def service():
    return MainFundFlowService()


def test_fetch_history_success(service, monkeypatch):
    """单次请求成功时返回解析后的 DataFrame"""
    monkeypatch.setattr(
        main_fund_flow.requests, "get",
        lambda *a, **kw: FakeResponse(_ok_payload()),
    )
    df = service.fetch_history(use_cache=False, days=5)
    assert df is not None
    assert len(df) == 1
    assert pytest.approx(df.iloc[0]['main_net_flow'], rel=1e-6) == -12.34567890
    assert df.iloc[0]['sh_close'] == 3500.50


def test_fetch_history_retries_then_succeeds(service, monkeypatch, sleep_calls):
    """前两次连接中断、第三次成功：返回数据，退避 5s/10s"""
    call_count = {'n': 0}

    def fake_get(*args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] < 3:
            raise requests.exceptions.ConnectionError("RemoteDisconnected")
        return FakeResponse(_ok_payload())

    monkeypatch.setattr(main_fund_flow.requests, "get", fake_get)
    df = service.fetch_history(use_cache=False, days=5)
    assert df is not None
    assert call_count['n'] == 3
    assert sleep_calls == [5, 10]


def test_fetch_history_all_retries_fail_returns_none(service, monkeypatch, sleep_calls):
    """3次全部失败：返回 None（调用方据此降级），不抛异常"""
    def fake_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("RemoteDisconnected")

    monkeypatch.setattr(main_fund_flow.requests, "get", fake_get)
    df = service.fetch_history(use_cache=False, days=5)
    assert df is None
    assert sleep_calls == [5, 10]


def test_fetch_history_bad_rc_returns_none(service, monkeypatch, sleep_calls):
    """接口返回 rc != 0：重试3次后返回 None"""
    monkeypatch.setattr(
        main_fund_flow.requests, "get",
        lambda *a, **kw: FakeResponse({"rc": 1, "data": None}),
    )
    df = service.fetch_history(use_cache=False, days=5)
    assert df is None


def test_get_main_fund_trend_unavailable_flag(monkeypatch):
    """fetch_history 返回 None 时 available=False 且数值为 0"""
    import a_stock_comprehensive_analysis as aca
    monkeypatch.setattr(aca, 'MainFundFlowService', lambda: type('S', (), {'fetch_history': lambda self: None})())
    result = aca.get_main_fund_trend()
    assert result['available'] is False
    assert result['net_flow'] == 0
    assert result['net_flow_5d_sum'] == 0


def test_get_main_fund_trend_available_flag(monkeypatch):
    """fetch_history 返回数据时 available=True"""
    import a_stock_comprehensive_analysis as aca
    df = pd.DataFrame({
        'main_net_flow': [10.0, 20.0, 30.0],
        'super_large': [5.0, 5.0, 5.0],
        'large': [5.0, 5.0, 5.0],
        'mid': [1.0, 1.0, 1.0],
        'small': [1.0, 1.0, 1.0],
        'main_net_pct': [0.5, 0.6, 0.7],
        'small_pct': [0.1] * 3, 'mid_pct': [0.1] * 3,
        'large_pct': [0.2] * 3, 'super_large_pct': [0.3] * 3,
        'sh_close': [3500.0] * 3, 'sh_change_pct': [1.0] * 3,
        'sz_close': [11000.0] * 3, 'sz_change_pct': [-0.5] * 3,
    }, index=pd.date_range('2026-07-29', periods=3))
    monkeypatch.setattr(aca, 'MainFundFlowService', lambda: type('S', (), {'fetch_history': lambda self: df})())
    result = aca.get_main_fund_trend()
    assert result['available'] is True
    assert result['net_flow'] == 30.0
    assert result['net_flow_5d_sum'] == 60.0
    assert result['consecutive_inflow'] == 3


def _base_market_data():
    return {'sh_close': 3500.0, 'sh_change': 1.2, 'sh_ma20': 3450.0, 'sh_vs_ma20': 1.45}


def test_report_shows_warning_when_main_fund_unavailable():
    """market_data 无 main_fund_available 时报告显式输出获取失败警告"""
    import a_stock_comprehensive_analysis as aca
    report = aca.generate_comprehensive_report(None, None, {}, _base_market_data(), None)
    assert '### 1.2 主力资金' in report
    assert '主力资金数据获取失败' in report
    assert '净流入:' not in report.split('### 1.2 主力资金')[1].split('---')[0]


def test_report_shows_data_when_main_fund_available():
    """main_fund_available=True 时报告正常展示资金数据且无警告"""
    import a_stock_comprehensive_analysis as aca
    market_data = _base_market_data()
    market_data.update({
        'main_fund_available': True,
        'main_fund_net_flow': -12.35,
        'main_fund_super_large': -10.35,
        'main_fund_large': -2.0,
        'main_fund_mid': -3.0,
        'main_fund_small': -5.0,
        'main_fund_net_pct': -0.62,
        'main_fund_super_large_pct': -0.52,
        'main_fund_large_pct': -0.10,
        'main_fund_mid_pct': -0.15,
        'main_fund_small_pct': -0.25,
    })
    report = aca.generate_comprehensive_report(None, None, {}, market_data, None)
    assert '### 1.2 主力资金' in report
    assert '主力资金数据获取失败' not in report
    assert '-12.35 亿' in report
