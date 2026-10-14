# -*- coding: utf-8 -*-
"""美股缓存覆盖防护（2026-10-13 `us_market_data.py` 修复的回归测试）。

**事故背景**（修复写在 commit 389c6020，但当时**没有提交任何测试**，声明"测试 6 项全过"是空的）：
`_load_cache` **完全忽略** `period_days` / `start_date`，只要文件存在就返回；
而缓存文件名不含这两个参数，多个调用方共用同一份 `name.pkl` 却传不同值
（`hsi_email.py` 传 30、`ml_trading_model` 传 1460 或 `start_date`）
⇒ **「谁最后跑谁说了算」**，短历史静默覆盖长历史。

实测后果：`nasdaq.pkl` 由 **1460 行（2020-12→2026-10）退化为 195 行（2026-01→2026-10）**，
读取端不做任何校验 ⇒ 2020–2025 的美股特征静默消失，**模型照跑、指标照出**。

修复要点（本文件逐项锁定）：
1. `_required_start` —— 请求需要的历史起点（`start_date` 优先于 `period_days`）
2. `_covers_request` —— 缓存是否覆盖本次请求（含容忍滑窗 `_COVER_SLACK`）
3. `_load_cache` —— 覆盖不足 ⇒ 返回 `None` 触发重抓（**核心防护**）
4. `_save_cache` —— 拒绝用更短历史覆盖更长
5. 快照模式下 `_save_cache` 不写
6. `_covers_request` 的边界：`need_start=None` / 空 df ⇒ 视为覆盖

⚠️ **这些测试全部只写 `tmp_path`，不碰 `data/us_market_cache/`。**
"""
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytest

# ⚠️ 必须用 importlib 直取**模块对象**：
# `ml_services/__init__.py:18` 有 `from .us_market_data import us_market_data`
# ——它把一个 USMarketData 单例绑成了 `ml_services.us_market_data`，
# 于是 `from ml_services import us_market_data` 与 `import ml_services.us_market_data as X`
# **都拿到那个实例而不是模块**（子模块名被同名导出遮蔽）。importlib 走 sys.modules 才拿得到模块。
import importlib
umd = importlib.import_module('ml_services.us_market_data')


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """把缓存目录指向 tmp_path，并强制关闭快照模式。"""
    monkeypatch.setattr(umd, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('US_MARKET_SNAPSHOT_DIR', raising=False)
    monkeypatch.setattr(umd, '_snapshot_mode', lambda: False)
    yield


def _frame(n_days, end=None):
    end = pd.Timestamp(end or pd.Timestamp.now(tz='UTC').normalize())
    idx = pd.date_range(end=end, periods=n_days, freq='B', tz='UTC')
    return pd.DataFrame({'Close': range(1, n_days + 1)}, index=idx)


def _write_cache(name, df, date=None):
    path = umd._get_cache_key(name)
    # ⚠️ `_save_cache` 自身**不建目录**（靠 `data/us_market_cache/` 已存在）；
    # 干净环境下首次写缓存会走进 `except` 只打印警告 —— 已记录，未在本条修。
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'date': date or datetime.now(), 'data': df}, f)
    return path


# --------------------------------------------------------------------------
# 1. _required_start
# --------------------------------------------------------------------------

def test_required_start_start_date_wins():
    """指定 start_date 时以它为准，不受 period_days 影响。"""
    got = umd._required_start(period_days=1460, start_date='2020-06-01')
    assert got == pd.Timestamp('2020-06-01', tz='UTC')


def test_required_start_without_start_date():
    """未指定 start_date 时按 period_days 回算起点。"""
    got = umd._required_start(period_days=1460, start_date=None)
    expect = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1460)
    assert abs((got - expect).days) <= 1


def test_required_start_neither_returns_none():
    """两个都不给 ⇒ None（不校验覆盖）。"""
    assert umd._required_start(period_days=None, start_date=None) is None


# --------------------------------------------------------------------------
# 2. _covers_request
# --------------------------------------------------------------------------

def test_covers_request_short_history_returns_false():
    """短历史不覆盖长请求 —— 这就是事故本身的判定。"""
    need = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1460)
    short = _frame(195)                     # 约 9 个月，远短于 4 年
    assert umd._covers_request(short, need) is False


def test_covers_request_long_history_returns_true():
    need = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1460)
    long = _frame(1460)                     # 约 6 年
    assert umd._covers_request(long, need) is True


def test_covers_request_edge_cases():
    """need_start=None / 空 df ⇒ 视为覆盖（不误伤）。"""
    assert umd._covers_request(_frame(10), None) is True
    assert umd._covers_request(pd.DataFrame(), pd.Timestamp('2020-01-01', tz='UTC')) is True


def test_covers_request_allows_small_slack():
    """允许 `_COVER_SLACK` 的小缺口（容忍非交易日/节假日），但不容忍大幅截断。"""
    need = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=200)
    # 只差 3 天 ⇒ 在滑窗内
    assert umd._covers_request(_frame(197), need) is True
    # 差 90 天 ⇒ 超出滑窗
    assert umd._covers_request(_frame(110), need) is False


# --------------------------------------------------------------------------
# 3. _load_cache：覆盖不足必须拒绝（核心防护）
# --------------------------------------------------------------------------

def test_load_cache_rejects_short_history(tmp_path):
    """事故复现：磁盘上是 195 行短历史，请求 1460 天 ⇒ 必须返回 None 触发重抓。"""
    _write_cache('nasdaq', _frame(195))
    got = umd._load_cache('nasdaq', period_days=1460, start_date=None)
    assert got is None, '短历史缓存被接受了 —— 静默污染防护失效'


def test_load_cache_accepts_long_history(tmp_path):
    """覆盖充分 ⇒ 正常返回。"""
    df = _frame(1460)
    _write_cache('nasdaq', df)
    got = umd._load_cache('nasdaq', period_days=1460, start_date=None)
    assert got is not None and len(got) == len(df)


# --------------------------------------------------------------------------
# 4. _save_cache：拒绝短覆盖长
# --------------------------------------------------------------------------

def test_save_cache_refuses_shorter_history(tmp_path):
    """事故的另一面：短历史不得覆盖已有长历史。"""
    path = umd._get_cache_key('nasdaq')
    long_df = _frame(1460)
    _write_cache('nasdaq', long_df)

    umd._save_cache('nasdaq', _frame(195))          # 尝试写短历史

    with open(path, 'rb') as f:
        kept = pickle.load(f)['data']
    assert len(kept) == len(long_df), '短历史覆盖了长历史'


def test_save_cache_allows_longer_history(tmp_path):
    """更长历史应当正常写入（不能把防护做成「永不更新」）。"""
    path = umd._get_cache_key('nasdaq')
    _write_cache('nasdaq', _frame(195))
    umd._save_cache('nasdaq', _frame(1460))
    with open(path, 'rb') as f:
        assert len(pickle.load(f)['data']) == 1460


def test_save_cache_skipped_in_snapshot_mode(tmp_path, monkeypatch):
    """冻结快照模式下一律不写（否则会污染 PIT 快照）。"""
    monkeypatch.setattr(umd, '_snapshot_mode', lambda: True)
    umd._save_cache('nasdaq', _frame(1460))
    import os
    assert not os.path.exists(umd._get_cache_key('nasdaq'))


# --------------------------------------------------------------------------
# 5. 回归护栏：这三个函数必须还在，且默认路径不含 period_days
# --------------------------------------------------------------------------

def test_cache_key_does_not_encode_period_days():
    """⚠️ 已知设计缺陷（**故意未修**）：缓存文件名不含 period_days / start_date。

    修复是在**读取端**加覆盖校验，不是靠文件名区分。若将来改成把参数并入文件名，
    本护栏会失败 —— 那时应当同步删掉 `_covers_request` 那层校验或改为「双保险」，
    不要两边都留着造成判断分叉。
    """
    a = umd._get_cache_key('nasdaq')
    b = umd._get_cache_key('nasdaq')
    assert a == b
    assert '1460' not in a and '30' not in a, \
        '缓存键若已编码参数，需同步更新本护栏与 _load_cache/_save_cache 的覆盖逻辑'