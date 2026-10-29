#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股市场数据获取模块
提供美股指数、VIX恐慌指数、美国国债收益率等数据

数据源策略（AKShare 优先）：
- 标普500指数：优先 AKShare，失败后使用 yfinance
- 纳斯达克指数：优先 AKShare，失败后使用 yfinance
- VIX恐慌指数：仅使用 yfinance（AKShare 暂不支持）
- 美国国债收益率：仅使用 AKShare（稳定可靠）

缓存策略：
- 美股指数数据缓存1天（日线数据每天只更新一次）
- 当天有效，次日自动失效
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import warnings
import signal
from functools import wraps
import os
import pickle

warnings.filterwarnings('ignore')

# 缓存目录
CACHE_DIR = 'data/us_market_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存有效期（天）
CACHE_EXPIRE_DAYS = 1


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def timeout(seconds):
    """装饰器：为函数添加超时控制

    注意：使用 SIGALRM 信号实现，在多线程环境下可能有问题。
    如果超时触发，会抛出 TimeoutError。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否支持SIGALRM（Unix系统）
            if hasattr(signal, 'SIGALRM'):
                # 保存旧的信号处理器
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                    return result
                except TimeoutError:
                    # 重新抛出 TimeoutError，让调用者处理
                    raise
                except Exception as e:
                    # 其他异常也要重新抛出
                    raise
                finally:
                    # 确保取消闹钟并恢复旧的信号处理器
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Windows 系统或其他不支持 SIGALRM 的系统使用线程方法
                import threading
                from queue import Queue
                
                def target(queue):
                    try:
                        result = func(*args, **kwargs)
                        queue.put((True, result))
                    except Exception as e:
                        queue.put((False, e))
                
                queue = Queue()
                thread = threading.Thread(target=target, args=(queue,))
                thread.daemon = True
                thread.start()
                thread.join(timeout=seconds)
                
                if thread.is_alive():
                    # 线程仍在运行，视为超时
                    raise TimeoutError(f"Function call timed out after {seconds} seconds")
                
                success, result = queue.get()
                if not success:
                    raise result
                return result
        return wrapper
    return decorator


def _get_cache_key(name: str) -> str:
    """生成缓存文件路径

    若设置 US_MARKET_SNAPSHOT_DIR（冻结 PIT 快照，lessons 三.29），则读写快照目录，
    使宏观特征输入固定 → 跨日回测可复现。快照模式下忽略日期校验、永不自动刷新。
    """
    base = os.environ.get('US_MARKET_SNAPSHOT_DIR') or CACHE_DIR
    return os.path.join(base, f"{name}.pkl")


def _snapshot_mode() -> bool:
    return bool(os.environ.get('US_MARKET_SNAPSHOT_DIR'))


def _required_start(period_days, start_date):
    """本次请求需要的历史起点（UTC）。

    ⚠️ 2026-10-13（缓存污染修复）：此前 `_load_cache` **完全忽略** period_days / start_date，
    只要文件存在就返回。多个调用方共用同一份 `name.pkl` 却传不同 period_days
    （hsi_email.py 传 30、ml_trading_model 传 1460 或 start_date），
    于是「谁最后跑谁说了算」—— 短历史静默覆盖长历史。
    实测：nasdaq.pkl 由 1460 行(2020-12→2026-10) 退化为 195 行(2026-01→2026-10)，
    读取端不做任何校验 ⇒ 2020–2025 的美股特征静默消失，模型照跑、指标照出。
    """
    if start_date is not None:
        sd = pd.Timestamp(start_date)
        return sd.tz_localize('UTC') if sd.tzinfo is None else sd
    if period_days:
        return pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=int(period_days))
    return None


# 交易日/日历日比值约 0.69；留 10% 余量容忍非交易日与节假日
_COVER_SLACK = 0.10


def _covers_request(df, need_start) -> bool:
    """缓存的历史是否覆盖本次请求。覆盖不足 ⇒ 返回 False（触发重抓）。"""
    if need_start is None or df is None or len(df) == 0:
        return True
    have = df.index.min()
    if have.tzinfo is None:
        have = have.tz_localize('UTC')
    slack = pd.Timedelta(days=max(int((pd.Timestamp.now(tz="UTC") - need_start).days * _COVER_SLACK), 5))
    return have <= need_start + slack


def _load_cache(name: str, period_days=None, start_date=None) -> pd.DataFrame:
    """加载缓存数据

    ⚠️ 2026-10-13：`period_days` / `start_date` 现在**参与校验**。
    此前它们被完全忽略 —— 短历史缓存能满足长历史请求（见 `_required_start` 的说明）。

    Returns:
        DataFrame if cache valid AND covers the requested range, else None
    """
    cache_file = _get_cache_key(name)

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # 冻结快照模式：忽略日期，永久有效（可复现性优先）
        # 但**仍校验覆盖**：快照若被写短过，一样会导致特征缺失。
        if _snapshot_mode():
            data = cache_data.get('data')
            need = _required_start(period_days, start_date)
            if not _covers_request(data, need):
                print(f"⚠️ 快照 {name} 历史不足（起点 "
                      f"{data.index.min().date() if len(data) else 'N/A'} vs 需要 "
                      f"{need.date() if need is not None else 'N/A'}）⇒ 视为无效")
                return None
            return data

        # 检查缓存日期
        cache_date = cache_data.get('date')
        if cache_date:
            cache_date = pd.to_datetime(cache_date)
            today = pd.Timestamp.now().normalize()

            # 缓存有效期：当天有效
            if cache_date.date() >= today.date():
                data = cache_data.get('data')
                need = _required_start(period_days, start_date)
                if not _covers_request(data, need):
                    print(f"⚠️ 缓存 {name} 历史不足（起点 "
                          f"{data.index.min().date() if data is not None and len(data) else 'N/A'} vs 需要 "
                          f"{need.date() if need is not None else 'N/A'}）⇒ 触发重抓")
                    return None
                return data

        return None
    except Exception:
        return None


def _save_cache(name: str, data: pd.DataFrame):
    """保存缓存数据（冻结快照模式下不写，避免污染 PIT 快照）

    ⚠️ 2026-10-13：**拒绝用更短的历史覆盖更长的**。
    此前是无条件覆盖，而缓存文件名不含 period_days，多个调用方传不同参数
    （hsi_email.py 传 30、ml_trading_model 传 1460）⇒ 短历史静默覆盖长历史。
    实测 nasdaq.pkl 因此由 1460 行退化为 195 行。
    """
    if _snapshot_mode():
        return
    cache_file = _get_cache_key(name)

    # 不覆盖更长历史：若磁盘上已有起点更早（或行数更多）的数据，保留它
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                old = pickle.load(f).get('data')
            if old is not None and len(old) > len(data):
                print(f"⚠️ 拒绝覆盖：缓存 {name} 已有 {len(old)} 行"
                      f"（起点 {old.index.min().date()}），本次仅 {len(data)} 行"
                      f"（起点 {data.index.min().date() if len(data) else 'N/A'}）⇒ 保留原文件")
                return
        except Exception:
            pass  # 读不了就当没有，交给下面的正常写入

    try:
        cache_data = {
            'date': datetime.now(),
            'data': data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")


class USMarketData:
    """美股市场数据获取类（带缓存，AKShare 优先）"""

    def __init__(self):
        pass

    def get_sp500_data(self, period_days=1460, start_date=None):
        """获取标普500指数数据

        Args:
            period_days: 获取天数（默认730天，约2年；仅当 start_date 未指定时用于截断）
            start_date: 起始日期（'YYYY-MM-DD'）；指定后按历史区间取数，支持长历史（修复三.26）
        
        Returns:
            DataFrame: 包含标普500指数数据
        """
        # 尝试从缓存加载
        cached = _load_cache('sp500', period_days=period_days, start_date=start_date)
        if cached is not None:
            print("✅ 从缓存加载标普500数据")
            return cached

        # 策略1: 优先尝试 AKShare
        try:
            from akshare.index import index_global_em
            
            # 使用超时控制包装器（60秒超时，适应 GitHub Actions 网络环境）
            @timeout(60)
            def fetch_data():
                return index_global_em.index_global_hist_em(symbol="标普500")
            
            df = fetch_data()
            
            if not df.empty:
                # 重命名列以保持一致性
                df.rename(columns={'日期': 'Date', '收盘': 'Close'}, inplace=True)
                
                # 转换日期格式并设置为UTC时区
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
                df.set_index('Date', inplace=True)
                
                # 按起始日期截取（若指定），否则保留最近 period_days 天
                if start_date is not None:
                    sd = pd.Timestamp(start_date)
                    if sd.tzinfo is None:
                        sd = sd.tz_localize('UTC')
                    df = df[df.index >= sd]
                elif len(df) > period_days:
                    df = df.tail(period_days)
                
                # 计算收益率
                df['SP500_Return'] = df['Close'].pct_change()
                df['SP500_Return_5d'] = df['Close'].pct_change(5)
                df['SP500_Return_20d'] = df['Close'].pct_change(20)

                print("✅ 使用 AKShare 获取标普500数据成功")
                _save_cache('sp500', df)
                return df
        except TimeoutError as e:
            print(f"⚠️ AKShare 获取标普500数据超时: {e}")
        except Exception as e:
            print(f"⚠️ AKShare 获取标普500数据失败: {e}")

        # 策略2: AKShare 失败，使用 yfinance 作为备选
        try:
            import yfinance as yf

            ticker = yf.Ticker('^GSPC')
            if start_date is not None:
                df = ticker.history(start=pd.Timestamp(start_date).tz_localize(None),
                                    end=datetime.now(), auto_adjust=True, timeout=15)
            else:
                df = ticker.history(period=f'{period_days}d', timeout=15)

            if df.empty:
                print("⚠️ yfinance 也无法获取标普500指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率
            df['SP500_Return'] = df['Close'].pct_change()
            df['SP500_Return_5d'] = df['Close'].pct_change(5)
            df['SP500_Return_20d'] = df['Close'].pct_change(20)

            print("✅ 使用 yfinance 获取标普500数据成功")
            _save_cache('sp500', df)
            return df

        except Exception as e:
            print(f"⚠️ yfinance 获取标普500指数数据也失败: {e}")
            return None

    def get_nasdaq_data(self, period_days=1460, start_date=None):
        """获取纳斯达克指数数据

        Args:
            period_days: 获取天数（默认730天，约2年；仅当 start_date 未指定时用于截断）
            start_date: 起始日期（'YYYY-MM-DD'）；指定后按历史区间取数（修复三.26）
        
        Returns:
            DataFrame: 包含纳斯达克指数数据
        """
        # 尝试从缓存加载
        cached = _load_cache('nasdaq', period_days=period_days, start_date=start_date)
        if cached is not None:
            print("✅ 从缓存加载纳斯达克数据")
            return cached

        # 策略1: 优先尝试 AKShare
        try:
            from akshare.index import index_global_em

            # 使用超时控制包装器
            @timeout(60)  # 60秒超时，适应 GitHub Actions 网络环境
            def fetch_data():
                return index_global_em.index_global_hist_em(symbol="纳斯达克")

            df = fetch_data()

            if not df.empty:
                # 重命名列以保持一致性
                df.rename(columns={'日期': 'Date', '收盘': 'Close'}, inplace=True)

                # 转换日期格式并设置为UTC时区
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
                df.set_index('Date', inplace=True)

                # 按起始日期截取（若指定），否则保留最近 period_days 天
                if start_date is not None:
                    sd = pd.Timestamp(start_date)
                    if sd.tzinfo is None:
                        sd = sd.tz_localize('UTC')
                    df = df[df.index >= sd]
                elif len(df) > period_days:
                    df = df.tail(period_days)
                
                # 计算收益率
                df['NASDAQ_Return'] = df['Close'].pct_change()
                df['NASDAQ_Return_5d'] = df['Close'].pct_change(5)
                df['NASDAQ_Return_20d'] = df['Close'].pct_change(20)

                print("✅ 使用 AKShare 获取纳斯达克数据成功")
                _save_cache('nasdaq', df)
                return df
        except TimeoutError as e:
            print(f"⚠️ AKShare 获取纳斯达克数据超时: {e}")
        except Exception as e:
            print(f"⚠️ AKShare 获取纳斯达克数据失败: {e}")

        # 策略2: AKShare 失败，使用 yfinance 作为备选
        try:
            import yfinance as yf

            ticker = yf.Ticker('^IXIC')
            if start_date is not None:
                df = ticker.history(start=pd.Timestamp(start_date).tz_localize(None),
                                    end=datetime.now(), auto_adjust=True, timeout=15)
            else:
                df = ticker.history(period=f'{period_days}d', timeout=15)

            if df.empty:
                print("⚠️ yfinance 也无法获取纳斯达克指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率
            df['NASDAQ_Return'] = df['Close'].pct_change()
            df['NASDAQ_Return_5d'] = df['Close'].pct_change(5)
            df['NASDAQ_Return_20d'] = df['Close'].pct_change(20)

            print("✅ 使用 yfinance 获取纳斯达克数据成功")
            _save_cache('nasdaq', df)
            return df

        except Exception as e:
            print(f"⚠️ yfinance 获取纳斯达克指数数据也失败: {e}")
            return None

    def get_vix_data(self, period_days=1460, start_date=None):
        """获取VIX恐慌指数数据

        注意：AKShare 暂不支持 VIX 恐慌指数，仅使用 yfinance

        Args:
            period_days: 获取天数（默认730天，约2年）
            start_date: 起始日期（'YYYY-MM-DD'）；指定后按历史区间取数（修复三.26）
        
        Returns:
            DataFrame: 包含VIX恐慌指数数据
        """
        # 尝试从缓存加载
        cached = _load_cache('vix', period_days=period_days, start_date=start_date)
        if cached is not None:
            print("✅ 从缓存加载VIX数据")
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker('^VIX')
            if start_date is not None:
                df = ticker.history(start=pd.Timestamp(start_date).tz_localize(None),
                                    end=datetime.now(), auto_adjust=True, timeout=15)
            else:
                df = ticker.history(period=f'{period_days}d', timeout=15)

            if df.empty:
                print("⚠️ 无法获取VIX恐慌指数数据")
                return None

            # 重置索引，将日期作为列
            df = df.reset_index()
            # 移除原始时区信息，然后设置为UTC时区（与港股数据一致）
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算VIX变化
            df['VIX_Change'] = df['Close'].pct_change()
            df['VIX_MA5'] = df['Close'].rolling(window=5).mean()
            df['VIX_MA20'] = df['Close'].rolling(window=20).mean()

            # VIX相对位置（相对于20日均值）
            df['VIX_Ratio_MA20'] = df['Close'] / df['VIX_MA20']

            print("✅ 使用 yfinance 获取VIX数据成功")
            _save_cache('vix', df)
            return df

        except Exception as e:
            print(f"⚠️ 获取VIX恐慌指数数据失败: {e}")
            return None

    def get_us_treasury_yield(self, period_days=1460, start_date=None):
        """获取美国多期限国债收益率数据（2Y, 10Y, 30Y）和中国国债收益率

        Args:
            period_days: 获取天数（默认1460天，约4年；仅当 start_date 未指定时用于推算起始）
            start_date: 起始日期（'YYYY-MM-DD'）；指定后按历史区间取数（修复三.26）

        Returns:
            DataFrame: 包含美国和中国多期限国债收益率数据，及利差特征
        """
        # 尝试从缓存加载
        cached = _load_cache('treasury_yield', period_days=period_days, start_date=start_date)
        if cached is not None:
            print("✅ 从缓存加载国债收益率数据")
            return cached

        try:
            # 使用超时控制包装器
            @timeout(60)  # 60秒超时，适应 GitHub Actions 网络环境
            def fetch_data():
                if start_date is not None:
                    start_date_str = pd.Timestamp(start_date).strftime('%Y%m%d')
                else:
                    start_date_str = (datetime.now() - timedelta(days=period_days)).strftime('%Y%m%d')
                return ak.bond_zh_us_rate(start_date=start_date_str)

            df = fetch_data()

            if df.empty:
                print("⚠️ 无法获取国债收益率数据")
                return None

            # 重命名列以保持一致性
            df.rename(columns={'日期': 'Date'}, inplace=True)

            # 美国国债收益率（已经是百分比形式，需要除以100转换为小数）
            df['US_2Y_Yield'] = df['美国国债收益率2年'] / 100
            df['US_10Y_Yield'] = df['美国国债收益率10年'] / 100
            df['US_30Y_Yield'] = df['美国国债收益率30年'] / 100

            # 中国国债收益率（已经是百分比形式）
            df['CN_2Y_Yield'] = df['中国国债收益率2年'] / 100
            df['CN_10Y_Yield'] = df['中国国债收益率10年'] / 100
            df['CN_30Y_Yield'] = df['中国国债收益率30年'] / 100

            # 转换日期格式并设置为UTC时区
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率变化
            df['US_2Y_Yield_Change'] = df['US_2Y_Yield'].pct_change()
            df['US_10Y_Yield_Change'] = df['US_10Y_Yield'].pct_change()
            df['US_30Y_Yield_Change'] = df['US_30Y_Yield'].pct_change()
            df['CN_10Y_Yield_Change'] = df['CN_10Y_Yield'].pct_change()

            # 计算美债期限利差（收益率曲线形态）
            df['US_2Y_10Y_Spread'] = df['US_10Y_Yield'] - df['US_2Y_Yield']
            df['US_10Y_30Y_Spread'] = df['US_30Y_Yield'] - df['US_10Y_Yield']

            # 计算中美利差（核心特征）
            df['CN_US_10Y_Spread'] = df['CN_10Y_Yield'] - df['US_10Y_Yield']
            df['CN_US_Spread_Change_5d'] = df['CN_US_10Y_Spread'].diff(5)
            # 利差 Z-Score（60日滚动）
            spread_mean = df['CN_US_10Y_Spread'].rolling(window=60).mean()
            spread_std = df['CN_US_10Y_Spread'].rolling(window=60).std()
            df['CN_US_Spread_Z_Score'] = (df['CN_US_10Y_Spread'] - spread_mean) / spread_std

            # 移动平均
            df['US_10Y_Yield_MA5'] = df['US_10Y_Yield'].rolling(window=5).mean()
            df['US_10Y_Yield_MA20'] = df['US_10Y_Yield'].rolling(window=20).mean()

            print("✅ 使用 AKShare 获取多期限国债收益率数据成功（含中美利差）")
            _save_cache('treasury_yield', df)
            return df

        except TimeoutError as e:
            print(f"⚠️ 获取国债收益率数据超时: {e}")
            return None
        except Exception as e:
            print(f"⚠️ 获取国债收益率数据失败: {e}")
            return None

    def get_all_us_market_data(self, period_days=1460, start_date=None):
        """获取所有美股市场数据

        Args:
            period_days: 获取天数（默认730天，约2年；仅当 start_date 未指定时用于截断）
            start_date: 起始日期（'YYYY-MM-DD'）；指定后按历史区间取数（修复三.26）
        
        Returns:
            DataFrame: 合并后的美股市场数据
        """
        # 获取各项数据（每个都有独立的超时处理和备选方案）
        sp500_df = None
        nasdaq_df = None
        vix_df = None
        treasury_df = None

        try:
            sp500_df = self.get_sp500_data(period_days, start_date)
        except TimeoutError:
            print("⚠️ 获取标普500数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取标普500数据失败: {e}")

        try:
            nasdaq_df = self.get_nasdaq_data(period_days, start_date)
        except TimeoutError:
            print("⚠️ 获取纳斯达克数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取纳斯达克数据失败: {e}")

        try:
            vix_df = self.get_vix_data(period_days, start_date)
        except TimeoutError:
            print("⚠️ 获取VIX数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取VIX数据失败: {e}")

        try:
            treasury_df = self.get_us_treasury_yield(period_days, start_date)
        except TimeoutError:
            print("⚠️ 获取国债收益率数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取国债收益率数据失败: {e}")

        # 合并数据
        if sp500_df is not None:
            merged_df = sp500_df[['SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d']].copy()
        else:
            return None

        if nasdaq_df is not None:
            merged_df = merged_df.merge(
                nasdaq_df[['NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d']],
                left_index=True, right_index=True, how='left'
            )

        if vix_df is not None:
            merged_df = merged_df.merge(
                vix_df[['VIX_Change', 'VIX_Ratio_MA20', 'Close']],
                left_index=True, right_index=True, how='left'
            )
            # 重命名VIX绝对值
            merged_df.rename(columns={'Close': 'VIX_Level'}, inplace=True)

        if treasury_df is not None:
            # 新增多期限利率和利差特征
            treasury_cols = [
                'US_2Y_Yield', 'US_2Y_Yield_Change',
                'US_10Y_Yield', 'US_10Y_Yield_Change',
                'US_30Y_Yield', 'US_30Y_Yield_Change',
                'CN_10Y_Yield', 'CN_10Y_Yield_Change',
                'US_2Y_10Y_Spread', 'US_10Y_30Y_Spread',
                'CN_US_10Y_Spread', 'CN_US_Spread_Change_5d', 'CN_US_Spread_Z_Score',
            ]
            existing_cols = [c for c in treasury_cols if c in treasury_df.columns]
            if existing_cols:
                merged_df = merged_df.merge(
                    treasury_df[existing_cols],
                    left_index=True, right_index=True, how='left'
                )

        # HIBOR 短端（默认并入=D22 例外观察采用；HIBOR_FEATURES=0 一键回退=复现旧基线必设）
        # 数据源 data/cnh_hibor/hibor_hkd.csv（git 入库，本地读不走网络/缓存）；
        # T-1 由下游 us_features shift(shift_val) 统一处理
        from ml_services.run_manifest import resolve_flags
        if resolve_flags()['HIBOR_FEATURES'] == '1':
            try:
                hibor_df = self._load_hibor_features()
                if hibor_df is not None and not hibor_df.empty:
                    if merged_df.index.tz is not None and hibor_df.index.tz is None:
                        hibor_df.index = hibor_df.index.tz_localize('UTC')
                    elif merged_df.index.tz is None and hibor_df.index.tz is not None:
                        hibor_df.index = hibor_df.index.tz_convert(None)
                    merged_df = merged_df.merge(
                        hibor_df, left_index=True, right_index=True, how='left'
                    )
            except Exception as e:
                print(f"⚠️ HIBOR 特征加载失败: {e}")

        return merged_df

    @staticmethod
    def _load_hibor_features():
        """data/cnh_hibor/hibor_hkd.csv → HIBOR_ON_Z / HIBOR_1M_Z（trailing 252d z-score）"""
        import os
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cnh_hibor', 'hibor_hkd.csv'
        )
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
        out = pd.DataFrame(index=df.index)
        for src, col in (('hibor_on', 'HIBOR_ON_Z'), ('hibor_1m', 'HIBOR_1M_Z')):
            if src not in df.columns:
                continue
            s = pd.to_numeric(df[src], errors='coerce')
            mean = s.rolling(252, min_periods=60).mean()
            std = s.rolling(252, min_periods=60).std()
            out[col] = (s - mean) / std.replace(0, pd.NA)
        # 与 merged_df（美股侧 tz-aware 索引）对齐（lessons 三.27）
        if out.index.tz is None:
            out.index = out.index.tz_localize('UTC')
        return out if not out.empty else None

    def calculate_systemic_crash_risk(self, indicators):
        """
        计算系统性崩盘风险评分
        
        Args:
            indicators: 市场指标字典，包含以下键：
                - VIX: VIX恐慌指数
                - HSI_Return_1d: 恒指1日收益率
                - Avg_Vol_Ratio: 平均成交量比率
                - SP500_Return_1d: 标普500 1日收益率
                - Decline_Ratio: 下跌股票占比（可选，默认0.5）
        
        Returns:
            dict: 包含以下键：
                - risk_score: 风险评分（0-100分）
                - risk_level: 风险等级（"低"/"中"/"高"/"极高"）
                - factors: 风险因素列表
                - recommendations: 建议措施列表
        """
        risk_score = 0
        factors = []
        recommendations = []
        
        # 1. VIX 恐慌指数（权重30%）
        vix = indicators.get('VIX', 15)
        if vix > 40:
            risk_score += 30
            factors.append(f"VIX严重恐慌({vix:.1f})")
            recommendations.append("立即清仓，观望为主")
        elif vix > 30:
            risk_score += 20
            factors.append(f"VIX恐慌({vix:.1f})")
            recommendations.append("大幅降低仓位至30%以下")
        elif vix > 20:
            risk_score += 10
            factors.append(f"VIX轻度恐慌({vix:.1f})")
            recommendations.append("谨慎交易，降低仓位至50%以下")
        elif vix < 15:
            risk_score += 5
            factors.append(f"VIX过度乐观({vix:.1f})")
            recommendations.append("警惕回调风险，适度降低仓位")
        
        # 2. 恒指跌幅（权重25%）
        hsi_change = indicators.get('HSI_Return_1d', 0)
        if hsi_change < -5:
            risk_score += 25
            factors.append(f"恒指暴跌({hsi_change:.2f}%)")
            recommendations.append("恒指暴跌，暂停所有买入操作")
        elif hsi_change < -3:
            risk_score += 15
            factors.append(f"恒指大跌({hsi_change:.2f}%)")
            recommendations.append("恒指大跌，谨慎建仓")
        elif hsi_change < -1:
            risk_score += 5
            factors.append(f"恒指下跌({hsi_change:.2f}%)")
            recommendations.append("恒指下跌，降低仓位")
        elif hsi_change > 3:
            risk_score += 3
            factors.append(f"恒指大涨({hsi_change:.2f}%)")
            recommendations.append("恒指大涨，注意回调风险")
        
        # 3. 成交额萎缩（权重20%）
        vol_ratio = indicators.get('Avg_Vol_Ratio', 1.0)
        if vol_ratio < 0.5:
            risk_score += 20
            factors.append(f"成交额严重萎缩({vol_ratio:.2f})")
            recommendations.append("成交额严重萎缩，市场流动性枯竭，观望为主")
        elif vol_ratio < 0.8:
            risk_score += 10
            factors.append(f"成交额萎缩({vol_ratio:.2f})")
            recommendations.append("成交额萎缩，减少交易频率")
        elif vol_ratio > 2.0:
            risk_score += 8
            factors.append(f"成交额异常放大({vol_ratio:.2f})")
            recommendations.append("成交额异常放大，可能存在恐慌性抛售")
        
        # 4. 美股联动（权重15%）
        sp500_change = indicators.get('SP500_Return_1d', 0)
        if sp500_change < -3:
            risk_score += 15
            factors.append(f"美股暴跌({sp500_change:.2f}%)")
            recommendations.append("美股暴跌，港股跟随下跌概率高")
        elif sp500_change < -1:
            risk_score += 5
            factors.append(f"美股下跌({sp500_change:.2f}%)")
            recommendations.append("美股下跌，警惕港股跟随")
        elif sp500_change > 3:
            risk_score += 3
            factors.append(f"美股大涨({sp500_change:.2f}%)")
            recommendations.append("美股大涨，关注港股跟涨")
        
        # 5. 广度指标（权重10%，可选）
        decline_ratio = indicators.get('Decline_Ratio', 0.5)  # 下跌股票占比
        if decline_ratio > 0.8:
            risk_score += 10
            factors.append(f"普跌({decline_ratio:.1%})")
            recommendations.append("市场普跌，系统性风险高")
        elif decline_ratio > 0.6:
            risk_score += 5
            factors.append(f"多数下跌({decline_ratio:.1%})")
            recommendations.append("多数股票下跌，谨慎选股")
        elif decline_ratio < 0.2:
            risk_score += 3
            factors.append(f"普涨({decline_ratio:.1%})")
            recommendations.append("市场普涨，注意过热风险")
        
        # 限制最大评分为100
        risk_score = min(risk_score, 100)
        
        # 风险等级
        if risk_score >= 80:
            risk_level = "极高"
            if "立即清仓，观望为主" not in recommendations:
                recommendations.insert(0, "立即清仓，观望为主")
        elif risk_score >= 60:
            risk_level = "高"
            if "大幅降低仓位至30%以下" not in recommendations:
                recommendations.insert(0, "大幅降低仓位至30%以下")
        elif risk_score >= 40:
            risk_level = "中"
            if "降低仓位至50%以下" not in recommendations:
                recommendations.insert(0, "降低仓位至50%以下")
        else:
            risk_level = "低"
            if "正常交易" not in recommendations:
                recommendations.insert(0, "正常交易，可适当建仓")
        
        # 去重建议
        recommendations = list(dict.fromkeys(recommendations))
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'factors': factors,
            'recommendations': recommendations
        }

    # 移除 clear_cache 方法（不再使用缓存）


# 全局实例
us_market_data = USMarketData()


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("测试美股市场数据获取")
    print("=" * 70)

    # 获取所有美股市场数据
    print("\n📊 获取美股市场数据...")
    us_df = us_market_data.get_all_us_market_data(period_days=30)

    if us_df is not None:
        print(f"\n✅ 成功获取 {len(us_df)} 天的美股市场数据")
        print("\n📊 数据预览:")
        print(us_df.tail(10))

        print("\n📊 数据统计:")
        print(us_df.describe())

        print("\n📊 缺失值统计:")
        print(us_df.isnull().sum())
    else:
        print("\n❌ 获取美股市场数据失败")
