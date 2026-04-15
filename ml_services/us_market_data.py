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

注：不使用缓存机制，每次都实时获取最新数据
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import warnings
import signal
from functools import wraps

warnings.filterwarnings('ignore')


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


class USMarketData:
    """美股市场数据获取类（无缓存，AKShare 优先）"""

    def __init__(self):
        # 不使用缓存
        pass

    def get_sp500_data(self, period_days=730):
        """获取标普500指数数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含标普500指数数据
        """
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
                
                # 只保留最近N天的数据
                if len(df) > period_days:
                    df = df.tail(period_days)
                
                # 计算收益率
                df['SP500_Return'] = df['Close'].pct_change()
                df['SP500_Return_5d'] = df['Close'].pct_change(5)
                df['SP500_Return_20d'] = df['Close'].pct_change(20)
                
                print("✅ 使用 AKShare 获取标普500数据成功")
                return df
        except TimeoutError as e:
            print(f"⚠️ AKShare 获取标普500数据超时: {e}")
        except Exception as e:
            print(f"⚠️ AKShare 获取标普500数据失败: {e}")
        
        # 策略2: AKShare 失败，使用 yfinance 作为备选
        try:
            import yfinance as yf
            
            ticker = yf.Ticker('^GSPC')
            df = ticker.history(period=f'{period_days}d')

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
            return df

        except Exception as e:
            print(f"⚠️ yfinance 获取标普500指数数据也失败: {e}")
            return None

    def get_nasdaq_data(self, period_days=730):
        """获取纳斯达克指数数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含纳斯达克指数数据
        """
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
                
                # 只保留最近N天的数据
                if len(df) > period_days:
                    df = df.tail(period_days)
                
                # 计算收益率
                df['NASDAQ_Return'] = df['Close'].pct_change()
                df['NASDAQ_Return_5d'] = df['Close'].pct_change(5)
                df['NASDAQ_Return_20d'] = df['Close'].pct_change(20)
                
                print("✅ 使用 AKShare 获取纳斯达克数据成功")
                return df
        except TimeoutError as e:
            print(f"⚠️ AKShare 获取纳斯达克数据超时: {e}")
        except Exception as e:
            print(f"⚠️ AKShare 获取纳斯达克数据失败: {e}")
        
        # 策略2: AKShare 失败，使用 yfinance 作为备选
        try:
            import yfinance as yf
            
            ticker = yf.Ticker('^IXIC')
            df = ticker.history(period=f'{period_days}d')

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
            return df

        except Exception as e:
            print(f"⚠️ yfinance 获取纳斯达克指数数据也失败: {e}")
            return None

    def get_vix_data(self, period_days=730):
        """获取VIX恐慌指数数据
        
        注意：AKShare 暂不支持 VIX 恐慌指数，仅使用 yfinance
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含VIX恐慌指数数据
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker('^VIX')
            df = ticker.history(period=f'{period_days}d')

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
            return df

        except Exception as e:
            print(f"⚠️ 获取VIX恐慌指数数据失败: {e}")
            return None

    def get_us_treasury_yield(self, period_days=730):
        """获取美国10年期国债收益率数据
        
        Args:
            period_days: 获取天数（默认730天，约2年）
        
        Returns:
            DataFrame: 包含美国10年期国债收益率数据
        """
        try:
            # 使用超时控制包装器
            @timeout(60)  # 60秒超时，适应 GitHub Actions 网络环境
            def fetch_data():
                start_date_str = (datetime.now() - timedelta(days=period_days)).strftime('%Y%m%d')
                return ak.bond_zh_us_rate(start_date=start_date_str)

            df = fetch_data()

            if df.empty:
                print("⚠️ 无法获取美国10年期国债收益率数据")
                return None

            # 重命名列以保持一致性
            df.rename(columns={'日期': 'Date'}, inplace=True)
            
            # 提取美国10年期国债收益率（已经是百分比形式，需要除以100转换为小数）
            df['US_10Y_Yield'] = df['美国国债收益率10年'] / 100
            
            # 转换日期格式并设置为UTC时区
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
            df.set_index('Date', inplace=True)

            # 计算收益率变化
            df['US_10Y_Yield_Change'] = df['US_10Y_Yield'].pct_change()
            df['US_10Y_Yield_MA5'] = df['US_10Y_Yield'].rolling(window=5).mean()
            df['US_10Y_Yield_MA20'] = df['US_10Y_Yield'].rolling(window=20).mean()

            print("✅ 使用 AKShare 获取美国国债收益率数据成功")
            return df

        except TimeoutError as e:
            print(f"⚠️ 获取美国10年期国债收益率数据超时: {e}")
            return None
        except Exception as e:
            print(f"⚠️ 获取美国10年期国债收益率数据失败: {e}")
            return None

    def get_all_us_market_data(self, period_days=730):
        """获取所有美股市场数据

        Args:
            period_days: 获取天数（默认730天，约2年）

        Returns:
            DataFrame: 合并后的美股市场数据
        """
        # 获取各项数据（每个都有独立的超时处理和备选方案）
        sp500_df = None
        nasdaq_df = None
        vix_df = None
        treasury_df = None

        try:
            sp500_df = self.get_sp500_data(period_days)
        except TimeoutError:
            print("⚠️ 获取标普500数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取标普500数据失败: {e}")

        try:
            nasdaq_df = self.get_nasdaq_data(period_days)
        except TimeoutError:
            print("⚠️ 获取纳斯达克数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取纳斯达克数据失败: {e}")

        try:
            vix_df = self.get_vix_data(period_days)
        except TimeoutError:
            print("⚠️ 获取VIX数据超时，跳过")
        except Exception as e:
            print(f"⚠️ 获取VIX数据失败: {e}")

        try:
            treasury_df = self.get_us_treasury_yield(period_days)
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
            merged_df = merged_df.merge(
                treasury_df[['US_10Y_Yield', 'US_10Y_Yield_Change']],
                left_index=True, right_index=True, how='left'
            )

        return merged_df

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
