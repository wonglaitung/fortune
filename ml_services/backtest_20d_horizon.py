#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20天持有期回测脚本 - 正确评估CatBoost 20天模型的实际性能
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import CatBoostModel
from ml_services.logger_config import get_logger
from ml_services.dynamic_risk_control import DynamicRiskControl, calculate_market_beta
from config import WATCHLIST as STOCK_LIST

logger = get_logger('backtest_20d_horizon')

# 股票名称映射
STOCK_NAMES = STOCK_LIST


class Backtest20DHoldPeriod:
    """20天持有期回测器 - 符合CatBoost 20天模型的实际预测逻辑
    
    新增功能（业界标准）：
    - 动态仓位管理
    - 极端市场环境识别
    - 多层级风险控制
    - 真实市场数据获取（恒生指数、VIX）
    """

    def __init__(self, model, confidence_threshold=0.55, commission=0.001, slippage=0.001, 
                 enable_dynamic_risk_control=True):
        """
        初始化回测器

        参数:
        - model: 训练好的模型
        - confidence_threshold: 置信度阈值（基础阈值，实际会根据市场环境动态调整）
        - commission: 交易佣金
        - slippage: 滑点
        - enable_dynamic_risk_control: 是否启用动态风险控制（业界标准）
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.commission = commission
        self.slippage = slippage
        self.enable_dynamic_risk_control = enable_dynamic_risk_control
        
        # 初始化动态风险控制系统（业界标准）
        if self.enable_dynamic_risk_control:
            self.risk_control = DynamicRiskControl()
            logger.info("动态风险控制系统已启用（符合业界标准）")
        else:
            self.risk_control = None
            logger.info("动态风险控制系统未启用，使用固定置信度阈值")
        
        # 市场数据缓存
        self.hsi_cache = None
        self.vix_cache = None
        self.cache_start_date = None
        self.cache_end_date = None

    def backtest_single_stock(self, stock_code, test_df, feature_columns, start_date, end_date):
        """
        对单只股票进行20天持有期回测

        关键逻辑：
        - 第i天的预测是"第i+20天是否会上涨"
        - 如果预测上涨且置信度>阈值，则在第i天买入，第i+20天卖出
        - 持有20天，不考虑中间的信号变化

        参数:
        - stock_code: 股票代码
        - test_df: 测试数据（包含多只股票）
        - feature_columns: 特征列名列表
        - start_date: 开始日期
        - end_date: 结束日期

        返回:
        dict: 回测结果
        """
        # 获取单只股票的数据
        single_stock_df = test_df[test_df['Code'] == stock_code].sort_index()
        prices = single_stock_df['Close']

        # 确保索引是 tz-naive（移除时区）
        if hasattr(single_stock_df.index, 'tz') and single_stock_df.index.tz is not None:
            single_stock_df.index = single_stock_df.index.tz_localize(None)
            prices.index = prices.index.tz_localize(None)

        # 转换日期为 tz-naive Timestamp
        start_ts = pd.Timestamp(start_date).tz_localize(None)
        end_ts = pd.Timestamp(end_date).tz_localize(None)

        # 过滤日期范围
        mask = (single_stock_df.index >= start_ts) & (single_stock_df.index <= end_ts)
        single_stock_df_filtered = single_stock_df[mask]
        prices_filtered = prices[mask]

        if len(prices_filtered) < 22:  # 至少需要22天数据（1天买入 + 20天持有 + 1天卖出）
            logger.warning(f"{stock_code}: 日期范围内数据不足（{len(prices_filtered)} 天）")
            return None

        # 准备测试数据
        X_test = single_stock_df_filtered[feature_columns].copy()
        y_test = single_stock_df_filtered['Label'].values

        # 生成预测
        if hasattr(self.model, 'catboost_model'):
            from catboost import Pool
            categorical_encoders = getattr(self.model, 'categorical_encoders', {})
            model_features = getattr(self.model, 'feature_columns', [])
            catboost_model = self.model.catboost_model

            # 确保特征列正确
            available_features = [col for col in model_features if col in X_test.columns]
            if len(available_features) < len(model_features):
                missing_cols = [col for col in model_features if col not in X_test.columns]
                X_test = X_test.copy()
                for col in missing_cols:
                    X_test[col] = 0.0

            X_test = X_test[model_features]

            # 处理分类特征
            categorical_features = [model_features.index(col) for col in categorical_encoders.keys() if col in model_features]
            for cat_idx in categorical_features:
                col_name = model_features[cat_idx]
                if col_name in X_test.columns:
                    X_test[col_name] = X_test[col_name].astype(np.int32)

            # 使用 Pool 对象进行预测
            test_pool = Pool(data=X_test)
            predictions = catboost_model.predict_proba(test_pool)[:, 1]
        else:
            predictions = self.model.predict_proba(X_test)[:, 1]

        horizon = 20
        capital = 100000
        trades = []

        # 逐个交易机会进行回测
        for i in range(len(prices_filtered) - horizon):
            buy_date = prices_filtered.index[i]
            sell_date = prices_filtered.index[i + horizon]
            buy_price = prices_filtered.iloc[i]
            sell_price = prices_filtered.iloc[i + horizon]

            # 检查日期是否在范围内
            if buy_date > end_ts or sell_date > end_ts:
                continue

            # 计算实际涨跌（不考虑交易成本，用于预测准确性评估）
            actual_change = (sell_price - buy_price) / buy_price
            actual_direction = 1 if actual_change > 0 else 0

            # 计算实际交易价格（考虑滑点和佣金）
            # 买入：价格通常比收盘价高（滑点），还需要支付佣金
            actual_buy_price = buy_price * (1 + self.slippage) * (1 + self.commission / 2)
            
            # 卖出：价格通常比收盘价低（滑点），还需要支付佣金
            actual_sell_price = sell_price * (1 - self.slippage) * (1 - self.commission / 2)
            
            # 计算实际收益率（考虑交易成本）
            actual_change_with_cost = (actual_sell_price - actual_buy_price) / actual_buy_price
            actual_direction_with_cost = 1 if actual_change_with_cost > 0 else 0

            # 模型预测
            prob = predictions[i]
            
            # 动态风险控制（业界标准）
            if self.enable_dynamic_risk_control and self.risk_control is not None:
                # 1. 获取市场环境数据
                market_data = self.get_market_data_at_date(buy_date, single_stock_df, i)
                
                # 2. 检测极端市场环境
                is_extreme, extreme_conditions, extreme_count = self.risk_control.detect_extreme_market_conditions(
                    market_data['hsi_data'],
                    market_data['vix_level'],
                    market_data['stock_data']
                )
                
                # 3. 极端市场环境：停止交易
                if is_extreme:
                    continue  # 跳过该交易
                
                # 4. 计算市场环境评分
                market_env_score = self.risk_control.assess_market_environment(
                    market_data['hsi_data'],
                    market_data['vix_level']
                )
                
                # 5. 动态仓位管理（业界标准）
                adjusted_prob, position_size, risk_level = self.risk_control.get_dynamic_position_size(
                    prob,
                    market_data['market_regime'],
                    market_data['vix_level'],
                    market_env_score
                )
                
                # 仓位为0：停止交易
                if position_size <= 0:
                    continue
                
                # 使用调整后的概率作为决策依据
                final_threshold = 0.5  # 动态风险管理下，使用0.5作为基础阈值
                signal = 1 if adjusted_prob > final_threshold else 0
                
                # 添加风险控制信息到交易记录
                trades.append({
                    'stock_code': stock_code,
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'sell_date': sell_date.strftime('%Y-%m-%d'),
                    'buy_price': buy_price,  # 原始买入价格
                    'sell_price': sell_price,  # 原始卖出价格
                    'actual_buy_price': actual_buy_price,  # 实际买入价格（考虑滑点和佣金）
                    'actual_sell_price': actual_sell_price,  # 实际卖出价格（考虑滑点和佣金）
                    'prediction': signal,
                    'probability': prob,
                    'adjusted_probability': adjusted_prob,
                    'actual_change': actual_change,  # 原始收益率（不考虑成本）
                    'actual_change_with_cost': actual_change_with_cost,  # 实际收益率（考虑成本）
                    'actual_direction': actual_direction,  # 原始方向
                    'actual_direction_with_cost': actual_direction_with_cost,  # 实际方向（考虑成本）
                    'prediction_correct': signal == actual_direction,  # 基于原始方向判断
                    'prediction_correct_with_cost': signal == actual_direction_with_cost,  # 基于实际方向判断
                    'position_size': position_size,
                    'risk_level': risk_level,
                    'market_env_score': market_env_score,
                    'market_regime': market_data['market_regime'],
                    'vix_level': market_data['vix_level']
                })
            else:
                # 传统模式：使用固定置信度阈值
                signal = 1 if prob > self.confidence_threshold else 0
                trades.append({
                    'stock_code': stock_code,
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'sell_date': sell_date.strftime('%Y-%m-%d'),
                    'buy_price': buy_price,  # 原始买入价格
                    'sell_price': sell_price,  # 原始卖出价格
                    'actual_buy_price': actual_buy_price,  # 实际买入价格（考虑滑点和佣金）
                    'actual_sell_price': actual_sell_price,  # 实际卖出价格（考虑滑点和佣金）
                    'prediction': signal,
                    'probability': prob,
                    'actual_change': actual_change,  # 原始收益率（不考虑成本）
                    'actual_change_with_cost': actual_change_with_cost,  # 实际收益率（考虑成本）
                    'actual_direction': actual_direction,  # 原始方向
                    'actual_direction_with_cost': actual_direction_with_cost,  # 实际方向（考虑成本）
                    'prediction_correct': signal == actual_direction,  # 基于原始方向判断
                    'prediction_correct_with_cost': signal == actual_direction_with_cost  # 基于实际方向判断
                })

        return trades

    def get_market_data_at_date(self, buy_date, single_stock_df, current_index):
        """
        获取指定日期的市场环境数据（使用真实市场数据）

        参数:
        - buy_date: 买入日期
        - single_stock_df: 单只股票数据
        - current_index: 当前索引位置

        返回:
        dict: 市场环境数据，包括恒生指数、VIX、市场状态等
        """
        try:
            # 1. 检查是否需要更新市场数据缓存
            end_date = buy_date
            start_date = end_date - timedelta(days=90)  # 缓存90天数据
            
            # 初始化缓存或需要更新缓存
            if self.hsi_cache is None or self.vix_cache is None or \
               end_date > self.cache_end_date or start_date < self.cache_start_date:
                logger.info(f"更新市场数据缓存: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
                self._update_market_data_cache(start_date, end_date)
            
            # 2. 从缓存中获取恒生指数数据
            if self.hsi_cache is not None and len(self.hsi_cache) > 0:
                # 确保日期格式匹配
                buy_date_normalized = pd.Timestamp(buy_date).normalize()
                
                # 找到最近的交易日
                hsi_data = self.hsi_cache[self.hsi_cache['Date'] <= buy_date_normalized].tail(30)
                
                if len(hsi_data) == 0:
                    # 缓存中没有足够的历史数据，使用默认值
                    raise Exception("恒生指数缓存数据不足")
            else:
                raise Exception("恒生指数缓存为空")

            # 3. 从缓存中获取VIX数据
            if self.vix_cache is not None and len(self.vix_cache) > 0:
                vix_row = self.vix_cache[self.vix_cache['Date'] <= buy_date_normalized].tail(1)
                if len(vix_row) > 0:
                    vix_level = float(vix_row['Close'].iloc[-1])
                else:
                    vix_level = 50
            else:
                vix_level = 50

            # 4. 计算市场状态
            lookback = min(20, len(hsi_data))
            if lookback >= 5:
                hsi_return_5d = hsi_data['Close'].pct_change(5).iloc[-1] if len(hsi_data) >= 6 else 0
                hsi_return_20d = hsi_data['Close'].pct_change(20).iloc[-1] if len(hsi_data) >= 21 else 0
                
                # 计算市场状态
                if hsi_return_20d > 0.05:
                    market_regime = 'bull'
                elif hsi_return_20d < -0.05:
                    market_regime = 'bear'
                else:
                    market_regime = 'neutral'
            else:
                hsi_return_5d = 0
                hsi_return_20d = 0
                market_regime = 'neutral'

            # 5. 获取股票数据
            stock_data = pd.DataFrame({
                'Return': np.random.normal(0, 0.02, 5)  # 模拟5只股票的收益率
            })

            return {
                'hsi_data': hsi_data,
                'vix_level': vix_level,
                'market_regime': market_regime,
                'stock_data': stock_data
            }

        except Exception as e:
            logger.warning(f"获取市场数据失败: {e}")
            # 返回默认值
            hsi_data = pd.DataFrame({
                'Close': [10000] * 30,
                'Volume': [1000000] * 30
            })
            stock_data = pd.DataFrame({
                'Return': [0] * 5
            })
            return {
                'hsi_data': hsi_data,
                'vix_level': 50,
                'market_regime': 'neutral',
                'stock_data': stock_data
            }

    def _update_market_data_cache(self, start_date, end_date):
        """更新市场数据缓存"""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # 获取恒生指数数据
            hsi_ticker = yf.Ticker("^HSI")
            hsi_df = hsi_ticker.history(start=start_str, end=end_str)
            
            if len(hsi_df) > 0:
                hsi_df = hsi_df.reset_index()
                hsi_df['Date'] = pd.to_datetime(hsi_df['Date']).dt.normalize()
                self.hsi_cache = hsi_df
                self.cache_start_date = hsi_df['Date'].min()
                self.cache_end_date = hsi_df['Date'].max()
                logger.info(f"恒生指数缓存已更新: {len(hsi_df)} 行数据")
            else:
                logger.warning("恒生指数数据获取失败，使用默认值")
                self.hsi_cache = None

            # 获取VIX数据
            vix_ticker = yf.Ticker("^VIX")
            vix_df = vix_ticker.history(start=start_str, end=end_str)
            
            if len(vix_df) > 0:
                vix_df = vix_df.reset_index()
                vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.normalize()
                self.vix_cache = vix_df
                logger.info(f"VIX缓存已更新: {len(vix_df)} 行数据")
            else:
                logger.warning("VIX数据获取失败，使用默认值")
                self.vix_cache = None

        except Exception as e:
            logger.error(f"更新市场数据缓存失败: {e}")
            self.hsi_cache = None
            self.vix_cache = None

    def backtest_all_stocks(self, test_df, feature_columns, start_date, end_date):
        """
        对所有股票进行回测

        参数:
        - test_df: 测试数据
        - feature_columns: 特征列名列表
        - start_date: 开始日期
        - end_date: 结束日期

        返回:
        dict: 所有股票的回测结果
        """
        unique_stocks = test_df['Code'].unique()
        logger.info(f"开始20天持有期回测，共 {len(unique_stocks)} 只股票")

        all_trades = []

        for i, stock_code in enumerate(unique_stocks, 1):
            print(f"[{i}/{len(unique_stocks)}] 回测股票: {stock_code}")

            trades = self.backtest_single_stock(
                stock_code, test_df, feature_columns, start_date, end_date
            )

            if trades:
                all_trades.extend(trades)

        return all_trades


def calculate_performance_metrics(all_trades):
    """
    计算性能指标

    参数:
    - all_trades: 所有交易记录

    返回:
    dict: 性能指标
    """
    if not all_trades:
        return {}

    df = pd.DataFrame(all_trades)
    df['buy_date'] = pd.to_datetime(df['buy_date'])

    # 基本统计
    total_trades = len(df)
    correct_predictions = df['prediction_correct'].sum()
    accuracy = correct_predictions / total_trades if total_trades > 0 else 0

    # 收益统计（不考虑交易成本 - 用于预测准确性评估）
    buy_signals = df[df['prediction'] == 1]

    if len(buy_signals) > 0:
        # 不考虑交易成本的指标
        avg_return = buy_signals['actual_change'].mean()
        median_return = buy_signals['actual_change'].median()
        std_return = buy_signals['actual_change'].std()
        positive_trades = (buy_signals['actual_change'] > 0).sum()
        negative_trades = (buy_signals['actual_change'] <= 0).sum()

        # 考虑交易成本的指标
        avg_return_with_cost = buy_signals['actual_change_with_cost'].mean()
        median_return_with_cost = buy_signals['actual_change_with_cost'].median()
        std_return_with_cost = buy_signals['actual_change_with_cost'].std()
        positive_trades_with_cost = (buy_signals['actual_change_with_cost'] > 0).sum()
        negative_trades_with_cost = (buy_signals['actual_change_with_cost'] <= 0).sum()

        # 夏普比率（年化）- 不考虑成本
        returns = buy_signals['actual_change'].values
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252/20) if returns.std() > 0 else 0

        # 夏普比率（年化）- 考虑成本
        returns_with_cost = buy_signals['actual_change_with_cost'].values
        sharpe_ratio_with_cost = returns_with_cost.mean() / returns_with_cost.std() * np.sqrt(252/20) if returns_with_cost.std() > 0 else 0

        # 胜率（基于买入信号）
        win_rate = positive_trades / len(buy_signals) if len(buy_signals) > 0 else 0
        win_rate_with_cost = positive_trades_with_cost / len(buy_signals) if len(buy_signals) > 0 else 0

        # F1分数（不考虑成本）
        precision = correct_predictions / len(buy_signals) if len(buy_signals) > 0 else 0
        recall = correct_predictions / (df['actual_direction'] == 1).sum() if (df['actual_direction'] == 1).sum() > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # F1分数（考虑成本）
        correct_predictions_with_cost = df['prediction_correct_with_cost'].sum()
        precision_with_cost = correct_predictions_with_cost / len(buy_signals) if len(buy_signals) > 0 else 0
        recall_with_cost = correct_predictions_with_cost / (df['actual_direction_with_cost'] == 1).sum() if (df['actual_direction_with_cost'] == 1).sum() > 0 else 0
        f1_score_with_cost = 2 * precision_with_cost * recall_with_cost / (precision_with_cost + recall_with_cost) if (precision_with_cost + recall_with_cost) > 0 else 0

        # 最大回撤（不考虑成本）
        cumulative_returns = (1 + buy_signals.sort_values('buy_date')['actual_change']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 最大回撤（考虑成本）
        cumulative_returns_with_cost = (1 + buy_signals.sort_values('buy_date')['actual_change_with_cost']).cumprod()
        running_max_with_cost = cumulative_returns_with_cost.expanding().max()
        drawdown_with_cost = (cumulative_returns_with_cost - running_max_with_cost) / running_max_with_cost
        max_drawdown_with_cost = drawdown_with_cost.min()

        # 每日性能统计
        daily_metrics = []
        for buy_date in sorted(buy_signals['buy_date'].unique()):
            daily_df = buy_signals[buy_signals['buy_date'] == buy_date]
            daily_total = len(daily_df)
            daily_correct = daily_df['prediction_correct'].sum()
            daily_accuracy = daily_correct / daily_total if daily_total > 0 else 0
            daily_avg_return = daily_df['actual_change'].mean()
            daily_median_return = daily_df['actual_change'].median()
            daily_std_return = daily_df['actual_change'].std()
            daily_positive = (daily_df['actual_change'] > 0).sum()
            daily_win_rate = daily_positive / daily_total if daily_total > 0 else 0

            daily_metrics.append({
                'buy_date': buy_date.strftime('%Y-%m-%d'),
                'total_trades': daily_total,
                'correct_predictions': int(daily_correct),
                'accuracy': float(daily_accuracy),
                'avg_return': float(daily_avg_return) if not np.isnan(daily_avg_return) else 0.0,
                'median_return': float(daily_median_return) if not np.isnan(daily_median_return) else 0.0,
                'std_return': float(daily_std_return) if not np.isnan(daily_std_return) else 0.0,
                'positive_trades': int(daily_positive),
                'win_rate': float(daily_win_rate)
            })

        return {
            'total_trades': total_trades,
            'buy_signals': len(buy_signals),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_return': avg_return,
            'median_return': median_return,
            'std_return': std_return,
            'positive_trades': positive_trades,
            'negative_trades': negative_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            # 考虑交易成本的指标
            'correct_predictions_with_cost': correct_predictions_with_cost,
            'accuracy_with_cost': correct_predictions_with_cost / total_trades if total_trades > 0 else 0,
            'precision_with_cost': precision_with_cost,
            'recall_with_cost': recall_with_cost,
            'f1_score_with_cost': f1_score_with_cost,
            'avg_return_with_cost': avg_return_with_cost,
            'median_return_with_cost': median_return_with_cost,
            'std_return_with_cost': std_return_with_cost,
            'positive_trades_with_cost': positive_trades_with_cost,
            'negative_trades_with_cost': negative_trades_with_cost,
            'win_rate_with_cost': win_rate_with_cost,
            'sharpe_ratio_with_cost': sharpe_ratio_with_cost,
            'max_drawdown_with_cost': max_drawdown_with_cost,
            'daily_metrics': daily_metrics
        }
    else:
        return {
            'total_trades': total_trades,
            'buy_signals': 0,
            'accuracy': accuracy,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'win_rate': 0,
            'daily_metrics': []
        }


def save_results(all_trades, metrics, output_dir='output'):
    """
    保存回测结果

    参数:
    - all_trades: 所有交易记录
    - metrics: 性能指标
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细交易记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = os.path.join(output_dir, f"backtest_20d_trades_{timestamp}.csv")

    df = pd.DataFrame(all_trades)
    df.to_csv(trades_file, index=False, encoding='utf-8')
    print(f"\n✅ 交易记录已保存到: {trades_file}")

    # 保存性能指标
    metrics_file = os.path.join(output_dir, f"backtest_20d_metrics_{timestamp}.json")

    # 转换为可序列化的格式
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ 性能指标已保存到: {metrics_file}")

    # 生成股票级别汇总CSV
    stock_summary_file = os.path.join(output_dir, f"backtest_20d_stock_summary_{timestamp}.csv")
    
    stock_summary = []
    for stock_code in df['stock_code'].unique():
        stock_df = df[df['stock_code'] == stock_code]
        
        total_trades = len(stock_df)
        buy_signals = stock_df[stock_df['prediction'] == 1]
        
        if len(buy_signals) > 0:
            # 不考虑交易成本
            avg_return = buy_signals['actual_change'].mean()
            win_rate = (buy_signals['actual_change'] > 0).mean()
            
            # 考虑交易成本
            avg_return_with_cost = buy_signals['actual_change_with_cost'].mean()
            win_rate_with_cost = (buy_signals['actual_change_with_cost'] > 0).mean()
        else:
            avg_return = 0
            win_rate = 0
            avg_return_with_cost = 0
            win_rate_with_cost = 0
        
        accuracy = stock_df['prediction_correct'].mean()
        
        stock_summary.append({
            '股票代码': stock_code,
            '股票名称': STOCK_NAMES.get(stock_code, stock_code),
            '交易次数': total_trades,
            '平均收益率(无成本)': avg_return,
            '平均收益率(含成本)': avg_return_with_cost,
            '胜率(无成本)': win_rate,
            '胜率(含成本)': win_rate_with_cost,
            '准确率': accuracy
        })
    
    stock_summary_df = pd.DataFrame(stock_summary)
    stock_summary_df = stock_summary_df.sort_values('平均收益率(无成本)', ascending=False)
    stock_summary_df.to_csv(stock_summary_file, index=False, encoding='utf-8')
    print(f"✅ 股票汇总已保存到: {stock_summary_file}")

    # 生成文本报告
    report_file = os.path.join(output_dir, f"backtest_20d_report_{timestamp}.txt")

    report = generate_text_report(metrics, df)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 文本报告已保存到: {report_file}")

    return trades_file, metrics_file, report_file, stock_summary_file


def generate_text_report(metrics, df):
    """
    生成文本报告

    参数:
    - metrics: 性能指标
    - df: 交易数据

    返回:
    str: 文本报告
    """
    report = f"""
{'='*80}
20天持有期回测报告 - CatBoost 20天模型
{'='*80}

【回测参数】
  模型类型: CatBoost 20天
  持有期: 20个交易日
  评估方式: 买入后持有20天（符合模型预测周期）
  交易成本: 滑点 0.1% + 佣金 0.1% (双边 0.4%)

【性能指标】
  总交易机会: {metrics.get('total_trades', 0)}
  买入信号数: {metrics.get('buy_signals', 0)}

【预测准确率】（不考虑交易成本）
  准确率: {metrics.get('accuracy', 0):.2%} ({metrics.get('correct_predictions', 0)}/{metrics.get('total_trades', 0)})
  精确率: {metrics.get('precision', 0):.2%}
  召回率: {metrics.get('recall', 0):.2%}
  F1分数: {metrics.get('f1_score', 0):.4f}

【收益统计】（仅买入信号）
  不考虑交易成本:
    平均收益率: {metrics.get('avg_return', 0):.2%}
    收益率中位数: {metrics.get('median_return', 0):.2%}
    收益率标准差: {metrics.get('std_return', 0):.2%}
    上涨交易: {metrics.get('positive_trades', 0)} 笔
    下跌交易: {metrics.get('negative_trades', 0)} 笔
    胜率: {metrics.get('win_rate', 0):.2%}

  考虑交易成本:
    平均收益率: {metrics.get('avg_return_with_cost', 0):.2%} ⚠️
    收益率中位数: {metrics.get('median_return_with_cost', 0):.2%}
    收益率标准差: {metrics.get('std_return_with_cost', 0):.2%}
    上涨交易: {metrics.get('positive_trades_with_cost', 0)} 笔
    下跌交易: {metrics.get('negative_trades_with_cost', 0)} 笔
    胜率: {metrics.get('win_rate_with_cost', 0):.2%}

【风险指标】
  不考虑交易成本:
    夏普比率（年化）: {metrics.get('sharpe_ratio', 0):.2f}
    最大回撤: {metrics.get('max_drawdown', 0):.2%}

  考虑交易成本:
    夏普比率（年化）: {metrics.get('sharpe_ratio_with_cost', 0):.2f} ⚠️
    最大回撤: {metrics.get('max_drawdown_with_cost', 0):.2%} ⚠️

【交易成本影响分析】
  成本对收益率的影响: {metrics.get('avg_return', 0):.2%} → {metrics.get('avg_return_with_cost', 0):.2%}
  成本对胜率的影响: {metrics.get('win_rate', 0):.2%} → {metrics.get('win_rate_with_cost', 0):.2%}
  成本对夏普比率的影响: {metrics.get('sharpe_ratio', 0):.2f} → {metrics.get('sharpe_ratio_with_cost', 0):.2f}

{'='*80}
"""

    if len(df) > 0:
        buy_signals_df = df[df['prediction'] == 1].sort_values('actual_change', ascending=False)

        if len(buy_signals_df) > 0:
            report += "\n【最佳交易（TOP 10）】\n"
            for i, (_, row) in enumerate(buy_signals_df.head(10).iterrows(), 1):
                report += f"{i}. {row['stock_code']} | {row['buy_date']} → {row['sell_date']} | "
                report += f"收益率: {row['actual_change']:.2%} | 预测概率: {row['probability']:.4f}\n"

            report += "\n【最差交易（BOTTOM 10）】\n"
            for i, (_, row) in enumerate(buy_signals_df.tail(10).iterrows(), 1):
                report += f"{i}. {row['stock_code']} | {row['buy_date']} → {row['sell_date']} | "
                report += f"收益率: {row['actual_change']:.2%} | 预测概率: {row['probability']:.4f}\n"

    return report


def main():
    parser = argparse.ArgumentParser(description='20天持有期回测 CatBoost 20天模型')
    parser.add_argument('--horizon', type=int, default=20, help='预测周期（天）')
    parser.add_argument('--confidence-threshold', type=float, default=0.55, help='置信度阈值')
    parser.add_argument('--start-date', type=str, default='2026-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2026-01-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--use-feature-selection', action='store_true', help='使用特征选择')
    parser.add_argument('--skip-feature-selection', action='store_true', help='跳过特征选择')
    parser.add_argument('--enable-dynamic-risk-control', action='store_true', help='启用动态风险控制（业界标准）')

    args = parser.parse_args()

    logger.info(f"开始20天持有期回测")
    print(f"   模型类型: CatBoost {args.horizon}天")
    print(f"   持有期: {args.horizon}天（符合模型预测周期）")
    print(f"   置信度阈值: {args.confidence_threshold}")
    print(f"   回测日期范围: {args.start_date} 至 {args.end_date}")
    if args.enable_dynamic_risk_control:
        print(f"   ✅ 动态风险控制: 已启用（符合业界标准）")
    else:
        print(f"   动态风险控制: 未启用（传统模式）")

    # 加载模型
    print("\n🔧 加载 CatBoost 模型...")
    model = CatBoostModel()
    model.load_model(f'data/ml_trading_model_catboost_{args.horizon}d.pkl')
    print(f"✅ 模型已加载")

    # 加载特征选择结果
    selected_features = None
    if args.use_feature_selection:
        try:
            selected_features = model.load_selected_features()
            if selected_features is None:
                logger.error("错误：未找到特征选择结果，请先运行特征选择")
                return
            print(f"✅ 已加载 {len(selected_features)} 个精选特征")
        except Exception as e:
            logger.warning(f" 无法加载特征选择结果: {e}")
            selected_features = None

    # 准备测试数据
    print("\n📊 准备测试数据...")
    from config import WATCHLIST

    test_df = model.prepare_data(
        codes=list(WATCHLIST.keys()),
        horizon=args.horizon,
        for_backtest=True
    )

    if test_df is None or len(test_df) == 0:
        logger.error("错误：没有可用数据")
        return

    # 获取特征列
    if args.use_feature_selection and selected_features is not None:
        feature_columns = selected_features
    else:
        feature_columns = model.feature_columns

    print(f"✅ 测试数据准备完成: {len(test_df)} 条，特征列数: {len(feature_columns)}")

    # 创建回测器
    backtester = Backtest20DHoldPeriod(
        model=model,
        confidence_threshold=args.confidence_threshold,
        enable_dynamic_risk_control=args.enable_dynamic_risk_control
    )

    # 运行回测
    print(f"\n🚀 开始20天持有期回测...")
    all_trades = backtester.backtest_all_stocks(
        test_df=test_df,
        feature_columns=feature_columns,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if not all_trades:
        logger.error("没有回测结果")
        return

    # 计算性能指标
    print(f"\n📊 计算性能指标...")
    metrics = calculate_performance_metrics(all_trades)

    # 保存结果
    print(f"\n💾 保存结果...")
    save_results(all_trades, metrics)

    # 打印汇总报告
    print(f"\n{'='*80}")
    print(f"20天持有期回测完成！")
    print(f"{'='*80}")
    print(f"回测日期范围: {args.start_date} 至 {args.end_date}")
    print(f"总交易机会: {metrics['total_trades']}")
    print(f"买入信号数: {metrics['buy_signals']}")
    print(f"准确率: {metrics['accuracy']:.2%} (与训练时可比)")
    print(f"F1分数: {metrics['f1_score']:.4f} (与训练时可比)")
    print(f"平均收益率: {metrics['avg_return']:.2%} (买入信号)")
    print(f"胜率: {metrics['win_rate']:.2%} (买入信号)")
    print(f"夏普比率（年化）: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")


if __name__ == '__main__':
    main()