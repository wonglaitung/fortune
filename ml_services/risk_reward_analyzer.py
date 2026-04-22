#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险回报率分析器 - 多股票风险回报率对比分析工具

功能：
1. 分析多只股票的风险指标（VaR、最大回撤、波动率、Beta、流动性）
2. 分析多只股票的回报指标（趋势、动量、夏普比率、技术形态、实时状态）
3. 根据投资风格动态调整权重
4. 生成综合排名报告

使用方式：
    python3 ml_services/risk_reward_analyzer.py --stocks watchlist --style moderate
    python3 ml_services/risk_reward_analyzer.py --stocks "0700.HK,9988.HK" --style conservative
    python3 ml_services/risk_reward_analyzer.py --sector bank --style aggressive
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SECTOR_MAPPING, SECTOR_NAME_MAPPING, WATCHLIST
from data_services.tencent_finance import get_hk_stock_data_tencent, get_hsi_data_tencent
from data_services.technical_analysis import TechnicalAnalyzer


# ==================== 投资风格配置 ====================

INVESTMENT_STYLES = {
    'conservative': {
        'name': '保守型',
        'description': '优先控制风险，适合防御性资产',
        'risk_weight': 0.60,
        'return_weight': 0.40,
    },
    'moderate': {
        'name': '平衡型',
        'description': '风险回报并重',
        'risk_weight': 0.50,
        'return_weight': 0.50,
    },
    'aggressive': {
        'name': '激进型',
        'description': '追求回报，可接受高波动',
        'risk_weight': 0.30,
        'return_weight': 0.70,
    }
}


# ==================== 风险指标计算函数 ====================

def calculate_var(df: pd.DataFrame, confidence: float = 0.95, window: int = 20) -> float:
    """
    计算风险价值 (VaR) - 历史模拟法

    Args:
        df: 包含Close列的DataFrame
        confidence: 置信水平，默认0.95
        window: 计算窗口（天数），默认20

    Returns:
        VaR值（正数，表示可能的最大损失比例）
    """
    if df.empty or len(df) < window:
        return 0.0

    returns = df['Close'].pct_change(window).dropna()
    if len(returns) < 10:
        return 0.0

    var_percentile = (1 - confidence) * 100
    var = np.percentile(returns, var_percentile)

    return abs(var)


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """
    计算最大回撤

    Args:
        df: 包含Close列的DataFrame

    Returns:
        最大回撤值（正数，例如0.25表示25%）
    """
    if df.empty or len(df) < 2:
        return 0.0

    cumulative_max = df['Close'].cummax()
    drawdown = (df['Close'] - cumulative_max) / cumulative_max
    max_dd = drawdown.min()

    return abs(max_dd) if not np.isnan(max_dd) else 0.0


def calculate_volatility(df: pd.DataFrame, annualize: bool = True) -> float:
    """
    计算波动率

    Args:
        df: 包含Close列的DataFrame
        annualize: 是否年化，默认True

    Returns:
        波动率（年化或日波动率）
    """
    if df.empty or len(df) < 2:
        return 0.0

    daily_returns = df['Close'].pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    daily_std = daily_returns.std()

    if annualize:
        return daily_std * np.sqrt(252)
    return daily_std


def calculate_beta(stock_df: pd.DataFrame, index_df: pd.DataFrame, window: int = 60) -> float:
    """
    计算Beta系数

    Args:
        stock_df: 股票价格DataFrame
        index_df: 指数价格DataFrame
        window: 计算窗口（天数），默认60

    Returns:
        Beta系数
    """
    if stock_df.empty or index_df.empty:
        return 1.0

    stock_returns = stock_df['Close'].pct_change().dropna()
    index_returns = index_df['Close'].pct_change().dropna()

    # 对齐日期
    aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()

    if len(aligned) < window:
        window = len(aligned)

    if window < 10:
        return 1.0

    covariance = aligned.iloc[-window:, 0].cov(aligned.iloc[-window:, 1])
    variance = aligned.iloc[-window:, 1].var()

    if variance == 0 or np.isnan(variance):
        return 1.0

    beta = covariance / variance

    # 限制在合理范围内
    return max(-2.0, min(3.0, beta))


def calculate_liquidity_score(df: pd.DataFrame, window: int = 20) -> float:
    """
    计算流动性评分

    Args:
        df: 包含Volume列的DataFrame
        window: 计算窗口（天数），默认20

    Returns:
        流动性评分（0-100）
    """
    if df.empty or 'Volume' not in df.columns or len(df) < window:
        return 50.0

    recent_volume = df['Volume'].iloc[-window:]

    # 计算平均成交量
    if len(df) > 60:
        avg_volume = df['Volume'].iloc[-60:-window].mean()
    else:
        avg_volume = df['Volume'].mean()

    if avg_volume == 0 or np.isnan(avg_volume):
        return 50.0

    # 成交量比率
    volume_ratio = recent_volume.mean() / avg_volume

    # 成交量稳定性（越稳定越好）
    cv = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 1.0
    stability = max(0, 1 - cv)

    # 综合评分
    score = min(100, (volume_ratio * 30 + stability * 70))

    return score


# ==================== 回报指标计算函数 ====================

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率

    Args:
        df: 包含Close列的DataFrame
        risk_free_rate: 无风险利率（年化），默认2%

    Returns:
        夏普比率
    """
    if df.empty or len(df) < 30:
        return 0.0

    daily_returns = df['Close'].pct_change().dropna()
    if daily_returns.empty:
        return 0.0

    # 年化收益率
    annual_return = (1 + daily_returns.mean()) ** 252 - 1

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252)

    if annual_vol == 0:
        return 0.0

    return (annual_return - risk_free_rate) / annual_vol


def calculate_price_percentile(df: pd.DataFrame, window: int = 90) -> float:
    """
    计算价格分位数

    Args:
        df: 包含Close列的DataFrame
        window: 计算窗口（天数），默认90

    Returns:
        价格分位数（0-100）
    """
    if df.empty or len(df) < window:
        return 50.0

    recent_prices = df['Close'].iloc[-window:]
    current_price = df['Close'].iloc[-1]

    percentile = (recent_prices < current_price).sum() / len(recent_prices) * 100

    return percentile


def calculate_recent_performance(df: pd.DataFrame, index_df: pd.DataFrame = None, days: int = 20) -> Tuple[float, float]:
    """
    计算近期表现

    Args:
        df: 股票价格DataFrame
        index_df: 指数价格DataFrame（可选）
        days: 计算天数，默认20

    Returns:
        (股票收益率%, 相对收益率%)
    """
    if df.empty or len(df) < days:
        return 0.0, 0.0

    stock_return = (df['Close'].iloc[-1] / df['Close'].iloc[-days] - 1) * 100

    if index_df is not None and not index_df.empty and len(index_df) >= days:
        index_return = (index_df['Close'].iloc[-1] / index_df['Close'].iloc[-days] - 1) * 100
        relative_return = stock_return - index_return
    else:
        relative_return = 0.0

    return stock_return, relative_return


def calculate_overbought_oversold_score(df: pd.DataFrame) -> float:
    """
    计算超买超卖评分

    RSI 30-50区间最优（健康），极端区域扣分
    布林带中轨附近最优

    Args:
        df: 包含RSI和BB_position列的DataFrame

    Returns:
        超买超卖评分（0-100）
    """
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
    bb_position = df['BB_position'].iloc[-1] if 'BB_position' in df.columns and not pd.isna(df['BB_position'].iloc[-1]) else 0.5

    # RSI评分：30-50区间最优，极端区域扣分
    if 30 <= rsi <= 50:
        rsi_score = 100
    elif 50 < rsi <= 70:
        rsi_score = 70 - (rsi - 50) * 1.5
    elif rsi > 70:
        rsi_score = max(0, 40 - (rsi - 70) * 2)
    elif 20 <= rsi < 30:
        rsi_score = 70 - (30 - rsi) * 1.5
    else:
        rsi_score = max(0, 40 - (20 - rsi) * 2)

    # 布林带位置评分：中轨附近（0.5）最优
    bb_score = 100 - abs(bb_position - 0.5) * 100

    return (rsi_score + bb_score) / 2


def calculate_trend_score(df: pd.DataFrame) -> float:
    """
    计算趋势评分

    基于均线排列和趋势方向

    Args:
        df: 包含MA列的DataFrame

    Returns:
        趋势评分（0-100）
    """
    if df.empty or len(df) < 50:
        return 50.0

    score = 50.0
    current_price = df['Close'].iloc[-1]

    # 均线排列评分
    ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else None
    ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns and not pd.isna(df['MA50'].iloc[-1]) else None

    if ma20 is not None and ma50 is not None:
        if current_price > ma20 > ma50:
            # 多头排列
            score = 90
        elif current_price < ma20 < ma50:
            # 空头排列
            score = 20
        elif current_price > ma20 and current_price > ma50:
            score = 70
        elif current_price < ma20 and current_price < ma50:
            score = 30

    return score


def calculate_momentum_score(df: pd.DataFrame) -> float:
    """
    计算动量评分

    基于RSI、MACD、KDJ

    Args:
        df: 包含技术指标的DataFrame

    Returns:
        动量评分（0-100）
    """
    if df.empty:
        return 50.0

    score = 50.0

    # RSI评分
    if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
        rsi = df['RSI'].iloc[-1]
        if 50 < rsi <= 70:
            score += 15
        elif rsi > 70:
            score -= 10  # 超买风险
        elif 30 <= rsi < 50:
            score -= 5
        elif rsi < 30:
            score += 10  # 超卖可能反弹

    # MACD评分
    if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]
        macd_hist = df['MACD_histogram'].iloc[-1]

        if not any(pd.isna([macd, macd_signal, macd_hist])):
            if macd > macd_signal and macd_hist > 0:
                score += 20  # 金叉
            elif macd < macd_signal and macd_hist < 0:
                score -= 15  # 死叉

    # KDJ评分
    if all(col in df.columns for col in ['K', 'D']):
        k = df['K'].iloc[-1]
        d = df['D'].iloc[-1]

        if not any(pd.isna([k, d])):
            if k > d and k < 80:
                score += 10  # 上升趋势
            elif k < d and k > 20:
                score -= 10  # 下降趋势

    return max(0, min(100, score))


def calculate_technical_pattern_score(df: pd.DataFrame) -> Tuple[float, str]:
    """
    计算技术形态评分

    Args:
        df: 包含技术指标的DataFrame

    Returns:
        (评分, 形态描述)
    """
    if df.empty:
        return 50.0, "数据不足"

    score = 50.0
    patterns = []

    # 均线排列
    current_price = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else None
    ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns and not pd.isna(df['MA50'].iloc[-1]) else None

    if ma20 is not None and ma50 is not None:
        if current_price > ma20 > ma50:
            patterns.append("多头排列")
            score += 25
        elif current_price < ma20 < ma50:
            patterns.append("空头排列")
            score -= 15
        else:
            patterns.append("均线交织")

    # 布林带形态
    if 'BB_position' in df.columns and not pd.isna(df['BB_position'].iloc[-1]):
        bb_pos = df['BB_position'].iloc[-1]
        if bb_pos < 0.2:
            patterns.append("触及下轨")
            score += 15  # 可能反弹
        elif bb_pos > 0.8:
            patterns.append("触及上轨")
            score -= 10  # 可能回调

    # 放量信号
    if 'Volume_Ratio' in df.columns and not pd.isna(df['Volume_Ratio'].iloc[-1]):
        vol_ratio = df['Volume_Ratio'].iloc[-1]
        if vol_ratio > 2.0:
            patterns.append("放量")
        elif vol_ratio < 0.5:
            patterns.append("缩量")

    pattern_desc = ", ".join(patterns) if patterns else "中性"

    return max(0, min(100, score)), pattern_desc


def calculate_technical_signal_score(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    计算技术信号评分

    Args:
        df: 包含技术指标的DataFrame

    Returns:
        (评分, 信号列表)
    """
    if df.empty:
        return 50.0, []

    score = 50.0
    signals = []

    # MACD信号
    if all(col in df.columns for col in ['MACD', 'MACD_signal']):
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]

        if not any(pd.isna([macd, macd_signal])):
            if macd > macd_signal:
                if len(df) > 1 and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
                    signals.append("MACD金叉")
                    score += 15
                else:
                    score += 5
            else:
                if len(df) > 1 and df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
                    signals.append("MACD死叉")
                    score -= 10
                else:
                    score -= 5

    # RSI信号
    if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signals.append("RSI超卖")
            score += 10
        elif rsi > 70:
            signals.append("RSI超买")
            score -= 10

    # 布林带信号
    if 'BB_position' in df.columns and not pd.isna(df['BB_position'].iloc[-1]):
        bb_pos = df['BB_position'].iloc[-1]
        if bb_pos < 0.1:
            signals.append("跌破下轨")
            score += 10
        elif bb_pos > 0.9:
            signals.append("突破上轨")
            score -= 5

    return max(0, min(100, score)), signals


def calculate_anomaly_score(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    计算异常状态评分

    Args:
        df: 包含价格和成交量数据的DataFrame

    Returns:
        (评分, 异常列表)
    """
    if df.empty or len(df) < 20:
        return 100.0, []

    score = 100.0
    anomalies = []

    # 计算Z-Score异常
    returns = df['Close'].pct_change().dropna()
    if len(returns) >= 20:
        mean_return = returns.iloc[-20:].mean()
        std_return = returns.iloc[-20:].std()
        if std_return > 0:
            z_score = (returns.iloc[-1] - mean_return) / std_return
            if abs(z_score) > 2.0:
                anomalies.append(f"价格异常(Z={z_score:.2f})")
                score -= 15

    # 成交量异常
    if 'Volume' in df.columns:
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-20:-1].mean()
        if avg_volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 3.0:
                anomalies.append(f"成交量暴增({vol_ratio:.1f}倍)")
                score -= 10
            elif vol_ratio > 2.0:
                anomalies.append(f"成交量放大({vol_ratio:.1f}倍)")
                score -= 5

    return max(0, score), anomalies


# ==================== 主分析类 ====================

class RiskRewardAnalyzer:
    """风险回报率分析器"""

    def __init__(self, style: str = 'moderate', period_days: int = 90):
        """
        初始化分析器

        Args:
            style: 投资风格（conservative/moderate/aggressive）
            period_days: 分析周期（天数）
        """
        self.style = style
        self.style_config = INVESTMENT_STYLES.get(style, INVESTMENT_STYLES['moderate'])
        self.period_days = period_days
        self.technical_analyzer = TechnicalAnalyzer()
        self.hsi_data = None

    def fetch_hsi_data(self) -> Optional[pd.DataFrame]:
        """获取恒生指数数据"""
        if self.hsi_data is not None:
            return self.hsi_data

        try:
            self.hsi_data = get_hsi_data_tencent(period_days=self.period_days + 30)
            return self.hsi_data
        except Exception as e:
            print(f"获取恒生指数数据失败: {e}")
            return None

    def fetch_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据

        Args:
            stock_code: 股票代码（例如：0700.HK）

        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 转换股票代码格式（0700.HK -> 00700）
            code_num = stock_code.replace('.HK', '').zfill(5)
            df = get_hk_stock_data_tencent(code_num, period_days=self.period_days + 30)
            return df
        except Exception as e:
            print(f"获取股票 {stock_code} 数据失败: {e}")
            return None

    def calculate_risk_metrics(self, df: pd.DataFrame, hsi_df: pd.DataFrame = None) -> Dict:
        """
        计算风险指标

        Args:
            df: 股票价格DataFrame
            hsi_df: 恒生指数DataFrame（可选）

        Returns:
            风险指标字典
        """
        if df.empty:
            return {}

        metrics = {
            'var_5d': calculate_var(df, confidence=0.95, window=5),
            'var_20d': calculate_var(df, confidence=0.95, window=20),
            'max_drawdown': calculate_max_drawdown(df),
            'volatility': calculate_volatility(df, annualize=True),
            'beta': calculate_beta(df, hsi_df, window=60) if hsi_df is not None else 1.0,
            'liquidity_score': calculate_liquidity_score(df, window=20),
        }

        return metrics

    def calculate_return_metrics(self, df: pd.DataFrame, hsi_df: pd.DataFrame = None) -> Dict:
        """
        计算回报指标

        Args:
            df: 股票价格DataFrame
            hsi_df: 恒生指数DataFrame（可选）

        Returns:
            回报指标字典
        """
        if df.empty:
            return {}

        # 先计算技术指标
        df = self.technical_analyzer.calculate_all_indicators(df)

        # 计算各项指标
        trend_score = calculate_trend_score(df)
        momentum_score = calculate_momentum_score(df)
        sharpe_ratio = calculate_sharpe_ratio(df)
        tech_pattern_score, tech_pattern = calculate_technical_pattern_score(df)
        overbought_oversold_score = calculate_overbought_oversold_score(df)
        price_percentile = calculate_price_percentile(df)
        tech_signal_score, tech_signals = calculate_technical_signal_score(df)
        anomaly_score, anomalies = calculate_anomaly_score(df)
        recent_return, relative_return = calculate_recent_performance(df, hsi_df, days=20)
        recent_return_5d, relative_return_5d = calculate_recent_performance(df, hsi_df, days=5)

        metrics = {
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'sharpe_ratio': sharpe_ratio,
            'tech_pattern_score': tech_pattern_score,
            'tech_pattern': tech_pattern,
            'overbought_oversold_score': overbought_oversold_score,
            'price_percentile': price_percentile,
            'tech_signal_score': tech_signal_score,
            'tech_signals': tech_signals,
            'anomaly_score': anomaly_score,
            'anomalies': anomalies,
            'recent_return_20d': recent_return,
            'relative_return_20d': relative_return,
            'recent_return_5d': recent_return_5d,
            'relative_return_5d': relative_return_5d,
        }

        return metrics

    def calculate_risk_score(self, risk_metrics: Dict) -> float:
        """
        计算风险得分（风险越低得分越高）

        Args:
            risk_metrics: 风险指标字典

        Returns:
            风险得分（0-100）
        """
        if not risk_metrics:
            return 50.0

        score = 0.0

        # VaR评分（VaR越低得分越高）
        var_20d = risk_metrics.get('var_20d', 0.1)
        if var_20d < 0.05:
            score += 25
        elif var_20d < 0.10:
            score += 20
        elif var_20d < 0.15:
            score += 15
        elif var_20d < 0.20:
            score += 10
        else:
            score += 5

        # 最大回撤评分
        max_dd = risk_metrics.get('max_drawdown', 0.3)
        if max_dd < 0.10:
            score += 25
        elif max_dd < 0.20:
            score += 20
        elif max_dd < 0.30:
            score += 15
        elif max_dd < 0.40:
            score += 10
        else:
            score += 5

        # 波动率评分
        volatility = risk_metrics.get('volatility', 0.3)
        if volatility < 0.20:
            score += 20
        elif volatility < 0.30:
            score += 15
        elif volatility < 0.40:
            score += 10
        else:
            score += 5

        # Beta评分（Beta越接近0越好，但适度暴露可以接受）
        beta = abs(risk_metrics.get('beta', 1.0))
        if beta < 0.5:
            score += 15
        elif beta < 1.0:
            score += 12
        elif beta < 1.5:
            score += 8
        else:
            score += 5

        # 流动性评分
        liquidity = risk_metrics.get('liquidity_score', 50)
        score += liquidity * 0.15

        return min(100, max(0, score))

    def calculate_return_score(self, return_metrics: Dict) -> float:
        """
        计算回报得分（回报潜力越高得分越高）

        Args:
            return_metrics: 回报指标字典

        Returns:
            回报得分（0-100）
        """
        if not return_metrics:
            return 50.0

        score = 0.0

        # 趋势评分（权重20%）
        score += return_metrics.get('trend_score', 50) * 0.20

        # 动量评分（权重15%）
        score += return_metrics.get('momentum_score', 50) * 0.15

        # 夏普比率评分（权重10%）
        sharpe = return_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            sharpe_score = 100
        elif sharpe > 1.0:
            sharpe_score = 80
        elif sharpe > 0.5:
            sharpe_score = 60
        elif sharpe > 0:
            sharpe_score = 40
        else:
            sharpe_score = 20
        score += sharpe_score * 0.10

        # 技术形态评分（权重10%）
        score += return_metrics.get('tech_pattern_score', 50) * 0.10

        # 超买超卖评分（权重15%）
        score += return_metrics.get('overbought_oversold_score', 50) * 0.15

        # 近期表现评分（权重10%）
        recent_return = return_metrics.get('recent_return_20d', 0)
        if 2 <= recent_return <= 8:
            recent_score = 80  # 适度涨幅
        elif 0 <= recent_return < 2:
            recent_score = 60  # 小幅上涨
        elif 8 < recent_return <= 15:
            recent_score = 50  # 涨幅较大，谨慎
        elif recent_return > 15:
            recent_score = 30  # 涨幅过大，风险
        elif -5 <= recent_return < 0:
            recent_score = 50  # 小幅回调
        elif -15 <= recent_return < -5:
            recent_score = 70  # 回调可能反弹
        else:
            recent_score = 40  # 大跌
        score += recent_score * 0.10

        # 价格分位数评分（权重8%）
        percentile = return_metrics.get('price_percentile', 50)
        if 30 <= percentile <= 70:
            percentile_score = 80
        elif 20 <= percentile < 30 or 70 < percentile <= 80:
            percentile_score = 60
        else:
            percentile_score = 40
        score += percentile_score * 0.08

        # 技术信号评分（权重8%）
        score += return_metrics.get('tech_signal_score', 50) * 0.08

        # 异常状态评分（权重4%）
        score += return_metrics.get('anomaly_score', 100) * 0.04

        return min(100, max(0, score))

    def calculate_comprehensive_score(self, risk_score: float, return_score: float) -> float:
        """
        计算综合评分

        Args:
            risk_score: 风险得分
            return_score: 回报得分

        Returns:
            综合评分（0-100）
        """
        risk_weight = self.style_config['risk_weight']
        return_weight = self.style_config['return_weight']

        comprehensive = risk_score * risk_weight + return_score * return_weight

        return comprehensive

    def analyze_single_stock(self, stock_code: str, stock_name: str = "") -> Dict:
        """
        分析单只股票

        Args:
            stock_code: 股票代码
            stock_name: 股票名称

        Returns:
            分析结果字典
        """
        # 获取数据
        df = self.fetch_stock_data(stock_code)
        if df is None or df.empty:
            return {
                'code': stock_code,
                'name': stock_name,
                'error': '无法获取数据'
            }

        hsi_df = self.fetch_hsi_data()

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(df, hsi_df)

        # 计算回报指标
        return_metrics = self.calculate_return_metrics(df, hsi_df)

        # 计算得分
        risk_score = self.calculate_risk_score(risk_metrics)
        return_score = self.calculate_return_score(return_metrics)
        comprehensive_score = self.calculate_comprehensive_score(risk_score, return_score)

        # 获取最新价格
        latest_price = df['Close'].iloc[-1]

        result = {
            'code': stock_code,
            'name': stock_name,
            'latest_price': latest_price,
            'risk_metrics': risk_metrics,
            'return_metrics': return_metrics,
            'risk_score': round(risk_score, 1),
            'return_score': round(return_score, 1),
            'comprehensive_score': round(comprehensive_score, 1),
        }

        return result

    def analyze_stocks(self, stock_list: Dict[str, str]) -> List[Dict]:
        """
        分析多只股票

        Args:
            stock_list: 股票字典 {代码: 名称}

        Returns:
            分析结果列表
        """
        results = []
        total = len(stock_list)

        print(f"\n{'='*60}")
        print(f"风险回报率分析器 - {self.style_config['name']}模式")
        print(f"{'='*60}")
        print(f"分析股票数: {total}")
        print(f"分析周期: {self.period_days} 天")
        print(f"{'='*60}\n")

        for i, (code, name) in enumerate(stock_list.items(), 1):
            print(f"[{i}/{total}] 分析 {name} ({code})...")

            result = self.analyze_single_stock(code, name)
            results.append(result)

            if 'error' in result:
                print(f"  ❌ {result['error']}")
            else:
                print(f"  ✅ 综合得分: {result['comprehensive_score']}")

        # 按综合得分排序
        results.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)

        return results

    def generate_report(self, results: List[Dict], output_dir: str = 'output') -> str:
        """
        生成分析报告

        Args:
            results: 分析结果列表
            output_dir: 输出目录

        Returns:
            报告文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'risk_reward_analysis_{timestamp}.md')

        lines = []
        lines.append(f"# 风险回报率分析报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n投资风格: {self.style_config['name']} ({self.style})")
        lines.append(f"- {self.style_config['description']}")
        lines.append(f"- 风险权重: {self.style_config['risk_weight']*100:.0f}%")
        lines.append(f"- 回报权重: {self.style_config['return_weight']*100:.0f}%")
        lines.append(f"\n分析周期: {self.period_days} 天")

        # 综合排名
        lines.append(f"\n## 综合排名\n")
        lines.append("| 排名 | 股票代码 | 股票名称 | 综合得分 | 风险得分 | 回报得分 | 建议 |")
        lines.append("|------|----------|----------|----------|----------|----------|------|")

        for i, r in enumerate(results, 1):
            if 'error' in r:
                continue

            score = r['comprehensive_score']
            if score >= 75:
                suggestion = "⭐ 优选"
            elif score >= 60:
                suggestion = "🟢 推荐"
            elif score >= 45:
                suggestion = "🟡 观察"
            else:
                suggestion = "🔴 暂缓"

            lines.append(f"| {i} | {r['code']} | {r['name']} | {r['comprehensive_score']} | {r['risk_score']} | {r['return_score']} | {suggestion} |")

        # 风险指标详情
        lines.append(f"\n## 风险指标详情\n")
        lines.append("| 股票代码 | VaR(5日) | VaR(20日) | 最大回撤 | 波动率 | Beta | 流动性 |")
        lines.append("|----------|----------|-----------|----------|--------|------|--------|")

        for r in results:
            if 'error' in r:
                continue
            m = r['risk_metrics']
            lines.append(f"| {r['code']} | {m['var_5d']:.2%} | {m['var_20d']:.2%} | {m['max_drawdown']:.2%} | {m['volatility']:.2%} | {m['beta']:.2f} | {m['liquidity_score']:.0f} |")

        # 回报指标详情
        lines.append(f"\n## 回报指标详情\n")
        lines.append("| 股票代码 | 趋势评分 | 动量评分 | 夏普比率 | 超买超卖 | 价格分位 |")
        lines.append("|----------|----------|----------|----------|----------|----------|")

        for r in results:
            if 'error' in r:
                continue
            m = r['return_metrics']
            lines.append(f"| {r['code']} | {m['trend_score']:.0f} | {m['momentum_score']:.0f} | {m['sharpe_ratio']:.2f} | {m['overbought_oversold_score']:.0f} | {m['price_percentile']:.0f}% |")

        # 实时状态详情
        lines.append(f"\n## 实时状态详情\n")
        lines.append("| 股票代码 | 近5日涨幅 | 近20日涨幅 | 相对表现 | 技术形态 | 技术信号 |")
        lines.append("|----------|-----------|------------|----------|----------|----------|")

        for r in results:
            if 'error' in r:
                continue
            m = r['return_metrics']
            signals = ", ".join(m['tech_signals']) if m['tech_signals'] else "无"
            lines.append(f"| {r['code']} | {m['recent_return_5d']:.2f}% | {m['recent_return_20d']:.2f}% | {m['relative_return_20d']:+.2f}% | {m['tech_pattern']} | {signals} |")

        # 异常状态
        has_anomalies = any(r.get('return_metrics', {}).get('anomalies') for r in results if 'error' not in r)
        if has_anomalies:
            lines.append(f"\n## 异常状态警告\n")
            for r in results:
                if 'error' in r:
                    continue
                anomalies = r['return_metrics'].get('anomalies', [])
                if anomalies:
                    lines.append(f"- **{r['name']} ({r['code']})**: {', '.join(anomalies)}")

        # 投资建议
        lines.append(f"\n## 投资建议\n")

        excellent = [r for r in results if 'error' not in r and r['comprehensive_score'] >= 75]
        good = [r for r in results if 'error' not in r and 60 <= r['comprehensive_score'] < 75]
        moderate = [r for r in results if 'error' not in r and 45 <= r['comprehensive_score'] < 60]
        poor = [r for r in results if 'error' not in r and r['comprehensive_score'] < 45]

        if excellent:
            lines.append(f"### 🟢 推荐关注（综合得分 ≥ 75）\n")
            for r in excellent:
                lines.append(f"- **{r['name']} ({r['code']})**: 综合得分 {r['comprehensive_score']}，风险得分 {r['risk_score']}，回报得分 {r['return_score']}")

        if good:
            lines.append(f"\n### 🟡 谨慎观察（综合得分 60-75）\n")
            for r in good:
                lines.append(f"- **{r['name']} ({r['code']})**: 综合得分 {r['comprehensive_score']}")

        if moderate:
            lines.append(f"\n### 🟠 需要审慎（综合得分 45-60）\n")
            for r in moderate:
                lines.append(f"- **{r['name']} ({r['code']})**: 综合得分 {r['comprehensive_score']}")

        if poor:
            lines.append(f"\n### 🔴 暂不考虑（综合得分 < 45）\n")
            for r in poor:
                lines.append(f"- **{r['name']} ({r['code']})**: 综合得分 {r['comprehensive_score']}")

        # 写入文件
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n✅ 报告已生成: {report_file}")

        return report_file

    def _get_suggestion(self, score: float) -> str:
        """根据综合得分获取建议"""
        if score >= 75:
            return "⭐ 优选"
        elif score >= 60:
            return "🟢 推荐"
        elif score >= 45:
            return "🟡 观察"
        else:
            return "🔴 暂缓"

    def save_results_json(self, results: List[Dict], output_path: str) -> str:
        """
        保存结果为JSON格式（用于其他脚本调用）

        Args:
            results: 分析结果列表
            output_path: JSON输出路径

        Returns:
            JSON文件路径
        """
        json_results = []
        for r in results:
            if 'error' not in r:
                json_results.append({
                    'code': r['code'],
                    'name': r['name'],
                    'comprehensive_score': r['comprehensive_score'],
                    'risk_score': r['risk_score'],
                    'return_score': r['return_score'],
                    'suggestion': self._get_suggestion(r['comprehensive_score'])
                })

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON结果已保存: {output_path}")

        return output_path


def get_stock_list_by_sector(sector: str) -> Dict[str, str]:
    """
    根据板块获取股票列表

    Args:
        sector: 板块名称

    Returns:
        股票字典 {代码: 名称}
    """
    stocks = {}
    for code, info in STOCK_SECTOR_MAPPING.items():
        if info.get('sector') == sector or info.get('type') == sector:
            stocks[code] = info.get('name', code)
    return stocks


def main():
    parser = argparse.ArgumentParser(description='风险回报率分析器')
    parser.add_argument('--stocks', type=str, default='watchlist',
                        help='股票列表（逗号分隔）或 watchlist')
    parser.add_argument('--sector', type=str, default=None,
                        help='板块名称（bank/tech/semiconductor等）')
    parser.add_argument('--style', type=str, default='moderate',
                        choices=['conservative', 'moderate', 'aggressive'],
                        help='投资风格')
    parser.add_argument('--period', type=int, default=90,
                        help='分析周期（天数）')
    parser.add_argument('--output', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--output-json', type=str, default=None,
                        help='JSON输出文件路径（用于其他脚本调用）')

    args = parser.parse_args()

    # 确定股票列表
    if args.sector:
        stock_list = get_stock_list_by_sector(args.sector)
        if not stock_list:
            print(f"未找到板块 {args.sector} 的股票")
            return
        print(f"分析板块: {SECTOR_NAME_MAPPING.get(args.sector, args.sector)} ({len(stock_list)} 只股票)")
    elif args.stocks == 'watchlist':
        stock_list = WATCHLIST
    else:
        stock_list = {}
        for code in args.stocks.split(','):
            code = code.strip()
            if code in STOCK_SECTOR_MAPPING:
                stock_list[code] = STOCK_SECTOR_MAPPING[code].get('name', code)
            else:
                stock_list[code] = code

    # 创建分析器并执行分析
    analyzer = RiskRewardAnalyzer(style=args.style, period_days=args.period)
    results = analyzer.analyze_stocks(stock_list)

    # 生成报告
    analyzer.generate_report(results, output_dir=args.output)

    # 如果指定了JSON输出路径，保存JSON结果
    if args.output_json:
        analyzer.save_results_json(results, args.output_json)

    # 打印简要结果
    print(f"\n{'='*60}")
    print("分析完成！综合排名 Top 5:")
    print(f"{'='*60}")
    for i, r in enumerate(results[:5], 1):
        if 'error' not in r:
            print(f"{i}. {r['name']} ({r['code']}): {r['comprehensive_score']} 分")


if __name__ == '__main__':
    main()
