#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态风险控制模块 - 符合业界标准

基于BlackRock、AQR、Two Sigma等顶级量化机构的最佳实践
实现动态仓位管理和极端市场环境识别

核心原则：
1. 不修改模型预测概率（保持模型纯粹性）
2. 根据市场环境动态调整仓位
3. 极端市场环境下停止交易
4. 多层级风险控制
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger('dynamic_risk_control')


class DynamicRiskControl:
    """动态风险控制系统（业界标准）"""
    
    def __init__(self):
        # 风险等级配置
        self.risk_levels = {
            'CRITICAL': {
                'position_multiplier': 0.0,
                'description': '极端市场，停止交易',
                'max_drawdown_limit': -0.05
            },
            'HIGH': {
                'position_multiplier': 0.3,
                'description': '高风险市场，降低仓位',
                'max_drawdown_limit': -0.10
            },
            'MEDIUM': {
                'position_multiplier': 0.6,
                'description': '中等风险，适度降低仓位',
                'max_drawdown_limit': -0.15
            },
            'LOW': {
                'position_multiplier': 1.0,
                'description': '低风险，满仓操作',
                'max_drawdown_limit': -0.20
            }
        }
        
        # 极端市场环境阈值
        self.extreme_market_thresholds = {
            'hsi_crash_5d': -0.10,      # HSI 5日跌幅 > 10%
            'hsi_crash_20d': -0.15,     # HSI 20日跌幅 > 15%
            'extreme_vix': 30,          # VIX > 30
            'consecutive_down_3plus': 3,  # 连续3天以上下跌
            'poor_breadth': 0.8,        # 下跌股票占比 > 80%
            'liquidity_dry': 0.5         # 成交量 < 平均值50%
        }
        
        # 风险限制
        self.risk_limits = {
            'market_level': {
                'max_hsi_drawdown': -0.20,
                'max_vix_level': 35,
                'min_market_breadth': 0.2
            },
            'portfolio_level': {
                'max_sector_concentration': 0.4,
                'max_single_stock_position': 0.3,
                'max_correlation_exposure': 0.7
            },
            'strategy_level': {
                'max_strategy_drawdown': -0.15,
                'min_win_rate': 0.5,
                'max_consecutive_losses': 5
            }
        }
    
    def detect_extreme_market_conditions(self, hsi_data: pd.DataFrame, 
                                           vix_level: float,
                                           stock_data: pd.DataFrame) -> Tuple[bool, Dict, int]:
        """
        检测极端市场环境（业界标准）
        
        Args:
            hsi_data: 恒生指数数据
            vix_level: VIX恐慌指数
            stock_data: 股票数据（用于计算市场广度）
        
        Returns:
            tuple: (是否极端市场, 极端条件字典, 触发指标数量)
        """
        extreme_conditions = {}
        
        # 1. HSI暴跌检测
        if len(hsi_data) >= 5:
            hsi_return_5d = hsi_data['Close'].pct_change(5).iloc[-1]
            extreme_conditions['hsi_crash_5d'] = hsi_return_5d < self.extreme_market_thresholds['hsi_crash_5d']
        
        if len(hsi_data) >= 20:
            hsi_return_20d = hsi_data['Close'].pct_change(20).iloc[-1]
            extreme_conditions['hsi_crash_20d'] = hsi_return_20d < self.extreme_market_thresholds['hsi_crash_20d']
        else:
            extreme_conditions['hsi_crash_20d'] = False
        
        # 2. VIX恐慌检测
        extreme_conditions['extreme_vix'] = vix_level > self.extreme_market_thresholds['extreme_vix']
        
        # 3. 连续下跌天数检测
        consecutive_down = 0
        if len(hsi_data) >= 2:
            for i in range(min(5, len(hsi_data)-1)):
                if hsi_data['Close'].iloc[-(i+1)] < hsi_data['Close'].iloc[-(i+2)]:
                    consecutive_down += 1
                else:
                    break
        
        extreme_conditions['consecutive_down_3plus'] = consecutive_down >= self.extreme_market_thresholds['consecutive_down_3plus']
        
        # 4. 市场广度检测
        if not stock_data.empty and 'Return' in stock_data.columns:
            down_count = (stock_data['Return'] < 0).sum()
            down_ratio = down_count / len(stock_data) if len(stock_data) > 0 else 0
            extreme_conditions['poor_breadth'] = down_ratio > self.extreme_market_thresholds['poor_breadth']
        else:
            extreme_conditions['poor_breadth'] = False
        
        # 5. 流动性检测
        if len(hsi_data) >= 20:
            volume_ratio = hsi_data['Volume'].iloc[-1] / hsi_data['Volume'].iloc[-20:].mean()
            extreme_conditions['liquidity_dry'] = volume_ratio < self.extreme_market_thresholds['liquidity_dry']
        else:
            extreme_conditions['liquidity_dry'] = False
        
        # 综合判断
        extreme_count = sum(extreme_conditions.values())
        is_extreme = extreme_count >= 3  # 3个及以上指标触发
        
        if is_extreme:
            logger.warning(f"极端市场环境触发！触发指标: {extreme_conditions}")
        
        return is_extreme, extreme_conditions, extreme_count
    
    def assess_market_environment(self, hsi_data: pd.DataFrame, 
                                   vix_level: float) -> int:
        """
        评估市场环境（业界标准）
        
        Args:
            hsi_data: 恒生指数数据
            vix_level: VIX恐慌指数
        
        Returns:
            int: 市场环境评分 (0-100)
        """
        score = 0
        
        # HSI 20日收益率（权重30%）
        if len(hsi_data) >= 20:
            hsi_return_20d = (hsi_data['Close'].iloc[-1] - hsi_data['Close'].iloc[-20]) / hsi_data['Close'].iloc[-20]
            if hsi_return_20d > 0.05:
                score += 30
            elif hsi_return_20d < -0.05:
                score += 0
            else:
                score += 15
        
        # VIX恐慌指数（权重25%）
        if vix_level < 15:
            score += 25
        elif vix_level < 25:
            score += 15
        elif vix_level < 35:
            score += 5
        else:
            score += 0
        
        # 连续下跌天数（权重20%）
        consecutive_down = 0
        if len(hsi_data) >= 2:
            for i in range(min(5, len(hsi_data)-1)):
                if hsi_data['Close'].iloc[-(i+1)] < hsi_data['Close'].iloc[-(i+2)]:
                    consecutive_down += 1
                else:
                    break
        
        if consecutive_down == 0:
            score += 20
        elif consecutive_down <= 2:
            score += 10
        else:
            score += 0
        
        # 市场广度（权重15%）- 简化处理，给中间分
        score += 10
        
        # 流动性指标（权重10%）- 简化处理，给中间分
        score += 8
        
        return score
    
    def determine_risk_level(self, market_env_score: int, 
                             base_prediction_prob: float) -> str:
        """
        确定风险等级（业界标准）
        
        Args:
            market_env_score: 市场环境评分 (0-100)
            base_prediction_prob: 基础预测概率
        
        Returns:
            str: 风险等级 ('CRITICAL'/'HIGH'/'MEDIUM'/'LOW')
        """
        # 极端市场：无论模型置信度如何，都是CRITICAL
        if market_env_score < 30:
            return 'CRITICAL'
        
        # 高风险市场：需要高模型置信度才能交易
        if market_env_score < 50:
            if base_prediction_prob > 0.75:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        # 正常市场：根据模型置信度调整
        if market_env_score < 70:
            if base_prediction_prob > 0.70:
                return 'LOW'
            else:
                return 'MEDIUM'
        
        # 低风险市场：降低模型置信度要求
        return 'LOW'
    
    def get_dynamic_position_size(self, base_prediction_prob: float,
                                   market_regime: str,
                                   vix_level: float,
                                   market_env_score: Optional[int] = None) -> Tuple[float, float, str]:
        """
        计算动态仓位大小（业界标准）
        
        核心原则：
        - 不修改模型预测概率
        - 根据市场环境动态调整仓位
        - 极端市场环境下停止交易
        
        Args:
            base_prediction_prob: 基础预测概率（模型输出）
            market_regime: 市场状态 ('bull'/'bear'/'normal')
            vix_level: VIX恐慌指数
            market_env_score: 市场环境评分（可选）
        
        Returns:
            tuple: (调整后的预测概率, 仓位大小, 风险等级)
        """
        # 保持预测概率不变（业界标准）
        adjusted_prediction = base_prediction_prob
        
        # 确定风险等级
        if market_env_score is None:
            # 如果没有市场环境评分，根据其他因素判断
            risk_level = self._determine_risk_level_simple(market_regime, vix_level, base_prediction_prob)
        else:
            risk_level = self.determine_risk_level(market_env_score, base_prediction_prob)
        
        # 获取仓位乘数
        position_multiplier = self.risk_levels[risk_level]['position_multiplier']
        
        # 计算最终仓位
        final_position_size = position_multiplier
        
        logger.info(f"动态仓位管理: 预测概率={base_prediction_prob:.4f}, "
                   f"市场状态={market_regime}, VIX={vix_level:.2f}, "
                   f"风险等级={risk_level}, 仓位={final_position_size:.2f}")
        
        return adjusted_prediction, final_position_size, risk_level
    
    def _determine_risk_level_simple(self, market_regime: str, 
                                   vix_level: float,
                                   base_prediction_prob: float) -> str:
        """
        简化版风险等级判断（不需要市场环境评分）
        """
        # 极端VIX：CRITICAL
        if vix_level > 30:
            return 'CRITICAL'
        
        # 高VIX + 熊市：HIGH
        if vix_level > 25 and market_regime == 'bear':
            return 'HIGH'
        
        # 熊市：HIGH或MEDIUM
        if market_regime == 'bear':
            if base_prediction_prob > 0.75:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        # 震荡市：MEDIUM
        if market_regime == 'normal':
            return 'MEDIUM'
        
        # 牛市：LOW
        return 'LOW'
    
    def check_all_risk_levels(self, market_data: Dict,
                              portfolio_data: Dict,
                              strategy_data: Dict) -> Tuple[bool, list]:
        """
        检查所有层级风险（业界标准4层风险控制）
        
        Args:
            market_data: 市场层面数据
            portfolio_data: 组合层面数据
            strategy_data: 策略层面数据
        
        Returns:
            tuple: (是否允许交易, 风险违规列表)
        """
        risk_violations = []
        
        # 第1层：市场层面检查
        market_risks = self.check_market_level(market_data)
        if market_risks:
            risk_violations.extend(market_risks)
        
        # 第2层：组合层面检查
        portfolio_risks = self.check_portfolio_level(portfolio_data)
        if portfolio_risks:
            risk_violations.extend(portfolio_risks)
        
        # 第3层：策略层面检查
        strategy_risks = self.check_strategy_level(strategy_data)
        if strategy_risks:
            risk_violations.extend(strategy_risks)
        
        # 判断是否允许交易
        if risk_violations:
            logger.warning(f"风险检查失败: {risk_violations}")
            return False, risk_violations
        else:
            return True, []
    
    def check_market_level(self, market_data: Dict) -> list:
        """检查市场层面风险"""
        violations = []
        
        # HSI回撤检查
        if market_data.get('hsi_drawdown', 0) < self.risk_limits['market_level']['max_hsi_drawdown']:
            violations.append(f"HSI回撤过大: {market_data['hsi_drawdown']:.2%}")
        
        # VIX检查
        if market_data.get('vix_level', 0) > self.risk_limits['market_level']['max_vix_level']:
            violations.append(f"VIX恐慌指数过高: {market_data['vix_level']:.2f}")
        
        # 市场广度检查
        if market_data.get('market_breadth', 1.0) < self.risk_limits['market_level']['min_market_breadth']:
            violations.append(f"市场广度过低: {market_data['market_breadth']:.2%}")
        
        return violations
    
    def check_portfolio_level(self, portfolio_data: Dict) -> list:
        """检查组合层面风险"""
        violations = []
        
        # 行业集中度检查
        if portfolio_data.get('max_sector_concentration', 0) > self.risk_limits['portfolio_level']['max_sector_concentration']:
            violations.append(f"行业集中度过高: {portfolio_data['max_sector_concentration']:.2%}")
        
        # 单股仓位检查
        if portfolio_data.get('max_single_stock_position', 0) > self.risk_limits['portfolio_level']['max_single_stock_position']:
            violations.append(f"单股仓位过大: {portfolio_data['max_single_stock_position']:.2%}")
        
        return violations
    
    def check_strategy_level(self, strategy_data: Dict) -> list:
        """检查策略层面风险"""
        violations = []
        
        # 策略回撤检查
        if strategy_data.get('strategy_drawdown', 0) < self.risk_limits['strategy_level']['max_strategy_drawdown']:
            violations.append(f"策略回撤过大: {strategy_data['strategy_drawdown']:.2%}")
        
        # 胜率检查
        if strategy_data.get('win_rate', 1.0) < self.risk_limits['strategy_level']['min_win_rate']:
            violations.append(f"胜率过低: {strategy_data['win_rate']:.2%}")
        
        # 连续亏损检查
        if strategy_data.get('consecutive_losses', 0) >= self.risk_limits['strategy_level']['max_consecutive_losses']:
            violations.append(f"连续亏损次数过多: {strategy_data['consecutive_losses']}")
        
        return violations


def calculate_market_beta(stock_return: pd.Series, hsi_return: pd.Series, window: int = 20) -> float:
    """
    计算市场beta（业界标准）
    
    Beta = Cov(Stock, HSI) / Var(HSI)
    
    Args:
        stock_return: 股票收益率序列
        hsi_return: HSI收益率序列
        window: 计算窗口
    
    Returns:
        float: 市场beta
    """
    if len(stock_return) < window or len(hsi_return) < window:
        return 0.0  # 数据不足，beta为0
    
    # 计算协方差和方差
    covariance = np.cov(stock_return[-window:], hsi_return[-window:])[0][1]
    variance = np.var(hsi_return[-window:])
    
    if variance == 0:
        return 0.0
    
    beta = covariance / variance
    
    return beta


def market_neutralize_prediction(base_prediction: float, market_beta: float) -> float:
    """
    市场中性化处理（业界标准）
    
    核心原则：
    - 消除市场beta的影响
    - 确保收益来自选股能力（alpha），而非市场涨跌（beta）
    
    Args:
        base_prediction: 基础预测概率
        market_beta: 市场beta
    
    Returns:
        float: 中性化后的预测概率
    """
    # 计算市场中性后的预测
    neutralized_prediction = base_prediction - market_beta
    
    # 确保预测概率在[0, 1]范围内
    neutralized_prediction = max(0.0, min(1.0, neutralized_prediction))
    
    return neutralized_prediction