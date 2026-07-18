#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股综合买卖建议生成器

功能：
1. 解析大模型建议文件
2. 整合ML预测结果
3. 生成综合买卖建议（强烈买入、买入、持有、卖出）
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# A股配置
from a_stock_config import A_STOCK_WATCHLIST, A_STOCK_SECTOR_MAPPING, get_limit_rate


class AStockRecommendationGenerator:
    """A股综合买卖建议生成器"""

    def __init__(self):
        self.stock_names = A_STOCK_WATCHLIST
        self.sector_mapping = A_STOCK_SECTOR_MAPPING

    def generate_recommendations(
        self,
        llm_report: str,
        ml_predictions: Dict,
        stock_analyses: Dict,
        market_data: Dict,
        northbound_data: Dict
    ) -> Dict:
        """
        生成综合买卖建议

        Args:
            llm_report: 大模型分析报告文本
            ml_predictions: ML预测结果 {stock_code: {'predictions': {1: {}, 5: {}, 20: {}}, 'pattern': str}}
            stock_analyses: 股票技术分析 {stock_code: {...}}
            market_data: 市场数据
            northbound_data: 北向资金数据

        Returns:
            Dict: {
                'strong_buy': [...],
                'buy': [...],
                'hold': [...],
                'sell': [...],
                'risk_control': {...}
            }
        """
        recommendations = {
            'strong_buy': [],
            'buy': [],
            'hold': [],
            'sell': [],
            'risk_control': {}
        }

        # 解析大模型建议
        llm_recommendations = self._parse_llm_report(llm_report)

        # 遍历所有股票生成建议
        for stock_code, analysis in stock_analyses.items():
            recommendation = self._generate_single_recommendation(
                stock_code=stock_code,
                analysis=analysis,
                ml_pred=ml_predictions.get(stock_code, {}),
                llm_rec=llm_recommendations.get(stock_code, {}),
                market_data=market_data
            )

            # 分类添加
            signal_type = recommendation.get('signal_type', 'hold')
            if signal_type == 'strong_buy':
                recommendations['strong_buy'].append(recommendation)
            elif signal_type == 'buy':
                recommendations['buy'].append(recommendation)
            elif signal_type == 'sell':
                recommendations['sell'].append(recommendation)
            else:
                recommendations['hold'].append(recommendation)

        # 按概率排序
        for key in ['strong_buy', 'buy', 'hold', 'sell']:
            recommendations[key].sort(
                key=lambda x: x.get('probability_20d', 0.5),
                reverse=(key in ['strong_buy', 'buy'])
            )

        # 生成风险控制建议
        recommendations['risk_control'] = self._generate_risk_control(
            recommendations=recommendations,
            market_data=market_data,
            northbound_data=northbound_data
        )

        return recommendations

    def _parse_llm_report(self, llm_report: str) -> Dict:
        """
        解析大模型报告，提取每只股票的建议

        Args:
            llm_report: 大模型报告文本

        Returns:
            Dict: {stock_code: {'short_term': str, 'mid_term': str, 'action': str}}
        """
        recommendations = {}

        # 尝试解析结构化的股票建议
        # 格式：股票名称 (代码): 建议内容
        for stock_code, stock_name in self.stock_names.items():
            # 查找该股票的相关内容
            pattern = rf'{stock_name}.*?{stock_code}'
            if re.search(pattern, llm_report):
                # 提取建议关键词
                short_term = self._extract_term_advice(llm_report, stock_name, '短期')
                mid_term = self._extract_term_advice(llm_report, stock_name, '中期')

                # 判断建议类型
                action = '观望'
                if '买入' in short_term or '建议买入' in short_term:
                    action = '买入'
                elif '卖出' in short_term or '减仓' in short_term:
                    action = '卖出'

                recommendations[stock_code] = {
                    'short_term': short_term,
                    'mid_term': mid_term,
                    'action': action
                }

        return recommendations

    def _extract_term_advice(self, text: str, stock_name: str, term: str) -> str:
        """提取特定期限的建议"""
        # 简单实现：查找股票名称附近的关键词
        pattern = rf'{stock_name}[^{{]*?{term}[^{{]*?(买入|卖出|观望|持有)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return '观望'

    def _generate_single_recommendation(
        self,
        stock_code: str,
        analysis: Dict,
        ml_pred: Dict,
        llm_rec: Dict,
        market_data: Dict
    ) -> Dict:
        """
        生成单只股票的建议

        决策逻辑：
        - 强烈买入：短期买入 + 中期买入 + CatBoost概率 ≥ 0.55
        - 买入：短期买入 + 中期买入 + 0.50 < CatBoost概率 < 0.55
        - 持有：其他情况
        - 卖出：短期卖出 + 中期卖出 + CatBoost概率 ≤ 0.45
        """
        stock_name = self.stock_names.get(stock_code, stock_code)
        current_price = analysis.get('current_price', 0)

        # 获取ML预测概率
        predictions = ml_pred.get('predictions', {})
        prob_1d = predictions.get(1, {}).get('probability', 0.5)
        prob_5d = predictions.get(5, {}).get('probability', 0.5)
        prob_20d = predictions.get(20, {}).get('probability', 0.5)

        # 获取大模型建议
        short_term = llm_rec.get('short_term', '观望')
        mid_term = llm_rec.get('mid_term', '观望')

        # 决策逻辑
        signal_type = 'hold'
        reason = ''

        # 短期和中期一致买入
        if '买入' in short_term and '买入' in mid_term:
            if prob_20d >= 0.55:
                signal_type = 'strong_buy'
                reason = f"短期建议买入，中期建议买入，CatBoost预测上涨概率{prob_20d:.2f}（高置信度），方向一致"
            elif prob_20d > 0.50:
                signal_type = 'buy'
                reason = f"短期建议买入，中期建议买入，CatBoost预测上涨概率{prob_20d:.2f}（中等置信度）"
            else:
                signal_type = 'hold'
                reason = f"短期建议买入，中期建议买入，但CatBoost预测上涨概率{prob_20d:.2f}（≤0.50），违反硬约束，建议观望"

        # 短期和中期一致卖出
        elif '卖出' in short_term or '卖出' in mid_term:
            if prob_20d <= 0.45:
                signal_type = 'sell'
                reason = f"短期建议{short_term}，中期建议{mid_term}，CatBoost预测下跌概率{1-prob_20d:.2f}"
            else:
                signal_type = 'hold'
                reason = f"短期建议{short_term}，中期建议{mid_term}，但CatBoost概率{prob_20d:.2f}，建议观望"

        # 观望
        else:
            signal_type = 'hold'
            reason = f"短期建议{short_term}，中期建议{mid_term}，建议观望"

        # 计算价格指引
        limit_rate = get_limit_rate(stock_code)
        buy_price = current_price
        stop_loss = current_price * 0.92  # -8%
        target_price = current_price * 1.10  # +10%

        # 建议仓位
        position_pct = 0
        if signal_type == 'strong_buy':
            position_pct = 4
        elif signal_type == 'buy':
            position_pct = 3

        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'signal_type': signal_type,
            'reason': reason,
            'action': '买入' if signal_type in ['strong_buy', 'buy'] else ('卖出' if signal_type == 'sell' else '观望'),
            'position_pct': position_pct,
            'current_price': current_price,
            'buy_price': buy_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'probability_1d': prob_1d,
            'probability_5d': prob_5d,
            'probability_20d': prob_20d,
            'limit_rate': limit_rate,
            'change_percent': analysis.get('change_percent', 0)
        }

    def _generate_risk_control(
        self,
        recommendations: Dict,
        market_data: Dict,
        northbound_data: Dict
    ) -> Dict:
        """生成风险控制建议"""
        # 计算总仓位
        total_position = 0
        for rec in recommendations['strong_buy']:
            total_position += rec.get('position_pct', 0)
        for rec in recommendations['buy']:
            total_position += rec.get('position_pct', 0)

        # 判断市场风险
        market_risk = '中'
        if northbound_data:
            nb_buy = northbound_data.get('northbound_net_buy', 0)
            if nb_buy < -50:
                market_risk = '高'
            elif nb_buy > 50:
                market_risk = '低'

        return {
            'market_risk': market_risk,
            'total_position': total_position,
            'stop_loss_strategy': '单只股票最大亏损不超过-8%，触及止损位无条件平仓',
            'position_strategy': f'建议总仓位{min(total_position, 50)}%，保留{(50-total_position) if total_position < 50 else 0}%现金'
        }

    def format_recommendations_html(self, recommendations: Dict) -> str:
        """生成HTML格式的综合买卖建议"""
        html = ""

        # 强烈买入
        if recommendations['strong_buy']:
            html += """
    <h2>🟢 强烈买入信号</h2>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议仓位</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
            for rec in recommendations['strong_buy']:
                html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{'positive' if rec['change_percent'] >= 0 else 'negative'}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #16a34a; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['position_pct']}%</td>
            <td>{rec['stop_loss']:.2f}</td>
            <td>{rec['target_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
            html += "    </table>\n"

        # 买入
        if recommendations['buy']:
            html += """
    <h2>🟡 买入信号</h2>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议仓位</th><th>止损位</th><th>目标价</th>
            <th>推荐理由</th>
        </tr>
"""
            for rec in recommendations['buy']:
                html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{'positive' if rec['change_percent'] >= 0 else 'negative'}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #ea580c; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['position_pct']}%</td>
            <td>{rec['stop_loss']:.2f}</td>
            <td>{rec['target_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
            html += "    </table>\n"

        # 持有/观望
        if recommendations['hold']:
            html += """
    <h2>⚪ 持有/观望</h2>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议</th><th>推荐理由</th>
        </tr>
"""
            for rec in recommendations['hold']:
                html += f"""        <tr>
            <td>{rec['stock_name']}</td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{'positive' if rec['change_percent'] >= 0 else 'negative'}">{rec['change_percent']:+.2f}%</td>
            <td>{rec['probability_20d']:.2f}</td>
            <td>观望</td>
            <td>{rec['reason']}</td>
        </tr>
"""
            html += "    </table>\n"

        # 卖出
        if recommendations['sell']:
            html += """
    <h2>🔴 卖出信号</h2>
    <table>
        <tr>
            <th>股票</th><th>代码</th><th>现价</th><th>涨跌</th>
            <th>20天概率</th><th>建议卖出价</th><th>推荐理由</th>
        </tr>
"""
            for rec in recommendations['sell']:
                html += f"""        <tr>
            <td><strong>{rec['stock_name']}</strong></td>
            <td>{rec['stock_code']}</td>
            <td>{rec['current_price']:.2f}</td>
            <td class="{'positive' if rec['change_percent'] >= 0 else 'negative'}">{rec['change_percent']:+.2f}%</td>
            <td style="color: #dc2626; font-weight: bold;">{rec['probability_20d']:.2f}</td>
            <td>{rec['current_price']:.2f}</td>
            <td>{rec['reason']}</td>
        </tr>
"""
            html += "    </table>\n"

        # 风险控制建议
        risk = recommendations['risk_control']
        html += f"""
    <h2>⚠️ 风险控制建议</h2>
    <table>
        <tr><th>指标</th><th>建议</th></tr>
        <tr><td>当前市场风险</td><td>{risk['market_risk']}</td></tr>
        <tr><td>建议总仓位</td><td>{risk['total_position']}%</td></tr>
        <tr><td>止损策略</td><td>{risk['stop_loss_strategy']}</td></tr>
        <tr><td>仓位策略</td><td>{risk['position_strategy']}</td></tr>
    </table>
"""

        return html

    def format_recommendations_text(self, recommendations: Dict) -> str:
        """生成文本格式的综合买卖建议"""
        text = "# 综合买卖建议\n\n"

        # 强烈买入
        if recommendations['strong_buy']:
            text += "## 强烈买入信号\n"
            for rec in recommendations['strong_buy']:
                text += f"""
{rec['stock_name']} ({rec['stock_code']})
   - 推荐理由：{rec['reason']}
   - 操作建议：买入
   - 建议仓位：{rec['position_pct']}%
   - 价格指引：
     * 建议买入价：{rec['buy_price']:.2f} 元
     * 止损位：{rec['stop_loss']:.2f} 元（-8.00%）
     * 目标价：{rec['target_price']:.2f} 元（+10.00%）
"""

        # 买入
        if recommendations['buy']:
            text += "\n## 买入信号\n"
            for rec in recommendations['buy']:
                text += f"""
{rec['stock_name']} ({rec['stock_code']})
   - 推荐理由：{rec['reason']}
   - 操作建议：买入
   - 建议仓位：{rec['position_pct']}%
   - 价格指引：
     * 建议买入价：{rec['buy_price']:.2f} 元
     * 止损位：{rec['stop_loss']:.2f} 元（-8.00%）
     * 目标价：{rec['target_price']:.2f} 元（+10.00%）
"""

        # 持有/观望
        if recommendations['hold']:
            text += "\n## 持有/观望\n"
            for rec in recommendations['hold']:
                text += f"""
{rec['stock_name']} ({rec['stock_code']})
   - 推荐理由：{rec['reason']}
   - 操作建议：观望
"""

        # 卖出
        if recommendations['sell']:
            text += "\n## 卖出信号\n"
            for rec in recommendations['sell']:
                text += f"""
{rec['stock_name']} ({rec['stock_code']})
   - 推荐理由：{rec['reason']}
   - 操作建议：卖出
   - 建议卖出价：{rec['current_price']:.2f} 元
"""

        # 风险控制
        risk = recommendations['risk_control']
        text += f"""
## 风险控制建议
- 当前市场风险：{risk['market_risk']}
- 建议仓位百分比：{risk['total_position']}%
- 止损位设置：{risk['stop_loss_strategy']}
- 仓位策略：{risk['position_strategy']}
"""

        return text