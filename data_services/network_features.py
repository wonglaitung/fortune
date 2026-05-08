#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络特征模块 - 实时计算或加载跨截面网络特征

用途：
1. 综合分析：实时计算网络洞察（用于邮件展示）
2. 独立分析：运行 stock_network_analysis.py 生成详细报告

网络洞察（用于综合分析展示）：
- 社区归属：股票的网络群落
- 枢纽等级：低/中/高（基于中心性）
- 桥梁股标记：是否跨社区连接

创建时间：2026-04-30
更新时间：2026-04-30（改为实时计算，不依赖文件）
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 跃径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class NetworkInsightCalculator:
    """网络洞察计算器 - 实时计算网络特征用于综合分析展示"""

    def __init__(self):
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600  # 缓存有效期：1小时

    def calculate_network_insights(self, stock_codes, force_refresh=False):
        """
        实时计算网络洞察

        参数:
            stock_codes: 股票代码列表
            force_refresh: 是否强制刷新缓存

        返回:
            dict: {股票代码: 网络洞察信息}
        """
        # 检查缓存
        if not force_refresh and self._cache and self._cache_time:
            elapsed = (datetime.now() - self._cache_time).total_seconds()
            if elapsed < self._cache_ttl:
                logger.info("使用缓存的网络洞察数据")
                return self._cache

        logger.info("开始计算网络洞察...")
        print("  📊 计算网络洞察...")

        try:
            # 导入网络分析模块
            from ml_services.stock_network_analysis import (
                fetch_all_stock_data,
                build_returns_dataframe,
                compute_correlation_matrices,
                build_minimum_spanning_tree,
                calculate_centrality_metrics,
                detect_communities,
                identify_bridge_stocks,
                get_stock_name,
                RANDOM_SEED
            )
            import networkx as nx

            # 获取股票数据
            stock_data = fetch_all_stock_data(list(stock_codes))
            if not stock_data:
                logger.warning("无法获取股票数据，返回默认值")
                return self._get_default_insights(stock_codes)

            # 构建收益率 DataFrame
            returns_df = build_returns_dataframe(stock_data)
            if returns_df.empty or len(returns_df.columns) < 2:
                logger.warning("数据不足，返回默认值")
                return self._get_default_insights(stock_codes)

            # 计算相关性矩阵
            pearson_corr, spearman_corr = compute_correlation_matrices(returns_df)
            corr_matrix = pearson_corr

            # 构建距离矩阵
            import numpy as np
            distance_matrix = np.sqrt(2 * (1 - corr_matrix))

            # 构建 MST
            mst_graph = build_minimum_spanning_tree(distance_matrix, list(returns_df.columns))

            # 计算中心性
            centrality_dict = calculate_centrality_metrics(mst_graph)

            # 检测社区
            communities, modularity = detect_communities(mst_graph)

            # 识别桥梁股
            bridge_stocks = identify_bridge_stocks(mst_graph, communities)

            # 构建洞察结果
            insights = {}
            bridge_set = set(b['stock'] for b in bridge_stocks)

            # 计算中心性排名
            composite_scores = [(code, c.get('composite', 0))
                               for code, c in centrality_dict.items()]
            composite_scores.sort(key=lambda x: x[1], reverse=True)
            total_stocks = len(composite_scores)

            for i, (code, score) in enumerate(composite_scores):
                rank = i + 1
                c = centrality_dict.get(code, {})
                comm = communities.get(code, -1)

                # 计算枢纽等级
                percentile = rank / total_stocks
                if percentile <= 0.1:  # Top 10%
                    hub_level = '高'
                elif percentile <= 0.3:  # Top 30%
                    hub_level = '中'
                else:
                    hub_level = '低'

                # 构建洞察字符串
                is_bridge = code in bridge_set
                insight_str = f"社区{comm}/{hub_level}枢纽"
                if is_bridge:
                    insight_str += "/桥梁股⚠️"

                insights[code] = {
                    'community': comm,
                    'hub_level': hub_level,
                    'is_bridge': is_bridge,
                    'composite_centrality': c.get('composite', 0),
                    'betweenness_centrality': c.get('betweenness', 0),
                    'rank': rank,
                    'insight_str': insight_str
                }

            # 添加系统性风险信息
            insights['_meta'] = {
                'core_hubs': [s[0] for s in composite_scores[:3]],  # Top 3 核心枢纽
                'bridge_count': len(bridge_set),
                'community_count': len(set(communities.values())),
                'modularity': modularity,
                'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 更新缓存
            self._cache = insights
            self._cache_time = datetime.now()

            print(f"    ✅ 网络洞察计算完成: {len(insights)} 只股票")
            logger.info(f"网络洞察计算完成: {len(insights)} 只股票")

            return insights

        except Exception as e:
            logger.warning(f"网络洞察计算失败: {e}")
            print(f"    ⚠️ 网络洞察计算失败: {e}")
            return self._get_default_insights(stock_codes)

    def _get_default_insights(self, stock_codes):
        """返回默认洞察值"""
        insights = {}
        for code in stock_codes:
            insights[code] = {
                'community': -1,
                'hub_level': '未知',
                'is_bridge': False,
                'composite_centrality': 0,
                'betweenness_centrality': 0,
                'rank': 0,
                'insight_str': '未知'
            }
        insights['_meta'] = {
            'core_hubs': [],
            'bridge_count': 0,
            'community_count': 0,
            'modularity': 0,
            'calculation_time': 'N/A'
        }
        return insights

    def get_insight_for_stock(self, insights, stock_code):
        """获取单只股票的网络洞察字符串"""
        if stock_code in insights:
            return insights[stock_code].get('insight_str', '未知')
        return '未知'

    def get_risk_warning(self, insights, stock_code):
        """获取网络风险警告"""
        if stock_code not in insights:
            return None

        stock_insight = insights[stock_code]
        warnings = []

        # 高枢纽警告
        if stock_insight.get('hub_level') == '高':
            warnings.append("核心枢纽，波动影响市场")

        # 桥梁股警告
        if stock_insight.get('is_bridge'):
            warnings.append("桥梁股，波动跨板块传导")

        return warnings if warnings else None

    def generate_insights_table(self, insights):
        """
        生成网络洞察预警表格（Markdown格式）

        参数:
            insights: 网络洞察信息字典（包含各股票洞察和_meta元数据）

        返回:
            str: Markdown格式的预警表格
        """
        if not insights or '_meta' not in insights:
            return ""

        meta = insights.get('_meta', {})

        # 统计各社区股票数量
        community_stats = {}
        hub_stats = {'高': 0, '中': 0, '低': 0, '未知': 0}
        bridge_count = 0

        for key, value in insights.items():
            if key == '_meta':
                continue
            if isinstance(value, dict):
                # 统计社区
                comm = value.get('community', -1)
                if comm not in community_stats:
                    community_stats[comm] = 0
                community_stats[comm] += 1

                # 统计枢纽等级
                hub_level = value.get('hub_level', '未知')
                if hub_level in hub_stats:
                    hub_stats[hub_level] += 1

                # 统计桥梁股
                if value.get('is_bridge', False):
                    bridge_count += 1

        total_stocks = sum(community_stats.values())
        modularity = meta.get('modularity', 0)
        community_count = meta.get('community_count', 0)

        # 构建表格
        table = "\n### 🕸️ 网络洞察预警\n\n"
        table += "**核心逻辑**：模块度高 → 社区分化明显 → 个股独立性强 → 选股模型有效 → 正常操作；模块度低 → 市场同涨同跌 → 系统性风险高 → 降低仓位\n\n"

        # 网络指标数据表
        table += "**网络指标数据**\n\n"
        table += "| 指标 | 数值 |\n"
        table += "|------|------|\n"
        table += f"| 社区数量 | {community_count} |\n"
        table += f"| 桥梁股数量 | {bridge_count} |\n"
        table += f"| 模块度 | {modularity:.4f} |\n"
        table += f"| 高枢纽股票 | {hub_stats['高']} |\n"

        # 状态判断
        if modularity >= 0.4:
            status = "✅ 正常（模块度高，社区分化明显）"
        elif modularity >= 0.2:
            status = "⚠️ 关注（模块度适中，市场有一定联动）"
        else:
            status = "🔴 预警（模块度低，市场同涨同跌）"
        table += f"| 状态 | {status} |\n\n"

        # 趋势说明
        table += "**趋势**\n\n"
        if modularity >= 0.4:
            table += "模块度较高，股票分化明显，选股模型有效性高\n\n"
        elif modularity >= 0.2:
            table += "模块度适中，需关注核心枢纽和桥梁股的信号传导\n\n"
        else:
            table += "模块度较低，市场联动性强，系统性风险高\n\n"

        # 预警阈值建议表
        table += "**预警阈值建议**\n\n"
        table += "| 级别 | 阈值 | 操作 |\n"
        table += "|------|------|------|\n"
        table += "| ✅ 正常 | > 0.40 | 社区分化明显，正常操作 |\n"
        table += "| ⚠️ 关注 | 0.20 ~ 0.40 | 关注系统性风险，适度降低仓位 |\n"
        table += "| 🔴 预警 | < 0.20 | 市场同涨同跌，降低仓位 30% |\n\n"

        # 操作建议
        if modularity >= 0.4:
            table += "**操作建议**：市场分化明显，选股模型有效，维持正常策略"
        elif modularity >= 0.2:
            table += "**操作建议**：市场有一定联动，关注桥梁股信号，适度控制仓位"
        else:
            table += "**操作建议**：市场同涨同跌风险高，建议降低仓位，以防御性配置为主"

        return table


# 保留旧的 NetworkFeatureLoader 用于向后兼容
class NetworkFeatureLoader:
    """网络特征加载器 - 保留用于向后兼容

    特征列表（12个）：
    - 中心性特征(5): degree, betweenness, eigenvector, closeness, composite
    - 社区特征(4): community_id, community_size, community_centrality_rank, sector_cohesion
    - MST特征(2): mst_degree, mst_neighbor_sectors
    - 跨社区特征(1): inter_community_ratio

    已移除的二元特征：
    - net_sector_community_match (0/1) -> 替换为 net_sector_cohesion (连续值)
    - net_is_bridge_stock (0/1) -> 替换为 net_inter_community_ratio (连续值)
    - net_systemic_risk_score -> 移除（与 betweenness 高度相关）
    """

    @staticmethod
    def get_feature_names():
        return [
            'net_degree_centrality', 'net_betweenness_centrality',
            'net_eigenvector_centrality', 'net_closeness_centrality',
            'net_composite_centrality', 'net_community_id',
            'net_community_size', 'net_community_centrality_rank',
            'net_sector_cohesion', 'net_mst_degree',
            'net_mst_neighbor_sectors', 'net_inter_community_ratio'
        ]

    @staticmethod
    def get_default_values():
        return {
            'net_degree_centrality': 0.0,
            'net_betweenness_centrality': 0.0,
            'net_eigenvector_centrality': 0.0,
            'net_closeness_centrality': 0.0,
            'net_composite_centrality': 0.0,
            'net_community_id': 'unknown',
            'net_community_size': 0,
            'net_community_centrality_rank': -1,  # -1表示未知社区
            'net_sector_cohesion': 0.0,
            'net_mst_degree': 0,
            'net_mst_neighbor_sectors': 0,
            'net_inter_community_ratio': 0.0
        }

    def __init__(self):
        self._features_cache = None

    def is_available(self):
        return False  # 不再使用文件加载

    def load_features(self):
        return False

    def get_features(self, stock_code):
        return self.get_default_values()


class VolatilityNetworkDensityCalculator:
    """波动率网络密度计算器 - 用于系统性风险预警"""

    def __init__(self):
        self._density_history = []
        self._current_density = None
        self._cache_time = None
        self._cache_ttl = 3600  # 缓存有效期：1小时

    def calculate_volatility_network_density(self, stock_codes, stock_data=None, window=20, threshold=0.5):
        """
        计算波动率网络密度

        参数:
            stock_codes: 股票代码列表
            stock_data: 股票数据字典（可选，如不提供则自动获取）
            window: 波动率计算窗口（天）
            threshold: 相关系数阈值

        返回:
            dict: {
                'current_density': 当前密度,
                'history_30d': 过去30天密度列表,
                'mean': 平均值,
                'std': 标准差,
                'status': 状态（正常/关注/预警/极端）,
                'trend': 趋势描述,
                'suggestion': 操作建议
            }
        """
        # 检查缓存
        if self._cache_time and self._current_density:
            elapsed = (datetime.now() - self._cache_time).total_seconds()
            if elapsed < self._cache_ttl:
                logger.info("使用缓存的波动率网络密度数据")
                return self._current_density

        logger.info("开始计算波动率网络密度...")
        print("  📉 计算波动率网络密度预警...")

        try:
            import networkx as nx
            import numpy as np

            # 获取股票数据
            if stock_data is None:
                from ml_services.stock_network_analysis import fetch_all_stock_data
                stock_data = fetch_all_stock_data(list(stock_codes))

            if not stock_data:
                logger.warning("无法获取股票数据，返回默认值")
                return self._get_default_density()

            # 计算滚动波动率
            vol_data = {}
            for stock_code, df in stock_data.items():
                if 'Return' not in df.columns and 'Close' in df.columns:
                    df['Return'] = df['Close'].pct_change()
                if 'Return' in df.columns:
                    vol_data[stock_code] = df['Return'].rolling(window).std()

            vol_df = pd.DataFrame(vol_data).dropna()

            if len(vol_df.columns) < 2 or len(vol_df) < 30:
                logger.warning("波动率数据不足，返回默认值")
                return self._get_default_density()

            # 计算过去30天的网络密度
            density_history = []
            for i in range(min(30, len(vol_df) - 1)):
                end_idx = len(vol_df) - i
                start_idx = max(0, end_idx - window)
                vol_window = vol_df.iloc[start_idx:end_idx]

                if len(vol_window) < 5:
                    continue

                # 计算相关系数矩阵
                vol_corr = vol_window.corr(method='pearson')

                # 构建网络
                G = nx.Graph()
                stock_list = list(vol_corr.columns)

                for code in stock_list:
                    G.add_node(code)

                # 添加边
                n = len(stock_list)
                edge_count = 0
                for j in range(n):
                    for k in range(j + 1, n):
                        corr_val = vol_corr.iloc[j, k]
                        if abs(corr_val) >= threshold:
                            G.add_edge(stock_list[j], stock_list[k])
                            edge_count += 1

                density = nx.density(G)
                density_history.append(density)

            if not density_history:
                return self._get_default_density()

            # 计算统计值
            current_density = density_history[0]  # 最新一天的密度
            mean_density = np.mean(density_history)
            std_density = np.std(density_history)

            # 判断状态（基于历史数据的统计分布）
            if std_density < 0.001:
                std_density = 0.01  # 防止除零

            deviation = (current_density - mean_density) / std_density

            # 动态计算阈值（基于历史均值 + 标准差）
            # 关注：均值 + 1σ
            # 预警：均值 + 2σ
            # 极端：均值 + 3σ 或 历史最大值
            watch_threshold = mean_density + std_density
            warning_threshold = mean_density + 2 * std_density
            extreme_threshold = min(mean_density + 3 * std_density, max(density_history) * 0.95)

            if current_density >= extreme_threshold:
                status = '🔴🔴 极端'
                status_icon = '🔴🔴'
            elif current_density >= warning_threshold:
                status = '🔴 预警'
                status_icon = '🔴'
            elif current_density >= watch_threshold:
                status = '⚠️ 关注'
                status_icon = '⚠️'
            elif abs(deviation) <= 1:
                status = '✅ 正常'
                status_icon = '✅'
            else:
                status = '⚠️ 异常'
                status_icon = '⚠️'

            # 判断趋势
            if len(density_history) >= 5:
                recent = density_history[:5]
                older = density_history[5:10] if len(density_history) >= 10 else density_history[5:]
                if len(older) > 0:
                    recent_mean = np.mean(recent)
                    older_mean = np.mean(older)
                    if recent_mean > older_mean + 0.02:
                        trend = '📈 密度上升，系统性风险增加'
                    elif recent_mean < older_mean - 0.02:
                        trend = '📉 密度下降，市场分化度提高'
                    else:
                        trend = f'过去30天密度稳定在 {mean_density:.3f} 左右'
                else:
                    trend = f'过去30天密度稳定在 {mean_density:.3f} 左右'
            else:
                trend = '数据不足，无法判断趋势'

            # 操作建议
            if status_icon == '🔴🔴':
                suggestion = '市场高度联动，降低仓位 50%，避免系统性风险'
            elif status_icon == '🔴':
                suggestion = '市场联动性较高，降低仓位 20%，关注系统性风险'
            elif status_icon == '⚠️':
                suggestion = '市场联动性上升，关注系统性风险，谨慎操作'
            else:
                suggestion = '市场状态稳定，维持正常策略'

            result = {
                'current_density': round(current_density, 4),
                'mean': round(mean_density, 4),
                'std': round(std_density, 4),
                'history_30d': [round(d, 4) for d in density_history],
                'status': status,
                'status_icon': status_icon,
                'trend': trend,
                'suggestion': suggestion,
                'thresholds': {
                    'watch': round(watch_threshold, 4),
                    'warning': round(warning_threshold, 4),
                    'extreme': round(extreme_threshold, 4)
                }
            }

            # 更新缓存
            self._current_density = result
            self._cache_time = datetime.now()

            print(f"    ✅ 波动率网络密度: {current_density:.4f} ({status})")
            logger.info(f"波动率网络密度计算完成: {current_density:.4f}")

            return result

        except Exception as e:
            logger.warning(f"波动率网络密度计算失败: {e}")
            print(f"    ⚠️ 波动率网络密度计算失败: {e}")
            return self._get_default_density()

    def _get_default_density(self):
        """返回默认密度值"""
        return {
            'current_density': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'history_30d': [],
            'status': '未知',
            'status_icon': '⚠️',
            'trend': '数据不足',
            'suggestion': '无法判断，维持正常策略',
            'thresholds': {
                'watch': 0.0,
                'warning': 0.0,
                'extreme': 0.0
            }
        }

    def generate_warning_table(self, density_info):
        """
        生成波动率网络密度预警表格（Markdown格式）

        参数:
            density_info: 波动率网络密度信息字典

        返回:
            str: Markdown格式的预警表格
        """
        if not density_info or density_info.get('current_density', 0) == 0:
            return ""

        current = density_info.get('current_density', 0)
        mean = density_info.get('mean', 0)
        std = density_info.get('std', 0)
        status = density_info.get('status', '未知')
        trend = density_info.get('trend', '未知')
        suggestion = density_info.get('suggestion', '')
        thresholds = density_info.get('thresholds', {})

        table = "\n### 📉 波动率网络密度预警\n\n"
        table += "**核心逻辑**：密度高 → 市场进入「同涨同跌」模式 → 个股分化度低 → 选股模型失效 → 降低仓位\n\n"
        table += "**过去30天数据**\n\n"
        table += "| 指标 | 数值 |\n"
        table += "|------|------|\n"
        table += f"| 平均值 | {mean:.4f} |\n"
        table += f"| 标准差 | {std:.4f} |\n"
        table += f"| 当前值 | {current:.4f} |\n"
        table += f"| 状态 | {status} |\n"
        table += f"| 趋势 | {trend} |\n\n"

        table += "**预警阈值**（基于历史数据动态计算：均值 + N×标准差）\n\n"
        table += "| 级别 | 阈值 | 计算方式 | 操作 |\n"
        table += "|------|------|----------|------|\n"
        table += f"| ⚠️ 关注 | > {thresholds.get('watch', 0):.4f} | μ + 1σ | 关注系统性风险 |\n"
        table += f"| 🔴 预警 | > {thresholds.get('warning', 0):.4f} | μ + 2σ | 降低仓位 20% |\n"
        table += f"| 🔴🔴 极端 | > {thresholds.get('extreme', 0):.4f} | μ + 3σ | 降低仓位 50% |\n\n"
        table += f"**操作建议**：{suggestion}\n"

        return table


# 全局网络洞察计算器实例
_network_calculator = None

def get_network_calculator():
    """获取网络洞察计算器实例"""
    global _network_calculator
    if _network_calculator is None:
        _network_calculator = NetworkInsightCalculator()
    return _network_calculator


# 全局波动率密度计算器实例
_volatility_density_calculator = None

def get_volatility_density_calculator():
    """获取波动率网络密度计算器实例"""
    global _volatility_density_calculator
    if _volatility_density_calculator is None:
        _volatility_density_calculator = VolatilityNetworkDensityCalculator()
    return _volatility_density_calculator
