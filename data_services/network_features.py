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


# 保留旧的 NetworkFeatureLoader 用于向后兼容
class NetworkFeatureLoader:
    """网络特征加载器 - 保留用于向后兼容"""

    @staticmethod
    def get_feature_names():
        return [
            'net_degree_centrality', 'net_betweenness_centrality',
            'net_eigenvector_centrality', 'net_closeness_centrality',
            'net_composite_centrality', 'net_community_id',
            'net_community_size', 'net_sector_community_match',
            'net_mst_degree', 'net_mst_neighbor_sectors',
            'net_systemic_risk_score', 'net_is_bridge_stock'
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
            'net_sector_community_match': 0,
            'net_mst_degree': 0,
            'net_mst_neighbor_sectors': 0,
            'net_systemic_risk_score': 0.0,
            'net_is_bridge_stock': 0
        }

    def __init__(self):
        self._features_cache = None

    def is_available(self):
        return False  # 不再使用文件加载

    def load_features(self):
        return False

    def get_features(self, stock_code):
        return self.get_default_values()


# 全局计算器实例
_network_calculator = None

def get_network_calculator():
    """获取网络洞察计算器实例"""
    global _network_calculator
    if _network_calculator is None:
        _network_calculator = NetworkInsightCalculator()
    return _network_calculator
