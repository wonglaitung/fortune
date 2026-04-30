#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络特征加载模块 - 从 stock_network_analysis.py 的输出加载跨截面网络特征

12个特征：
- 中心性(5): net_degree_centrality, net_betweenness_centrality,
              net_eigenvector_centrality, net_closeness_centrality,
              net_composite_centrality
- 社区(3): net_community_id, net_community_size, net_sector_community_match
- MST(2): net_mst_degree, net_mst_neighbor_sectors
- 风险(2): net_systemic_risk_score, net_is_bridge_stock

数据源：output/network_features_for_ml.json（由 ml_services/stock_network_analysis.py 生成）

创建时间：2026-04-30
"""

import os
import sys
import json
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 网络特征文件路径
NETWORK_FEATURES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'network_features_for_ml.json'
)


class NetworkFeatureLoader:
    """网络特征加载器 - 从 stock_network_analysis.py 的输出加载跨截面特征"""

    @staticmethod
    def get_feature_names():
        """返回所有12个网络特征名称"""
        return [
            'net_degree_centrality', 'net_betweenness_centrality',
            'net_eigenvector_centrality', 'net_closeness_centrality',
            'net_composite_centrality', 'net_community_id',
            'net_community_size', 'net_sector_community_match',
            'net_mst_degree', 'net_mst_neighbor_sectors',
            'net_systemic_risk_score', 'net_is_bridge_stock'
        ]

    @staticmethod
    def get_numeric_feature_names():
        """返回11个数值型特征（不包括 net_community_id）"""
        return [f for f in NetworkFeatureLoader.get_feature_names()
                if f != 'net_community_id']

    @staticmethod
    def get_categorical_feature_names():
        """返回1个分类特征"""
        return ['net_community_id']

    @staticmethod
    def get_default_values():
        """返回所有特征的默认值（用于缺失股票）"""
        return {
            'net_degree_centrality': 0.0,
            'net_betweenness_centrality': 0.0,
            'net_eigenvector_centrality': 0.0,
            'net_closeness_centrality': 0.0,
            'net_composite_centrality': 0.0,
            'net_community_id': 'unknown',  # 字符串，与其他分类特征一致
            'net_community_size': 0,
            'net_sector_community_match': 0,
            'net_mst_degree': 0,
            'net_mst_neighbor_sectors': 0,
            'net_systemic_risk_score': 0.0,
            'net_is_bridge_stock': 0
        }

    def __init__(self):
        self._features_cache = None  # {stock_code: {feature: value}}

    def is_available(self):
        """检查网络特征文件是否存在"""
        return os.path.exists(NETWORK_FEATURES_FILE)

    def load_features(self):
        """从 JSON 文件加载网络特征（带内部缓存）

        返回:
            bool: 是否成功加载
        """
        if self._features_cache is not None:
            return True

        if not self.is_available():
            logger.warning("网络特征文件不存在: %s", NETWORK_FEATURES_FILE)
            self._features_cache = {}
            return False

        try:
            with open(NETWORK_FEATURES_FILE, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 转换 net_community_id 为字符串，确保 CatBoost 分类特征处理一致
            self._features_cache = {}
            for code, feats in raw_data.items():
                normalized = dict(feats)
                if 'net_community_id' in normalized:
                    normalized['net_community_id'] = str(normalized['net_community_id'])
                self._features_cache[code] = normalized

            logger.info("网络特征加载完成: %d 只股票", len(self._features_cache))
            return True

        except (json.JSONDecodeError, IOError) as e:
            logger.warning("网络特征文件加载失败: %s", e)
            self._features_cache = {}
            return False

    def get_features(self, stock_code):
        """获取指定股票的网络特征字典

        参数:
            stock_code: 股票代码，如 '0005.HK' 或 '0005'

        返回:
            dict: 12个网络特征，股票未找到时返回默认值
        """
        if self._features_cache is None:
            self.load_features()

        # 直接匹配
        if stock_code in self._features_cache:
            return self._features_cache[stock_code]

        # 兼容不带 .HK 后缀的代码
        if not stock_code.endswith('.HK'):
            alt_code = f"{stock_code}.HK"
            if alt_code in self._features_cache:
                return self._features_cache[alt_code]

        # 兼容带 .HK 后缀但缓存中无后缀的情况
        if stock_code.endswith('.HK'):
            alt_code = stock_code.replace('.HK', '')
            if alt_code in self._features_cache:
                return self._features_cache[alt_code]

        # 未找到，返回默认值
        return self.get_default_values()


# 网络特征配置（供模型 FEATURE_CONFIG 引用）
NETWORK_FEATURE_CONFIG = {
    'network_features': NetworkFeatureLoader.get_feature_names(),
    'network_numeric_features': NetworkFeatureLoader.get_numeric_feature_names(),
    'network_categorical_features': NetworkFeatureLoader.get_categorical_feature_names()
}
