#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股网络分析模块

基于港股网络分析模块，适配A股股票池：
- 使用滚动窗口计算社区ID（防止时序泄漏）
- 构建MST/PMFG网络
- 计算中心性、结构洞等特征
- 输出到 data/a_stock_network_features/

创建时间：2026-07-17
"""

import os
import sys
import json
import argparse
import warnings
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import (
    A_STOCK_TRAINING_LIST,
    A_STOCK_SECTOR_MAPPING,
    A_STOCK_SECTOR_NAME_MAPPING,
    A_STOCK_NETWORK_FEATURES_DIR,
)

logger = logging.getLogger(__name__)

# 创建输出目录
os.makedirs(A_STOCK_NETWORK_FEATURES_DIR, exist_ok=True)

# 中心性权重
CENTRALITY_WEIGHTS = {
    'degree': 0.2,
    'betweenness': 0.3,
    'eigenvector': 0.3,
    'closeness': 0.2
}


def get_stock_list():
    """获取A股股票列表"""
    return list(A_STOCK_TRAINING_LIST.keys())


def get_stock_name(stock_code):
    """获取股票名称"""
    return A_STOCK_TRAINING_LIST.get(stock_code, stock_code)


def get_stock_sector(stock_code):
    """获取股票板块"""
    return A_STOCK_SECTOR_MAPPING.get(stock_code, {}).get('sector', 'unknown')


def fetch_stock_data(stock_codes, period_days=500):
    """
    获取多只股票数据

    Args:
        stock_codes: 股票代码列表
        period_days: 获取天数

    Returns:
        dict: {股票代码: DataFrame}
    """
    from data_services.a_stock_data import get_a_stock_data

    print(f"📊 正在获取 {len(stock_codes)} 只A股数据...")
    stock_data = {}

    for i, code in enumerate(stock_codes):
        df = get_a_stock_data(code, period_days=period_days, use_cache=True)
        if df is not None and len(df) >= 100:
            df['Return'] = df['Close'].pct_change()
            stock_data[code] = df
        else:
            logger.warning(f"股票 {code} 数据不足")

        if (i + 1) % 10 == 0:
            print(f"  已获取 {i + 1}/{len(stock_codes)} 只股票...")

    print(f"  ✅ 成功获取 {len(stock_data)} 只股票数据")
    return stock_data


def build_returns_dataframe(stock_data):
    """构建收益率 DataFrame"""
    returns_dict = {}
    for stock_code, df in stock_data.items():
        returns_dict[stock_code] = df['Return']

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    return returns_df


def compute_correlation_matrix(returns_df):
    """计算相关性矩阵"""
    return returns_df.corr(method='pearson')


def build_distance_matrix(corr_matrix):
    """
    构建 Mantegna 距离矩阵
    d_ij = sqrt(2 * (1 - rho_ij))
    """
    distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 0)
    return distance_matrix


def build_minimum_spanning_tree(distance_matrix, stock_codes):
    """
    构建最小生成树（MST）
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = len(stock_codes)
    sparse_dist = csr_matrix(distance_matrix)
    mst_sparse = minimum_spanning_tree(sparse_dist)

    mst_dense = mst_sparse.toarray()
    G = nx.Graph()

    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    for i in range(n):
        for j in range(i + 1, n):
            if mst_dense[i, j] > 0 or mst_dense[j, i] > 0:
                weight = max(mst_dense[i, j], mst_dense[j, i])
                G.add_edge(stock_codes[i], stock_codes[j], weight=weight)

    return G


def calculate_centrality_metrics(G):
    """
    计算中心性指标
    """
    centrality_dict = {}

    # 度中心性
    degree_centrality = nx.degree_centrality(G)

    # 介数中心性
    betweenness_centrality = nx.betweenness_centrality(G)

    # 特征向量中心性
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {n: 0 for n in G.nodes()}

    # 接近中心性
    closeness_centrality = nx.closeness_centrality(G)

    # 综合中心性
    for node in G.nodes():
        composite = (
            degree_centrality.get(node, 0) * CENTRALITY_WEIGHTS['degree'] +
            betweenness_centrality.get(node, 0) * CENTRALITY_WEIGHTS['betweenness'] +
            eigenvector_centrality.get(node, 0) * CENTRALITY_WEIGHTS['eigenvector'] +
            closeness_centrality.get(node, 0) * CENTRALITY_WEIGHTS['closeness']
        )

        centrality_dict[node] = {
            'degree': degree_centrality.get(node, 0),
            'betweenness': betweenness_centrality.get(node, 0),
            'eigenvector': eigenvector_centrality.get(node, 0),
            'closeness': closeness_centrality.get(node, 0),
            'composite': composite
        }

    return centrality_dict


def detect_communities(G):
    """
    检测社区（使用 Louvain 算法或贪婪算法）
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        communities = partition
        modularity = community_louvain.modularity(partition, G)
    except ImportError:
        # 降级使用贪婪算法
        from networkx.algorithms.community import greedy_modularity_communities
        communities_list = greedy_modularity_communities(G)
        communities = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                communities[node] = i
        modularity = nx.algorithms.community.quality.modularity(G, communities_list)

    return communities, modularity


def calculate_structural_holes(G, communities):
    """
    计算结构洞特征
    """
    structural_holes = {}

    for node in G.nodes():
        # 约束度
        try:
            constraint = nx.constraint(G, node)
        except:
            constraint = 1.0

        # 有效规模
        neighbors = set(G.neighbors(node))
        if len(neighbors) > 0:
            subgraph = G.subgraph(neighbors)
            effective_size = len(neighbors) - (2 * subgraph.number_of_edges() / len(neighbors))
        else:
            effective_size = 0

        # 局部聚类系数
        clustering = nx.clustering(G, node)

        structural_holes[node] = {
            'constraint': float(constraint),
            'effective_size': float(effective_size),
            'local_clustering': float(clustering)
        }

    return structural_holes


def calculate_inter_community_ratio(G, communities):
    """
    计算跨社区连接比例
    """
    inter_community_ratio = {}

    for node in G.nodes():
        node_community = communities.get(node, -1)
        neighbors = list(G.neighbors(node))

        if len(neighbors) > 0:
            inter_neighbors = sum(1 for n in neighbors if communities.get(n, -1) != node_community)
            ratio = inter_neighbors / len(neighbors)
        else:
            ratio = 0.0

        inter_community_ratio[node] = ratio

    return inter_community_ratio


def calculate_sector_cohesion(G):
    """
    计算板块内聚度
    """
    sector_cohesion = {}

    for node in G.nodes():
        node_sector = get_stock_sector(node)
        neighbors = list(G.neighbors(node))

        if len(neighbors) > 0:
            same_sector_neighbors = sum(1 for n in neighbors if get_stock_sector(n) == node_sector)
            cohesion = same_sector_neighbors / len(neighbors)
        else:
            cohesion = 0.0

        sector_cohesion[node] = cohesion

    return sector_cohesion


def calculate_community_centrality_rank(G, centrality_dict, communities):
    """
    计算社区内中心性排名
    """
    community_rank = {}

    # 按社区分组
    community_nodes = defaultdict(list)
    for node, comm_id in communities.items():
        community_nodes[comm_id].append(node)

    # 计算每个社区内的排名
    for comm_id, nodes in community_nodes.items():
        # 按综合中心性排序
        nodes_with_centrality = [(n, centrality_dict.get(n, {}).get('composite', 0)) for n in nodes]
        nodes_with_centrality.sort(key=lambda x: x[1], reverse=True)

        for rank, (node, _) in enumerate(nodes_with_centrality):
            community_rank[node] = rank

    return community_rank


def build_network_features(stock_codes, period_days=500, rolling_window=60):
    """
    构建网络特征（使用滚动窗口防止时序泄漏）

    Args:
        stock_codes: 股票代码列表
        period_days: 总数据天数
        rolling_window: 滚动窗口大小（天）

    Returns:
        dict: {股票代码: 网络特征字典}
    """
    print(f"\n🕸️ 构建A股网络特征...")
    print(f"  股票数量: {len(stock_codes)}")
    print(f"  滚动窗口: {rolling_window} 天")

    # 获取股票数据
    stock_data = fetch_stock_data(stock_codes, period_days)

    if len(stock_data) < 10:
        print("  ⚠️ 股票数据不足，无法构建网络")
        return {}

    # 构建收益率 DataFrame
    returns_df = build_returns_dataframe(stock_data)

    if returns_df.empty:
        print("  ⚠️ 收益率数据为空")
        return {}

    # 使用最近rolling_window天数据构建网络
    returns_window = returns_df.tail(rolling_window)

    print(f"  使用最近 {len(returns_window)} 天数据")

    # 计算相关性矩阵
    corr_matrix = compute_correlation_matrix(returns_window)

    # 构建距离矩阵
    distance_matrix = build_distance_matrix(corr_matrix)

    # 构建MST
    stock_list = list(returns_window.columns)
    G = build_minimum_spanning_tree(distance_matrix, stock_list)

    print(f"  MST: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 计算中心性
    centrality_dict = calculate_centrality_metrics(G)

    # 检测社区
    communities, modularity = detect_communities(G)

    print(f"  社区数量: {len(set(communities.values()))}")
    print(f"  模块度: {modularity:.4f}")

    # 计算结构洞特征
    structural_holes = calculate_structural_holes(G, communities)

    # 计算跨社区连接比例
    inter_community_ratio = calculate_inter_community_ratio(G, communities)

    # 计算板块内聚度
    sector_cohesion = calculate_sector_cohesion(G)

    # 计算社区内中心性排名
    community_rank = calculate_community_centrality_rank(G, centrality_dict, communities)

    # 组装特征
    network_features = {}

    for stock_code in stock_codes:
        if stock_code not in G.nodes():
            # 股票不在网络中，使用默认值
            network_features[stock_code] = {
                'net_degree_centrality': 0.0,
                'net_betweenness_centrality': 0.0,
                'net_eigenvector_centrality': 0.0,
                'net_closeness_centrality': 0.0,
                'net_composite_centrality': 0.0,
                'net_community_id': -1,
                'net_community_size': 0,
                'net_community_centrality_rank': -1,
                'net_sector_cohesion': 0.0,
                'net_mst_degree': 0,
                'net_mst_neighbor_sectors': 0,
                'net_inter_community_ratio': 0.0,
                'net_constraint': 1.0,
                'net_effective_size': 0.0,
                'net_local_clustering': 0.0,
            }
            continue

        # MST度数
        mst_degree = G.degree(stock_code)

        # MST邻居板块数
        neighbors = list(G.neighbors(stock_code))
        neighbor_sectors = len(set(get_stock_sector(n) for n in neighbors))

        # 社区大小
        comm_id = communities.get(stock_code, -1)
        community_size = sum(1 for n in communities.values() if n == comm_id)

        network_features[stock_code] = {
            'net_degree_centrality': centrality_dict.get(stock_code, {}).get('degree', 0),
            'net_betweenness_centrality': centrality_dict.get(stock_code, {}).get('betweenness', 0),
            'net_eigenvector_centrality': centrality_dict.get(stock_code, {}).get('eigenvector', 0),
            'net_closeness_centrality': centrality_dict.get(stock_code, {}).get('closeness', 0),
            'net_composite_centrality': centrality_dict.get(stock_code, {}).get('composite', 0),
            'net_community_id': int(comm_id),
            'net_community_size': community_size,
            'net_community_centrality_rank': community_rank.get(stock_code, -1),
            'net_sector_cohesion': sector_cohesion.get(stock_code, 0),
            'net_mst_degree': mst_degree,
            'net_mst_neighbor_sectors': neighbor_sectors,
            'net_inter_community_ratio': inter_community_ratio.get(stock_code, 0),
            'net_constraint': structural_holes.get(stock_code, {}).get('constraint', 1.0),
            'net_effective_size': structural_holes.get(stock_code, {}).get('effective_size', 0),
            'net_local_clustering': structural_holes.get(stock_code, {}).get('local_clustering', 0),
        }

    print(f"  ✅ 网络特征计算完成: {len(network_features)} 只股票")

    return network_features


def save_network_features(network_features, output_file=None):
    """
    保存网络特征到JSON文件
    """
    if output_file is None:
        output_file = os.path.join(A_STOCK_NETWORK_FEATURES_DIR, 'network_features_for_ml.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_features, f, indent=2, ensure_ascii=False)

    print(f"  💾 网络特征已保存: {output_file}")

    # 同时保存社区ID列表
    community_ids = sorted(set(
        features['net_community_id']
        for features in network_features.values()
        if features['net_community_id'] >= 0
    ))

    community_file = os.path.join(A_STOCK_NETWORK_FEATURES_DIR, 'community_ids.json')
    with open(community_file, 'w') as f:
        json.dump(community_ids, f, indent=2)

    print(f"  💾 社区ID列表已保存: {community_file}")

    return output_file


def generate_network_report(network_features, output_file=None):
    """
    生成网络分析报告
    """
    if output_file is None:
        output_file = os.path.join(A_STOCK_NETWORK_FEATURES_DIR, 'network_report.md')

    # 统计信息
    community_counts = defaultdict(int)
    for features in network_features.values():
        comm_id = features['net_community_id']
        if comm_id >= 0:
            community_counts[comm_id] += 1

    # 中心性排名
    centrality_ranking = sorted(
        network_features.items(),
        key=lambda x: x[1]['net_composite_centrality'],
        reverse=True
    )

    # 生成报告
    report = "# A股网络分析报告\n\n"
    report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## 网络概览\n\n"
    report += f"| 指标 | 数值 |\n"
    report += f"|------|------|\n"
    report += f"| 股票数量 | {len(network_features)} |\n"
    report += f"| 社区数量 | {len(community_counts)} |\n"

    report += "\n## 社区分布\n\n"
    report += "| 社区ID | 股票数量 |\n"
    report += "|--------|----------|\n"
    for comm_id, count in sorted(community_counts.items()):
        report += f"| {comm_id} | {count} |\n"

    report += "\n## 中心性排名（Top 10）\n\n"
    report += "| 排名 | 股票代码 | 名称 | 综合中心性 |\n"
    report += "|------|----------|------|------------|\n"
    for i, (code, features) in enumerate(centrality_ranking[:10], 1):
        name = get_stock_name(code)
        report += f"| {i} | {code} | {name} | {features['net_composite_centrality']:.4f} |\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  📝 网络报告已生成: {output_file}")

    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股网络分析')
    parser.add_argument('--period', type=int, default=500, help='数据天数')
    parser.add_argument('--window', type=int, default=60, help='滚动窗口大小')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')

    args = parser.parse_args()

    # 获取股票列表
    stock_codes = get_stock_list()

    # 构建网络特征
    network_features = build_network_features(
        stock_codes,
        period_days=args.period,
        rolling_window=args.window
    )

    if not network_features:
        print("❌ 网络特征构建失败")
        return

    # 保存网络特征
    save_network_features(network_features, args.output)

    # 生成报告
    generate_network_report(network_features)

    print("\n✅ A股网络分析完成")


if __name__ == '__main__':
    main()