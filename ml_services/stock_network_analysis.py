#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股网络分析 - 基于图论的股票关系网络

分析内容：
1. 网络构建 - MST(最小生成树)、PMFG(平面最大过滤图)、阈值网络、偏相关网络
2. 网络指标 - 中心性(度/介数/特征向量/接近)、拓扑统计、社区检测
3. 应用功能 - 系统性风险识别、分散化组合、领先滞后网络
4. ML特征 - 网络特征导出供CatBoost使用

参考：Mantegna (1999), Onnela et al. (2003)

创建时间：2026-04-30
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csgraph
from scipy.sparse import triu
import yfinance as yf

# 设置 matplotlib 后端（支持无头环境）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx

# Louvain 社区检测（可选依赖，有降级方案）
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("⚠️ python-louvain 未安装，将使用 NetworkX 贪婪模块度算法")

from statsmodels.tsa.stattools import grangercausalitytests

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SECTOR_MAPPING, SECTOR_NAME_MAPPING

RANDOM_SEED = 42
OUTPUT_DIR = "output"

# 板块固定色板（16个板块）
SECTOR_COLORS = {
    'bank': '#1f77b4', 'tech': '#ff7f0e', 'semiconductor': '#2ca02c',
    'ai': '#d62728', 'new_energy': '#9467bd', 'utility': '#8c564b',
    'real_estate': '#e377c2', 'consumer': '#7f7f7f', 'biotech': '#bcbd22',
    'auto': '#17becf', 'exchange': '#aec7e8', 'index': '#ffbb78',
    'energy': '#98df8a', 'environmental': '#ff9896', 'shipping': '#c5b0d5',
    'insurance': '#c49c94'
}

# 中心性综合权重
CENTRALITY_WEIGHTS = {
    'degree': 0.2,
    'betweenness': 0.3,
    'eigenvector': 0.3,
    'closeness': 0.2
}


def get_stock_list():
    """获取股票列表"""
    return list(STOCK_SECTOR_MAPPING.keys())


def get_sector_list():
    """获取板块列表"""
    sectors = set()
    for stock_code, info in STOCK_SECTOR_MAPPING.items():
        sectors.add(info['sector'])
    return sorted(list(sectors))


def get_stock_name(stock_code):
    """获取股票中文名"""
    return STOCK_SECTOR_MAPPING.get(stock_code, {}).get('name', stock_code)


def get_stock_sector(stock_code):
    """获取股票板块"""
    return STOCK_SECTOR_MAPPING.get(stock_code, {}).get('sector', 'unknown')


def fetch_stock_data(stock_code, period="2y"):
    """获取单只股票数据"""
    try:
        ticker = yf.Ticker(stock_code)
        df = ticker.history(period=period, interval="1d")
        if len(df) < 100:
            return None
        df['Return'] = df['Close'].pct_change()
        return df
    except Exception as e:
        print(f"  ⚠️ 获取 {stock_code} 数据失败: {e}")
        return None


def fetch_all_stock_data(stock_list, period="2y"):
    """获取所有股票数据"""
    print(f"📊 正在获取 {len(stock_list)} 只股票数据...")

    stock_data = {}
    for i, stock_code in enumerate(stock_list):
        df = fetch_stock_data(stock_code, period)
        if df is not None:
            stock_data[stock_code] = df
        if (i + 1) % 10 == 0:
            print(f"  已获取 {i + 1}/{len(stock_list)} 只股票...")

    print(f"  ✅ 成功获取 {len(stock_data)} 只股票数据")
    return stock_data


def build_returns_dataframe(stock_data):
    """构建对齐的收益率 DataFrame"""
    returns_dict = {}
    for stock_code, df in stock_data.items():
        returns_dict[stock_code] = df['Return']

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    return returns_df


def compute_correlation_matrices(returns_df):
    """计算 Pearson 和 Spearman 相关系数矩阵"""
    pearson_corr = returns_df.corr(method='pearson')
    spearman_corr = returns_df.corr(method='spearman')
    return pearson_corr, spearman_corr


def build_correlation_distance_matrix(corr_matrix):
    """
    构建 Mantegna 距离矩阵
    d_ij = sqrt(2 * (1 - rho_ij))
    参考：Mantegna (1999)
    """
    n = len(corr_matrix)
    distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))
    # 确保对角线为0
    np.fill_diagonal(distance_matrix, 0)
    # 处理数值误差导致的负值
    distance_matrix = np.maximum(distance_matrix, 0)
    return distance_matrix


# ============================================================
# 网络构建函数
# ============================================================

def build_minimum_spanning_tree(distance_matrix, stock_codes):
    """
    构建最小生成树（MST）
    使用 scipy 稀疏矩阵实现（高效），转 NetworkX Graph
    """
    print("  🌳 构建最小生成树（MST）...")

    n = len(stock_codes)
    # scipy 的 MST 需要 dense 或 sparse 矩阵
    from scipy.sparse import csr_matrix
    sparse_dist = csr_matrix(distance_matrix)
    mst_sparse = csgraph.minimum_spanning_tree(sparse_dist)

    # 转为 NetworkX Graph
    mst_dense = mst_sparse.toarray()
    G = nx.Graph()

    # 添加节点（含板块属性）
    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # 添加边
    for i in range(n):
        for j in range(i + 1, n):
            if mst_dense[i, j] > 0 or mst_dense[j, i] > 0:
                weight = max(mst_dense[i, j], mst_dense[j, i])
                G.add_edge(stock_codes[i], stock_codes[j], weight=weight)

    print(f"    ✅ MST: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def build_planar_maximally_filtered_graph(distance_matrix, stock_codes):
    """
    构建平面最大过滤图（PMFG）
    算法：按距离排序边，逐条添加，保持平面性
    """
    print("  🔷 构建平面最大过滤图（PMFG）...")

    n = len(stock_codes)

    # 收集所有边并按距离排序
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((stock_codes[i], stock_codes[j], distance_matrix[i, j]))
    edges.sort(key=lambda x: x[2])

    # 构建图并逐条添加边
    G = nx.Graph()
    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # PMFG 最大边数为 3*(N-2)
    max_edges = 3 * (n - 2)
    added = 0

    for u, v, w in edges:
        if added >= max_edges:
            break
        G.add_edge(u, v, weight=w)
        # 检查平面性
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            G.remove_edge(u, v)
        else:
            added += 1

    print(f"    ✅ PMFG: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def build_threshold_network(corr_matrix, stock_codes, threshold=0.5):
    """
    构建阈值网络
    保留 |rho| >= threshold 的边
    """
    print(f"  🔗 构建阈值网络（阈值={threshold}）...")

    G = nx.Graph()
    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    n = len(stock_codes)
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                G.add_edge(stock_codes[i], stock_codes[j],
                           weight=corr_val, distance=1 - abs(corr_val))

    print(f"    ✅ 阈值网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def build_partial_correlation_network(returns_df, stock_codes, threshold=0.3):
    """
    构建偏相关网络
    方法：去除市场因子（所有股票平均收益）后，计算残差相关
    """
    print(f"  🔄 构建偏相关网络（阈值={threshold}）...")

    # 市场因子：所有股票平均收益率
    market_factor = returns_df.mean(axis=1)

    # 对每只股票回归去市场因子
    residuals = pd.DataFrame(index=returns_df.index)
    for code in stock_codes:
        if code in returns_df.columns:
            y = returns_df[code].values
            X = np.column_stack([market_factor.values, np.ones(len(market_factor))])
            # OLS 回归
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals[code] = y - X @ beta

    # 计算残差相关矩阵
    partial_corr = residuals.corr()

    # 构建网络
    G = nx.Graph()
    for code in stock_codes:
        if code in partial_corr.columns:
            G.add_node(code,
                       sector=get_stock_sector(code),
                       name=get_stock_name(code))

    codes_in_graph = list(G.nodes())
    n = len(codes_in_graph)
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = partial_corr.loc[codes_in_graph[i], codes_in_graph[j]]
            if abs(corr_val) >= threshold:
                G.add_edge(codes_in_graph[i], codes_in_graph[j],
                           weight=corr_val)

    print(f"    ✅ 偏相关网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def build_volume_correlation_network(stock_data, threshold=0.5, method='change'):
    """
    构建成交量相关性网络

    参数:
    - stock_data: 股票数据字典
    - threshold: 相关系数阈值
    - method: 成交量处理方法
        - 'change': 成交量变化率（推荐，更平稳）
        - 'log': 对数成交量
        - 'ratio': 成交量比率（相对于MA20）

    返回:
    - G: NetworkX 图
    - volume_corr: 成交量相关系数矩阵
    """
    print(f"  📊 构建成交量相关性网络（阈值={threshold}，方法={method}）...")

    volume_data = {}
    for stock_code, df in stock_data.items():
        if 'Volume' not in df.columns:
            continue

        if method == 'change':
            # 成交量变化率（最平稳）
            volume_data[stock_code] = df['Volume'].pct_change()
        elif method == 'log':
            # 对数成交量
            volume_data[stock_code] = np.log(df['Volume'] + 1)
        elif method == 'ratio':
            # 成交量比率（相对于20日均值）
            vol_ma = df['Volume'].rolling(20).mean()
            volume_data[stock_code] = df['Volume'] / vol_ma

    volume_df = pd.DataFrame(volume_data).dropna()

    if len(volume_df.columns) < 2:
        print("    ⚠️ 成交量数据不足")
        return nx.Graph(), pd.DataFrame()

    # 计算相关系数矩阵
    volume_corr = volume_df.corr(method='pearson')

    # 构建网络
    G = nx.Graph()
    stock_codes = list(volume_corr.columns)

    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # 添加边
    n = len(stock_codes)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = volume_corr.iloc[i, j]
            if abs(corr_val) >= threshold:
                G.add_edge(stock_codes[i], stock_codes[j],
                           weight=corr_val,
                           distance=1 - abs(corr_val))
                edge_count += 1

    print(f"    ✅ 成交量网络: {G.number_of_nodes()} 节点, {edge_count} 边")
    return G, volume_corr


def build_momentum_correlation_network(stock_data, horizon=20, threshold=0.5):
    """
    构建动量相关性网络

    参数:
    - stock_data: 股票数据字典
    - horizon: 动量周期（天数）
    - threshold: 相关系数阈值

    返回:
    - G: NetworkX 图
    - momentum_corr: 动量相关系数矩阵
    """
    print(f"  📈 构建动量相关性网络（周期={horizon}天，阈值={threshold}）...")

    momentum_data = {}
    for stock_code, df in stock_data.items():
        if 'Close' not in df.columns:
            continue
        # N日动量（收益率）
        momentum_data[stock_code] = df['Close'].pct_change(horizon)

    momentum_df = pd.DataFrame(momentum_data).dropna()

    if len(momentum_df.columns) < 2:
        print("    ⚠️ 动量数据不足")
        return nx.Graph(), pd.DataFrame()

    # 计算相关系数矩阵
    momentum_corr = momentum_df.corr(method='pearson')

    # 构建网络
    G = nx.Graph()
    stock_codes = list(momentum_corr.columns)

    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # 添加边
    n = len(stock_codes)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = momentum_corr.iloc[i, j]
            if abs(corr_val) >= threshold:
                G.add_edge(stock_codes[i], stock_codes[j],
                           weight=corr_val,
                           distance=1 - abs(corr_val))
                edge_count += 1

    print(f"    ✅ 动量网络: {G.number_of_nodes()} 节点, {edge_count} 边")
    return G, momentum_corr


def build_volatility_correlation_network(stock_data, window=20, threshold=0.5):
    """
    构建波动率相关性网络

    参数:
    - stock_data: 股票数据字典
    - window: 波动率计算窗口
    - threshold: 相关系数阈值

    返回:
    - G: NetworkX 图
    - vol_corr: 波动率相关系数矩阵
    """
    print(f"  📉 构建波动率相关性网络（窗口={window}天，阈值={threshold}）...")

    vol_data = {}
    for stock_code, df in stock_data.items():
        if 'Return' not in df.columns:
            continue
        # 滚动波动率（标准差）
        vol_data[stock_code] = df['Return'].rolling(window).std()

    vol_df = pd.DataFrame(vol_data).dropna()

    if len(vol_df.columns) < 2:
        print("    ⚠️ 波动率数据不足")
        return nx.Graph(), pd.DataFrame()

    # 计算相关系数矩阵
    vol_corr = vol_df.corr(method='pearson')

    # 构建网络
    G = nx.Graph()
    stock_codes = list(vol_corr.columns)

    for code in stock_codes:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # 添加边
    n = len(stock_codes)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = vol_corr.iloc[i, j]
            if abs(corr_val) >= threshold:
                G.add_edge(stock_codes[i], stock_codes[j],
                           weight=corr_val,
                           distance=1 - abs(corr_val))
                edge_count += 1

    print(f"    ✅ 波动率网络: {G.number_of_nodes()} 节点, {edge_count} 边")
    return G, vol_corr


def build_multiplex_network(stock_data, returns_df, thresholds=None):
    """
    构建多层网络（价格 + 成交量 + 波动率）

    参数:
    - stock_data: 股票数据字典
    - returns_df: 收益率 DataFrame
    - thresholds: 各层阈值字典 {'price': 0.5, 'volume': 0.5, 'volatility': 0.5}

    返回:
    - layers: 各层网络字典
    - interlayer_edges: 层间边列表
    """
    print("  🔀 构建多层网络...")

    if thresholds is None:
        thresholds = {'price': 0.5, 'volume': 0.5, 'volatility': 0.5}

    layers = {}

    # 层1: 价格相关性网络
    price_corr = returns_df.corr()
    layers['price'] = build_threshold_network(price_corr, list(price_corr.columns),
                                               thresholds['price'])

    # 层2: 成交量相关性网络
    layers['volume'], _ = build_volume_correlation_network(stock_data, thresholds['volume'])

    # 层3: 波动率相关性网络
    layers['volatility'], _ = build_volatility_correlation_network(stock_data, 20, thresholds['volatility'])

    # 计算层间重叠
    interlayer_edges = []
    layer_names = list(layers.keys())

    for i, layer1 in enumerate(layer_names):
        for layer2 in layer_names[i+1:]:
            G1 = layers[layer1]
            G2 = layers[layer2]
            common_edges = set(G1.edges()) & set(G2.edges())
            interlayer_edges.append({
                'layer1': layer1,
                'layer2': layer2,
                'common_edges': len(common_edges),
                'overlap_ratio': len(common_edges) / max(G1.number_of_edges(), 1)
            })

    print(f"    ✅ 多层网络: {len(layers)} 层")
    for layer, G in layers.items():
        print(f"       - {layer}: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    return layers, interlayer_edges


# ============================================================
# 网络指标函数
# ============================================================

def calculate_centrality_metrics(graph):
    """计算4种中心性指标 + 综合得分"""
    print("  📏 计算中心性指标...")

    if graph.number_of_nodes() == 0:
        return {}

    # 度中心性
    degree_cen = nx.degree_centrality(graph)

    # 介数中心性
    betweenness_cen = nx.betweenness_centrality(graph)

    # 特征向量中心性
    try:
        eigenvector_cen = nx.eigenvector_centrality_numpy(graph, max_iter=500)
    except Exception:
        # 断开图降级为 Katz 中心性
        try:
            eigenvector_cen = nx.katz_centrality_numpy(graph, alpha=0.1)
        except Exception:
            eigenvector_cen = {n: 0 for n in graph.nodes()}

    # 接近中心性
    if nx.is_connected(graph):
        closeness_cen = nx.closeness_centrality(graph)
    else:
        # 对断开图，在各连通分量上计算
        closeness_cen = {}
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            sub_closeness = nx.closeness_centrality(subgraph)
            closeness_cen.update(sub_closeness)

    # 综合得分
    centrality_dict = {}
    for node in graph.nodes():
        d = degree_cen.get(node, 0)
        b = betweenness_cen.get(node, 0)
        e = eigenvector_cen.get(node, 0)
        c = closeness_cen.get(node, 0)
        composite = (CENTRALITY_WEIGHTS['degree'] * d +
                     CENTRALITY_WEIGHTS['betweenness'] * b +
                     CENTRALITY_WEIGHTS['eigenvector'] * e +
                     CENTRALITY_WEIGHTS['closeness'] * c)
        centrality_dict[node] = {
            'degree': d,
            'betweenness': b,
            'eigenvector': e,
            'closeness': c,
            'composite': composite
        }

    print(f"    ✅ 中心性计算完成（{len(centrality_dict)} 只股票）")
    return centrality_dict


def detect_communities(graph, resolution=1.0):
    """
    社区检测：Louvain 算法（优先）或 NetworkX 贪婪模块度
    返回：(partition_dict, modularity_score)
    """
    print("  🏘️ 检测社区结构...")

    if graph.number_of_nodes() == 0:
        return {}, 0.0

    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(
            graph, resolution=resolution, random_state=RANDOM_SEED)
        modularity = community_louvain.modularity(partition, graph)
    else:
        # 降级方案：NetworkX 贪婪模块度
        communities_sets = nx.community.greedy_modularity_communities(graph)
        partition = {}
        for idx, comm in enumerate(communities_sets):
            for node in comm:
                partition[node] = idx
        modularity = nx.community.modularity(graph, communities_sets)

    print(f"    ✅ 检测到 {len(set(partition.values()))} 个社区，模块度={modularity:.4f}")
    return partition, modularity


def calculate_topology_stats(graph):
    """计算网络拓扑统计"""
    if graph.number_of_nodes() == 0:
        return {}

    stats_dict = {
        'node_count': graph.number_of_nodes(),
        'edge_count': graph.number_of_edges(),
        'density': nx.density(graph),
        'connected_components': nx.number_connected_components(graph),
        'transitivity': nx.transitivity(graph),  # 全局聚类系数
        'assortativity': nx.degree_assortativity_coefficient(graph)
                          if graph.number_of_edges() > 0 else 0
    }

    # 平均聚类系数
    try:
        stats_dict['avg_clustering'] = nx.average_clustering(graph)
    except Exception:
        stats_dict['avg_clustering'] = 0

    # 对连通图计算路径相关指标
    if nx.is_connected(graph):
        stats_dict['avg_path_length'] = nx.average_shortest_path_length(graph)
        stats_dict['diameter'] = nx.diameter(graph)
    else:
        # 在最大连通分量上计算
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        try:
            stats_dict['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            stats_dict['diameter'] = nx.diameter(subgraph)
            stats_dict['avg_path_length_note'] = '基于最大连通分量'
        except Exception:
            stats_dict['avg_path_length'] = 0
            stats_dict['diameter'] = 0

    return stats_dict


# ============================================================
# 应用功能
# ============================================================

def identify_systemically_important_stocks(centrality_dict, top_n=10):
    """
    识别系统性重要股票
    综合中心性排名 Top N = "大而不倒"
    """
    if not centrality_dict:
        return []

    # 按综合得分排序
    sorted_stocks = sorted(
        centrality_dict.items(),
        key=lambda x: x[1]['composite'],
        reverse=True
    )

    return sorted_stocks[:top_n]


def identify_bridge_stocks(graph, communities):
    """
    识别跨社区桥梁股票
    桥梁股票：MST邻居属于不同社区的数量 >= 2
    """
    if not communities or graph.number_of_nodes() == 0:
        return []

    bridge_stocks = []
    for node in graph.nodes():
        neighbor_communities = set()
        for neighbor in graph.neighbors(node):
            if neighbor in communities:
                neighbor_communities.add(communities[neighbor])
        # 节点自身社区
        own_community = communities.get(node, -1)
        # 邻居社区数 >= 2 或 邻居中有不同社区的
        other_communities = neighbor_communities - {own_community}
        if len(other_communities) >= 1:
            bridge_stocks.append({
                'stock': node,
                'name': get_stock_name(node),
                'sector': get_stock_sector(node),
                'own_community': own_community,
                'bridge_communities': sorted(list(other_communities)),
                'bridge_count': len(other_communities)
            })

    bridge_stocks.sort(key=lambda x: x['bridge_count'], reverse=True)
    return bridge_stocks


def analyze_community_vs_sector(communities):
    """
    对比 Louvain 社区与官方板块分类
    计算 ARI（调整兰德指数）和 NMI（标准化互信息）
    """
    if not communities:
        return {'ari': 0, 'nmi': 0, 'misaligned_stocks': []}

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    stocks = [s for s in communities.keys() if s in STOCK_SECTOR_MAPPING]
    if len(stocks) < 2:
        return {'ari': 0, 'nmi': 0, 'misaligned_stocks': []}

    # 官方板块标签
    sector_labels = [STOCK_SECTOR_MAPPING[s]['sector'] for s in stocks]
    # 社区标签
    community_labels = [communities[s] for s in stocks]

    ari = adjusted_rand_score(sector_labels, community_labels)
    nmi = normalized_mutual_info_score(sector_labels, community_labels)

    # 找出板块错位股票
    misaligned = []
    for stock in stocks:
        sector = STOCK_SECTOR_MAPPING[stock]['sector']
        comm = communities[stock]
        # 同板块的大多数股票属于哪个社区
        same_sector_stocks = [s for s in stocks
                              if STOCK_SECTOR_MAPPING[s]['sector'] == sector]
        same_sector_comms = [communities[s] for s in same_sector_stocks]
        # 众数社区
        if same_sector_comms:
            from collections import Counter
            comm_counter = Counter(same_sector_comms)
            majority_comm = comm_counter.most_common(1)[0][0]
            if comm != majority_comm:
                misaligned.append({
                    'stock': stock,
                    'name': get_stock_name(stock),
                    'sector': sector,
                    'sector_name': SECTOR_NAME_MAPPING.get(sector, sector),
                    'community': comm,
                    'majority_community': majority_comm
                })

    # 社区组成
    community_composition = defaultdict(list)
    for stock in stocks:
        comm = communities[stock]
        community_composition[comm].append({
            'stock': stock,
            'name': get_stock_name(stock),
            'sector': STOCK_SECTOR_MAPPING[stock]['sector']
        })

    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'misaligned_stocks': misaligned,
        'community_composition': {str(k): v for k, v in community_composition.items()}
    }


def generate_diversification_recommendations(mst_graph, communities, centrality_dict,
                                              portfolio_size=10):
    """
    分散化组合推荐
    策略：每社区选1只低介数股票，最大化MST距离
    """
    if not communities or not centrality_dict or mst_graph.number_of_nodes() == 0:
        return {'recommended_stocks': [], 'diversification_score': 0}

    # 按社区分组
    community_stocks = defaultdict(list)
    for stock, comm in communities.items():
        if stock in centrality_dict:
            community_stocks[comm].append(stock)

    # 每社区选介数最低的股票（独立性最强）
    selected = []
    for comm in sorted(community_stocks.keys()):
        stocks = community_stocks[comm]
        # 按介数排序，选最低的
        stocks.sort(key=lambda s: centrality_dict[s].get('betweenness', 0))
        selected.append({
            'stock': stocks[0],
            'name': get_stock_name(stocks[0]),
            'sector': get_stock_sector(stocks[0]),
            'community': comm,
            'betweenness': centrality_dict[stocks[0]].get('betweenness', 0)
        })

    # 按社区大小排序，优先大社区
    community_sizes = {comm: len(v) for comm, v in community_stocks.items()}
    selected.sort(key=lambda x: community_sizes.get(x['community'], 0), reverse=True)

    # 限制组合大小
    selected = selected[:portfolio_size]

    # 计算分散化得分：选中股票在MST中的平均距离
    total_dist = 0
    pair_count = 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            s1 = selected[i]['stock']
            s2 = selected[j]['stock']
            try:
                dist = nx.shortest_path_length(mst_graph, s1, s2, weight='weight')
                total_dist += dist
                pair_count += 1
            except nx.NetworkXNoPath:
                pass

    avg_dist = total_dist / pair_count if pair_count > 0 else 0
    # 归一化到 [0, 1]（经验值：MST 最大距离约 4-6）
    div_score = min(avg_dist / 5.0, 1.0)

    # 覆盖统计
    sectors_covered = len(set(s['sector'] for s in selected))
    communities_covered = len(set(s['community'] for s in selected))

    return {
        'recommended_stocks': selected,
        'diversification_score': round(div_score, 4),
        'avg_mst_distance': round(avg_dist, 4),
        'sectors_covered': sectors_covered,
        'communities_covered': communities_covered,
        'total_sectors': len(set(get_stock_sector(s['stock']) for s in selected)),
        'total_communities': len(set(s['community'] for s in selected))
    }


def build_lead_lag_network(stock_data, max_lag=5, use_all_stocks=False):
    """
    构建领先滞后有向网络
    使用 Granger 因果检验，边从领先股票指向滞后股票
    """
    print(f"  🔀 构建领先滞后网络（max_lag={max_lag}）...")

    returns_dict = {}
    for stock_code, df in stock_data.items():
        returns_dict[stock_code] = df['Return']
    returns_df = pd.DataFrame(returns_dict).dropna()

    # 选择股票
    if use_all_stocks:
        test_stocks = list(returns_df.columns)
    else:
        # 每板块选1只代表
        sector_rep = {}
        for code in returns_df.columns:
            sector = get_stock_sector(code)
            if sector not in sector_rep:
                sector_rep[sector] = code
        test_stocks = list(sector_rep.values())

    print(f"    使用 {len(test_stocks)} 只股票进行 Granger 检验...")

    G = nx.DiGraph()
    for code in test_stocks:
        G.add_node(code,
                   sector=get_stock_sector(code),
                   name=get_stock_name(code))

    # Granger 因果检验
    for i, stock1 in enumerate(test_stocks):
        for j, stock2 in enumerate(test_stocks):
            if i != j and stock1 in returns_df.columns and stock2 in returns_df.columns:
                try:
                    test_data = returns_df[[stock2, stock1]].values
                    result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                    for lag in range(1, max_lag + 1):
                        p_value = result[lag][0]['ssr_ftest'][1]
                        if p_value < 0.05:
                            # 边权 = -log(p)，权重越大越显著
                            weight = -np.log10(p_value)
                            G.add_edge(stock1, stock2,
                                       lag=lag, p_value=p_value,
                                       weight=weight)
                            break  # 只保留最显著的滞后期
                except Exception:
                    continue

    print(f"    ✅ 领先滞后网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


# ============================================================
# 动态网络分析
# ============================================================

def analyze_network_evolution(stock_data, window_days=120, step_days=20):
    """滚动窗口网络演变分析"""
    print(f"  📈 滚动窗口分析（窗口={window_days}天，步长={step_days}天）...")

    # 构建收益率 DataFrame
    all_returns = {}
    for code, df in stock_data.items():
        all_returns[code] = df['Return']
    returns_df = pd.DataFrame(all_returns).dropna()

    if len(returns_df) < window_days:
        print("    ⚠️ 数据不足，无法进行滚动窗口分析")
        return []

    evolution = []
    start_idx = 0

    while start_idx + window_days <= len(returns_df):
        window_data = returns_df.iloc[start_idx:start_idx + window_days]
        end_date = window_data.index[-1]

        # 计算相关矩阵和距离矩阵
        corr = window_data.corr()
        stock_codes = list(corr.columns)
        dist = build_correlation_distance_matrix(corr)

        # 构建 MST
        mst = build_minimum_spanning_tree(dist, stock_codes)

        # 计算指标
        topology = calculate_topology_stats(mst)
        centrality = calculate_centrality_metrics(mst)
        partition, mod = detect_communities(mst)

        # Top 3 中心性股票
        top_centrality = sorted(
            centrality.items(), key=lambda x: x[1]['composite'], reverse=True)[:3]
        top_names = [get_stock_name(s) for s, _ in top_centrality]

        evolution.append({
            'date': end_date.strftime('%Y-%m-%d'),
            'topology': topology,
            'modularity': mod,
            'community_count': len(set(partition.values())) if partition else 0,
            'top_centrality_stocks': top_names
        })

        start_idx += step_days

    print(f"    ✅ 完成 {len(evolution)} 个窗口分析")
    return evolution


def calculate_network_stability(evolution):
    """计算网络稳定性指标（相邻窗口间）"""
    if len(evolution) < 2:
        return {'avg_jaccard': 0, 'stability_timeline': []}

    stability_timeline = []

    for i in range(1, len(evolution)):
        # 拓扑变化率
        prev_edges = evolution[i - 1]['topology'].get('edge_count', 0)
        curr_edges = evolution[i]['topology'].get('edge_count', 0)
        edge_change = abs(curr_edges - prev_edges) / max(prev_edges, 1)

        # 聚类系数变化
        prev_cc = evolution[i - 1]['topology'].get('avg_clustering', 0)
        curr_cc = evolution[i]['topology'].get('avg_clustering', 0)
        cc_change = abs(curr_cc - prev_cc)

        # 社区数变化
        prev_comm = evolution[i - 1]['community_count']
        curr_comm = evolution[i]['community_count']
        comm_change = abs(curr_comm - prev_comm)

        stability_timeline.append({
            'date': evolution[i]['date'],
            'edge_change_rate': round(edge_change, 4),
            'clustering_change': round(cc_change, 4),
            'community_change': comm_change
        })

    avg_edge_change = np.mean([s['edge_change_rate'] for s in stability_timeline])
    avg_cc_change = np.mean([s['clustering_change'] for s in stability_timeline])

    return {
        'avg_edge_change_rate': round(avg_edge_change, 4),
        'avg_clustering_change': round(avg_cc_change, 4),
        'stability_timeline': stability_timeline
    }


# ============================================================
# ML 特征导出
# ============================================================

def export_network_features(centrality_dict, communities, bridge_stocks, stock_codes):
    """
    导出网络特征供 ML 模型使用
    12个特征：中心性(5) + 社区(3) + MST(2) + 风险(2)
    """
    print("  🤖 导出 ML 网络特征...")

    # 桥梁股票集合
    bridge_set = set(b['stock'] for b in bridge_stocks)

    features = {}
    for code in stock_codes:
        c = centrality_dict.get(code, {})
        comm = communities.get(code, -1)

        # 计算社区大小
        community_size = sum(1 for v in communities.values() if v == comm) if communities else 0

        # 判断社区是否匹配板块
        sector = get_stock_sector(code)
        sector_match = 0
        if communities:
            # 同板块的社区众数
            same_sector_comms = [communities[s] for s in communities
                                 if s in STOCK_SECTOR_MAPPING
                                 and STOCK_SECTOR_MAPPING[s]['sector'] == sector]
            if same_sector_comms:
                from collections import Counter
                majority = Counter(same_sector_comms).most_common(1)[0][0]
                sector_match = 1 if comm == majority else 0

        features[code] = {
            'net_degree_centrality': c.get('degree', 0),
            'net_betweenness_centrality': c.get('betweenness', 0),
            'net_eigenvector_centrality': c.get('eigenvector', 0),
            'net_closeness_centrality': c.get('closeness', 0),
            'net_composite_centrality': c.get('composite', 0),
            'net_community_id': comm,
            'net_community_size': community_size,
            'net_sector_community_match': sector_match,
            'net_mst_degree': 0,  # 将在下面填充
            'net_mst_neighbor_sectors': 0,
            'net_systemic_risk_score': c.get('betweenness', 0) * 0.5 + c.get('eigenvector', 0) * 0.5,
            'net_is_bridge_stock': 1 if code in bridge_set else 0
        }

    print(f"    ✅ 导出 {len(features)} 只股票的网络特征（12个）")
    return features


def add_mst_degree_features(features, mst_graph):
    """补充 MST 度数相关特征"""
    for code in features:
        if code in mst_graph:
            features[code]['net_mst_degree'] = mst_graph.degree(code)
            # 邻居板块多样性
            neighbor_sectors = set()
            for neighbor in mst_graph.neighbors(code):
                neighbor_sectors.add(get_stock_sector(neighbor))
            features[code]['net_mst_neighbor_sectors'] = len(neighbor_sectors)
    return features


# ============================================================
# 可视化
# ============================================================

def setup_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei',
                                        'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def get_node_colors(graph):
    """获取节点板块颜色"""
    return [SECTOR_COLORS.get(graph.nodes[n].get('sector', ''), '#cccccc')
            for n in graph.nodes()]


def get_node_sizes(graph, centrality_dict, scale=800, min_size=100):
    """获取节点大小（基于介数中心性）"""
    if not centrality_dict:
        return [300] * graph.number_of_nodes()
    max_b = max(c.get('betweenness', 0) for c in centrality_dict.values()) or 1
    sizes = []
    for n in graph.nodes():
        b = centrality_dict.get(n, {}).get('betweenness', 0)
        sizes.append(min_size + scale * (b / max_b))
    return sizes


def visualize_mst(mst_graph, communities, centrality_dict, output_dir):
    """可视化 MST 网络"""
    setup_chinese_font()
    print("  📊 生成 MST 可视化...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(mst_graph, seed=RANDOM_SEED, weight='weight')

    colors = get_node_colors(mst_graph)
    sizes = get_node_sizes(mst_graph, centrality_dict)

    # 绘制边
    nx.draw_networkx_edges(mst_graph, pos, ax=ax, alpha=0.4, width=1.5)

    # 绘制社区凸包
    if communities:
        comm_groups = defaultdict(list)
        for node, comm in communities.items():
            if node in pos:
                comm_groups[comm].append(pos[node])

        for comm, positions in comm_groups.items():
            if len(positions) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    points = np.array(positions)
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    # 扩展凸包
                    center = hull_points.mean(axis=0)
                    expanded = center + 1.1 * (hull_points - center)
                    polygon = plt.Polygon(expanded, alpha=0.08,
                                          color=f'C{comm % 10}')
                    ax.add_patch(polygon)
                except Exception:
                    pass

    # 绘制节点
    nx.draw_networkx_nodes(mst_graph, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors='white', linewidths=1)

    # 绘制标签
    labels = {n: get_stock_name(n) for n in mst_graph.nodes()}
    nx.draw_networkx_labels(mst_graph, pos, labels, ax=ax, font_size=7,
                            font_family='WenQuanYi Micro Hei')

    # 图例
    legend_patches = []
    seen_sectors = set()
    for n in mst_graph.nodes():
        sector = mst_graph.nodes[n].get('sector', '')
        if sector and sector not in seen_sectors:
            seen_sectors.add(sector)
            sector_name = SECTOR_NAME_MAPPING.get(sector, sector)
            legend_patches.append(mpatches.Patch(
                color=SECTOR_COLORS.get(sector, '#cccccc'), label=sector_name))

    ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
              title='板块', title_fontsize=9)

    ax.set_title('港股最小生成树网络（MST）', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'network_mst.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_pmfg(pmfg_graph, communities, centrality_dict, output_dir):
    """可视化 PMFG 网络"""
    setup_chinese_font()
    print("  📊 生成 PMFG 可视化...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    # 使用 kamada_kawai 布局（基于距离）
    try:
        pos = nx.kamada_kawai_layout(pmfg_graph, weight='weight')
    except Exception:
        pos = nx.spring_layout(pmfg_graph, seed=RANDOM_SEED)

    colors = get_node_colors(pmfg_graph)
    sizes = get_node_sizes(pmfg_graph, centrality_dict)

    nx.draw_networkx_edges(pmfg_graph, pos, ax=ax, alpha=0.2, width=0.8)
    nx.draw_networkx_nodes(pmfg_graph, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in pmfg_graph.nodes()}
    nx.draw_networkx_labels(pmfg_graph, pos, labels, ax=ax, font_size=6,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('港股平面最大过滤图（PMFG）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_pmfg.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_threshold_network(threshold_graph, output_dir):
    """可视化阈值网络（环形布局，按板块分组）"""
    setup_chinese_font()
    print("  📊 生成阈值网络可视化...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))

    # 按板块分组排序节点
    nodes_by_sector = defaultdict(list)
    for n in threshold_graph.nodes():
        sector = get_stock_sector(n)
        nodes_by_sector[sector].append(n)

    sorted_nodes = []
    for sector in sorted(nodes_by_sector.keys()):
        sorted_nodes.extend(nodes_by_sector[sector])

    # 环形布局
    pos = {}
    n = len(sorted_nodes)
    for i, node in enumerate(sorted_nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))

    # 边颜色：正相关蓝色，负相关红色
    edge_colors = []
    for u, v, data in threshold_graph.edges(data=True):
        w = data.get('weight', 0)
        edge_colors.append('#2196F3' if w > 0 else '#F44336')

    node_colors = get_node_colors(threshold_graph)

    nx.draw_networkx_edges(threshold_graph, pos, ax=ax, alpha=0.3,
                           edge_color=edge_colors, width=0.5)
    nx.draw_networkx_nodes(threshold_graph, pos, ax=ax, node_color=node_colors,
                           node_size=200, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in threshold_graph.nodes()}
    nx.draw_networkx_labels(threshold_graph, pos, labels, ax=ax, font_size=5,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('港股阈值网络（环形布局，按板块分组）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_threshold.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_community_comparison(communities, output_dir):
    """可视化社区 vs 板块对比热力图"""
    setup_chinese_font()
    print("  📊 生成社区-板块对比图...")

    if not communities:
        return

    # 构建板块-社区矩阵
    sector_comm_count = defaultdict(lambda: defaultdict(int))
    for stock, comm in communities.items():
        if stock in STOCK_SECTOR_MAPPING:
            sector = STOCK_SECTOR_MAPPING[stock]['sector']
            sector_comm_count[sector][comm] += 1

    sectors = sorted(sector_comm_count.keys())
    all_comms = sorted(set(comm for c in communities.values()))

    matrix = np.zeros((len(sectors), len(all_comms)))
    for i, sector in enumerate(sectors):
        for j, comm in enumerate(all_comms):
            matrix[i, j] = sector_comm_count[sector].get(comm, 0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(all_comms)))
    ax.set_xticklabels([f'社区{c}' for c in all_comms], fontsize=9)
    ax.set_yticks(range(len(sectors)))
    ax.set_yticklabels([SECTOR_NAME_MAPPING.get(s, s) for s in sectors], fontsize=9)

    # 标注数值
    for i in range(len(sectors)):
        for j in range(len(all_comms)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center',
                        fontsize=8, color='white' if matrix[i, j] > matrix.max() / 2 else 'black')

    ax.set_title('官方板块 vs 网络社区 对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('Louvain 社区', fontsize=11)
    ax.set_ylabel('官方板块', fontsize=11)

    plt.colorbar(im, ax=ax, label='股票数量')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_community_vs_sector.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_centrality_ranking(centrality_dict, output_dir, top_n=20):
    """可视化中心性排名条形图"""
    setup_chinese_font()
    print("  📊 生成中心性排名图...")

    if not centrality_dict:
        return

    sorted_stocks = sorted(
        centrality_dict.items(),
        key=lambda x: x[1]['composite'],
        reverse=True
    )[:top_n]

    names = [get_stock_name(s) for s, _ in sorted_stocks]
    scores = [c['composite'] for _, c in sorted_stocks]
    sectors = [get_stock_sector(s) for s, _ in sorted_stocks]
    bar_colors = [SECTOR_COLORS.get(s, '#cccccc') for s in sectors]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bars = ax.barh(range(len(names)), scores, color=bar_colors, edgecolor='white')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, fontfamily='WenQuanYi Micro Hei')
    ax.invert_yaxis()
    ax.set_xlabel('综合中心性得分', fontsize=11)
    ax.set_title('系统性重要股票排名（网络中心性）', fontsize=14, fontweight='bold')

    # 图例
    seen = set()
    legend_patches = []
    for i, s in enumerate(sectors):
        if s not in seen:
            seen.add(s)
            legend_patches.append(mpatches.Patch(
                color=SECTOR_COLORS.get(s, '#cccccc'),
                label=SECTOR_NAME_MAPPING.get(s, s)))
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'network_centrality_ranking.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_lead_lag_network(digraph, output_dir):
    """可视化领先滞后有向网络"""
    setup_chinese_font()
    print("  📊 生成领先滞后网络图...")

    if digraph.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    pos = nx.spring_layout(digraph, seed=RANDOM_SEED)

    # 节点大小 = 出度（领导力）
    out_degrees = dict(digraph.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    node_sizes = [200 + 800 * (out_degrees.get(n, 0) / max(max_out, 1))
                  for n in digraph.nodes()]

    node_colors = get_node_colors(digraph)

    # 边颜色深浅 = 显著性
    edge_weights = [digraph[u][v].get('weight', 1) for u, v in digraph.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.2 + 0.8 * (w / max(max_w, 1)) for w in edge_weights]

    nx.draw_networkx_edges(digraph, pos, ax=ax, alpha=0.4,
                           edge_color=edge_alphas, width=1,
                           arrowsize=10, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(digraph, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in digraph.nodes()}
    nx.draw_networkx_labels(digraph, pos, labels, ax=ax, font_size=7,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('领先滞后网络（Granger因果）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_lead_lag.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_network_evolution(evolution, output_dir):
    """可视化网络演变时序图"""
    setup_chinese_font()
    print("  📊 生成网络演变图...")

    if not evolution:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    dates = [e['date'] for e in evolution]

    # 边数
    ax = axes[0, 0]
    edges = [e['topology'].get('edge_count', 0) for e in evolution]
    ax.plot(dates, edges, 'b-o', markersize=3)
    ax.set_title('MST 边数', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # 聚类系数
    ax = axes[0, 1]
    clustering = [e['topology'].get('avg_clustering', 0) for e in evolution]
    ax.plot(dates, clustering, 'g-o', markersize=3)
    ax.set_title('平均聚类系数', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # 路径长度
    ax = axes[1, 0]
    path_len = [e['topology'].get('avg_path_length', 0) for e in evolution]
    ax.plot(dates, path_len, 'r-o', markersize=3)
    ax.set_title('平均路径长度', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # 模块度
    ax = axes[1, 1]
    modularity = [e.get('modularity', 0) for e in evolution]
    ax.plot(dates, modularity, 'm-o', markersize=3)
    ax.set_title('社区模块度', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    fig.suptitle('港股网络结构演变', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_evolution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_volume_network(volume_graph, output_dir):
    """可视化成交量相关性网络"""
    setup_chinese_font()
    print("  📊 生成成交量网络图...")

    if volume_graph.number_of_nodes() == 0 or volume_graph.number_of_edges() == 0:
        print("    ⚠️ 成交量网络无边，跳过可视化")
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # 使用 spring 布局
    pos = nx.spring_layout(volume_graph, seed=RANDOM_SEED, k=2)

    # 节点大小 = 度中心性
    degrees = dict(volume_graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 700 * (degrees.get(n, 0) / max(max_deg, 1))
                  for n in volume_graph.nodes()]

    node_colors = get_node_colors(volume_graph)

    # 边颜色 = 相关性强度
    edge_weights = [volume_graph[u][v].get('weight', 0.5) for u, v in volume_graph.edges()]
    edge_colors = plt.cm.YlOrRd([w for w in edge_weights])

    nx.draw_networkx_edges(volume_graph, pos, ax=ax, edge_color=edge_colors,
                           width=2, alpha=0.7)
    nx.draw_networkx_nodes(volume_graph, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in volume_graph.nodes()}
    nx.draw_networkx_labels(volume_graph, pos, labels, ax=ax, font_size=8,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('成交量相关性网络（流动性联动）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_volume.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_momentum_network(momentum_graph, output_dir):
    """可视化动量相关性网络"""
    setup_chinese_font()
    print("  📊 生成动量网络图...")

    if momentum_graph.number_of_nodes() == 0 or momentum_graph.number_of_edges() == 0:
        print("    ⚠️ 动量网络无边，跳过可视化")
        return

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    # 使用 spring 布局
    pos = nx.spring_layout(momentum_graph, seed=RANDOM_SEED, k=1.5)

    # 节点大小 = 度中心性
    degrees = dict(momentum_graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [200 + 800 * (degrees.get(n, 0) / max(max_deg, 1))
                  for n in momentum_graph.nodes()]

    node_colors = get_node_colors(momentum_graph)

    # 边颜色 = 相关性强度
    edge_weights = [momentum_graph[u][v].get('weight', 0.5) for u, v in momentum_graph.edges()]
    edge_colors = plt.cm.Greens([w for w in edge_weights])

    nx.draw_networkx_edges(momentum_graph, pos, ax=ax, edge_color=edge_colors,
                           width=1.5, alpha=0.5)
    nx.draw_networkx_nodes(momentum_graph, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in momentum_graph.nodes()}
    nx.draw_networkx_labels(momentum_graph, pos, labels, ax=ax, font_size=7,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('动量相关性网络（趋势同步）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_momentum.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_volatility_network(volatility_graph, output_dir):
    """可视化波动率相关性网络"""
    setup_chinese_font()
    print("  📊 生成波动率网络图...")

    if volatility_graph.number_of_nodes() == 0 or volatility_graph.number_of_edges() == 0:
        print("    ⚠️ 波动率网络无边，跳过可视化")
        return

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    # 使用 spring 布局
    pos = nx.spring_layout(volatility_graph, seed=RANDOM_SEED, k=1.2)

    # 节点大小 = 度中心性
    degrees = dict(volatility_graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [200 + 800 * (degrees.get(n, 0) / max(max_deg, 1))
                  for n in volatility_graph.nodes()]

    node_colors = get_node_colors(volatility_graph)

    # 边颜色 = 相关性强度
    edge_weights = [volatility_graph[u][v].get('weight', 0.5) for u, v in volatility_graph.edges()]
    edge_colors = plt.cm.Reds([w for w in edge_weights])

    nx.draw_networkx_edges(volatility_graph, pos, ax=ax, edge_color=edge_colors,
                           width=1, alpha=0.4)
    nx.draw_networkx_nodes(volatility_graph, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='white', linewidths=0.5)

    labels = {n: get_stock_name(n) for n in volatility_graph.nodes()}
    nx.draw_networkx_labels(volatility_graph, pos, labels, ax=ax, font_size=7,
                            font_family='WenQuanYi Micro Hei')

    ax.set_title('波动率相关性网络（风险传导）', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_volatility.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


def visualize_multiplex_comparison(volume_topo, momentum_topo, volatility_topo,
                                    threshold_topo, output_dir):
    """可视化四种网络对比"""
    setup_chinese_font()
    print("  📊 生成多网络对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    networks = [
        ('价格相关性', threshold_topo, '#3498db'),
        ('成交量相关性', volume_topo, '#e74c3c'),
        ('动量相关性', momentum_topo, '#2ecc71'),
        ('波动率相关性', volatility_topo, '#9b59b6')
    ]

    for idx, (name, topo, color) in enumerate(networks):
        ax = axes[idx // 2, idx % 2]

        metrics = ['边数', '密度', '聚类系数']
        values = [
            topo.get('edge_count', 0) / 10,  # 缩放以便显示
            topo.get('density', 0) * 100,     # 转为百分比
            topo.get('avg_clustering', 0) * 100  # 转为百分比
        ]

        bars = ax.bar(metrics, values, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}网络', fontsize=12, fontweight='bold')
        ax.set_ylabel('数值')

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    fig.suptitle('四种网络拓扑指标对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'network_multiplex_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 已保存: {path}")


# ============================================================
# 报告生成
# ============================================================

def generate_markdown_report(mst_topo, pmfg_topo, threshold_topo, partial_topo,
                              volume_topo, momentum_topo, volatility_topo,
                              centrality_dict, communities, modularity,
                              community_comparison, systemic_stocks, bridge_stocks,
                              diversification, lead_lag_graph,
                              mst_graph, pmfg_graph, volume_graph, momentum_graph, volatility_graph,
                              evolution, stability,
                              threshold, partial_threshold, volume_threshold,
                              stock_count, sector_count):
    """生成 Markdown 分析报告"""
    print("\n📝 生成分析报告...")

    L = []
    L.append("# 港股网络分析报告")
    L.append(f"\n**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"**分析范围**: {stock_count} 只股票，{sector_count} 个板块")
    L.append(f"**网络方法**: MST + PMFG + 阈值网络({threshold}) + 偏相关网络({partial_threshold})")
    L.append("\n---")

    # ========== 一、网络拓扑概览 ==========
    L.append("\n## 一、网络拓扑概览")
    L.append("\n| 指标 | MST | 阈值网络 | 偏相关网络 | 成交量网络 | 动量网络 | 波动率网络 |")
    L.append("|------|-----|----------|-----------|-----------|---------|-----------|")

    def topo_row(key, fmt='.4f'):
        vals = []
        for topo in [mst_topo, threshold_topo, partial_topo, volume_topo, momentum_topo, volatility_topo]:
            v = topo.get(key, 0)
            vals.append(f"{v:{fmt}}" if isinstance(v, float) else str(v))
        return f"| {key} | {' | '.join(vals)} |"

    L.append(topo_row('node_count', 'd'))
    L.append(topo_row('edge_count', 'd'))
    L.append(topo_row('density'))
    L.append(topo_row('avg_clustering'))
    L.append(topo_row('avg_path_length'))
    L.append(topo_row('connected_components', 'd'))
    L.append(topo_row('transitivity'))

    # ========== 二、系统性重要股票 ==========
    L.append("\n## 二、系统性重要股票（网络中心性）")

    L.append("\n### 2.1 综合中心性排名 Top 20")
    L.append("\n| 排名 | 股票 | 板块 | 度 | 介数 | 特征向量 | 接近 | 综合 |")
    L.append("|------|------|------|-----|------|----------|------|------|")

    sorted_cen = sorted(centrality_dict.items(),
                        key=lambda x: x[1]['composite'], reverse=True)
    for i, (code, c) in enumerate(sorted_cen[:20]):
        name = get_stock_name(code)
        sector = get_stock_sector(code)
        sector_name = SECTOR_NAME_MAPPING.get(sector, sector)
        L.append(f"| {i+1} | {name} | {sector_name} | "
                 f"{c['degree']:.4f} | {c['betweenness']:.4f} | "
                 f"{c['eigenvector']:.4f} | {c['closeness']:.4f} | "
                 f"{c['composite']:.4f} |")

    # "大而不倒"
    L.append("\n### 2.2 \"大而不倒\"股票（高介数中心性）")
    L.append("\n这些股票在网络中承担信息传递枢纽角色，其波动可能传导至整个市场：")
    L.append("\n| 排名 | 股票 | 板块 | 介数中心性 | 说明 |")
    L.append("|------|------|------|-----------|------|")

    top_betweenness = sorted(centrality_dict.items(),
                             key=lambda x: x[1]['betweenness'], reverse=True)[:5]
    for i, (code, c) in enumerate(top_betweenness):
        name = get_stock_name(code)
        sector = get_stock_sector(code)
        sector_name = SECTOR_NAME_MAPPING.get(sector, sector)
        desc = "超级枢纽" if c['betweenness'] > 0.3 else "重要枢纽" if c['betweenness'] > 0.15 else "局部枢纽"
        L.append(f"| {i+1} | {name} | {sector_name} | "
                 f"{c['betweenness']:.4f} | {desc} |")

    # ========== 三、社区检测 ==========
    L.append("\n## 三、社区检测分析")

    L.append(f"\n**模块度**: {modularity:.4f}")
    L.append(f"**社区数量**: {len(set(communities.values())) if communities else 0}")

    L.append("\n### 3.1 Louvain 社区构成")
    L.append("\n| 社区ID | 股票数 | 主要板块 | 包含股票 |")
    L.append("|--------|--------|----------|----------|")

    comm_groups = defaultdict(list)
    for stock, comm in communities.items():
        comm_groups[comm].append(stock)

    for comm_id in sorted(comm_groups.keys()):
        stocks = comm_groups[comm_id]
        # 主要板块
        sector_counts = defaultdict(int)
        for s in stocks:
            sector_counts[get_stock_sector(s)] += 1
        main_sector = max(sector_counts, key=sector_counts.get)
        main_sector_name = SECTOR_NAME_MAPPING.get(main_sector, main_sector)
        names = ', '.join(get_stock_name(s) for s in stocks[:8])
        if len(stocks) > 8:
            names += f'... 等{len(stocks)}只'
        L.append(f"| {comm_id} | {len(stocks)} | {main_sector_name} | {names} |")

    # 社区vs板块对齐度
    if community_comparison:
        L.append("\n### 3.2 社区与板块对齐度")
        L.append(f"\n- **ARI（调整兰德指数）**: {community_comparison.get('ari', 0):.4f}")
        L.append(f"- **NMI（标准化互信息）**: {community_comparison.get('nmi', 0):.4f}")

        misaligned = community_comparison.get('misaligned_stocks', [])
        if misaligned:
            L.append(f"\n### 3.3 板块错位股票（{len(misaligned)}只）")
            L.append("\n这些股票的网络行为与其官方板块不一致，可能受其他因素驱动：")
            L.append("\n| 股票 | 官方板块 | 所属社区 | 板块主流社区 |")
            L.append("|------|----------|----------|-------------|")
            for m in misaligned[:10]:
                L.append(f"| {m['name']} | {m['sector_name']} | "
                         f"社区{m['community']} | 社区{m['majority_community']} |")

    # 桥梁股票
    if bridge_stocks:
        L.append(f"\n### 3.4 跨社区桥梁股票（{len(bridge_stocks)}只）")
        L.append("\n这些股票连接不同社区，是风险传导的关键节点：")
        L.append("\n| 股票 | 板块 | 连接社区数 | 说明 |")
        L.append("|------|------|-----------|------|")
        for b in bridge_stocks[:10]:
            sector_name = SECTOR_NAME_MAPPING.get(b['sector'], b['sector'])
            L.append(f"| {b['name']} | {sector_name} | "
                     f"{b['bridge_count']} | 连接社区 {b['bridge_communities']} |")

    # ========== 四、MST 结构 ==========
    L.append("\n## 四、MST 结构分析")

    if mst_graph.number_of_nodes() > 0:
        # 中心节点（度最高的）
        mst_degrees = dict(mst_graph.degree())
        hub_nodes = sorted(mst_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

        L.append("\n### 4.1 MST 核心枢纽节点")
        L.append("\n| 排名 | 股票 | 板块 | MST度数 | 说明 |")
        L.append("|------|------|------|---------|------|")
        for i, (code, deg) in enumerate(hub_nodes):
            name = get_stock_name(code)
            sector_name = SECTOR_NAME_MAPPING.get(get_stock_sector(code), get_stock_sector(code))
            L.append(f"| {i+1} | {name} | {sector_name} | {deg} | "
                     f"{'核心枢纽' if deg >= 4 else '次级枢纽'} |")

        # MST 子树与板块
        L.append("\n### 4.2 MST 度分布")
        degree_dist = defaultdict(int)
        for d in mst_degrees.values():
            degree_dist[d] += 1
        L.append("\n| 度数 | 节点数 |")
        L.append("|------|--------|")
        for d in sorted(degree_dist.keys()):
            L.append(f"| {d} | {degree_dist[d]} |")

    # ========== 五、分散化建议 ==========
    L.append("\n## 五、投资组合分散化建议")

    if diversification and diversification.get('recommended_stocks'):
        recs = diversification['recommended_stocks']
        L.append(f"\n### 5.1 推荐分散化组合（{len(recs)}只股票）")
        L.append("\n| 排名 | 股票 | 板块 | 社区 | 介数 | 分散化贡献 |")
        L.append("|------|------|------|------|------|----------|")
        for i, r in enumerate(recs):
            sector_name = SECTOR_NAME_MAPPING.get(r['sector'], r['sector'])
            L.append(f"| {i+1} | {r['name']} | {sector_name} | "
                     f"社区{r['community']} | {r['betweenness']:.4f} | "
                     f"低介数=高独立性 |")

        L.append(f"\n### 5.2 分散化评分")
        L.append(f"- **分散化得分**: {diversification.get('diversification_score', 0):.4f}（0-1，越高越好）")
        L.append(f"- **平均MST距离**: {diversification.get('avg_mst_distance', 0):.4f}")
        L.append(f"- **板块覆盖**: {diversification.get('sectors_covered', 0)} 个")
        L.append(f"- **社区覆盖**: {diversification.get('communities_covered', 0)} 个")

    # ========== 六、领先滞后 ==========
    if lead_lag_graph and lead_lag_graph.number_of_edges() > 0:
        L.append("\n## 六、领先滞后网络")

        # 出度最高的领导股票
        out_degrees = dict(lead_lag_graph.out_degree())
        leaders = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        L.append("\n### 6.1 领导力排名（出度最高）")
        L.append("\n| 排名 | 股票 | 板块 | 领先股票数 |")
        L.append("|------|------|------|-----------|")
        for i, (code, deg) in enumerate(leaders):
            name = get_stock_name(code)
            sector_name = SECTOR_NAME_MAPPING.get(get_stock_sector(code), get_stock_sector(code))
            L.append(f"| {i+1} | {name} | {sector_name} | {deg} |")

        # 最显著的领先滞后关系
        L.append("\n### 6.2 最显著领先滞后关系")
        L.append("\n| 领先股票 | 滞后股票 | 滞后天数 | p值 |")
        L.append("|----------|----------|----------|-----|")
        edges_sorted = sorted(lead_lag_graph.edges(data=True),
                              key=lambda x: x[2].get('p_value', 1))
        for u, v, data in edges_sorted[:15]:
            L.append(f"| {get_stock_name(u)} | {get_stock_name(v)} | "
                     f"{data.get('lag', '?')}天 | {data.get('p_value', 1):.4f} |")

    # ========== 七、动态分析 ==========
    if evolution:
        L.append("\n## 七、动态网络分析")

        L.append("\n### 7.1 网络结构演变")
        L.append("\n| 日期 | 边数 | 聚类系数 | 路径长度 | 社区数 | 模块度 | 中心股票 |")
        L.append("|------|------|----------|----------|--------|--------|----------|")
        for e in evolution:
            top_names = ', '.join(e.get('top_centrality_stocks', [])[:3])
            L.append(f"| {e['date']} | {e['topology'].get('edge_count', 0)} | "
                     f"{e['topology'].get('avg_clustering', 0):.4f} | "
                     f"{e['topology'].get('avg_path_length', 0):.4f} | "
                     f"{e['community_count']} | {e['modularity']:.4f} | "
                     f"{top_names} |")

        if stability and stability.get('stability_timeline'):
            L.append("\n### 7.2 网络稳定性")
            L.append(f"\n- **平均边变化率**: {stability.get('avg_edge_change_rate', 0):.4f}")
            L.append(f"- **平均聚类系数变化**: {stability.get('avg_clustering_change', 0):.4f}")

    # ========== 七B、成交量/动量/波动率网络分析 ==========
    L.append("\n## 七B、多维度网络分析")

    # 成交量网络
    L.append("\n### 7B.1 成交量相关性网络")
    L.append("\n成交量网络反映股票间的流动性联动关系：")
    L.append(f"\n- **节点数**: {volume_topo.get('node_count', 0)}")
    L.append(f"- **边数**: {volume_topo.get('edge_count', 0)}")
    L.append(f"- **密度**: {volume_topo.get('density', 0):.4f}")
    L.append(f"- **平均聚类系数**: {volume_topo.get('avg_clustering', 0):.4f}")
    L.append("\n**解读**：成交量相关性高的股票，资金流入/流出同步，可能受相同因素驱动。")

    # 动量网络
    L.append("\n### 7B.2 动量相关性网络")
    L.append("\n动量网络反映股票间的趋势同步性：")
    L.append(f"\n- **节点数**: {momentum_topo.get('node_count', 0)}")
    L.append(f"- **边数**: {momentum_topo.get('edge_count', 0)}")
    L.append(f"- **密度**: {momentum_topo.get('density', 0):.4f}")
    L.append(f"- **平均聚类系数**: {momentum_topo.get('avg_clustering', 0):.4f}")
    L.append("\n**解读**：动量相关性高的股票，趋势方向一致，适合趋势跟踪策略。")

    # 波动率网络
    L.append("\n### 7B.3 波动率相关性网络")
    L.append("\n波动率网络反映股票间的风险传导关系：")
    L.append(f"\n- **节点数**: {volatility_topo.get('node_count', 0)}")
    L.append(f"- **边数**: {volatility_topo.get('edge_count', 0)}")
    L.append(f"- **密度**: {volatility_topo.get('density', 0):.4f}")
    L.append(f"- **平均聚类系数**: {volatility_topo.get('avg_clustering', 0):.4f}")
    L.append("\n**解读**：波动率相关性高的股票，风险传导性强，需警惕系统性风险。")

    # 网络对比
    L.append("\n### 7B.4 四种网络对比")
    L.append("\n| 网络类型 | 边数 | 密度 | 聚类系数 | 应用场景 |")
    L.append("|---------|------|------|----------|----------|")
    L.append(f"| 价格相关性 | {threshold_topo.get('edge_count', 0)} | {threshold_topo.get('density', 0):.4f} | {threshold_topo.get('avg_clustering', 0):.4f} | 系统性风险 |")
    L.append(f"| 成交量相关性 | {volume_topo.get('edge_count', 0)} | {volume_topo.get('density', 0):.4f} | {volume_topo.get('avg_clustering', 0):.4f} | 流动性风险 |")
    L.append(f"| 动量相关性 | {momentum_topo.get('edge_count', 0)} | {momentum_topo.get('density', 0):.4f} | {momentum_topo.get('avg_clustering', 0):.4f} | 趋势策略 |")
    L.append(f"| 波动率相关性 | {volatility_topo.get('edge_count', 0)} | {volatility_topo.get('density', 0):.4f} | {volatility_topo.get('avg_clustering', 0):.4f} | 风险传导 |")

    # ========== 八、关键发现 ==========
    L.append("\n## 八、关键发现")

    findings = []

    # 核心枢纽
    if systemic_stocks:
        top_name = get_stock_name(systemic_stocks[0][0])
        findings.append(f"- **核心枢纽**: {top_name} 综合中心性最高，是网络最关键节点")

    # 隐藏群落
    if community_comparison and community_comparison.get('misaligned_stocks'):
        n = len(community_comparison['misaligned_stocks'])
        findings.append(f"- **隐藏群落**: 发现 {n} 只板块错位股票，网络行为与官方板块不一致")

    # 系统性风险
    if bridge_stocks:
        top_bridge = bridge_stocks[0]
        findings.append(f"- **风险传导**: {top_bridge['name']} 连接 {top_bridge['bridge_count']} 个社区，"
                       f"是风险跨板块传导的关键通道")

    # 分散化
    if diversification and diversification.get('diversification_score', 0) > 0:
        score = diversification['diversification_score']
        findings.append(f"- **分散化启示**: 推荐组合分散化得分 {score:.2f}，"
                       f"覆盖 {diversification.get('sectors_covered', 0)} 个板块 "
                       f"{diversification.get('communities_covered', 0)} 个社区")

    # ARI/NMI
    if community_comparison:
        ari = community_comparison.get('ari', 0)
        nmi = community_comparison.get('nmi', 0)
        if ari < 0.5:
            findings.append(f"- **板块重构**: ARI={ari:.2f} 表明网络社区与官方板块差异较大，"
                           f"网络结构揭示隐藏关联")

    for f in findings:
        L.append(f)

    L.append("\n---")
    L.append(f"\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    report = "\n".join(L)
    return report


def save_json_results(mst_topo, pmfg_topo, threshold_topo, partial_topo,
                       centrality_dict, communities, modularity,
                       community_comparison, systemic_stocks, bridge_stocks,
                       diversification, lead_lag_graph,
                       evolution, stability, ml_features,
                       threshold, partial_threshold):
    """保存 JSON 格式结果"""
    print("\n💾 保存 JSON 结果...")

    # 格式化中心性
    centrality_out = {}
    for code, c in centrality_dict.items():
        centrality_out[code] = {
            'name': get_stock_name(code),
            'sector': get_stock_sector(code),
            **{k: round(v, 6) for k, v in c.items()}
        }

    # 格式化领先滞后
    lead_lag_out = {}
    if lead_lag_graph:
        for u, v, data in lead_lag_graph.edges(data=True):
            key = f"{get_stock_name(u)}->{get_stock_name(v)}"
            lead_lag_out[key] = {
                'leader': u, 'leader_name': get_stock_name(u),
                'follower': v, 'follower_name': get_stock_name(v),
                'lag': data.get('lag', 0),
                'p_value': round(data.get('p_value', 1), 6)
            }

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold': threshold,
        'partial_threshold': partial_threshold,
        'network_construction': {
            'mst': mst_topo,
            'pmfg': pmfg_topo,
            'threshold_network': {**threshold_topo, 'threshold': threshold},
            'partial_corr_network': {**partial_topo, 'threshold': partial_threshold}
        },
        'centrality': centrality_out,
        'communities': {
            'partition': {str(k): v for k, v in communities.items()},
            'modularity': round(modularity, 6),
            'community_count': len(set(communities.values())) if communities else 0
        },
        'community_vs_sector': community_comparison,
        'systemic_risk': {
            'systemic_stocks': [
                {'stock': s, 'name': get_stock_name(s), 'score': round(c['composite'], 6)}
                for s, c in systemic_stocks
            ],
            'bridge_stocks': bridge_stocks[:10]
        },
        'diversification': diversification,
        'lead_lag_network': lead_lag_out,
        'network_evolution': evolution,
        'network_stability': stability,
        'ml_features': ml_features
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'stock_network_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"  ✅ JSON 结果已保存到: {json_path}")


def save_ml_features(ml_features, output_dir):
    """保存 ML 特征为独立 JSON"""
    if not ml_features:
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'network_features_for_ml.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ml_features, f, indent=2, ensure_ascii=False)

    print(f"  ✅ ML 特征已保存到: {path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='港股网络分析 - MST、PMFG、社区检测、系统性风险')

    # 网络构建参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='阈值网络相关系数阈值（默认0.5）')
    parser.add_argument('--partial-corr-threshold', type=float, default=0.3,
                        help='偏相关网络阈值（默认0.3）')
    parser.add_argument('--volume-threshold', type=float, default=0.5,
                        help='成交量网络阈值（默认0.5）')
    parser.add_argument('--momentum-horizon', type=int, default=20,
                        help='动量周期天数（默认20）')
    parser.add_argument('--skip-pmfg', action='store_true',
                        help='跳过PMFG构建（节省计算时间）')
    parser.add_argument('--multiplex', action='store_true',
                        help='构建多层网络（价格+成交量+波动率）')

    # 分析参数
    parser.add_argument('--max-lag', type=int, default=5,
                        help='Granger因果检验最大滞后期（默认5）')
    parser.add_argument('--full-granger', action='store_true',
                        help='使用全部股票进行Granger检验（默认仅代表股票）')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Louvain社区检测分辨率参数（默认1.0）')
    parser.add_argument('--portfolio-size', type=int, default=10,
                        help='分散化组合推荐股票数（默认10）')

    # 滚动窗口分析
    parser.add_argument('--rolling', action='store_true',
                        help='启用滚动窗口网络演变分析')
    parser.add_argument('--window-days', type=int, default=120,
                        help='滚动窗口天数（默认120）')
    parser.add_argument('--step-days', type=int, default=20,
                        help='滚动窗口步长（默认20）')

    # 输出控制
    parser.add_argument('--no-visualization', action='store_true',
                        help='不生成可视化图表')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录（默认output）')

    args = parser.parse_args()

    print("=" * 80)
    print("港股网络分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数: threshold={args.threshold}, skip_pmfg={args.skip_pmfg}, "
          f"rolling={args.rolling}")
    print()

    start_time = time.time()

    # 1. 获取数据
    stock_list = get_stock_list()
    print(f"股票列表: {len(stock_list)} 只股票")

    stock_data = fetch_all_stock_data(stock_list)
    if len(stock_data) < 10:
        print("❌ 数据不足，无法分析")
        return

    stock_codes = list(stock_data.keys())
    sector_count = len(set(get_stock_sector(c) for c in stock_codes))

    # 2. 构建收益率和相关矩阵
    returns_df = build_returns_dataframe(stock_data)
    pearson_corr, _ = compute_correlation_matrices(returns_df)
    distance_matrix = build_correlation_distance_matrix(pearson_corr)

    # 3. 网络构建
    print("\n📊 构建网络...")
    mst_graph = build_minimum_spanning_tree(distance_matrix, stock_codes)

    if args.skip_pmfg:
        pmfg_graph = nx.Graph()
        print("  ⏭️ 跳过 PMFG")
    else:
        pmfg_graph = build_planar_maximally_filtered_graph(distance_matrix, stock_codes)

    threshold_graph = build_threshold_network(pearson_corr, stock_codes, args.threshold)
    partial_graph = build_partial_correlation_network(
        returns_df, stock_codes, args.partial_corr_threshold)

    # 成交量网络
    volume_graph, volume_corr = build_volume_correlation_network(
        stock_data, args.volume_threshold, method='change')

    # 动量网络
    momentum_graph, momentum_corr = build_momentum_correlation_network(
        stock_data, args.momentum_horizon, args.threshold)

    # 波动率网络
    volatility_graph, vol_corr = build_volatility_correlation_network(
        stock_data, window=20, threshold=args.threshold)

    # 多层网络（可选）
    layers = {}
    interlayer_edges = []
    if args.multiplex:
        layers, interlayer_edges = build_multiplex_network(
            stock_data, returns_df,
            thresholds={'price': args.threshold, 'volume': args.volume_threshold, 'volatility': args.threshold})

    # 4. 网络指标
    print("\n📊 计算网络指标...")
    mst_topo = calculate_topology_stats(mst_graph)
    pmfg_topo = calculate_topology_stats(pmfg_graph) if pmfg_graph.number_of_nodes() > 0 else {}
    threshold_topo = calculate_topology_stats(threshold_graph)
    partial_topo = calculate_topology_stats(partial_graph)
    volume_topo = calculate_topology_stats(volume_graph)
    momentum_topo = calculate_topology_stats(momentum_graph)
    volatility_topo = calculate_topology_stats(volatility_graph)

    centrality_dict = calculate_centrality_metrics(mst_graph)
    communities, modularity = detect_communities(mst_graph, args.resolution)

    # 5. 应用功能
    print("\n📊 应用分析...")
    systemic_stocks = identify_systemically_important_stocks(centrality_dict, top_n=10)
    bridge_stocks = identify_bridge_stocks(mst_graph, communities)
    community_comparison = analyze_community_vs_sector(communities)
    diversification = generate_diversification_recommendations(
        mst_graph, communities, centrality_dict, args.portfolio_size)
    lead_lag_graph = build_lead_lag_network(
        stock_data, args.max_lag, args.full_granger)

    # 6. 动态分析（可选）
    evolution = []
    stability = {}
    if args.rolling:
        evolution = analyze_network_evolution(
            stock_data, args.window_days, args.step_days)
        if evolution:
            stability = calculate_network_stability(evolution)

    # 7. ML 特征导出
    ml_features = export_network_features(centrality_dict, communities,
                                           bridge_stocks, stock_codes)
    ml_features = add_mst_degree_features(ml_features, mst_graph)
    save_ml_features(ml_features, args.output_dir)

    # 8. 可视化
    if not args.no_visualization:
        print("\n📊 生成可视化图表...")
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_mst(mst_graph, communities, centrality_dict, args.output_dir)

        if pmfg_graph.number_of_nodes() > 0:
            visualize_pmfg(pmfg_graph, communities, centrality_dict, args.output_dir)

        visualize_threshold_network(threshold_graph, args.output_dir)
        visualize_community_comparison(communities, args.output_dir)
        visualize_centrality_ranking(centrality_dict, args.output_dir)

        if lead_lag_graph.number_of_edges() > 0:
            visualize_lead_lag_network(lead_lag_graph, args.output_dir)

        if evolution:
            visualize_network_evolution(evolution, args.output_dir)

        # 新增：多维度网络可视化
        if volume_graph.number_of_edges() > 0:
            visualize_volume_network(volume_graph, args.output_dir)

        if momentum_graph.number_of_edges() > 0:
            visualize_momentum_network(momentum_graph, args.output_dir)

        if volatility_graph.number_of_edges() > 0:
            visualize_volatility_network(volatility_graph, args.output_dir)

        # 四种网络对比图
        visualize_multiplex_comparison(volume_topo, momentum_topo, volatility_topo,
                                        threshold_topo, args.output_dir)

    # 9. 报告
    report = generate_markdown_report(
        mst_topo, pmfg_topo, threshold_topo, partial_topo,
        volume_topo, momentum_topo, volatility_topo,
        centrality_dict, communities, modularity,
        community_comparison, systemic_stocks, bridge_stocks,
        diversification, lead_lag_graph,
        mst_graph, pmfg_graph, volume_graph, momentum_graph, volatility_graph,
        evolution, stability,
        args.threshold, args.partial_corr_threshold, args.volume_threshold,
        len(stock_codes), sector_count)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'stock_network_analysis.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✅ 报告已保存到: {report_path}")

    # 10. JSON
    save_json_results(
        mst_topo, pmfg_topo, threshold_topo, partial_topo,
        centrality_dict, communities, modularity,
        community_comparison, systemic_stocks, bridge_stocks,
        diversification, lead_lag_graph,
        evolution, stability, ml_features,
        args.threshold, args.partial_corr_threshold)

    # 摘要
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("📊 分析摘要")
    print("=" * 80)
    print(f"  分析股票: {len(stock_codes)} 只")
    print(f"  MST 节点/边: {mst_topo.get('node_count', 0)}/{mst_topo.get('edge_count', 0)}")
    print(f"  检测社区: {len(set(communities.values())) if communities else 0} 个")
    if systemic_stocks:
        top_name = get_stock_name(systemic_stocks[0][0])
        print(f"  核心枢纽: {top_name}")
    if bridge_stocks:
        print(f"  桥梁股票: {len(bridge_stocks)} 只")
    print(f"  ML 特征: 12 个/股票")
    print(f"\n  耗时: {elapsed:.1f} 秒")
    print("\n" + "=" * 80)
    print(f"分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
