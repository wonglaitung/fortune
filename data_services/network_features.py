#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络特征模块 - 实时计算或加载跨截面网络特征

用途：
1. 综合分析：实时计算网络洞察（用于邮件展示）
2. 独立分析：运行 stock_network_analysis.py 生成详细报告
3. 机器学习特征：为模型提供网络相关特征

网络洞察（用于综合分析展示）：
- 社区归属：股票的网络群落
- 枢纽等级：低/中/高（基于中心性）
- 桥梁股标记：是否跨社区连接
- 波动率网络密度预警：系统性风险预警

二阶网络特征（用于机器学习模型）：
- 节点偏离度 (Node Deviation)：个股动量与邻居平均动量的差值，捕捉"掉队补涨"或"领涨回调"

创建时间：2026-04-30
更新时间：2026-05-02（新增节点偏离度特征）
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

# 添加项目根目录到 Python 跃径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 波动率网络密度预警阈值（动态阈值，基于历史均值±标准差）
# 当历史数据不足时使用默认阈值
DENSITY_DEFAULT_THRESHOLDS = {
    'watch_sigma': 1.0,     # 关注：均值 + 1σ
    'warning_sigma': 1.5,   # 预警：均值 + 1.5σ
    'extreme_sigma': 2.0    # 极端：均值 + 2σ
}

# 历史数据存储路径
DENSITY_HISTORY_PATH = 'data/network_density_history.json'


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
            # 数据泄漏防护：for_prediction=True 排除当日数据
            returns_df = build_returns_dataframe(stock_data, for_prediction=True)
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

    def calculate_node_deviation(self, stock_codes, score_type='momentum', window=20):
        """
        计算节点偏离度特征 (Node Deviation)

        核心逻辑：
        Feature_diff = Score_i - Average(Score_neighbors)

        用途：
        - 正值：个股跑赢邻居 → 可能回调（领涨回调）
        - 负值：个股跑输邻居 → 可能补涨（掉队补涨）

        参数:
            stock_codes: 股票代码列表
            score_type: 评分类型，可选 'momentum'(动量) 或 'return'(收益率)
            window: 计算窗口（天数）

        返回:
            dict: {股票代码: {
                'node_deviation': 偏离度值,
                'own_score': 个股评分,
                'neighbor_avg_score': 邻居平均评分,
                'neighbor_count': 邻居数量,
                'signal': 信号描述
            }}
        """
        import numpy as np

        logger.info(f"开始计算节点偏离度特征 (score_type={score_type}, window={window})...")
        print(f"  📊 计算节点偏离度特征...")

        try:
            # 导入网络分析模块
            from ml_services.stock_network_analysis import (
                fetch_all_stock_data,
                build_returns_dataframe,
                compute_correlation_matrices,
                build_minimum_spanning_tree
            )
            import networkx as nx

            # 获取股票数据
            stock_data = fetch_all_stock_data(list(stock_codes))
            if not stock_data:
                logger.warning("无法获取股票数据，返回默认值")
                return self._get_default_node_deviations(stock_codes)

            # 构建收益率 DataFrame
            # 数据泄漏防护：for_prediction=True 排除当日数据
            returns_df = build_returns_dataframe(stock_data, for_prediction=True)
            if returns_df.empty or len(returns_df.columns) < 2:
                logger.warning("数据不足，返回默认值")
                return self._get_default_node_deviations(stock_codes)

            # 计算个股评分（动量或收益率）
            # 数据泄漏防护：使用 T-1 日数据，不包含当日
            if score_type == 'momentum':
                # 动量：过去 window 天的累计收益率（不含当日）
                # 使用 [-window-1:-1] 而非 [-window:]，确保不包含当日数据
                if len(returns_df) >= window + 1:
                    scores = returns_df.iloc[-window-1:-1].sum()
                else:
                    # 数据不足时使用可用数据
                    scores = returns_df.iloc[:-1].sum() if len(returns_df) > 1 else pd.Series(0, index=returns_df.columns)
            else:
                # 收益率：T-1 日的收益率（不使用当日）
                scores = returns_df.iloc[-2] if len(returns_df) >= 2 else pd.Series(0, index=returns_df.columns)

            # 计算相关性矩阵
            pearson_corr, _ = compute_correlation_matrices(returns_df)
            corr_matrix = pearson_corr

            # 构建距离矩阵
            distance_matrix = np.sqrt(2 * (1 - corr_matrix))

            # 构建 MST
            mst_graph = build_minimum_spanning_tree(distance_matrix, list(returns_df.columns))

            # 计算每只股票的节点偏离度
            deviations = {}

            for stock_code in stock_codes:
                if stock_code not in mst_graph.nodes():
                    deviations[stock_code] = {
                        'node_deviation': 0.0,
                        'own_score': 0.0,
                        'neighbor_avg_score': 0.0,
                        'neighbor_count': 0,
                        'signal': '无邻居'
                    }
                    continue

                # 获取该股票的邻居
                neighbors = list(mst_graph.neighbors(stock_code))

                if len(neighbors) == 0:
                    deviations[stock_code] = {
                        'node_deviation': 0.0,
                        'own_score': float(scores.get(stock_code, 0)),
                        'neighbor_avg_score': 0.0,
                        'neighbor_count': 0,
                        'signal': '无邻居'
                    }
                    continue

                # 计算个股评分
                own_score = float(scores.get(stock_code, 0))

                # 计算邻居平均评分
                neighbor_scores = [float(scores.get(n, 0)) for n in neighbors]
                neighbor_avg_score = np.mean(neighbor_scores) if neighbor_scores else 0.0

                # 计算偏离度
                node_deviation = own_score - neighbor_avg_score

                # 生成信号描述
                if node_deviation > 0:
                    # 正值：跑赢邻居
                    if node_deviation > 0.02:  # 2% 以上
                        signal = '领涨回调⚠️'
                    else:
                        signal = '强势'
                elif node_deviation < 0:
                    # 负值：跑输邻居
                    if node_deviation < -0.02:  # -2% 以下
                        signal = '掉队补涨📈'
                    else:
                        signal = '弱势'
                else:
                    signal = '持平'

                deviations[stock_code] = {
                    'node_deviation': float(node_deviation),
                    'own_score': own_score,
                    'neighbor_avg_score': float(neighbor_avg_score),
                    'neighbor_count': len(neighbors),
                    'signal': signal
                }

            # 添加元数据
            deviations['_meta'] = {
                'score_type': score_type,
                'window': window,
                'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_count': len([k for k in deviations.keys() if k != '_meta'])
            }

            print(f"    ✅ 节点偏离度计算完成: {len(deviations) - 1} 只股票")
            logger.info(f"节点偏离度计算完成: {len(deviations) - 1} 只股票")

            return deviations

        except Exception as e:
            logger.warning(f"节点偏离度计算失败: {e}")
            print(f"    ⚠️ 节点偏离度计算失败: {e}")
            return self._get_default_node_deviations(stock_codes)

    def _get_default_node_deviations(self, stock_codes):
        """返回默认节点偏离度值"""
        deviations = {}
        for code in stock_codes:
            deviations[code] = {
                'node_deviation': 0.0,
                'own_score': 0.0,
                'neighbor_avg_score': 0.0,
                'neighbor_count': 0,
                'signal': '未知'
            }
        deviations['_meta'] = {
            'score_type': 'unknown',
            'window': 20,
            'calculation_time': 'N/A',
            'stock_count': len(stock_codes)
        }
        return deviations

    def get_node_deviation_for_features(self, stock_codes, window=20):
        """
        获取节点偏离度特征用于机器学习模型

        返回格式适合直接加入特征 DataFrame

        参数:
            stock_codes: 股票代码列表
            window: 计算窗口（天数）

        返回:
            dict: {股票代码: node_deviation值}，可直接用于 DataFrame 列
        """
        deviations = self.calculate_node_deviation(stock_codes, score_type='momentum', window=window)

        # 提取简洁格式
        result = {}
        for code in stock_codes:
            if code in deviations and isinstance(deviations[code], dict):
                result[code] = deviations[code].get('node_deviation', 0.0)
            else:
                result[code] = 0.0

        return result

    def get_node_deviation_signals(self, stock_codes, threshold=0.02):
        """
        获取节点偏离度信号（用于交易决策）

        参数:
            stock_codes: 股票代码列表
            threshold: 信号阈值（默认 2%）

        返回:
            dict: {股票代码: {
                'signal': 'buy'/'sell'/'hold',
                'strength': 信号强度,
                'description': 描述
            }}
        """
        deviations = self.calculate_node_deviation(stock_codes)

        signals = {}
        for code in stock_codes:
            if code not in deviations or code == '_meta':
                continue

            dev = deviations[code]
            node_dev = dev.get('node_deviation', 0.0)

            if node_dev < -threshold:
                # 跑输邻居超过阈值 → 补涨信号
                signals[code] = {
                    'signal': 'buy',
                    'strength': min(abs(node_dev) / threshold, 3.0),  # 1-3 倍强度
                    'description': f"掉队补涨: 跑输邻居 {abs(node_dev)*100:.1f}%"
                }
            elif node_dev > threshold:
                # 跑赢邻居超过阈值 → 回调风险
                signals[code] = {
                    'signal': 'sell',
                    'strength': min(node_dev / threshold, 3.0),
                    'description': f"领涨回调: 跑赢邻居 {node_dev*100:.1f}%"
                }
            else:
                signals[code] = {
                    'signal': 'hold',
                    'strength': 0.0,
                    'description': f"与邻居持平: 偏离 {node_dev*100:.1f}%"
                }

        return signals

    def calculate_volatility_network_density(self, stock_codes):
        """
        计算波动率相关性网络的密度

        数据泄漏防护：
        - 使用 T-1 日数据构建网络，确保不包含当日信息
        - 波动率计算基于历史窗口，本身不存在泄漏风险
        - 但为确保一致性，显式声明用于预测

        参数:
            stock_codes: 股票代码列表

        返回:
            float: 波动率网络密度（0-1之间）
        """
        try:
            from ml_services.stock_network_analysis import (
                fetch_all_stock_data,
                build_volatility_correlation_network,
                calculate_topology_stats
            )

            # 获取股票数据
            stock_data = fetch_all_stock_data(list(stock_codes))
            if not stock_data:
                logger.warning("无法获取股票数据用于密度计算")
                return None

            # 构建波动率相关性网络
            # 注意：波动率使用 rolling(window).std()，基于历史数据计算
            # 本身不存在数据泄漏，但保持一致性
            volatility_graph, _ = build_volatility_correlation_network(
                stock_data, window=20, threshold=0.5
            )

            if volatility_graph.number_of_nodes() == 0:
                logger.warning("波动率网络节点数为0")
                return None

            # 计算拓扑统计（包含密度）
            topo_stats = calculate_topology_stats(volatility_graph)
            density = topo_stats.get('density', 0)

            logger.info(f"波动率网络密度: {density:.4f}")
            return density

        except Exception as e:
            logger.warning(f"波动率网络密度计算失败: {e}")
            return None

    def save_density_history(self, density, date_str=None):
        """
        保存密度历史数据

        参数:
            density: 当前密度值
            date_str: 日期字符串，默认使用今天
        """
        if density is None:
            return

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        history_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            DENSITY_HISTORY_PATH
        )

        # 加载现有历史
        history_data = {'history': []}
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            except Exception as e:
                logger.warning(f"加载密度历史失败: {e}")

        # 检查是否已有今天的记录
        existing_dates = [h['date'] for h in history_data.get('history', [])]
        if date_str in existing_dates:
            # 更新今天的记录
            for h in history_data['history']:
                if h['date'] == date_str:
                    h['volatility_density'] = density
                    break
        else:
            # 添加新记录
            history_data['history'].append({
                'date': date_str,
                'volatility_density': density
            })

        # 按日期排序
        history_data['history'].sort(key=lambda x: x['date'])

        # 只保留最近60天的数据
        if len(history_data['history']) > 60:
            history_data['history'] = history_data['history'][-60:]

        # 保存
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            logger.info(f"密度历史已保存: {date_str} -> {density:.4f}")
        except Exception as e:
            logger.warning(f"保存密度历史失败: {e}")

    def load_density_history(self, days=30):
        """
        加载密度历史数据

        参数:
            days: 要加载的天数

        返回:
            list: 历史密度数据列表 [{date, volatility_density}, ...]
        """
        history_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            DENSITY_HISTORY_PATH
        )

        if not os.path.exists(history_path):
            return []

        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            history = history_data.get('history', [])
            # 返回最近N天的数据
            return history[-days:] if len(history) > days else history

        except Exception as e:
            logger.warning(f"加载密度历史失败: {e}")
            return []

    def calculate_density_warning(self, stock_codes):
        """
        计算波动率网络密度预警

        核心逻辑：
        密度高 → 市场进入"同涨同跌"模式 → 个股分化度低 → 选股模型失效 → 降低仓位

        参数:
            stock_codes: 股票代码列表

        返回:
            dict: {
                'current_density': 当前密度,
                'mean_30d': 30天平均值,
                'std_30d': 30天标准差,
                'first_density': 30天前密度,
                'status': 状态描述,
                'trend': 趋势方向,
                'warning_level': 预警级别,
                'recommendation': 操作建议
            }
        """
        import numpy as np

        # 计算当前密度
        current_density = self.calculate_volatility_network_density(stock_codes)
        if current_density is None:
            return None

        # 保存当前密度到历史
        self.save_density_history(current_density)

        # 加载历史数据
        history = self.load_density_history(days=30)

        # 计算统计值
        if len(history) >= 2:
            densities = [h['volatility_density'] for h in history]
            mean_30d = np.mean(densities)
            std_30d = np.std(densities)
            first_density = densities[0]  # 30天前的密度
        else:
            mean_30d = current_density
            std_30d = 0.0
            first_density = current_density

        # 判断状态（相对于均值）
        if current_density < mean_30d - std_30d:
            status = '偏低'
            status_icon = '✅'
            status_detail = '低于均值-1σ'
        elif current_density > mean_30d + std_30d:
            status = '偏高'
            status_icon = '⚠️'
            status_detail = '高于均值+1σ'
        else:
            status = '正常'
            status_icon = '✅'
            status_detail = '在均值±1σ范围内'

        # 判断预警级别（动态阈值：基于历史均值±标准差）
        warning_level = None
        thresholds = {}

        if std_30d > 0:
            # 有足够历史数据，使用动态阈值
            thresholds = {
                'watch': mean_30d + DENSITY_DEFAULT_THRESHOLDS['watch_sigma'] * std_30d,
                'warning': mean_30d + DENSITY_DEFAULT_THRESHOLDS['warning_sigma'] * std_30d,
                'extreme': mean_30d + DENSITY_DEFAULT_THRESHOLDS['extreme_sigma'] * std_30d
            }
        else:
            # 历史数据不足，使用固定阈值（基于当前密度的百分比）
            thresholds = {
                'watch': 0.45,
                'warning': 0.50,
                'extreme': 0.55
            }

        if current_density > thresholds['extreme']:
            warning_level = '🔴🔴 极端'
            status_icon = '🔴🔴'
            status = '极端'
        elif current_density > thresholds['warning']:
            warning_level = '🔴 预警'
            status_icon = '🔴'
            status = '预警'
        elif current_density > thresholds['watch']:
            warning_level = '⚠️ 关注'
            status_icon = '⚠️'
            status = '关注'

        # 判断趋势
        if len(history) >= 2:
            density_change = current_density - first_density
            if density_change > 0.01:
                trend = '上升'
                trend_detail = f'过去30天密度从 {first_density:.3f} → {current_density:.3f}，持续上升'
            elif density_change < -0.01:
                trend = '下降'
                trend_detail = f'过去30天密度从 {first_density:.3f} → {current_density:.3f}，持续下降'
            else:
                trend = '稳定'
                trend_detail = f'过去30天密度稳定在 {current_density:.3f} 左右'
        else:
            trend = '未知'
            trend_detail = '历史数据不足，无法判断趋势'

        # 生成建议
        if warning_level:
            if warning_level == '🔴🔴 极端':
                recommendation = '市场高度同步，降低仓位 50%，选股模型可能失效'
            elif warning_level == '🔴 预警':
                recommendation = '市场同步性增强，降低仓位 20%，警惕系统性风险'
            else:
                recommendation = '关注系统性风险，密切观察市场联动'
        elif trend == '下降':
            recommendation = '风险传导性减弱，市场从"同涨同跌"转向"个股分化"，选股模型有效性提高'
        elif trend == '上升':
            recommendation = '风险传导性增强，市场趋向"同涨同跌"，需关注系统性风险'
        else:
            recommendation = '市场状态稳定，维持正常策略'

        result = {
            'current_density': current_density,
            'mean_30d': mean_30d,
            'std_30d': std_30d,
            'first_density': first_density,
            'status': status,
            'status_icon': status_icon,
            'status_detail': status_detail,
            'trend': trend,
            'trend_detail': trend_detail,
            'warning_level': warning_level,
            'recommendation': recommendation,
            'thresholds': thresholds,
            'threshold_type': 'dynamic' if std_30d > 0 else 'default',
            'history_count': len(history)
        }

        print(f"    ✅ 波动率网络密度: {current_density:.4f} ({status})")
        logger.info(f"波动率网络密度预警: {result}")

        return result


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
