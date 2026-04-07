#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股异常因果关系分析工具

分析内容：
1. 检测一年期间的异常（批量方式）
2. 分析异常后不同时间窗口的股价表现
3. Granger因果检验（异常是否预示股价变化）
4. 时间序列交叉相关分析
5. 事件研究法（异常前后表现对比）
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import yfinance as yf

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WATCHLIST, STOCK_SECTOR_MAPPING
from anomaly_detector.zscore_detector import ZScoreDetector
from anomaly_detector.feature_extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        return super(NumpyJSONEncoder, self).default(obj)


class HKStockAnomalyCausalityAnalyzer:
    """港股异常因果关系分析器"""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None,
        window_size: int = 30,
        threshold_high: float = 4.0,
        threshold_medium: float = 3.0
    ):
        """
        初始化分析器
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            stocks: 股票列表（默认使用WATCHLIST）
            window_size: Z-Score窗口大小
            threshold_high: 高异常阈值
            threshold_medium: 中异常阈值
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.stocks = stocks or WATCHLIST
        self.window_size = window_size
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        
        # 初始化检测器
        self.zscore_detector = ZScoreDetector(
            window_size=window_size,
            threshold=threshold_medium  # 使用medium阈值作为基准
        )
        
        # 存储数据
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.all_anomalies: List[Dict] = []
        
        logger.info(f"初始化分析器: {start_date} 至 {end_date}")
        logger.info(f"股票数量: {len(self.stocks)}")
    
    def load_stock_data(self):
        """批量加载所有股票数据"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤1: 批量加载股票数据")
        logger.info(f"{'='*60}")
        
        # 计算扩展的时间范围（前后多加载60天用于分析）
        extended_start = self.start_date - timedelta(days=90)
        extended_end = self.end_date + timedelta(days=90)
        
        success_count = 0
        for i, stock in enumerate(self.stocks, 1):
            logger.info(f"[{i}/{len(self.stocks)}] 加载 {stock}...")
            
            try:
                ticker = yf.Ticker(stock)
                df = ticker.history(start=extended_start, end=extended_end, interval='1d')
                
                if df is None or len(df) == 0:
                    logger.warning(f"  {stock}: 无数据")
                    continue
                
                # 统一时区
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                self.stock_data[stock] = df
                success_count += 1
                logger.info(f"  {stock}: {len(df)} 天数据")
                
            except Exception as e:
                logger.error(f"  {stock}: 加载失败 - {e}")
        
        logger.info(f"\n成功加载 {success_count}/{len(self.stocks)} 只股票数据")
        return success_count > 0
    
    def detect_anomalies_batch(self):
        """批量检测所有股票的异常"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤2: 批量检测异常")
        logger.info(f"{'='*60}")
        
        total_anomalies = 0
        anomaly_by_date: Dict[str, List] = {}
        
        for i, (stock, df) in enumerate(self.stock_data.items(), 1):
            logger.info(f"[{i}/{len(self.stock_data)}] 检测 {stock}...")
            
            try:
                # 筛选日期范围
                df_period = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                
                if len(df_period) == 0:
                    continue
                
                # 检测每个交易日的异常
                for idx, row in df_period.iterrows():
                    current_date = idx
                    current_price = row['Close']
                    current_volume = row['Volume']
                    
                    # 获取历史数据（用于Z-Score计算）
                    hist_df = df[df.index < current_date].tail(self.window_size + 10)  # 多取一些确保足够
                    
                    if len(hist_df) < self.window_size:
                        continue
                    
                    # 价格异常检测
                    price_history = hist_df['Close']  # 传递 pd.Series 而非 numpy array
                    price_anomaly = self.zscore_detector.detect_anomaly(
                        metric_name='price',
                        current_value=current_price,
                        history=price_history,
                        timestamp=current_date
                    )
                    
                    # 成交量异常检测
                    volume_history = hist_df['Volume']  # 传递 pd.Series 而非 numpy array
                    volume_anomaly = self.zscore_detector.detect_anomaly(
                        metric_name='volume',
                        current_value=current_volume,
                        history=volume_history,
                        timestamp=current_date
                    )
                    
                    # 记录异常
                    for anomaly in [price_anomaly, volume_anomaly]:
                        if anomaly and anomaly.get('severity') in ['high', 'medium']:
                            date_str = current_date.strftime('%Y-%m-%d')
                            
                            anomaly_info = {
                                'stock': stock,
                                'name': STOCK_SECTOR_MAPPING.get(stock, {}).get('name', stock),
                                'date': date_str,
                                'type': anomaly['type'],
                                'severity': anomaly['severity'],
                                'z_score': anomaly.get('z_score'),
                                'value': anomaly.get('value'),
                                'close': current_price,
                                'volume': current_volume
                            }
                            
                            self.all_anomalies.append(anomaly_info)
                            total_anomalies += 1
                            
                            if date_str not in anomaly_by_date:
                                anomaly_by_date[date_str] = []
                            anomaly_by_date[date_str].append(anomaly_info)
                
            except Exception as e:
                logger.error(f"  {stock}: 检测失败 - {e}")
        
        # 统计结果
        unique_dates = len(anomaly_by_date)
        total_days = len(pd.bdate_range(self.start_date, self.end_date))
        
        logger.info(f"\n异常检测完成:")
        logger.info(f"  总异常数: {total_anomalies}")
        logger.info(f"  有异常的交易日: {unique_dates}/{total_days} ({unique_dates/total_days*100:.1f}%)")
        
        # 按严重程度统计
        high_count = len([a for a in self.all_anomalies if a['severity'] == 'high'])
        medium_count = len([a for a in self.all_anomalies if a['severity'] == 'medium'])
        logger.info(f"  高异常: {high_count}, 中异常: {medium_count}")
        
        # 按类型统计
        price_count = len([a for a in self.all_anomalies if a['type'] == 'price'])
        volume_count = len([a for a in self.all_anomalies if a['type'] == 'volume'])
        logger.info(f"  价格异常: {price_count}, 成交量异常: {volume_count}")
        
        return total_anomalies > 0
    
    def analyze_post_anomaly_performance(self) -> Dict:
        """分析异常后的股价表现"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤3: 分析异常后股价表现")
        logger.info(f"{'='*60}")
        
        performance_results = []
        windows = [1, 3, 5, 10, 20, 30, 60]
        
        for i, anomaly in enumerate(self.all_anomalies, 1):
            stock = anomaly['stock']
            date_str = anomaly['date']
            
            if i % 50 == 0 or i == len(self.all_anomalies):
                logger.info(f"[{i}/{len(self.all_anomalies)}] 分析 {stock} {date_str}...")
            
            if stock not in self.stock_data:
                continue
            
            df = self.stock_data[stock]
            anomaly_date = pd.to_datetime(date_str)
            
            try:
                # 确保异常日期在数据中
                if anomaly_date not in df.index:
                    idx = df.index.get_indexer([anomaly_date], method='nearest')[0]
                    anomaly_date = df.index[idx]
                
                idx = df.index.get_loc(anomaly_date)
                
                # 计算异常后收益率
                performance = {
                    'stock': stock,
                    'date': date_str,
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'z_score': anomaly['z_score']
                }
                
                for window in windows:
                    if idx + window < len(df):
                        future_close = df['Close'].iloc[idx + window]
                        current_close = df['Close'].iloc[idx]
                        return_pct = (future_close / current_close - 1) * 100
                        performance[f'return_{window}d'] = return_pct
                    else:
                        performance[f'return_{window}d'] = None
                
                # 计算异常后波动率（5日）
                if idx + 5 < len(df):
                    post_data = df['Close'].iloc[idx:idx+6]
                    performance['volatility_post_5d'] = post_data.pct_change().std() * 100
                else:
                    performance['volatility_post_5d'] = None
                
                # 计算异常前波动率（5日）
                if idx >= 5:
                    pre_data = df['Close'].iloc[idx-5:idx+1]
                    performance['volatility_pre_5d'] = pre_data.pct_change().std() * 100
                else:
                    performance['volatility_pre_5d'] = None
                
                # 计算异常当日涨跌幅
                if idx > 0:
                    prev_close = df['Close'].iloc[idx-1]
                    current_close = df['Close'].iloc[idx]
                    performance['daily_return'] = (current_close / prev_close - 1) * 100
                else:
                    performance['daily_return'] = None
                
                performance_results.append(performance)
                
            except Exception as e:
                logger.warning(f"  分析失败: {e}")
        
        logger.info(f"\n成功分析 {len(performance_results)} 个异常")
        
        # 统计汇总
        stats_summary = self._calculate_performance_stats(performance_results, windows)
        
        return {
            'performances': performance_results,
            'stats': stats_summary
        }
    
    def _calculate_performance_stats(self, results: List[Dict], windows: List[int]) -> Dict:
        """计算表现统计"""
        stats = {}
        
        # 各时间窗口收益率统计
        for window in windows:
            returns = [r.get(f'return_{window}d') for r in results 
                       if r.get(f'return_{window}d') is not None]
            
            if returns:
                stats[f'{window}d'] = {
                    'avg_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'win_rate': float(sum(1 for r in returns if r > 0) / len(returns) * 100),
                    'sample_count': len(returns)
                }
        
        # 波动率统计
        vol_post = [r.get('volatility_post_5d') for r in results 
                    if r.get('volatility_post_5d') is not None]
        vol_pre = [r.get('volatility_pre_5d') for r in results 
                   if r.get('volatility_pre_5d') is not None]
        
        if vol_post:
            stats['volatility_post'] = {
                'mean': float(np.mean(vol_post)),
                'sample_count': len(vol_post)
            }
        
        if vol_pre:
            stats['volatility_pre'] = {
                'mean': float(np.mean(vol_pre)),
                'sample_count': len(vol_pre)
            }
        
        if vol_post and vol_pre:
            vol_change = (np.mean(vol_post) - np.mean(vol_pre)) / np.mean(vol_pre) * 100
            stats['volatility_change'] = float(vol_change)
        
        # 异常当日涨跌幅
        daily_returns = [r.get('daily_return') for r in results 
                         if r.get('daily_return') is not None]
        if daily_returns:
            stats['daily_return'] = {
                'mean': float(np.mean(daily_returns)),
                'sample_count': len(daily_returns)
            }
        
        return stats
    
    def analyze_granger_causality(self, max_lag: int = 10) -> Dict:
        """Granger因果检验：异常是否预示股价变化"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤4: Granger因果检验")
        logger.info(f"{'='*60}")
        
        results = {}
        
        # 按股票分组异常
        for stock in self.stocks:
            if stock not in self.stock_data:
                continue
            
            df = self.stock_data[stock]
            
            # 创建异常序列
            stock_anomalies = [a for a in self.all_anomalies if a['stock'] == stock]
            anomaly_dates = set(pd.to_datetime(a['date']) for a in stock_anomalies)
            
            if len(anomaly_dates) < 5:
                continue
            
            # 创建异常指示序列
            df_analysis = df.copy()
            df_analysis['anomaly'] = 0
            for date in anomaly_dates:
                if date in df_analysis.index:
                    df_analysis.loc[date, 'anomaly'] = 1
            
            # 计算收益率
            df_analysis['return'] = df_analysis['Close'].pct_change()
            
            # 准备数据
            data = df_analysis[['return', 'anomaly']].dropna()
            
            if len(data) < 30:
                continue
            
            try:
                # Granger因果检验：异常是否预示收益率变化
                test_result = grangercausalitytests(
                    data[['return', 'anomaly']], 
                    maxlag=max_lag, 
                    verbose=False
                )
                
                p_values = []
                significant_lags = []
                
                for lag in range(1, max_lag + 1):
                    p_value = test_result[lag][0]['ssr_ftest'][1]
                    p_values.append(p_value)
                    if p_value < 0.05:
                        significant_lags.append(lag)
                
                results[stock] = {
                    'p_values': p_values,
                    'significant_lags': significant_lags,
                    'is_significant': len(significant_lags) > 0,
                    'min_p_value': float(min(p_values)),
                    'best_lag': int(np.argmin(p_values) + 1)
                }
                
                if results[stock]['is_significant']:
                    logger.info(f"  {stock}: 显著滞后期 = {significant_lags}")
                
            except Exception as e:
                logger.warning(f"  {stock}: Granger检验失败 - {e}")
        
        # 统计显著的股票
        significant_stocks = [s for s, r in results.items() if r['is_significant']]
        logger.info(f"\nGranger因果检验结果:")
        logger.info(f"  显著股票数: {len(significant_stocks)}/{len(results)}")
        
        return results
    
    def analyze_cross_correlation(self, max_lag: int = 20) -> Dict:
        """时间序列交叉相关分析"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤5: 交叉相关分析")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for stock in self.stocks:
            if stock not in self.stock_data:
                continue
            
            df = self.stock_data[stock]
            
            # 创建异常序列
            stock_anomalies = [a for a in self.all_anomalies if a['stock'] == stock]
            anomaly_dates = set(pd.to_datetime(a['date']) for a in stock_anomalies)
            
            if len(anomaly_dates) < 5:
                continue
            
            # 创建异常指示序列
            df_analysis = df.copy()
            df_analysis['anomaly'] = 0
            for date in anomaly_dates:
                if date in df_analysis.index:
                    df_analysis.loc[date, 'anomaly'] = 1
            
            # 计算收益率
            df_analysis['return'] = df_analysis['Close'].pct_change()
            
            # 准备数据
            data = df_analysis[['return', 'anomaly']].dropna()
            
            if len(data) < 30:
                continue
            
            try:
                # 交叉相关分析
                returns = data['return'].values
                anomalies = data['anomaly'].values
                
                # 标准化
                returns_std = (returns - np.mean(returns)) / np.std(returns)
                anomalies_std = (anomalies - np.mean(anomalies)) / np.std(anomalies)
                
                # 计算交叉相关
                cross_corr = ccf(anomalies_std, returns_std, adjusted=False)[:max_lag+1]
                
                results[stock] = {
                    'cross_correlations': cross_corr.tolist(),
                    'max_corr': float(np.max(np.abs(cross_corr))),
                    'max_corr_lag': int(np.argmax(np.abs(cross_corr))),
                    'leading_effect': cross_corr[1:6].tolist() if len(cross_corr) > 5 else []
                }
                
                # 检查前置效应（异常领先收益率）
                leading_corr = cross_corr[1:6]  # 异常领先1-5天
                if np.any(np.abs(leading_corr) > 0.1):
                    logger.info(f"  {stock}: 前置效应显著 (最大相关={np.max(np.abs(leading_corr)):.3f})")
                
            except Exception as e:
                logger.warning(f"  {stock}: 交叉相关分析失败 - {e}")
        
        return results
    
    def analyze_event_study(self) -> Dict:
        """事件研究法：异常前后表现对比"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤6: 事件研究法")
        logger.info(f"{'='*60}")
        
        results = {
            'pre_anomaly': [],
            'post_anomaly': []
        }
        
        for anomaly in self.all_anomalies:
            stock = anomaly['stock']
            date_str = anomaly['date']
            
            if stock not in self.stock_data:
                continue
            
            df = self.stock_data[stock]
            anomaly_date = pd.to_datetime(date_str)
            
            try:
                if anomaly_date not in df.index:
                    idx = df.index.get_indexer([anomaly_date], method='nearest')[0]
                    anomaly_date = df.index[idx]
                
                idx = df.index.get_loc(anomaly_date)
                
                # 异常前5日收益率
                if idx >= 5:
                    pre_return = (df['Close'].iloc[idx] / df['Close'].iloc[idx-5] - 1) * 100
                    results['pre_anomaly'].append(pre_return)
                
                # 异常后5日收益率
                if idx + 5 < len(df):
                    post_return = (df['Close'].iloc[idx+5] / df['Close'].iloc[idx] - 1) * 100
                    results['post_anomaly'].append(post_return)
                
            except Exception as e:
                pass
        
        # 统计分析
        if results['pre_anomaly']:
            results['pre_stats'] = {
                'mean': float(np.mean(results['pre_anomaly'])),
                'std': float(np.std(results['pre_anomaly'])),
                'win_rate': float(sum(1 for r in results['pre_anomaly'] if r > 0) / len(results['pre_anomaly']) * 100)
            }
        
        if results['post_anomaly']:
            results['post_stats'] = {
                'mean': float(np.mean(results['post_anomaly'])),
                'std': float(np.std(results['post_anomaly'])),
                'win_rate': float(sum(1 for r in results['post_anomaly'] if r > 0) / len(results['post_anomaly']) * 100)
            }
        
        # t检验
        if results['pre_anomaly'] and results['post_anomaly']:
            t_stat, p_value = stats.ttest_rel(
                results['pre_anomaly'][:len(results['post_anomaly'])],
                results['post_anomaly']
            )
            results['t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        logger.info(f"\n事件研究结果:")
        if 'pre_stats' in results:
            logger.info(f"  异常前5日: 平均收益率={results['pre_stats']['mean']:+.2f}%, 胜率={results['pre_stats']['win_rate']:.1f}%")
        if 'post_stats' in results:
            logger.info(f"  异常后5日: 平均收益率={results['post_stats']['mean']:+.2f}%, 胜率={results['post_stats']['win_rate']:.1f}%")
        if 't_test' in results:
            logger.info(f"  差异显著性: p={results['t_test']['p_value']:.4f} ({'显著' if results['t_test']['significant'] else '不显著'})")
        
        return results
    
    def generate_report(self, output_dir: str = 'output') -> str:
        """生成综合分析报告"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤7: 生成分析报告")
        logger.info(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行所有分析
        performance_result = self.analyze_post_anomaly_performance()
        granger_result = self.analyze_granger_causality()
        cross_corr_result = self.analyze_cross_correlation()
        event_study_result = self.analyze_event_study()
        
        # 汇总结果
        report_data = {
            'analysis_period': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d')
            },
            'anomaly_summary': {
                'total_anomalies': len(self.all_anomalies),
                'stocks_analyzed': len(self.stock_data),
                'high_severity': len([a for a in self.all_anomalies if a['severity'] == 'high']),
                'medium_severity': len([a for a in self.all_anomalies if a['severity'] == 'medium']),
                'price_anomalies': len([a for a in self.all_anomalies if a['type'] == 'price']),
                'volume_anomalies': len([a for a in self.all_anomalies if a['type'] == 'volume'])
            },
            'performance_stats': performance_result['stats'],
            'granger_causality': {
                'significant_stocks': len([s for s, r in granger_result.items() if r.get('is_significant')]),
                'total_stocks': len(granger_result),
                'details': {k: {
                    'is_significant': v['is_significant'],
                    'significant_lags': v['significant_lags'],
                    'min_p_value': v['min_p_value']
                } for k, v in granger_result.items()}
            },
            'cross_correlation': cross_corr_result,
            'event_study': event_study_result
        }
        
        # 保存JSON
        json_file = os.path.join(output_dir, 'hk_stock_anomaly_causality_analysis.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"  保存JSON: {json_file}")
        
        # 保存详细异常数据
        anomaly_file = os.path.join(output_dir, 'hk_stock_anomalies_detailed.json')
        with open(anomaly_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_anomalies, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"  保存异常数据: {anomaly_file}")
        
        # 保存表现数据
        performance_file = os.path.join(output_dir, 'hk_stock_anomaly_performance.json')
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_result['performances'], f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"  保存表现数据: {performance_file}")
        
        # 生成Markdown报告
        md_file = os.path.join(output_dir, 'hk_stock_anomaly_causality_report.md')
        self._write_markdown_report(md_file, report_data)
        logger.info(f"  保存Markdown报告: {md_file}")
        
        return md_file
    
    def _write_markdown_report(self, filepath: str, data: Dict):
        """写入Markdown格式的报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 港股异常因果关系分析报告\n\n")
            
            # 分析周期
            f.write("## 分析周期\n\n")
            f.write(f"- **开始日期**: {data['analysis_period']['start_date']}\n")
            f.write(f"- **结束日期**: {data['analysis_period']['end_date']}\n\n")
            
            # 异常汇总
            f.write("## 异常检测汇总\n\n")
            summary = data['anomaly_summary']
            f.write(f"- **总异常数**: {summary['total_anomalies']}\n")
            f.write(f"- **分析的股票数**: {summary['stocks_analyzed']}\n")
            f.write(f"- **高异常**: {summary['high_severity']}\n")
            f.write(f"- **中异常**: {summary['medium_severity']}\n")
            f.write(f"- **价格异常**: {summary['price_anomalies']}\n")
            f.write(f"- **成交量异常**: {summary['volume_anomalies']}\n\n")
            
            # 异常后股价表现
            f.write("## 异常后股价表现分析\n\n")
            f.write("### 各时间窗口收益率统计\n\n")
            f.write("| 时间窗口 | 平均收益率 | 标准差 | 上涨概率 | 样本数 |\n")
            f.write("|---------|-----------|--------|---------|--------|\n")
            
            for window in [1, 3, 5, 10, 20, 30, 60]:
                key = f'{window}d'
                if key in data['performance_stats']:
                    stats = data['performance_stats'][key]
                    f.write(f"| {window}天后 | {stats['avg_return']:+.2f}% | {stats['std_return']:.2f}% | {stats['win_rate']:.1f}% | {stats['sample_count']} |\n")
            
            f.write("\n### 波动率分析\n\n")
            if 'volatility_pre' in data['performance_stats'] and 'volatility_post' in data['performance_stats']:
                vol_pre = data['performance_stats']['volatility_pre']['mean']
                vol_post = data['performance_stats']['volatility_post']['mean']
                vol_change = data['performance_stats'].get('volatility_change', 0)
                f.write(f"- **异常前5日波动率**: {vol_pre:.2f}%\n")
                f.write(f"- **异常后5日波动率**: {vol_post:.2f}%\n")
                f.write(f"- **波动率变化**: {vol_change:+.1f}%\n\n")
            
            if 'daily_return' in data['performance_stats']:
                f.write(f"- **异常当日平均涨跌幅**: {data['performance_stats']['daily_return']['mean']:+.2f}%\n\n")
            
            # Granger因果检验
            f.write("## Granger因果检验\n\n")
            granger = data['granger_causality']
            f.write(f"- **显著股票数**: {granger['significant_stocks']}/{granger['total_stocks']}\n\n")
            
            if granger['significant_stocks'] > 0:
                f.write("### 显著股票详情\n\n")
                f.write("| 股票 | 最小p值 | 显著滞后期 |\n")
                f.write("|------|--------|-----------|\n")
                
                for stock, details in granger['details'].items():
                    if details['is_significant']:
                        lags = ', '.join(map(str, details['significant_lags']))
                        f.write(f"| {stock} | {details['min_p_value']:.4f} | {lags} |\n")
                f.write("\n")
            
            # 事件研究
            f.write("## 事件研究法\n\n")
            event = data['event_study']
            
            if 'pre_stats' in event:
                f.write(f"- **异常前5日**: 平均收益率 {event['pre_stats']['mean']:+.2f}%, 胜率 {event['pre_stats']['win_rate']:.1f}%\n")
            
            if 'post_stats' in event:
                f.write(f"- **异常后5日**: 平均收益率 {event['post_stats']['mean']:+.2f}%, 胜率 {event['post_stats']['win_rate']:.1f}%\n")
            
            if 't_test' in event:
                sig = "显著" if event['t_test']['significant'] else "不显著"
                f.write(f"- **差异检验**: t={event['t_test']['t_statistic']:.2f}, p={event['t_test']['p_value']:.4f} ({sig})\n\n")
            
            # 关键发现
            f.write("## 关键发现与结论\n\n")
            f.write("### 因果关系分析\n\n")
            
            # 基于Granger检验结果
            if granger['significant_stocks'] > granger['total_stocks'] * 0.3:
                f.write("1. **异常对股价有预测能力**: 超过30%的股票通过了Granger因果检验，表明异常可能是股价变化的先行指标。\n")
            else:
                f.write("1. **异常对股价的预测能力有限**: 仅少数股票通过了Granger因果检验。\n")
            
            # 基于表现统计
            stats_5d = data['performance_stats'].get('5d', {})
            if stats_5d:
                if stats_5d['avg_return'] < -1:
                    f.write(f"2. **异常后中期下跌趋势**: 异常后5日平均收益率为{stats_5d['avg_return']:+.2f}%，显示下跌倾向。\n")
                elif stats_5d['avg_return'] > 1:
                    f.write(f"2. **异常后中期上涨趋势**: 异常后5日平均收益率为{stats_5d['avg_return']:+.2f}%，显示上涨倾向。\n")
                else:
                    f.write(f"2. **异常后中期方向不明确**: 异常后5日平均收益率为{stats_5d['avg_return']:+.2f}%，接近随机。\n")
            
            # 基于波动率变化
            if 'volatility_change' in data['performance_stats']:
                vol_change = data['performance_stats']['volatility_change']
                if vol_change > 20:
                    f.write(f"3. **异常后波动率显著增加**: 波动率增加{vol_change:+.1f}%，风险加大。\n")
                elif vol_change < -20:
                    f.write(f"3. **异常后波动率显著下降**: 波动率下降{abs(vol_change):.1f}%，市场趋于平静。\n")
            
            f.write("\n### 时间关系分析\n\n")
            
            # 基于事件研究
            if 'pre_stats' in event and 'post_stats' in event:
                pre_mean = event['pre_stats']['mean']
                post_mean = event['post_stats']['mean']
                if pre_mean > 0 and post_mean < 0:
                    f.write(f"1. **异常可能是顶点**: 异常前上涨({pre_mean:+.2f}%)，异常后下跌({post_mean:+.2f}%)。\n")
                elif pre_mean < 0 and post_mean > 0:
                    f.write(f"1. **异常可能是底部**: 异常前下跌({pre_mean:+.2f}%)，异常后反弹({post_mean:+.2f}%)。\n")
                else:
                    f.write(f"1. **异常前后的趋势延续**: 异常前{pre_mean:+.2f}%，异常后{post_mean:+.2f}%。\n")
            
            f.write("\n### 交易策略建议\n\n")
            
            # 基于分析结果给出建议
            stats_1d = data['performance_stats'].get('1d', {})
            if stats_1d and abs(stats_1d['avg_return']) < 0.5 and abs(stats_1d['win_rate'] - 50) < 5:
                f.write("1. **异常后1-3天观望**: 短期收益率接近随机，方向不明确。\n")
            
            if stats_5d and stats_5d['avg_return'] < -1:
                f.write("2. **异常后5-10天可考虑做空方向**: 中期有下跌趋势，但需控制仓位。\n")
            
            if 'volatility_change' in data['performance_stats'] and data['performance_stats']['volatility_change'] > 20:
                f.write("3. **异常后降低仓位**: 波动率显著增加，风险加大。\n")
            
            f.write("\n---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    parser = argparse.ArgumentParser(description='港股异常因果关系分析')
    parser.add_argument('--start-date', type=str, default='2025-04-01',
                        help='开始日期（格式：YYYY-MM-DD）')
    parser.add_argument('--end-date', type=str, default='2026-04-01',
                        help='结束日期（格式：YYYY-MM-DD）')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--stocks', nargs='+', type=str, default=None,
                        help='指定股票列表（默认使用WATCHLIST）')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = HKStockAnomalyCausalityAnalyzer(
        start_date=args.start_date,
        end_date=args.end_date,
        stocks=args.stocks
    )
    
    # 加载数据
    if not analyzer.load_stock_data():
        logger.error("数据加载失败，退出分析")
        return
    
    # 检测异常
    if not analyzer.detect_anomalies_batch():
        logger.warning("未检测到异常，但仍继续分析")
    
    # 生成报告
    report_file = analyzer.generate_report(args.output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("分析完成！")
    logger.info(f"报告文件: {report_file}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
