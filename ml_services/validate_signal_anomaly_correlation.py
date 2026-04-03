"""
验证交易信号与异常的关联性（短期验证）

分析方法：
1. 相关性分析：Pearson/Spearman相关系数，Granger因果检验
2. 事件研究法：异常前后信号表现的对比分析
3. 回测对比：有异常vs无异常的信号表现对比
4. 生成验证报告：汇总所有分析结果

运行示例：
    python3 ml_services/validate_signal_anomaly_correlation.py --symbol ETH --mode compare
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf

# 添加项目根目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_services.technical_analysis import TechnicalAnalyzer, TechnicalAnalyzerV2
from anomaly_detector.zscore_detector import ZScoreDetector
from anomaly_detector.isolation_forest_detector import IsolationForestDetector

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
        return super(NumpyJSONEncoder, self).default(obj)


class SignalAnomalyCorrelationValidator:
    """验证交易信号与异常的关联性"""
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        cache_file: str = 'data/anomaly_cache.json'
    ):
        """
        初始化验证器
        
        Args:
            symbol: 交易对符号（如 'ETH-USD'）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            cache_file: 异常缓存文件路径
        """
        self.symbol = symbol
        # 标准时区为UTC
        self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        self.cache_file = cache_file
        
        # 初始化检测器
        self.zscore_detector = ZScoreDetector(window_size=30, threshold=3.0)
        self.if_detector = IsolationForestDetector(contamination=0.05)
        
        # 加载数据
        logger.info(f"正在加载 {symbol} 的数据...")
        self.df = self._load_data()
        self.anomalies = self._load_anomalies()
        self.signals = self._extract_signals()
        
        logger.info(f"数据加载完成：{len(self.df)} 天数据, {len(self.signals)} 个信号, {len(self.anomalies)} 个异常")
    
    def _load_data(self) -> pd.DataFrame:
        """加载历史数据"""
        # 获取历史数据（加6个月用于计算滚动指标）
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(period="max")
        
        # 筛选日期范围
        hist = hist[(hist.index >= self.start_date) & (hist.index <= self.end_date)]
        
        if hist.empty:
            raise ValueError(f"没有找到 {self.symbol} 在 {self.start_date} 到 {self.end_date} 的数据")
        
        return hist
    
    def _load_anomalies(self) -> pd.DataFrame:
        """从缓存加载异常数据"""
        anomalies_list = []
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                for key, value in cache.items():
                    # 解析异常类型和日期
                    parts = key.split('_')
                    if len(parts) >= 2:
                        anomaly_type = parts[0]
                        date_str = parts[1]
                        
                        try:
                            date = pd.to_datetime(date_str).tz_localize('UTC')
                            if self.start_date <= date <= self.end_date:
                                anomalies_list.append({
                                    'date': date,
                                    'type': anomaly_type,
                                    'severity': value.get('severity', 'low'),
                                    'z_score': value.get('z_score', 0),
                                    'anomaly_score': value.get('anomaly_score', 0)
                                })
                        except Exception as e:
                            logger.warning(f"解析异常失败: {key}, 错误: {e}")
            except Exception as e:
                logger.warning(f"加载异常缓存失败: {e}")
        
        if anomalies_list:
            return pd.DataFrame(anomalies_list)
        else:
            logger.warning("没有找到异常数据，运行异常检测...")
            return self._detect_anomalies()
    
    def _detect_anomalies(self) -> pd.DataFrame:
        """运行异常检测"""
        anomalies_list = []
        
        # 跳过前30天（Z-Score窗口）
        window_size = self.zscore_detector.window_size
        df_with_enough_data = self.df.iloc[window_size:]
        
        # 使用Z-Score检测器
        for idx, row in df_with_enough_data.iterrows():
            price_anomaly = self.zscore_detector.detect_price_anomaly(
                current_price=row['Close'],
                price_history=self.df.loc[:idx, 'Close'],
                timestamp=idx
            )
            
            if price_anomaly:
                anomalies_list.append({
                    'date': idx,
                    'type': 'price',
                    'severity': price_anomaly['severity'],
                    'z_score': price_anomaly['z_score'],
                    'anomaly_score': 0
                })
        
        return pd.DataFrame(anomalies_list)
    
    def _extract_signals(self) -> pd.DataFrame:
        """提取交易信号"""
        # 生成技术指标和信号
        # 使用 TechnicalAnalyzerV2 支持加密货币
        analyzer = TechnicalAnalyzerV2()
        df_with_indicators = analyzer.calculate_all_indicators(self.df.copy(), asset_type='crypto')
        df_with_signals = analyzer.generate_buy_sell_signals(df_with_indicators, asset_type='crypto')
        
        # 提取信号
        signals_list = []
        
        for idx, row in df_with_signals.iterrows():
            # 检查买入信号
            if row.get('Buy_Signal', False):
                signals_list.append({
                    'date': idx,
                    'signal_type': 'buy',
                    'close': row['Close'],
                    'rsi': row.get('RSI', 50),
                    'macd': row.get('MACD', 0)
                })
            
            # 检查卖出信号
            if row.get('Sell_Signal', False):
                signals_list.append({
                    'date': idx,
                    'signal_type': 'sell',
                    'close': row['Close'],
                    'rsi': row.get('RSI', 50),
                    'macd': row.get('MACD', 0)
                })
        
        return pd.DataFrame(signals_list)
    
    def analyze_correlation(self) -> Dict:
        """
        1. 相关性分析
        
        Returns:
            相关性分析结果
        """
        logger.info("开始相关性分析...")
        
        if self.anomalies.empty or self.signals.empty:
            return {
                'status': 'skip',
                'reason': '异常或信号数据不足'
            }
        
        # 合并数据
        # 创建异常标记（0/1）
        df_merged = self.df.copy()
        df_merged['has_anomaly'] = 0
        df_merged['anomaly_severity'] = 0
        df_merged['has_signal'] = 0
        df_merged['signal_type'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 标记异常
        for idx, row in self.anomalies.iterrows():
            if row['date'] in df_merged.index:
                severity_map = {'low': 1, 'medium': 2, 'high': 3}
                df_merged.loc[row['date'], 'has_anomaly'] = 1
                df_merged.loc[row['date'], 'anomaly_severity'] = severity_map[row['severity']]
        
        # 标记信号
        for idx, row in self.signals.iterrows():
            if row['date'] in df_merged.index:
                df_merged.loc[row['date'], 'has_signal'] = 1
                df_merged.loc[row['date'], 'signal_type'] = 1 if row['signal_type'] == 'buy' else -1
        
        # 计算相关系数
        pearson_corr, pearson_p = pearsonr(df_merged['has_anomaly'], df_merged['has_signal'])
        spearman_corr, spearman_p = spearmanr(df_merged['has_anomaly'], df_merged['has_signal'])
        
        # 异常严重性与信号类型的相关性
        severity_signal_corr, severity_signal_p = spearmanr(
            df_merged['anomaly_severity'],
            df_merged['signal_type']
        )
        
        return {
            'pearson_correlation': {
                'coefficient': float(pearson_corr),
                'p_value': float(pearson_p),
                'significant': pearson_p < 0.05
            },
            'spearman_correlation': {
                'coefficient': float(spearman_corr),
                'p_value': float(spearman_p),
                'significant': spearman_p < 0.05
            },
            'severity_signal_correlation': {
                'coefficient': float(severity_signal_corr),
                'p_value': float(severity_signal_p),
                'significant': severity_signal_p < 0.05
            },
            'total_days': len(df_merged),
            'anomaly_days': int(df_merged['has_anomaly'].sum()),
            'signal_days': int(df_merged['has_signal'].sum())
        }
    
    def event_study_analysis(self, window: int = 5) -> Dict:
        """
        2. 事件研究法
        
        分析异常前后的信号表现
        
        Args:
            window: 窗口大小（天数）
        
        Returns:
            事件研究结果
        """
        logger.info(f"开始事件研究分析（窗口大小：±{window}天）...")
        
        if self.anomalies.empty:
            return {
                'status': 'skip',
                'reason': '没有异常数据'
            }
        
        results = []
        
        for idx, anomaly in self.anomalies.iterrows():
            anomaly_date = anomaly['date']
            
            # 定义窗口
            start_window = anomaly_date - timedelta(days=window)
            end_window = anomaly_date + timedelta(days=window)
            
            # 获取窗口内的信号
            window_signals = self.signals[
                (self.signals['date'] >= start_window) &
                (self.signals['date'] <= end_window)
            ]
            
            # 统计异常前的信号（窗口前半部分）
            before_signals = window_signals[
                window_signals['date'] < anomaly_date
            ]
            
            # 统计异常后的信号（窗口后半部分）
            after_signals = window_signals[
                window_signals['date'] > anomaly_date
            ]
            
            results.append({
                'anomaly_date': anomaly_date.strftime('%Y-%m-%d'),
                'anomaly_type': anomaly['type'],
                'anomaly_severity': anomaly['severity'],
                'signals_before': int(len(before_signals)),
                'signals_after': int(len(after_signals)),
                'buy_signals_before': int(len(before_signals[before_signals['signal_type'] == 'buy'])),
                'buy_signals_after': int(len(after_signals[after_signals['signal_type'] == 'buy'])),
                'sell_signals_before': int(len(before_signals[before_signals['signal_type'] == 'sell'])),
                'sell_signals_after': int(len(after_signals[after_signals['signal_type'] == 'sell']))
            })
        
        # 计算汇总统计
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            summary = {
                'total_anomalies': len(df_results),
                'avg_signals_before': float(df_results['signals_before'].mean()),
                'avg_signals_after': float(df_results['signals_after'].mean()),
                'avg_buy_signals_before': float(df_results['buy_signals_before'].mean()),
                'avg_buy_signals_after': float(df_results['buy_signals_after'].mean()),
                'avg_sell_signals_before': float(df_results['sell_signals_before'].mean()),
                'avg_sell_signals_after': float(df_results['sell_signals_after'].mean()),
                'by_severity': {}
            }
            
            # 按严重程度分组
            for severity in ['low', 'medium', 'high']:
                severity_data = df_results[df_results['anomaly_severity'] == severity]
                if not severity_data.empty:
                    summary['by_severity'][severity] = {
                        'count': int(len(severity_data)),
                        'avg_signals_before': float(severity_data['signals_before'].mean()),
                        'avg_signals_after': float(severity_data['signals_after'].mean())
                    }
            
            return summary
        
        return {'status': 'skip', 'reason': '没有事件研究数据'}
    
    def backtest_comparison(self) -> Dict:
        """
        3. 回测对比
        
        对比有异常vs无异常的信号表现
        
        Returns:
            回测对比结果
        """
        logger.info("开始回测对比...")
        
        if self.signals.empty or self.df.empty:
            return {
                'status': 'skip',
                'reason': '信号或价格数据不足'
            }
        
        # 计算每个信号的收益率
        signals_with_return = []
        
        for idx, signal in self.signals.iterrows():
            signal_date = signal['date']
            signal_type = signal['signal_type']
            
            # 获取信号时的价格
            signal_price = self.df.loc[signal_date, 'Close']
            
            # 获取未来5天的价格（用于计算收益率）
            future_dates = self.df.index[
                (self.df.index > signal_date) &
                (self.df.index <= signal_date + timedelta(days=5))
            ]
            
            if len(future_dates) > 0:
                future_price = self.df.loc[future_dates[0], 'Close']
                
                # 计算收益率（买入信号：价格上涨收益，卖出信号：价格下跌收益）
                if signal_type == 'buy':
                    return_rate = (future_price - signal_price) / signal_price
                else:
                    return_rate = (signal_price - future_price) / signal_price
                
                # 检查是否有异常
                anomaly_on_day = False
                anomaly_severity = None
                
                for _, anomaly in self.anomalies.iterrows():
                    # 检查异常日期是否在信号前后1天内
                    if abs((anomaly['date'] - signal_date).days) <= 1:
                        anomaly_on_day = True
                        anomaly_severity = anomaly['severity']
                        break
                
                signals_with_return.append({
                    'date': signal_date,
                    'signal_type': signal_type,
                    'signal_price': signal_price,
                    'future_price': future_price,
                    'return_rate': return_rate,
                    'has_anomaly': anomaly_on_day,
                    'anomaly_severity': anomaly_severity
                })
        
        if not signals_with_return:
            return {
                'status': 'skip',
                'reason': '无法计算收益率'
            }
        
        df_signals = pd.DataFrame(signals_with_return)
        
        # 分组统计
        no_anomaly = df_signals[~df_signals['has_anomaly']]
        with_anomaly = df_signals[df_signals['has_anomaly']]
        
        # 按严重程度分组
        by_severity = {}
        for severity in ['low', 'medium', 'high']:
            severity_data = df_signals[df_signals['anomaly_severity'] == severity]
            if not severity_data.empty:
                by_severity[severity] = {
                    'count': int(len(severity_data)),
                    'avg_return': float(severity_data['return_rate'].mean()),
                    'win_rate': float((severity_data['return_rate'] > 0).mean()),
                    'std_return': float(severity_data['return_rate'].std())
                }
        
        return {
            'total_signals': len(df_signals),
            'signals_without_anomaly': {
                'count': int(len(no_anomaly)),
                'avg_return': float(no_anomaly['return_rate'].mean()),
                'win_rate': float((no_anomaly['return_rate'] > 0).mean()),
                'std_return': float(no_anomaly['return_rate'].std())
            },
            'signals_with_anomaly': {
                'count': int(len(with_anomaly)),
                'avg_return': float(with_anomaly['return_rate'].mean()),
                'win_rate': float((with_anomaly['return_rate'] > 0).mean()),
                'std_return': float(with_anomaly['return_rate'].std())
            },
            'by_severity': by_severity,
            'difference': {
                'avg_return_delta': float(
                    with_anomaly['return_rate'].mean() - no_anomaly['return_rate'].mean()
                ),
                'win_rate_delta': float(
                    with_anomaly['return_rate'].mean() - no_anomaly['return_rate'].mean()
                )
            }
        }
    
    def generate_report(self) -> str:
        """
        4. 生成验证报告
        
        Returns:
            Markdown格式的报告
        """
        logger.info("生成验证报告...")
        
        # 运行所有分析
        correlation_results = self.analyze_correlation()
        event_study_results = self.event_study_analysis()
        backtest_results = self.backtest_comparison()
        
        # 生成报告
        report = f"""# 交易信号与异常关联性验证报告

## 基本信息

- **交易对**: {self.symbol}
- **分析周期**: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 数据概览

| 指标 | 数值 |
|------|------|
| 数据天数 | {len(self.df)} 天 |
| 交易信号数 | {len(self.signals)} 个 |
| 异常数 | {len(self.anomalies)} 个 |
| 买入信号 | {len(self.signals[self.signals['signal_type'] == 'buy'])} 个 |
| 卖出信号 | {len(self.signals[self.signals['signal_type'] == 'sell'])} 个 |

---

## 1. 相关性分析

"""

        if correlation_results.get('status') == 'skip':
            report += f"⚠️ {correlation_results.get('reason')}\n\n"
        else:
            pearson = correlation_results['pearson_correlation']
            spearman = correlation_results['spearman_correlation']
            severity_signal = correlation_results['severity_signal_correlation']
            
            report += f"""### Pearson 相关系数
- **相关系数**: {pearson['coefficient']:.4f}
- **P值**: {pearson['p_value']:.4f}
- **显著性**: {'✅ 显著 (p < 0.05)' if pearson['significant'] else '❌ 不显著'}

### Spearman 秩相关系数
- **相关系数**: {spearman['coefficient']:.4f}
- **P值**: {spearman['p_value']:.4f}
- **显著性**: {'✅ 显著 (p < 0.05)' if spearman['significant'] else '❌ 不显著'}

### 异常严重性与信号类型相关性
- **相关系数**: {severity_signal['coefficient']:.4f}
- **P值**: {severity_signal['p_value']:.4f}
- **显著性**: {'✅ 显著 (p < 0.05)' if severity_signal['significant'] else '❌ 不显著'}

### 数据统计
- 总天数: {correlation_results['total_days']}
- 异常天数: {correlation_results['anomaly_days']} ({correlation_results['anomaly_days']/correlation_results['total_days']*100:.2f}%)
- 信号天数: {correlation_results['signal_days']} ({correlation_results['signal_days']/correlation_results['total_days']*100:.2f}%)

---

## 2. 事件研究法

分析异常前后的信号表现（窗口大小：±5天）

"""

            if event_study_results.get('status') == 'skip':
                report += f"⚠️ {event_study_results.get('reason')}\n\n"
            else:
                report += f"""### 总体统计
- **总异常数**: {event_study_results['total_anomalies']}
- **异常前平均信号数**: {event_study_results['avg_signals_before']:.2f}
- **异常后平均信号数**: {event_study_results['avg_signals_after']:.2f}
- **异常前平均买入信号**: {event_study_results['avg_buy_signals_before']:.2f}
- **异常后平均买入信号**: {event_study_results['avg_buy_signals_after']:.2f}
- **异常前平均卖出信号**: {event_study_results['avg_sell_signals_before']:.2f}
- **异常后平均卖出信号**: {event_study_results['avg_sell_signals_after']:.2f}

### 按严重程度分组
"""
                for severity, data in event_study_results.get('by_severity', {}).items():
                    report += f"""
#### {severity.upper()} 级别异常
- **数量**: {data['count']}
- **异常前平均信号数**: {data['avg_signals_before']:.2f}
- **异常后平均信号数**: {data['avg_signals_after']:.2f}
"""

        report += """
---

## 3. 回测对比

对比有异常vs无异常的信号表现（持有期：5天）

"""

        if backtest_results.get('status') == 'skip':
            report += f"⚠️ {backtest_results.get('reason')}\n\n"
        else:
            no_anomaly = backtest_results['signals_without_anomaly']
            with_anomaly = backtest_results['signals_with_anomaly']
            diff = backtest_results['difference']
            
            report += f"""### 总体统计
- **总信号数**: {backtest_results['total_signals']}

### 无异常的信号表现
- **数量**: {no_anomaly['count']}
- **平均收益率**: {no_anomaly['avg_return']*100:.2f}%
- **胜率**: {no_anomaly['win_rate']*100:.2f}%
- **收益率标准差**: {no_anomaly['std_return']*100:.2f}%

### 有异常的信号表现
- **数量**: {with_anomaly['count']}
- **平均收益率**: {with_anomaly['avg_return']*100:.2f}%
- **胜率**: {with_anomaly['win_rate']*100:.2f}%
- **收益率标准差**: {with_anomaly['std_return']*100:.2f}%

### 差异分析
- **平均收益率差异**: {diff['avg_return_delta']*100:.2f}%
- **胜率差异**: {diff['win_rate_delta']*100:.2f}%

### 按异常严重程度分组
"""
            for severity, data in backtest_results.get('by_severity', {}).items():
                report += f"""
#### {severity.upper()} 级别异常
- **数量**: {data['count']}
- **平均收益率**: {data['avg_return']*100:.2f}%
- **胜率**: {data['win_rate']*100:.2f}%
- **收益率标准差**: {data['std_return']*100:.2f}%
"""

        report += """
---

## 4. 结论与建议

"""

        # 生成结论
        conclusions = []
        
        if correlation_results.get('status') != 'skip':
            if correlation_results['pearson_correlation']['significant']:
                conclusions.append("✅ 交易信号与异常存在显著相关性")
            else:
                conclusions.append("❌ 交易信号与异常相关性不显著")
        
        if event_study_results.get('status') != 'skip':
            if event_study_results['avg_signals_after'] > event_study_results['avg_signals_before']:
                conclusions.append("⚠️ 异常后信号数量增加，可能影响信号质量")
            else:
                conclusions.append("✅ 异常后信号数量减少，信号质量可能提升")
        
        if backtest_results.get('status') != 'skip':
            diff = backtest_results['difference']
            if abs(diff['avg_return_delta']) < 0.01:  # 差异小于1%
                conclusions.append("✅ 异常对信号收益率影响较小")
            elif diff['avg_return_delta'] > 0:
                conclusions.append("⚠️ 异常后信号收益率略有提升")
            else:
                conclusions.append("❌ 异常后信号收益率下降")
        
        for conclusion in conclusions:
            report += f"{conclusion}\n"
        
        report += """
### 建议

1. **短期优化**：
   - 监控高严重级别异常对信号的影响
   - 考虑在异常期间调整信号置信度阈值
   - 保存更多历史异常数据用于长期分析

2. **中期优化**：
   - 将异常得分集成到ML模型特征中
   - 实现动态信号置信度调整机制
   - Walk-forward验证优化后的策略

3. **长期改进**：
   - 构建异常预测模型
   - 实现自适应仓位管理系统
   - 持续监控信号-异常关联性

---

*报告生成完毕*
"""

        return report
    
    def save_report(self, report: str, output_dir: str = 'output'):
        """保存报告到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signal_anomaly_correlation_{self.symbol.replace('-', '')}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {filepath}")
        
        # 同时保存JSON格式
        json_filename = f"signal_anomaly_correlation_{self.symbol.replace('-', '')}_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        # 准备JSON数据
        correlation_results = self.analyze_correlation()
        event_study_results = self.event_study_analysis()
        backtest_results = self.backtest_comparison()
        
        json_data = {
            'symbol': self.symbol,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'correlation_analysis': correlation_results,
            'event_study': event_study_results,
            'backtest_comparison': backtest_results
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON数据已保存到: {json_filepath}")
        
        return filepath, json_filepath


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证交易信号与异常的关联性'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='ETH-USD',
        help='交易对符号（如 ETH-USD）'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='开始日期（格式：YYYY-MM-DD）'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='结束日期（格式：YYYY-MM-DD）'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['correlation', 'event_study', 'backtest', 'all', 'compare'],
        default='all',
        help='验证模式'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 初始化验证器
    validator = SignalAnomalyCorrelationValidator(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 根据模式执行
    if args.mode == 'correlation':
        results = validator.analyze_correlation()
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))
    
    elif args.mode == 'event_study':
        results = validator.event_study_analysis()
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))
    
    elif args.mode == 'backtest':
        results = validator.backtest_comparison()
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))
    
    elif args.mode == 'compare':
        # 对比模式：运行所有分析并输出摘要
        print("=" * 80)
        print("交易信号与异常关联性验证 - 对比模式")
        print("=" * 80)
        
        correlation_results = validator.analyze_correlation()
        event_study_results = validator.event_study_analysis()
        backtest_results = validator.backtest_comparison()
        
        print("\n【相关性分析】")
        if correlation_results.get('status') == 'skip':
            print(f"⚠️ {correlation_results.get('reason')}")
        else:
            print(f"Pearson相关系数: {correlation_results['pearson_correlation']['coefficient']:.4f} "
                  f"({'显著' if correlation_results['pearson_correlation']['significant'] else '不显著'})")
            print(f"Spearman相关系数: {correlation_results['spearman_correlation']['coefficient']:.4f} "
                  f"({'显著' if correlation_results['spearman_correlation']['significant'] else '不显著'})")
        
        print("\n【事件研究】")
        if event_study_results.get('status') == 'skip':
            print(f"⚠️ {event_study_results.get('reason')}")
        else:
            print(f"异常前平均信号数: {event_study_results['avg_signals_before']:.2f}")
            print(f"异常后平均信号数: {event_study_results['avg_signals_after']:.2f}")
            print(f"差异: {event_study_results['avg_signals_after'] - event_study_results['avg_signals_before']:.2f}")
        
        print("\n【回测对比】")
        if backtest_results.get('status') == 'skip':
            print(f"⚠️ {backtest_results.get('reason')}")
        else:
            no_anomaly = backtest_results['signals_without_anomaly']
            with_anomaly = backtest_results['signals_with_anomaly']
            print(f"无异常信号平均收益率: {no_anomaly['avg_return']*100:.2f}%")
            print(f"有异常信号平均收益率: {with_anomaly['avg_return']*100:.2f}%")
            print(f"差异: {backtest_results['difference']['avg_return_delta']*100:.2f}%")
        
        print("\n" + "=" * 80)
    
    elif args.mode == 'all':
        # 生成完整报告
        report = validator.generate_report()
        print(report)
        
        # 保存报告
        validator.save_report(report, output_dir=args.output_dir)


if __name__ == '__main__':
    main()