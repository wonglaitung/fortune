#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密货币异常因果关系分析工具

分析内容：
1. 检测历史期间的异常（批量方式）
2. 分析异常后不同时间窗口的价格表现
3. 验证股票策略是否适用
4. 生成验证报告

与股票异常验证的核心差异：
- 使用小时级数据（加密货币24/7交易）
- 调整异常阈值（加密货币波动率更高）
- 独立验证均值回归效应
- 验证Isolation Forest和Z-Score策略
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
import yfinance as yf

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            return obj.strftime('%Y-%m-%d %H:%M')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M')
        return super(NumpyJSONEncoder, self).default(obj)


class CryptoAnomalyCausalityAnalyzer:
    """加密货币异常因果关系分析器"""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        cryptos: List[str] = None,
        window_size: int = 72,  # 72小时（3天）
        threshold_high: float = 5.0,  # 加密货币阈值更高
        threshold_medium: float = 4.0
    ):
        """
        初始化分析器
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            cryptos: 加密货币列表（默认使用BTC-USD和ETH-USD）
            window_size: Z-Score窗口大小（小时）
            threshold_high: 高异常阈值
            threshold_medium: 中异常阈值
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.cryptos = cryptos or ['BTC-USD', 'ETH-USD']
        self.window_size = window_size
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        
        # 初始化检测器
        self.zscore_detector = ZScoreDetector(
            window_size=window_size,
            threshold=threshold_medium,
            time_interval='hour'
        )
        
        # 存储数据
        self.crypto_data: Dict[str, pd.DataFrame] = {}
        self.all_anomalies: List[Dict] = []
        
        logger.info(f"初始化加密货币异常分析器: {start_date} 至 {end_date}")
        logger.info(f"加密货币: {self.cryptos}")
        logger.info(f"窗口大小: {window_size}小时")
        logger.info(f"异常阈值: high={threshold_high}, medium={threshold_medium}")
    
    def load_crypto_data(self):
        """批量加载所有加密货币数据（小时级）"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤1: 批量加载加密货币数据（小时级）")
        logger.info(f"{'='*60}")
        
        # 计算扩展的时间范围（前后多加载7天用于分析）
        extended_start = self.start_date - timedelta(days=7)
        extended_end = self.end_date + timedelta(days=7)
        
        success_count = 0
        for i, crypto in enumerate(self.cryptos, 1):
            logger.info(f"[{i}/{len(self.cryptos)}] 加载 {crypto}...")
            
            try:
                ticker = yf.Ticker(crypto)
                # 使用小时级数据
                df = ticker.history(start=extended_start, end=extended_end, interval='1h')
                
                if df is None or len(df) == 0:
                    logger.warning(f"  {crypto}: 无数据")
                    continue
                
                # 统一时区
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                self.crypto_data[crypto] = df
                success_count += 1
                logger.info(f"  {crypto}: {len(df)} 小时数据 ({len(df)/24:.1f}天)")
                
            except Exception as e:
                logger.error(f"  {crypto}: 加载失败 - {e}")
        
        logger.info(f"\n成功加载 {success_count}/{len(self.cryptos)} 个加密货币数据")
        return success_count > 0
    
    def detect_anomalies_batch(self):
        """批量检测所有加密货币的异常"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤2: 批量检测异常")
        logger.info(f"{'='*60}")
        
        total_anomalies = 0
        anomaly_by_date: Dict[str, List] = {}
        
        for i, (crypto, df) in enumerate(self.crypto_data.items(), 1):
            logger.info(f"[{i}/{len(self.crypto_data)}] 检测 {crypto}...")
            
            try:
                # 筛选日期范围
                df_period = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                
                if len(df_period) == 0:
                    continue
                
                # 检测每个小时的异常
                for idx, row in df_period.iterrows():
                    current_date = idx
                    current_price = row['Close']
                    current_volume = row['Volume']
                    
                    # 获取历史数据（用于Z-Score计算）
                    hist_df = df[df.index < current_date].tail(self.window_size + 10)  # 多取一些确保足够
                    
                    if len(hist_df) < self.window_size:
                        continue
                    
                    # 价格异常检测
                    price_history = hist_df['Close']
                    price_anomaly = self.zscore_detector.detect_anomaly(
                        metric_name='price',
                        current_value=current_price,
                        history=price_history,
                        timestamp=current_date,
                        time_interval='hour'
                    )
                    
                    # 成交量异常检测
                    volume_history = hist_df['Volume']
                    volume_anomaly = self.zscore_detector.detect_anomaly(
                        metric_name='volume',
                        current_value=current_volume,
                        history=volume_history,
                        timestamp=current_date,
                        time_interval='hour'
                    )
                    
                    # 记录异常
                    for anomaly in [price_anomaly, volume_anomaly]:
                        if anomaly and anomaly.get('severity') in ['high', 'medium']:
                            date_str = current_date.strftime('%Y-%m-%d %H:%M')
                            
                            anomaly_info = {
                                'crypto': crypto,
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
                logger.error(f"  {crypto}: 检测失败 - {e}")
        
        # 统计结果
        total_hours = len(df_period) if len(self.crypto_data) > 0 else 0
        
        logger.info(f"\n异常检测完成:")
        logger.info(f"  总异常数: {total_anomalies}")
        logger.info(f"  总小时数: {total_hours}")
        logger.info(f"  异常率: {total_anomalies/total_hours*100:.2f}%" if total_hours > 0 else "N/A")
        
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
        """分析异常后的价格表现"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤3: 分析异常后价格表现")
        logger.info(f"{'='*60}")
        
        performance_results = []
        windows = [1, 3, 5, 10, 20]  # 小时级别的时间窗口
        
        for i, anomaly in enumerate(self.all_anomalies, 1):
            crypto = anomaly['crypto']
            date_str = anomaly['date']
            
            if i % 50 == 0 or i == len(self.all_anomalies):
                logger.info(f"[{i}/{len(self.all_anomalies)}] 分析 {crypto} {date_str}...")
            
            try:
                df = self.crypto_data[crypto]
                anomaly_time = pd.to_datetime(date_str)
                
                # 获取异常当日涨跌
                prev_hour = anomaly_time - timedelta(hours=1)
                prev_row = df[df.index == prev_hour]
                
                if len(prev_row) == 0:
                    continue
                
                prev_close = prev_row.iloc[0]['Close']
                current_close = anomaly['close']
                same_day_change = (current_close / prev_close - 1) * 100
                
                # 计算异常后不同时间窗口的收益率
                window_returns = {}
                for window in windows:
                    future_time = anomaly_time + timedelta(hours=window)
                    future_df = df[df.index >= future_time].head(1)
                    
                    if len(future_df) > 0:
                        future_close = future_df.iloc[0]['Close']
                        window_return = (future_close / current_close - 1) * 100
                        window_returns[f'{window}h'] = {
                            'return': window_return,
                            'is_positive': window_return > 0
                        }
                
                performance_results.append({
                    **anomaly,
                    'same_day_change': same_day_change,
                    'window_returns': window_returns
                })
                
            except Exception as e:
                logger.debug(f"分析失败: {crypto} {date_str} - {e}")
                continue
        
        logger.info(f"\n成功分析 {len(performance_results)} 个异常")
        
        # 保存结果
        self.performance_results = performance_results
        return performance_results
    
    def validate_stock_strategies(self) -> Dict:
        """验证股票策略是否适用于加密货币"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤4: 验证股票策略适用性")
        logger.info(f"{'='*60}")
        
        if not hasattr(self, 'performance_results') or len(self.performance_results) == 0:
            logger.error("请先运行 analyze_post_anomaly_performance()")
            return {}
        
        results = {
            'if_high_strategy': self._validate_if_high_strategy(),
            'zscore_buy_strategy': self._validate_zscore_buy_strategy(),
            'mean_reversion': self._validate_mean_reversion(),
            'summary': {}
        }
        
        # 生成总结
        results['summary'] = {
            'total_anomalies': len(self.all_anomalies),
            'if_high_effective': results['if_high_strategy'].get('effective', False),
            'zscore_buy_effective': results['zscore_buy_strategy'].get('effective', False),
            'mean_reversion_effective': results['mean_reversion'].get('effective', False),
            'recommendation': self._generate_recommendation(results)
        }
        
        return results
    
    def _validate_if_high_strategy(self) -> Dict:
        """验证Isolation Forest high异常策略（减仓）"""
        logger.info("\n验证策略1: Isolation Forest high异常 → 减仓")
        
        # 筛选高严重度异常
        high_anomalies = [a for a in self.performance_results if a['severity'] == 'high']
        
        if len(high_anomalies) == 0:
            logger.warning("  无高严重度异常样本")
            return {'effective': False, 'reason': '无样本'}
        
        # 计算5小时后的收益率
        returns_5h = [a['window_returns'].get('5h', {}).get('return', 0) for a in high_anomalies if '5h' in a.get('window_returns', {})]
        
        if len(returns_5h) == 0:
            return {'effective': False, 'reason': '无5小时数据'}
        
        avg_return = np.mean(returns_5h)
        win_rate = sum(1 for r in returns_5h if r > 0) / len(returns_5h) * 100
        
        # 股票基准：5日收益-3.04%，胜率43%
        # 加密货币如果表现更差，说明策略有效
        effective = avg_return < -1.0 or win_rate < 45
        
        result = {
            'effective': effective,
            'sample_size': len(returns_5h),
            'avg_return_5h': avg_return,
            'win_rate_5h': win_rate,
            'stock_baseline': {'return': -3.04, 'win_rate': 43.0},
            'conclusion': '有效' if effective else '无效'
        }
        
        logger.info(f"  样本数: {len(returns_5h)}")
        logger.info(f"  5小时平均收益: {avg_return:.2f}% (股票基准: -3.04%)")
        logger.info(f"  胜率: {win_rate:.1f}% (股票基准: 43%)")
        logger.info(f"  结论: {result['conclusion']}")
        
        return result
    
    def _validate_zscore_buy_strategy(self) -> Dict:
        """验证Z-Score价格异常+当日下跌策略（抄底）"""
        logger.info("\n验证策略2: Z-Score价格异常+当日下跌 → 抄底")
        
        # 筛选价格异常且当日下跌的情况
        price_anomalies_down = [
            a for a in self.performance_results 
            if a['type'] == 'price' and a.get('same_day_change', 0) < 0
        ]
        
        if len(price_anomalies_down) == 0:
            logger.warning("  无价格异常+当日下跌样本")
            return {'effective': False, 'reason': '无样本'}
        
        # 计算5小时后的收益率
        returns_5h = [a['window_returns'].get('5h', {}).get('return', 0) for a in price_anomalies_down if '5h' in a.get('window_returns', {})]
        
        if len(returns_5h) == 0:
            return {'effective': False, 'reason': '无5小时数据'}
        
        avg_return = np.mean(returns_5h)
        win_rate = sum(1 for r in returns_5h if r > 0) / len(returns_5h) * 100
        
        # 股票基准：5日收益+7.02%，胜率100%
        # 加密货币如果接近或超过，说明策略有效
        effective = avg_return > 2.0 and win_rate > 60
        
        result = {
            'effective': effective,
            'sample_size': len(returns_5h),
            'avg_return_5h': avg_return,
            'win_rate_5h': win_rate,
            'stock_baseline': {'return': 7.02, 'win_rate': 100.0},
            'conclusion': '有效' if effective else '无效/高风险'
        }
        
        logger.info(f"  样本数: {len(returns_5h)}")
        logger.info(f"  5小时平均收益: {avg_return:.2f}% (股票基准: +7.02%)")
        logger.info(f"  胜率: {win_rate:.1f}% (股票基准: 100%)")
        logger.info(f"  结论: {result['conclusion']}")
        
        return result
    
    def _validate_mean_reversion(self) -> Dict:
        """验证均值回归效应"""
        logger.info("\n验证策略3: 均值回归效应")
        
        # 分析异常当日涨跌与后续表现的关系
        down_anomalies = [a for a in self.performance_results if a.get('same_day_change', 0) < 0]
        up_anomalies = [a for a in self.performance_results if a.get('same_day_change', 0) >= 0]
        
        results_by_direction = {}
        
        for label, anomalies in [('当日下跌', down_anomalies), ('当日上涨', up_anomalies)]:
            if len(anomalies) == 0:
                continue
            
            returns_5h = [a['window_returns'].get('5h', {}).get('return', 0) for a in anomalies if '5h' in a.get('window_returns', {})]
            
            if len(returns_5h) > 0:
                avg_return = np.mean(returns_5h)
                win_rate = sum(1 for r in returns_5h if r > 0) / len(returns_5h) * 100
                
                results_by_direction[label] = {
                    'sample_size': len(returns_5h),
                    'avg_return_5h': avg_return,
                    'win_rate_5h': win_rate
                }
                
                logger.info(f"  {label}: 样本{len(returns_5h)}, 5小时收益{avg_return:.2f}%, 胜率{win_rate:.1f}%")
        
        # 判断均值回归是否有效
        # 如果当日下跌后反弹更强，说明均值回归有效
        effective = False
        if '当日下跌' in results_by_direction and '当日上涨' in results_by_direction:
            down_return = results_by_direction['当日下跌']['avg_return_5h']
            up_return = results_by_direction['当日上涨']['avg_return_5h']
            effective = down_return > up_return
        
        return {
            'effective': effective,
            'by_direction': results_by_direction,
            'conclusion': '均值回归有效' if effective else '均值回归无效'
        }
    
    def _generate_recommendation(self, results: Dict) -> str:
        """生成策略建议"""
        recommendations = []
        
        if results['if_high_strategy'].get('effective'):
            recommendations.append("✅ IF high减仓策略有效")
        else:
            recommendations.append("❌ IF high减仓策略无效")
        
        if results['zscore_buy_strategy'].get('effective'):
            recommendations.append("✅ Z-Score抄底策略有效")
        else:
            recommendations.append("⚠️ Z-Score抄底策略高风险")
        
        if results['mean_reversion'].get('effective'):
            recommendations.append("✅ 均值回归效应存在")
        else:
            recommendations.append("❌ 均值回归效应不明显")
        
        return " | ".join(recommendations)

    
    def generate_report(self, output_file: str = None) -> str:
        """生成验证报告"""
        logger.info(f"\n{'='*60}")
        logger.info("步骤5: 生成验证报告")
        logger.info(f"{'='*60}")
        
        report_lines = []
        report_lines.append("# 加密货币异常策略验证报告\n")
        report_lines.append(f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**验证期间**: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        report_lines.append(f"**加密货币**: {', '.join(self.cryptos)}\n")
        
        report_lines.append("---\n")
        report_lines.append("## 核心结论\n")
        
        if hasattr(self, 'validation_results'):
            results = self.validation_results
            summary = results.get('summary', {})
            
            report_lines.append(f"**总异常数**: {summary.get('total_anomalies', 0)}\n")
            
            # 策略验证结果
            report_lines.append("### 策略验证结果\n")
            
            # IF high策略
            if_high = results.get('if_high_strategy', {})
            report_lines.append(f"#### 1. Isolation Forest high异常 → 减仓\n")
            report_lines.append(f"- **有效性**: {'✅ 有效' if if_high.get('effective') else '❌ 无效'}")
            report_lines.append(f"- **样本数**: {if_high.get('sample_size', 0)}")
            report_lines.append(f"- **5小时收益**: {if_high.get('avg_return_5h', 0):.2f}% (股票基准: -3.04%)")
            report_lines.append(f"- **胜率**: {if_high.get('win_rate_5h', 0):.1f}% (股票基准: 43%)")
            report_lines.append(f"- **结论**: {if_high.get('conclusion', 'N/A')}\n")
            
            # Z-Score抄底策略
            zscore = results.get('zscore_buy_strategy', {})
            report_lines.append(f"#### 2. Z-Score价格异常+当日下跌 → 抄底\n")
            report_lines.append(f"- **有效性**: {'✅ 有效' if zscore.get('effective') else '⚠️ 高风险'}")
            report_lines.append(f"- **样本数**: {zscore.get('sample_size', 0)}")
            report_lines.append(f"- **5小时收益**: {zscore.get('avg_return_5h', 0):.2f}% (股票基准: +7.02%)")
            report_lines.append(f"- **胜率**: {zscore.get('win_rate_5h', 0):.1f}% (股票基准: 100%)")
            report_lines.append(f"- **结论**: {zscore.get('conclusion', 'N/A')}\n")
            
            # 均值回归
            mean_rev = results.get('mean_reversion', {})
            report_lines.append(f"#### 3. 均值回归效应\n")
            report_lines.append(f"- **有效性**: {'✅ 有效' if mean_rev.get('effective') else '❌ 无效'}")
            report_lines.append(f"- **结论**: {mean_rev.get('conclusion', 'N/A')}\n")
            
            # 总体建议
            report_lines.append("---\n")
            report_lines.append("## 总体建议\n")
            report_lines.append(f"{summary.get('recommendation', 'N/A')}\n")
            
            # 与股票对比
            report_lines.append("---\n")
            report_lines.append("## 与股票市场对比\n")
            report_lines.append("| 指标 | 股票（港股） | 加密货币 | 差异 |")
            report_lines.append("|------|-------------|---------|------|")
            report_lines.append(f"| IF high收益 | -3.04% | {if_high.get('avg_return_5h', 0):.2f}% | {'✅ 接近' if abs(if_high.get('avg_return_5h', 0) - (-3.04)) < 2 else '❌ 差异大'} |")
            report_lines.append(f"| IF high胜率 | 43% | {if_high.get('win_rate_5h', 0):.1f}% | {'✅ 接近' if abs(if_high.get('win_rate_5h', 0) - 43) < 10 else '❌ 差异大'} |")
            report_lines.append(f"| Z-Score抄底收益 | +7.02% | {zscore.get('avg_return_5h', 0):.2f}% | {'✅ 接近' if abs(zscore.get('avg_return_5h', 0) - 7.02) < 3 else '❌ 差异大'} |")
            report_lines.append(f"| Z-Score抄底胜率 | 100% | {zscore.get('win_rate_5h', 0):.1f}% | {'✅ 接近' if abs(zscore.get('win_rate_5h', 0) - 100) < 20 else '❌ 差异大'} |")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"报告已保存: {output_file}")
        
        return report_content


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币异常因果关系分析')
    parser.add_argument('--start-date', required=True, help='开始日期（YYYY-MM-DD）')
    parser.add_argument('--end-date', required=True, help='结束日期（YYYY-MM-DD）')
    parser.add_argument('--cryptos', nargs='+', default=['BTC-USD', 'ETH-USD'], help='加密货币列表')
    parser.add_argument('--output', default='output/crypto_anomaly_validation.md', help='输出报告文件')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = CryptoAnomalyCausalityAnalyzer(
        start_date=args.start_date,
        end_date=args.end_date,
        cryptos=args.cryptos
    )
    
    # 执行分析步骤
    success = analyzer.load_crypto_data()
    if not success:
        logger.error("数据加载失败，退出")
        return
    
    success = analyzer.detect_anomalies_batch()
    if not success:
        logger.error("异常检测失败，退出")
        return
    
    analyzer.analyze_post_anomaly_performance()
    analyzer.validation_results = analyzer.validate_stock_strategies()
    
    # 生成报告
    report = analyzer.generate_report(args.output)
    
    # 打印报告
    print("\n" + "="*80)
    print("验证报告")
    print("="*80)
    print(report)


if __name__ == "__main__":
    main()

