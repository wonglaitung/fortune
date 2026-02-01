# -*- coding: utf-8 -*-
"""
港股板块分析模块 - 轻量级版本
功能：
1. 板块涨跌幅排名
2. 板块技术趋势分析
3. 板块龙头识别
4. 板块资金流向分析

日期：2026-02-01
"""

import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 导入腾讯财经接口
from tencent_finance import get_hk_stock_data_tencent

# 导入技术分析工具
try:
    from technical_analysis import TechnicalAnalyzer
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False
    print("⚠️ 技术分析工具不可用，部分功能将受限")

# ==============================
# 股票板块映射（复用 ml_trading_model.py）
# ==============================
STOCK_SECTOR_MAPPING = {
    # 银行股
    '0005.HK': {'sector': 'bank', 'name': '汇丰银行'},
    '0939.HK': {'sector': 'bank', 'name': '建设银行'},
    '1288.HK': {'sector': 'bank', 'name': '农业银行'},
    '1398.HK': {'sector': 'bank', 'name': '工商银行'},
    '3968.HK': {'sector': 'bank', 'name': '招商银行'},

    # 科技股
    '0700.HK': {'sector': 'tech', 'name': '腾讯控股'},
    '9988.HK': {'sector': 'tech', 'name': '阿里巴巴-SW'},
    '3690.HK': {'sector': 'tech', 'name': '美团-W'},
    '1810.HK': {'sector': 'tech', 'name': '小米集团-W'},

    # 半导体股
    '0981.HK': {'sector': 'semiconductor', 'name': '中芯国际'},
    '1347.HK': {'sector': 'semiconductor', 'name': '华虹半导体'},

    # 人工智能股
    '6682.HK': {'sector': 'ai', 'name': '第四范式'},
    '9660.HK': {'sector': 'ai', 'name': '地平线机器人'},
    '2533.HK': {'sector': 'ai', 'name': '黑芝麻智能'},

    # 新能源股
    '1211.HK': {'sector': 'new_energy', 'name': '比亚迪股份'},
    '1330.HK': {'sector': 'environmental', 'name': '绿色动力环保'},

    # 能源/周期股
    '0883.HK': {'sector': 'energy', 'name': '中国海洋石油'},
    '1088.HK': {'sector': 'energy', 'name': '中国神华'},
    '1138.HK': {'sector': 'shipping', 'name': '中远海能'},
    '0388.HK': {'sector': 'exchange', 'name': '香港交易所'},

    # 公用事业股
    '0728.HK': {'sector': 'utility', 'name': '中国电信'},
    '0941.HK': {'sector': 'utility', 'name': '中国移动'},

    # 保险股
    '1299.HK': {'sector': 'insurance', 'name': '友邦保险'},

    # 生物医药股
    '2269.HK': {'sector': 'biotech', 'name': '药明生物'},

    # 指数基金
    '2800.HK': {'sector': 'index', 'name': '盈富基金'},
}

# 板块中文名称映射
SECTOR_NAME_MAPPING = {
    'bank': '银行股',
    'tech': '科技股',
    'semiconductor': '半导体',
    'ai': '人工智能',
    'new_energy': '新能源',
    'environmental': '环保',
    'energy': '能源股',
    'shipping': '航运',
    'exchange': '交易所',
    'utility': '公用事业',
    'insurance': '保险',
    'biotech': '生物医药',
    'index': '指数基金',
}


class SectorAnalyzer:
    """板块分析器"""

    def __init__(self, stock_mapping: Optional[Dict] = None):
        """
        初始化板块分析器

        Args:
            stock_mapping: 股票板块映射字典，默认使用内置映射
        """
        self.stock_mapping = stock_mapping or STOCK_SECTOR_MAPPING
        self.sector_name_mapping = SECTOR_NAME_MAPPING

        # 构建板块到股票的反向映射
        self.sector_stocks = {}
        for code, info in self.stock_mapping.items():
            sector = info['sector']
            if sector not in self.sector_stocks:
                self.sector_stocks[sector] = []
            self.sector_stocks[sector].append(code)

    def get_sector_name(self, sector_code: str) -> str:
        """获取板块中文名称"""
        return self.sector_name_mapping.get(sector_code, sector_code)

    def calculate_sector_performance(self, period: int = 1) -> pd.DataFrame:
        """
        计算各板块涨跌幅排名

        Args:
            period: 计算周期（天数），默认1天

        Returns:
            DataFrame: 板块涨跌幅排名，包含板块名称、平均涨跌幅、股票数量
        """
        sector_results = []

        for sector, stocks in self.sector_stocks.items():
            sector_changes = []
            sector_volumes = []
            stock_details = []

            for stock_code in stocks:
                try:
                    # 获取股票数据
                    df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=period + 5)

                    if df is not None and len(df) > period:
                        # 计算涨跌幅
                        if len(df) > 0:
                            latest_close = df['Close'].iloc[-1]
                            prev_close = df['Close'].iloc[-1-period] if len(df) > period else df['Close'].iloc[0]
                            change_pct = (latest_close - prev_close) / prev_close * 100

                            # 获取成交量
                            latest_volume = df['Volume'].iloc[-1] if len(df) > 0 else 0

                            sector_changes.append(change_pct)
                            sector_volumes.append(latest_volume)

                            stock_details.append({
                                'code': stock_code,
                                'name': self.stock_mapping[stock_code]['name'],
                                'change_pct': change_pct,
                                'volume': latest_volume,
                            })
                except Exception as e:
                    print(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
                    continue

            if sector_changes:
                avg_change = np.mean(sector_changes)
                total_volume = sum(sector_volumes)

                # 排序股票详情
                stock_details_sorted = sorted(stock_details, key=lambda x: x['change_pct'], reverse=True)

                sector_results.append({
                    'sector_code': sector,
                    'sector_name': self.get_sector_name(sector),
                    'avg_change_pct': avg_change,
                    'total_volume': total_volume,
                    'stock_count': len(sector_changes),
                    'stocks': stock_details_sorted,
                    'best_stock': stock_details_sorted[0] if stock_details_sorted else None,
                    'worst_stock': stock_details_sorted[-1] if stock_details_sorted else None,
                })

        # 转换为DataFrame并排序
        if sector_results:
            df = pd.DataFrame(sector_results)
            df = df.sort_values('avg_change_pct', ascending=False)
            return df.reset_index(drop=True)
        else:
            return pd.DataFrame()

    def analyze_sector_trend(self, sector_code: str, days: int = 20) -> Dict:
        """
        分析板块技术趋势

        Args:
            sector_code: 板块代码
            days: 分析天数

        Returns:
            Dict: 板块趋势分析结果
        """
        stocks = self.sector_stocks.get(sector_code, [])

        if not stocks:
            return {
                'sector': sector_code,
                'error': '未找到该板块的股票'
            }

        # 获取板块内所有股票的数据
        all_data = []
        for stock_code in stocks:
            try:
                df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=days + 10)
                if df is not None and len(df) > days:
                    # 计算涨跌幅
                    if len(df) > 0:
                        latest_close = df['Close'].iloc[-1]
                        prev_close = df['Close'].iloc[-1-days] if len(df) > days else df['Close'].iloc[0]
                        change_pct = (latest_close - prev_close) / prev_close * 100

                        # 技术指标（如果可用）
                        ma20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
                        ma60 = df['Close'].rolling(window=60).mean().iloc[-1] if len(df) >= 60 else None

                        all_data.append({
                            'code': stock_code,
                            'name': self.stock_mapping[stock_code]['name'],
                            'change_pct': change_pct,
                            'price': latest_close,
                            'ma20': ma20,
                            'ma60': ma60,
                        })
            except Exception as e:
                print(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
                continue

        if not all_data:
            return {
                'sector': sector_code,
                'error': '无法获取该板块的数据'
            }

        # 计算板块平均指标
        avg_change = np.mean([d['change_pct'] for d in all_data])
        rising_count = sum(1 for d in all_data if d['change_pct'] > 0)
        total_count = len(all_data)

        # 判断趋势
        if avg_change > 2 and rising_count / total_count > 0.6:
            trend = '强势上涨'
        elif avg_change > 0 and rising_count / total_count > 0.5:
            trend = '温和上涨'
        elif avg_change < -2 and rising_count / total_count < 0.4:
            trend = '强势下跌'
        elif avg_change < 0 and rising_count / total_count < 0.5:
            trend = '温和下跌'
        else:
            trend = '震荡整理'

        return {
            'sector_code': sector_code,
            'sector_name': self.get_sector_name(sector_code),
            'trend': trend,
            'avg_change_pct': avg_change,
            'rising_count': rising_count,
            'total_count': total_count,
            'rising_ratio': rising_count / total_count * 100,
            'stocks': sorted(all_data, key=lambda x: x['change_pct'], reverse=True),
        }

    def identify_sector_leaders(self, sector_code: str, top_n: int = 3) -> pd.DataFrame:
        """
        识别板块龙头（涨幅最大、成交量最大）

        Args:
            sector_code: 板块代码
            top_n: 返回前N只股票

        Returns:
            DataFrame: 板块龙头股票
        """
        stocks = self.sector_stocks.get(sector_code, [])

        if not stocks:
            return pd.DataFrame()

        stock_data = []
        for stock_code in stocks:
            try:
                df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=5)
                if df is not None and len(df) > 0:
                    # 1日涨跌幅
                    change_pct = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) > 1 else 0

                    # 成交量
                    volume = df['Volume'].iloc[-1]

                    stock_data.append({
                        'code': stock_code,
                        'name': self.stock_mapping[stock_code]['name'],
                        'price': df['Close'].iloc[-1],
                        'change_pct': change_pct,
                        'volume': volume,
                    })
            except Exception as e:
                print(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
                continue

        if not stock_data:
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(stock_data)

        # 排序（按涨跌幅）
        df_sorted = df.sort_values('change_pct', ascending=False)

        # 排名
        df_sorted['rank_by_change'] = range(1, len(df_sorted) + 1)

        # 按成交量排序
        df_sorted_vol = df.sort_values('volume', ascending=False)
        df_sorted['rank_by_volume'] = df_sorted_vol.index.map(lambda x: list(df_sorted_vol.index).index(x) + 1)

        # 综合排名（涨跌幅权重60%，成交量权重40%）
        df_sorted['composite_score'] = (
            df_sorted['rank_by_change'] * 0.6 +
            df_sorted['rank_by_volume'] * 0.4
        )
        df_sorted = df_sorted.sort_values('composite_score')

        return df_sorted.head(top_n).reset_index(drop=True)

    def analyze_sector_fund_flow(self, sector_code: str, days: int = 5) -> Dict:
        """
        分析板块资金流向（基于成交量和涨跌幅）

        Args:
            sector_code: 板块代码
            days: 分析天数

        Returns:
            Dict: 板块资金流向分析
        """
        stocks = self.sector_stocks.get(sector_code, [])

        if not stocks:
            return {
                'sector': sector_code,
                'error': '未找到该板块的股票'
            }

        stock_flow_data = []
        for stock_code in stocks:
            try:
                df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=days + 2)
                if df is not None and len(df) > days:
                    # 计算平均成交量和涨跌幅
                    recent_volume = df['Volume'].iloc[-1] if len(df) > 0 else 0
                    avg_volume = df['Volume'].iloc[-days:].mean() if len(df) > days else 0
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

                    change_pct = (df['Close'].iloc[-1] - df['Close'].iloc[-days]) / df['Close'].iloc[-days] * 100 if len(df) > days else 0

                    stock_flow_data.append({
                        'code': stock_code,
                        'name': self.stock_mapping[stock_code]['name'],
                        'change_pct': change_pct,
                        'volume_ratio': volume_ratio,
                        'flow_score': change_pct * volume_ratio,  # 资金流向评分
                    })
            except Exception as e:
                print(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
                continue

        if not stock_flow_data:
            return {
                'sector': sector_code,
                'error': '无法获取该板块的数据'
            }

        # 计算板块整体资金流向
        avg_flow_score = np.mean([d['flow_score'] for d in stock_flow_data])
        inflow_count = sum(1 for d in stock_flow_data if d['flow_score'] > 0)
        total_count = len(stock_flow_data)

        # 判断资金流向
        if avg_flow_score > 10:
            flow_direction = '大幅流入'
        elif avg_flow_score > 0:
            flow_direction = '小幅流入'
        elif avg_flow_score < -10:
            flow_direction = '大幅流出'
        else:
            flow_direction = '小幅流出'

        return {
            'sector_code': sector_code,
            'sector_name': self.get_sector_name(sector_code),
            'flow_direction': flow_direction,
            'avg_flow_score': avg_flow_score,
            'inflow_count': inflow_count,
            'total_count': total_count,
            'inflow_ratio': inflow_count / total_count * 100,
            'stocks': sorted(stock_flow_data, key=lambda x: x['flow_score'], reverse=True),
        }

    def generate_sector_report(self, period: int = 1) -> str:
        """
        生成板块分析报告

        Args:
            period: 计算周期（天数）

        Returns:
            str: 板块分析报告文本
        """
        # 获取板块涨跌幅排名
        perf_df = self.calculate_sector_performance(period)

        if perf_df.empty:
            return "⚠️ 无法获取板块数据"

        report = []
        report.append("=" * 60)
        report.append(f"港股板块分析报告（{period}日涨跌幅排名）")
        report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")

        # 强势板块（前3名）
        report.append("📈 强势板块（TOP 3）")
        report.append("-" * 60)
        for idx, row in perf_df.head(3).iterrows():
            report.append(f"{idx+1}. {row['sector_name']}：{row['avg_change_pct']:.2f}%（{row['stock_count']}只股票）")
            if row['best_stock']:
                report.append(f"   领涨：{row['best_stock']['name']}（{row['best_stock']['change_pct']:.2f}%）")
            if row['worst_stock']:
                report.append(f"   殿后：{row['worst_stock']['name']}（{row['worst_stock']['change_pct']:.2f}%）")
            report.append("")

        # 弱势板块（后3名）
        report.append("📉 弱势板块（BOTTOM 3）")
        report.append("-" * 60)
        bottom_3 = perf_df.tail(3).copy()
        for i, (idx, row) in enumerate(bottom_3.iterrows(), 1):
            rank = len(perf_df) - len(bottom_3) + i
            report.append(f"{rank}. {row['sector_name']}：{row['avg_change_pct']:.2f}%（{row['stock_count']}只股票）")
            if row['best_stock']:
                report.append(f"   领涨：{row['best_stock']['name']}（{row['best_stock']['change_pct']:.2f}%）")
            if row['worst_stock']:
                report.append(f"   殿后：{row['worst_stock']['name']}（{row['worst_stock']['change_pct']:.2f}%）")
            report.append("")

        # 板块详细排名
        report.append("📊 板块详细排名")
        report.append("-" * 60)
        for idx, row in perf_df.iterrows():
            trend_icon = "🔥" if row['avg_change_pct'] > 2 else "📈" if row['avg_change_pct'] > 0 else "📉"
            report.append(f"{idx+1:2d}. {trend_icon} {row['sector_name']:8s} {row['avg_change_pct']:7.2f}%  ({row['stock_count']}只)")

        report.append("")
        report.append("=" * 60)
        report.append("💡 投资建议")
        report.append("-" * 60)

        if not perf_df.empty:
            top_sector = perf_df.iloc[0]
            bottom_sector = perf_df.iloc[-1]

            if top_sector['avg_change_pct'] > 1:
                report.append(f"• 当前热点板块：{top_sector['sector_name']}，平均涨幅 {top_sector['avg_change_pct']:.2f}%")
                if top_sector['best_stock']:
                    report.append(f"  建议关注该板块的龙头股：{top_sector['best_stock']['name']}")

            if bottom_sector['avg_change_pct'] < -1:
                report.append(f"• 当前弱势板块：{bottom_sector['sector_name']}，平均跌幅 {bottom_sector['avg_change_pct']:.2f}%")
                report.append(f"  建议谨慎操作该板块，等待企稳信号")

        report.append("=" * 60)
        return "\n".join(report)


# ==============================
# 命令行接口
# ==============================
def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='港股板块分析工具')
    parser.add_argument('--period', type=int, default=1, help='计算周期（天数）')
    parser.add_argument('--sector', type=str, help='分析指定板块（板块代码）')
    parser.add_argument('--leaders', type=str, help='识别板块龙头（板块代码）')
    parser.add_argument('--flow', type=str, help='分析板块资金流向（板块代码）')
    parser.add_argument('--trend', type=str, help='分析板块趋势（板块代码）')

    args = parser.parse_args()

    analyzer = SectorAnalyzer()

    if args.sector:
        # 分析指定板块
        result = analyzer.analyze_sector_trend(args.sector)
        print(f"\n板块趋势分析：{analyzer.get_sector_name(args.sector)}")
        print("-" * 60)
        print(f"趋势：{result.get('trend', '未知')}")
        print(f"平均涨跌幅：{result.get('avg_change_pct', 0):.2f}%")
        print(f"上涨股票数：{result.get('rising_count', 0)}/{result.get('total_count', 0)}")
        print(f"上涨比例：{result.get('rising_ratio', 0):.1f}%")
        if 'stocks' in result:
            print("\n个股表现：")
            for stock in result['stocks'][:5]:
                print(f"  {stock['name']}: {stock['change_pct']:.2f}%")

    elif args.leaders:
        # 识别板块龙头
        df = analyzer.identify_sector_leaders(args.leaders)
        print(f"\n板块龙头：{analyzer.get_sector_name(args.leaders)}")
        print("-" * 60)
        print(df.to_string(index=False))

    elif args.flow:
        # 分析板块资金流向
        result = analyzer.analyze_sector_fund_flow(args.flow)
        print(f"\n板块资金流向：{analyzer.get_sector_name(args.flow)}")
        print("-" * 60)
        print(f"资金流向：{result.get('flow_direction', '未知')}")
        print(f"资金流向评分：{result.get('avg_flow_score', 0):.2f}")
        print(f"流入股票数：{result.get('inflow_count', 0)}/{result.get('total_count', 0)}")
        if 'stocks' in result:
            print("\n个股资金流向：")
            for stock in result['stocks'][:5]:
                print(f"  {stock['name']}: {stock['flow_score']:.2f}")

    elif args.trend:
        # 分析板块趋势
        result = analyzer.analyze_sector_trend(args.trend)
        print(f"\n板块趋势分析：{analyzer.get_sector_name(args.trend)}")
        print("-" * 60)
        print(result)

    else:
        # 默认生成完整报告
        report = analyzer.generate_sector_report(args.period)
        print(report)


if __name__ == '__main__':
    main()
