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
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 支持直接运行和模块运行两种方式
if __name__ == '__main__':
    # 直接运行时，添加项目根目录到path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 导入腾讯财经接口
try:
    from .tencent_finance import get_hk_stock_data_tencent
except ImportError:
    # 直接运行时使用绝对导入
    from data_services.tencent_finance import get_hk_stock_data_tencent

# 导入技术分析工具
try:
    from .technical_analysis import TechnicalAnalyzer
    TECHNICAL_AVAILABLE = True
except ImportError:
    try:
        from data_services.technical_analysis import TechnicalAnalyzer
        TECHNICAL_AVAILABLE = True
    except ImportError:
        TECHNICAL_AVAILABLE = False
        print("⚠️ 技术分析工具不可用，部分功能将受限")

# 导入基本面数据模块
try:
    from .fundamental_data import get_comprehensive_fundamental_data
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    try:
        from data_services.fundamental_data import get_comprehensive_fundamental_data
        FUNDAMENTAL_AVAILABLE = True
    except ImportError:
        FUNDAMENTAL_AVAILABLE = False
        print("⚠️ 基本面数据模块不可用，部分功能将受限")

# ==============================
# 业界标准权重配置（基于MVP模型）
# ==============================
# 投资风格权重配置
INVESTMENT_STYLE_WEIGHTS = {
    'aggressive': {  # 进取型：关注动量和成交量
        'momentum': 0.6,
        'volume': 0.3,
        'fundamental': 0.1,
        'description': '进取型：重点关注短期动量和成交量，适合短线交易'
    },
    'moderate': {  # 稳健型：平衡动量、成交量、基本面
        'momentum': 0.4,
        'volume': 0.3,
        'fundamental': 0.3,
        'description': '稳健型：平衡动量、成交量、基本面，适合波段交易'
    },
    'conservative': {  # 保守型：关注基本面和成交量
        'momentum': 0.2,
        'volume': 0.3,
        'fundamental': 0.5,
        'description': '保守型：重点关注基本面和成交量，适合中长期投资'
    },
}

# 默认市值筛选阈值（亿港币）
DEFAULT_MIN_MARKET_CAP = 100  # 100亿港币

# ==============================
# 股票板块映射（扩展版：58只股票，覆盖13个板块）
# ==============================
STOCK_SECTOR_MAPPING = {
    # 银行股（8只）
    '0005.HK': {'sector': 'bank', 'name': '汇丰银行'},
    '0939.HK': {'sector': 'bank', 'name': '建设银行'},
    '1288.HK': {'sector': 'bank', 'name': '农业银行'},
    '1398.HK': {'sector': 'bank', 'name': '工商银行'},
    '3968.HK': {'sector': 'bank', 'name': '招商银行'},
    '2388.HK': {'sector': 'bank', 'name': '中银香港'},
    '0011.HK': {'sector': 'bank', 'name': '恒生银行'},
    '0881.HK': {'sector': 'bank', 'name': '中银香港'},

    # 科技股（8只）
    '0700.HK': {'sector': 'tech', 'name': '腾讯控股'},
    '9988.HK': {'sector': 'tech', 'name': '阿里巴巴-SW'},
    '3690.HK': {'sector': 'tech', 'name': '美团-W'},
    '1810.HK': {'sector': 'tech', 'name': '小米集团-W'},
    '9618.HK': {'sector': 'tech', 'name': '京东集团-SW'},
    '9999.HK': {'sector': 'tech', 'name': '网易-S'},
    '9888.HK': {'sector': 'tech', 'name': '百度集团-SW'},
    '1024.HK': {'sector': 'tech', 'name': '快手-W'},

    # 半导体股（4只）
    '0981.HK': {'sector': 'semiconductor', 'name': '中芯国际'},
    '1347.HK': {'sector': 'semiconductor', 'name': '华虹半导体'},
    '0960.HK': {'sector': 'semiconductor', 'name': '龙源电力'},
    '02382.HK': {'sector': 'semiconductor', 'name': '舜宇光学科技'},

    # 人工智能股（4只）
    '6682.HK': {'sector': 'ai', 'name': '第四范式'},
    '9660.HK': {'sector': 'ai', 'name': '地平线机器人'},
    '2533.HK': {'sector': 'ai', 'name': '黑芝麻智能'},
    '0020.HK': {'sector': 'ai', 'name': '商汤-W'},

    # 新能源股（4只）
    '1211.HK': {'sector': 'new_energy', 'name': '比亚迪股份'},
    '1798.HK': {'sector': 'new_energy', 'name': '赣锋锂业'},
    '3800.HK': {'sector': 'new_energy', 'name': '保利协鑫能源'},
    '2282.HK': {'sector': 'new_energy', 'name': '比亚迪电子'},

    # 环保股（3只）
    '1330.HK': {'sector': 'environmental', 'name': '绿色动力环保'},
    '01257.HK': {'sector': 'environmental', 'name': '中国光大环境'},
    '01387.HK': {'sector': 'environmental', 'name': '中国水务'},

    # 能源股（4只）
    '0883.HK': {'sector': 'energy', 'name': '中国海洋石油'},
    '1088.HK': {'sector': 'energy', 'name': '中国神华'},
    '1171.HK': {'sector': 'energy', 'name': '兖矿能源'},
    '02883.HK': {'sector': 'energy', 'name': '中海油服'},

    # 航运股（3只）
    '1138.HK': {'sector': 'shipping', 'name': '中远海能'},
    '01919.HK': {'sector': 'shipping', 'name': '中远海控'},
    '02866.HK': {'sector': 'shipping', 'name': '中远海运港口'},

    # 交易所（1只）
    '0388.HK': {'sector': 'exchange', 'name': '香港交易所'},

    # 公用事业股（4只）
    '0728.HK': {'sector': 'utility', 'name': '中国电信'},
    '0941.HK': {'sector': 'utility', 'name': '中国移动'},
    '0002.HK': {'sector': 'utility', 'name': '中电控股'},
    '0006.HK': {'sector': 'utility', 'name': '电能实业'},

    # 保险股（4只）
    '1299.HK': {'sector': 'insurance', 'name': '友邦保险'},
    '2318.HK': {'sector': 'insurance', 'name': '中国平安'},
    '2601.HK': {'sector': 'insurance', 'name': '中国太保'},
    '0966.HK': {'sector': 'insurance', 'name': '中国人寿'},

    # 生物医药股（4只）
    '2269.HK': {'sector': 'biotech', 'name': '药明生物'},
    '02269.HK': {'sector': 'biotech', 'name': '药明康德'},
    '01177.HK': {'sector': 'biotech', 'name': '中国生物制药'},
    '02186.HK': {'sector': 'biotech', 'name': '绿叶制药'},

    # 指数基金（2只）
    '2800.HK': {'sector': 'index', 'name': '盈富基金'},
    '08288.HK': {'sector': 'index', 'name': '恒生中国企业'},

    # 房地产股（3只）
    '01109.HK': {'sector': 'property', 'name': '华润置地'},
    '0012.HK': {'sector': 'property', 'name': '恒基地产'},
    '0016.HK': {'sector': 'property', 'name': '新鸿基地产'},

    # 消费股（4只）
    '02318.HK': {'sector': 'consumer', 'name': '中国平安证券'},
    '02314.HK': {'sector': 'consumer', 'name': '辉山乳业'},
    '00151.HK': {'sector': 'consumer', 'name': '中国旺旺'},
    '02228.HK': {'sector': 'consumer', 'name': '中国飞鹤'},

    # 汽车股（2只）
    '02333.HK': {'sector': 'auto', 'name': '长城汽车'},
    '1053.HK': {'sector': 'auto', 'name': '重庆长安汽车'},
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
    'property': '房地产',
    'consumer': '消费',
    'auto': '汽车',
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

    def identify_sector_leaders(
        self,
        sector_code: str,
        top_n: int = 3,
        period: int = 1,
        min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
        style: str = 'moderate'
    ) -> pd.DataFrame:
        """
        识别板块龙头（业界标准版本）

        Args:
            sector_code: 板块代码
            top_n: 返回前N只股票
            period: 计算周期（天数），1=1日，5=5日，20=20日
            min_market_cap: 最小市值阈值（亿港币）
            style: 投资风格（aggressive进取型、moderate稳健型、conservative保守型）

        Returns:
            DataFrame: 板块龙头股票
        """
        stocks = self.sector_stocks.get(sector_code, [])

        if not stocks:
            return pd.DataFrame()

        # 验证投资风格
        if style not in INVESTMENT_STYLE_WEIGHTS:
            print(f"⚠️ 未知的投资风格 '{style}'，使用默认风格 'moderate'")
            style = 'moderate'

        weights = INVESTMENT_STYLE_WEIGHTS[style]
        stock_data = []

        for stock_code in stocks:
            try:
                # 获取股票数据（根据周期调整天数）
                df = get_hk_stock_data_tencent(stock_code.replace('.HK', ''), period_days=period + 10)
                if df is not None and len(df) > period:
                    # 计算涨跌幅（支持多周期）
                    if period == 1 and len(df) > 1:
                        change_pct = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                    elif len(df) > period:
                        change_pct = (df['Close'].iloc[-1] - df['Close'].iloc[-1-period]) / df['Close'].iloc[-1-period] * 100
                    else:
                        change_pct = 0

                    # 成交量（使用最新交易日）
                    volume = df['Volume'].iloc[-1]

                    # 获取基本面数据
                    fundamental_data = {}
                    if FUNDAMENTAL_AVAILABLE:
                        try:
                            stock_num = stock_code.replace('.HK', '').replace('HK', '').lstrip('0')
                            fd = get_comprehensive_fundamental_data(stock_num)
                            if fd:
                                fundamental_data = {
                                    'market_cap': fd.get('fi_market_cap'),  # 市值（港币）
                                    'pe_ratio': fd.get('fi_pe_ratio'),      # 市盈率
                                    'pb_ratio': fd.get('fi_pb_ratio'),      # 市净率
                                }
                        except Exception as e:
                            print(f"  ⚠️ 获取 {stock_code} 基本面数据失败: {e}")

                    stock_info = {
                        'code': stock_code,
                        'name': self.stock_mapping[stock_code]['name'],
                        'price': df['Close'].iloc[-1],
                        'change_pct': change_pct,
                        'volume': volume,
                        'period': period,
                    }

                    # 添加基本面数据
                    stock_info.update(fundamental_data)

                    stock_data.append(stock_info)

            except Exception as e:
                print(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
                continue

        if not stock_data:
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(stock_data)

        # 市值筛选（如果提供了最小市值）
        if min_market_cap > 0 and 'market_cap' in df.columns:
            # 转换市值为亿港币
            df['market_cap_billion'] = df['market_cap'] / 1e8
            # 筛选市值大于最小市值的股票
            df = df[df['market_cap_billion'] >= min_market_cap]
            if df.empty:
                print(f"⚠️ 该板块没有市值 >= {min_market_cap}亿港币的股票")
                return pd.DataFrame()

        # 计算各项排名
        # 1. 动量排名（涨跌幅）
        df_sorted_momentum = df.sort_values('change_pct', ascending=False)
        df['rank_momentum'] = df_sorted_momentum.index.map(lambda x: list(df_sorted_momentum.index).index(x) + 1)

        # 2. 成交量排名
        df_sorted_volume = df.sort_values('volume', ascending=False)
        df['rank_volume'] = df_sorted_volume.index.map(lambda x: list(df_sorted_volume.index).index(x) + 1)

        # 3. 基本面排名（综合PE和PB）
        if 'pe_ratio' in df.columns and 'pb_ratio' in df.columns:
            # 计算基本面评分：PE和PB越低越好
            df['pe_ratio_norm'] = df['pe_ratio'].rank()
            df['pb_ratio_norm'] = df['pb_ratio'].rank()
            df['fundamental_score'] = (df['pe_ratio_norm'] + df['pb_ratio_norm']) / 2
            df_sorted_fundamental = df.sort_values('fundamental_score', ascending=True)
            df['rank_fundamental'] = df_sorted_fundamental.index.map(lambda x: list(df_sorted_fundamental.index).index(x) + 1)
        else:
            # 如果没有基本面数据，给所有股票相同的排名
            df['rank_fundamental'] = 1

        # 4. 综合评分（根据投资风格动态权重）
        df['composite_score'] = (
            df['rank_momentum'] * weights['momentum'] +
            df['rank_volume'] * weights['volume'] +
            df['rank_fundamental'] * weights['fundamental']
        )

        # 按综合评分排序（分数越低越好）
        df = df.sort_values('composite_score')

        # 选择前N只股票
        result = df.head(top_n).reset_index(drop=True)

        # 添加投资风格信息
        result['investment_style'] = style
        result['style_description'] = weights['description']
        result['min_market_cap'] = min_market_cap
        result['period_days'] = period

        return result

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

    parser = argparse.ArgumentParser(
        description='港股板块分析工具（业界标准版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
投资风格说明：
  aggressive  进取型：重点关注短期动量和成交量（动量60% 成交量30% 基本面10%）
  moderate    稳健型：平衡动量、成交量、基本面（动量40% 成交量30% 基本面30%）
  conservative 保守型：重点关注基本面和成交量（动量20% 成交量30% 基本面50%）

示例用法：
  # 识别科技板块龙头（1日周期，稳健型，市值>100亿）
  python hk_sector_analysis.py --leaders tech

  # 识别AI板块龙头（5日周期，进取型，市值>50亿）
  python hk_sector_analysis.py --leaders ai --period 5 --style aggressive --min-market-cap 50

  # 识别银行板块龙头（20日周期，保守型，市值>200亿）
  python hk_sector_analysis.py --leaders bank --period 20 --style conservative --min-market-cap 200
        """
    )

    parser.add_argument('--period', type=int, default=1,
                        choices=[1, 5, 20],
                        help='计算周期（天数）：1=1日（短线），5=5日（波段），20=20日（中线），默认：1')
    parser.add_argument('--sector', type=str, help='分析指定板块（板块代码）')
    parser.add_argument('--leaders', type=str, help='识别板块龙头（板块代码）')
    parser.add_argument('--flow', type=str, help='分析板块资金流向（板块代码）')
    parser.add_argument('--trend', type=str, help='分析板块趋势（板块代码）')
    parser.add_argument('--min-market-cap', type=float, default=DEFAULT_MIN_MARKET_CAP,
                        help=f'最小市值阈值（亿港币），默认：{DEFAULT_MIN_MARKET_CAP}亿')
    parser.add_argument('--style', type=str, default='moderate',
                        choices=['aggressive', 'moderate', 'conservative'],
                        help='投资风格：aggressive=进取型，moderate=稳健型，conservative=保守型，默认：moderate')
    parser.add_argument('--top-n', type=int, default=3, help='返回前N只龙头股，默认：3')

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
        # 识别板块龙头（业界标准版本）
        df = analyzer.identify_sector_leaders(
            sector_code=args.leaders,
            top_n=args.top_n,
            period=args.period,
            min_market_cap=args.min_market_cap,
            style=args.style
        )

        print(f"\n板块龙头：{analyzer.get_sector_name(args.leaders)}")
        print("-" * 80)
        print(f"配置：周期={args.period}日，投资风格={args.style}，最小市值={args.min_market_cap}亿港币，返回数量={args.top_n}")
        print("-" * 80)

        if df.empty:
            print("⚠️ 未找到符合条件的龙头股")
        else:
            # 显示投资风格描述
            style_desc = df.iloc[0]['style_description'] if 'style_description' in df.columns else ''
            print(f"投资风格：{style_desc}")
            print()

            # 显示结果
            columns_to_show = ['name', 'code', 'price', 'change_pct', 'volume', 'composite_score']
            if 'market_cap_billion' in df.columns:
                columns_to_show.insert(-1, 'market_cap_billion')
            if 'pe_ratio' in df.columns:
                columns_to_show.insert(-1, 'pe_ratio')
            if 'pb_ratio' in df.columns:
                columns_to_show.insert(-1, 'pb_ratio')

            # 重命名列以提高可读性
            display_df = df[columns_to_show].copy()
            display_df = display_df.rename(columns={
                'name': '股票名称',
                'code': '股票代码',
                'price': '最新价格',
                'change_pct': f'{args.period}日涨跌幅(%)',
                'volume': '成交量',
                'market_cap_billion': '市值(亿)',
                'pe_ratio': '市盈率',
                'pb_ratio': '市净率',
                'composite_score': '综合评分'
            })

            print(display_df.to_string(index=False))
            print()
            print("💡 综合评分越低表示表现越好（排名靠前）")

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
