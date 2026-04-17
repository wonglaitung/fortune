#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块轮动规律分析

分析不同板块之间的时间关系、因果关系和轮动规律：
1. 板块表现的相关性和领先滞后关系
2. 板块轮动的时间序列模式
3. 防御性vs周期性板块的轮动规律
4. 大盘与板块之间的因果关系
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from catboost import CatBoostClassifier

RANDOM_SEED = 42

# 板块定义
SECTORS = {
    'bank': {'name': '银行股', 'stocks': ['0005.HK', '0939.HK', '1288.HK', '1398.HK', '3968.HK', '2388.HK'], 'type': 'defensive'},
    'consumer': {'name': '消费股', 'stocks': ['0700.HK', '2313.HK', '1299.HK', '1876.HK', '9999.HK', '2015.HK'], 'type': 'cyclical'},
    'semiconductor': {'name': '半导体股', 'stocks': ['0981.HK', '1347.HK', '2333.HK'], 'type': 'cyclical'},
    'tech': {'name': '科技股', 'stocks': ['1810.HK', '9618.HK', '9988.HK', '3690.HK', '1024.HK', '9866.HK', '2015.HK', '9923.HK'], 'type': 'cyclical'},
    'biotech': {'name': '生物医药股', 'stocks': ['2269.HK', '6185.HK', '2359.HK', '1877.HK'], 'type': 'cyclical'},
    'insurance': {'name': '保险股', 'stocks': ['2318.HK', '1299.HK', '0966.HK'], 'type': 'defensive'},
    'real_estate': {'name': '房地产股', 'stocks': ['1109.HK', '0688.HK', '2007.HK', '0960.HK'], 'type': 'cyclical'},
    'energy': {'name': '能源股', 'stocks': ['0883.HK', '0386.HK', '0857.HK', '1088.HK'], 'type': 'cyclical'},
    'utility': {'name': '公用事业股', 'stocks': ['0002.HK', '0003.HK', '0006.HK', '1038.HK'], 'type': 'defensive'},
    'shipping': {'name': '航运股', 'stocks': ['1138.HK', '1919.HK', '2866.HK'], 'type': 'cyclical'},
}


def fetch_sector_data(sector_code, period="3y"):
    """获取板块数据（使用板块内股票平均）"""
    sector_info = SECTORS[sector_code]
    stocks = sector_info['stocks']

    all_returns = []
    for stock in stocks[:3]:  # 取前3只代表股票
        try:
            ticker = yf.Ticker(stock)
            df = ticker.history(period=period, interval="1d")
            if len(df) > 0:
                df['Return'] = df['Close'].pct_change()
                all_returns.append(df['Return'])
        except:
            continue

    if not all_returns:
        return None

    # 计算平均收益率
    avg_returns = pd.concat(all_returns, axis=1).mean(axis=1)
    return avg_returns


def fetch_hsi_data(period="3y"):
    """获取恒生指数数据"""
    hsi = yf.Ticker("^HSI")
    df = hsi.history(period=period, interval="1d")
    df['Return'] = df['Close'].pct_change()
    return df['Return']


def calculate_sector_features(returns):
    """计算板块特征"""
    features = {}

    # 收益率特征
    features['Return_1d'] = returns.shift(1)
    features['Return_5d'] = returns.rolling(5).sum().shift(1)
    features['Return_20d'] = returns.rolling(20).sum().shift(1)

    # 波动率
    features['Volatility_20d'] = returns.rolling(20).std().shift(1)

    # 趋势强度
    features['Trend_5d'] = (returns.rolling(5).sum() > 0).astype(int).shift(1)
    features['Trend_20d'] = (returns.rolling(20).sum() > 0).astype(int).shift(1)

    # 动量
    features['Momentum'] = returns.rolling(20).sum().shift(1) - returns.rolling(60).sum().shift(1)

    return pd.DataFrame(features)


def analyze_sector_performance():
    """分析各板块表现相关性"""
    print("=" * 80)
    print("📊 板块表现相关性分析")
    print("=" * 80)

    # 获取数据
    sector_returns = {}
    for code in SECTORS.keys():
        returns = fetch_sector_data(code)
        if returns is not None:
            sector_returns[code] = returns

    hsi_returns = fetch_hsi_data()
    sector_returns['HSI'] = hsi_returns

    # 构建DataFrame
    df = pd.DataFrame(sector_returns).dropna()

    print(f"\n数据期间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"样本数: {len(df)}")

    # 计算相关性矩阵
    corr_matrix = df.corr()

    print("\n1. 板块间相关性矩阵")
    print("-" * 80)

    # 只显示板块vs板块+HSI
    sectors_only = [s for s in corr_matrix.columns if s != 'HSI']
    hsi_corr = corr_matrix['HSI'].drop('HSI').sort_values(ascending=False)

    print("\n与恒指相关性排序:")
    for sector, corr in hsi_corr.items():
        sector_name = SECTORS.get(sector, {}).get('name', sector)
        print(f"  {sector_name:<12} ({sector}): {corr:.4f}")

    # 板块间相关性
    print("\n2. 板块间相关性（排除HSI）")
    print("-" * 80)

    sector_corr = corr_matrix.loc[sectors_only, sectors_only]

    # 找出相关性最高和最低的组合
    corr_pairs = []
    for i in range(len(sectors_only)):
        for j in range(i+1, len(sectors_only)):
            s1, s2 = sectors_only[i], sectors_only[j]
            corr_pairs.append((s1, s2, sector_corr.loc[s1, s2]))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\n相关性最高的5对板块:")
    for s1, s2, corr in corr_pairs[:5]:
        n1 = SECTORS.get(s1, {}).get('name', s1)
        n2 = SECTORS.get(s2, {}).get('name', s2)
        print(f"  {n1} - {n2}: {corr:.4f}")

    print("\n相关性最低的5对板块:")
    for s1, s2, corr in corr_pairs[-5:]:
        n1 = SECTORS.get(s1, {}).get('name', s1)
        n2 = SECTORS.get(s2, {}).get('name', s2)
        print(f"  {n1} - {n2}: {corr:.4f}")

    return df, corr_matrix


def analyze_sector_rotation_patterns(df):
    """分析板块轮动规律"""
    print("\n" + "=" * 80)
    print("🔄 板块轮动规律分析")
    print("=" * 80)

    sectors_only = [s for s in df.columns if s != 'HSI']

    # 计算20天滚动收益率
    rolling_returns = {}
    for sector in sectors_only:
        rolling_returns[sector] = df[sector].rolling(20).sum()

    rolling_df = pd.DataFrame(rolling_returns).dropna()

    print("\n1. 板块轮动频率分析")
    print("-" * 80)

    # 计算每个板块的领先/滞后次数
    rotation_stats = {}

    for sector in sectors_only:
        returns = rolling_df[sector]
        # 找出每个sector表现最好的时期
        is_leader = returns == rolling_df.max(axis=1)
        leader_count = is_leader.sum()
        leader_pct = leader_count / len(rolling_df) * 100

        rotation_stats[sector] = {
            'leader_count': leader_count,
            'leader_pct': leader_pct,
            'avg_return': returns.mean() * 100,
            'type': SECTORS.get(sector, {}).get('type', 'unknown')
        }

    # 按领涨次数排序
    sorted_sectors = sorted(rotation_stats.items(), key=lambda x: x[1]['leader_count'], reverse=True)

    print(f"\n{'板块':<15} {'领涨次数':<10} {'占比':<10} {'平均收益':<12} {'类型'}")
    print("-" * 60)
    for sector, stats in sorted_sectors:
        name = SECTORS.get(sector, {}).get('name', sector)
        print(f"{name:<15} {stats['leader_count']:<10} {stats['leader_pct']:<10.2f}% {stats['avg_return']:<12.3f}% {stats['type']}")

    print("\n2. 防御性 vs 周期性板块轮动")
    print("-" * 80)

    # 计算防御性和周期性板块的整体表现
    defensive_sectors = [s for s in sectors_only if SECTORS.get(s, {}).get('type') == 'defensive']
    cyclical_sectors = [s for s in sectors_only if SECTORS.get(s, {}).get('type') == 'cyclical']

    defensive_avg = rolling_df[defensive_sectors].mean(axis=1)
    cyclical_avg = rolling_df[cyclical_sectors].mean(axis=1)

    # 计算相对表现
    relative_performance = cyclical_avg - defensive_avg

    # 找出周期性板块占优和防御性板块占优的时期
    cyclical_dominant = relative_performance > relative_performance.quantile(0.75)
    defensive_dominant = relative_performance < relative_performance.quantile(0.25)

    print(f"\n周期性板块占优时期: {cyclical_dominant.sum()} 天 ({cyclical_dominant.sum()/len(relative_performance)*100:.1f}%)")
    print(f"防御性板块占优时期: {defensive_dominant.sum()} 天 ({defensive_dominant.sum()/len(relative_performance)*100:.1f}%)")

    # 分析恒指在这些时期的表现
    hsi_returns = df['HSI'].rolling(20).sum().reindex(relative_performance.index)

    print(f"\n周期性占优时恒指平均收益: {hsi_returns[cyclical_dominant].mean()*100:.3f}%")
    print(f"防御性占优时恒指平均收益: {hsi_returns[defensive_dominant].mean()*100:.3f}%")

    return rotation_stats, relative_performance


def analyze_lead_lag_relationships(df):
    """分析板块间的领先滞后关系"""
    print("\n" + "=" * 80)
    print("⏰ 板块领先-滞后关系分析")
    print("=" * 80)

    sectors_only = [s for s in df.columns if s != 'HSI']

    # 使用不同滞后计算相关性
    lag_correlations = {}

    print("\n1. 板块对恒指的领先/滞后关系")
    print("-" * 80)

    for sector in sectors_only:
        correlations = {}
        for lag in range(-5, 6):  # -5到+5天滞后
            if lag < 0:
                # 板块领先恒指（板块提前lag天）
                corr = df[sector].shift(-lag).corr(df['HSI'])
            elif lag > 0:
                # 恒指领先板块
                corr = df[sector].shift(lag).corr(df['HSI'])
            else:
                corr = df[sector].corr(df['HSI'])
            correlations[lag] = corr

        lag_correlations[sector] = correlations

        # 找出最大相关性对应的滞后
        max_lag = max(correlations.items(), key=lambda x: abs(x[1]))

        name = SECTORS.get(sector, {}).get('name', sector)
        if max_lag[0] < 0:
            lead_str = f"领先 {-max_lag[0]} 天"
        elif max_lag[0] > 0:
            lead_str = f"滞后 {max_lag[0]} 天"
        else:
            lead_str = "同步"

        print(f"  {name:<12}: {lead_str} (r={max_lag[1]:.4f})")

    print("\n2. 板块间的领先-滞后关系")
    print("-" * 80)

    # 分析板块对之间的领先滞后关系
    lead_lag_matrix = pd.DataFrame(index=sectors_only, columns=sectors_only)

    for s1 in sectors_only:
        for s2 in sectors_only:
            if s1 != s2:
                best_corr = 0
                best_lag = 0
                for lag in range(-5, 6):
                    if lag < 0:
                        corr = df[s1].shift(-lag).corr(df[s2])
                    elif lag > 0:
                        corr = df[s1].shift(lag).corr(df[s2])
                    else:
                        corr = df[s1].corr(df[s2])

                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                lead_lag_matrix.loc[s1, s2] = best_lag

    # 统计每个板块领先/滞后的次数
    lead_count = {}
    for sector in sectors_only:
        leads = (lead_lag_matrix.loc[sector] < 0).sum()
        lags = (lead_lag_matrix.loc[sector] > 0).sum()
        lead_count[sector] = {'leads': leads, 'lags': lags}

    print("\n板块领先/滞后统计:")
    print(f"{'板块':<15} {'领先次数':<10} {'滞后次数':<10} {'类型'}")
    print("-" * 50)
    for sector, counts in sorted(lead_count.items(), key=lambda x: x[1]['leads'], reverse=True):
        name = SECTORS.get(sector, {}).get('name', sector)
        stype = SECTORS.get(sector, {}).get('type', 'unknown')
        print(f"{name:<15} {counts['leads']:<10} {counts['lags']:<10} {stype}")

    return lead_lag_matrix


def analyze_market_regime_rotation(df, relative_performance):
    """分析不同市场状态下的板块轮动"""
    print("\n" + "=" * 80)
    print("📈 市场状态与板块轮动分析")
    print("=" * 80)

    hsi_returns = df['HSI'].rolling(20).sum()

    # 定义市场状态
    bull_market = hsi_returns > hsi_returns.quantile(0.75)
    bear_market = hsi_returns < hsi_returns.quantile(0.25)
    neutral_market = ~(bull_market | bear_market)

    print("\n1. 不同市场状态下的板块轮动偏好")
    print("-" * 80)

    # 计算各市场状态下的周期性/防御性偏好
    bull_cyclical_pct = (relative_performance[bull_market] > 0).mean() * 100
    bear_cyclical_pct = (relative_performance[bear_market] > 0).mean() * 100
    neutral_cyclical_pct = (relative_performance[neutral_market] > 0).mean() * 100

    print(f"\n牛市中周期性板块占优比例: {bull_cyclical_pct:.1f}%")
    print(f"熊市中周期性板块占优比例: {bear_cyclical_pct:.1f}%")
    print(f"震荡市中周期性板块占优比例: {neutral_cyclical_pct:.1f}%")

    # 轮动规律
    print("\n2. 板块轮动规律总结")
    print("-" * 80)

    regimes = []
    current_regime = None
    regime_start = None

    for date in relative_performance.index:
        if relative_performance[date] > relative_performance.quantile(0.6):
            regime = 'cyclical'
        elif relative_performance[date] < relative_performance.quantile(0.4):
            regime = 'defensive'
        else:
            regime = 'neutral'

        if regime != current_regime:
            if current_regime is not None:
                regimes.append({
                    'regime': current_regime,
                    'start': regime_start,
                    'end': date,
                    'duration': (date - regime_start).days
                })
            current_regime = regime
            regime_start = date

    # 统计各状态持续期
    cyclical_durations = [r['duration'] for r in regimes if r['regime'] == 'cyclical']
    defensive_durations = [r['duration'] for r in regimes if r['regime'] == 'defensive']

    if cyclical_durations:
        print(f"\n周期性占优平均持续: {np.mean(cyclical_durations):.1f} 天")
    if defensive_durations:
        print(f"防御性占优平均持续: {np.mean(defensive_durations):.1f} 天")

    print("\n3. 板块轮动策略启示")
    print("-" * 80)

    if bull_cyclical_pct > bear_cyclical_pct:
        print("✅ 牛市配周期，熊市配防御 - 策略有效")
        print(f"   牛市中周期/防御 = {bull_cyclical_pct/(100-bull_cyclical_pct):.2f}")
        print(f"   熊市中周期/防御 = {bear_cyclical_pct/(100-bear_cyclical_pct):.2f}")
    else:
        print("⚠️ 板块轮动规律不明显")

    return regimes


def analyze_sector_momentum(df):
    """分析板块动量效应"""
    print("\n" + "=" * 80)
    print("🚀 板块动量效应分析")
    print("=" * 80)

    sectors_only = [s for s in df.columns if s != 'HSI']

    print("\n1. 板块动量持续性分析")
    print("-" * 80)

    momentum_results = {}

    for sector in sectors_only:
        returns = df[sector]

        # 计算过去5天收益与未来5天收益的关系（动量效应）
        past_return = returns.rolling(5).sum().shift(5)
        future_return = returns.rolling(5).sum()

        correlation = past_return.corr(future_return)

        # 动量策略回测：过去5天涨则买入，跌则卖出
        signal = (past_return > 0).astype(int)
        strategy_return = signal * future_return

        momentum_results[sector] = {
            'correlation': correlation,
            'avg_strategy_return': strategy_return.mean() * 100,
            'buy_hold_return': future_return.mean() * 100
        }

    print(f"\n{'板块':<15} {'动量相关性':<12} {'动量策略收益':<12} {'买入持有'}")
    print("-" * 60)
    for sector, results in sorted(momentum_results.items(), key=lambda x: x[1]['correlation'], reverse=True):
        name = SECTORS.get(sector, {}).get('name', sector)
        print(f"{name:<15} {results['correlation']:<12.4f} {results['avg_strategy_return']:<12.3f}% {results['buy_hold_return']:<12.3f}%")

    print("\n2. 板块反转效应分析")
    print("-" * 80)

    reversal_results = {}

    for sector in sectors_only:
        returns = df[sector]

        # 反转策略：过去5天跌则买入（均值回归）
        past_return = returns.rolling(5).sum().shift(5)
        future_return = returns.rolling(5).sum()

        signal = (past_return < 0).astype(int)
        strategy_return = signal * future_return

        reversal_results[sector] = {
            'avg_strategy_return': strategy_return.mean() * 100
        }

    print(f"\n{'板块':<15} {'反转策略收益':<12} {'买入持有'}")
    print("-" * 40)
    for sector, results in sorted(reversal_results.items(), key=lambda x: x[1]['avg_strategy_return'], reverse=True):
        name = SECTORS.get(sector, {}).get('name', sector)
        momentum_ret = momentum_results[sector]['buy_hold_return']
        print(f"{name:<15} {results['avg_strategy_return']:<12.3f}% {momentum_ret:<12.3f}%")

    return momentum_results


def generate_sector_rotation_strategy(df, relative_performance):
    """生成板块轮动策略"""
    print("\n" + "=" * 80)
    print("💡 板块轮动策略建议")
    print("=" * 80)

    sectors_only = [s for s in df.columns if s != 'HSI']
    hsi_returns = df['HSI'].rolling(20).sum()

    print("\n1. 基于市场状态的轮动策略")
    print("-" * 80)

    print("""
策略框架:
┌─────────────────────────────────────────────────────────┐
│  市场判断 → 板块配置 → 个股选择                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 判断市场状态（使用HSI 20天收益）                     │
│     - HSI收益 > 75%分位 → 牛市                          │
│     - HSI收益 < 25%分位 → 熊市                          │
│     - 其他 → 震荡市                                      │
│                                                          │
│  2. 根据市场状态选择板块                                 │
│     - 牛市: 半导体、科技、消费（周期性）                  │
│     - 熊市: 银行、保险、公用事业（防御性）                │
│     - 震荡: 根据相对强弱动态调整                        │
│                                                          │
│  3. 板块内个股选择                                       │
│     - 使用CatBoost模型预测                              │
│     - 置信度>0.65时买入                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
    """)

    print("\n2. 板块轮动时机判断")
    print("-" * 80)

    print("""
轮动信号:
┌────────────────────────────────────────────────────────┐
│ 周期性 → 防御性                                         │
│ 触发条件:                                               │
│ 1. 恒指20天收益转负                                     │
│ 2. 周期性板块相对收益从高点回落                          │
│ 3. 防御性板块开始领涨                                   │
├────────────────────────────────────────────────────────┤
│ 防御性 → 周期性                                         │
│ 触发条件:                                               │
│ 1. 恒指20天收益转正且持续改善                           │
│ 2. 周期性板块相对收益触底反弹                           │
│ 3. 成交量放大配合                                       │
└────────────────────────────────────────────────────────┘
    """)

    print("\n3. 风险控制")
    print("-" * 80)

    print("""
风控措施:
1. 单板块仓位 ≤ 30%
2. 设置板块止损: 板块指数跌破20日均线减仓50%
3. 轮动失败保护: 如果轮动后2周收益为负，回归原配置
4. 大盘止损: 恒指跌破60日均线，整体仓位降至30%
    """)


def main():
    """主函数"""
    print("=" * 80)
    print("港股板块轮动规律深度分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 板块表现相关性分析
    df, corr_matrix = analyze_sector_performance()

    # 2. 板块轮动规律分析
    rotation_stats, relative_performance = analyze_sector_rotation_patterns(df)

    # 3. 领先滞后关系分析
    lead_lag_matrix = analyze_lead_lag_relationships(df)

    # 4. 市场状态与轮动分析
    regimes = analyze_market_regime_rotation(df, relative_performance)

    # 5. 板块动量效应分析
    momentum_results = analyze_sector_momentum(df)

    # 6. 生成策略建议
    generate_sector_rotation_strategy(df, relative_performance)

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
