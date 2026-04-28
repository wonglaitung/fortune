#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票关联与因果关系分析

分析内容：
1. 价格相关性分析 - 计算股票间 Pearson/Spearman 相关系数
2. 领先滞后关系分析 - 使用 Granger 因果检验识别领头羊股票
3. 板块联动分析 - 分析板块间相关性及轮动规律

分析范围：全部59只股票（16个板块）

创建时间：2026-04-28
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCK_SECTOR_MAPPING

RANDOM_SEED = 42
OUTPUT_DIR = "output"


def get_stock_list():
    """获取股票列表"""
    return list(STOCK_SECTOR_MAPPING.keys())


def get_sector_list():
    """获取板块列表"""
    sectors = set()
    for stock_code, info in STOCK_SECTOR_MAPPING.items():
        sectors.add(info['sector'])
    return list(sectors)


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


def calculate_price_correlation(stock_data):
    """计算价格相关性矩阵"""
    print("\n📊 计算价格相关性矩阵...")

    # 构建收益率 DataFrame
    returns_dict = {}
    for stock_code, df in stock_data.items():
        returns_dict[stock_code] = df['Return']

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    # 计算 Pearson 相关系数
    pearson_corr = returns_df.corr(method='pearson')

    # 计算 Spearman 相关系数
    spearman_corr = returns_df.corr(method='spearman')

    print(f"  ✅ 相关性矩阵计算完成（{len(pearson_corr)} x {len(pearson_corr)}）")

    return pearson_corr, spearman_corr


def find_high_correlation_pairs(corr_matrix, threshold=0.7):
    """找出高相关性股票组合"""
    high_corr_pairs = []

    stocks = list(corr_matrix.columns)
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            if i < j:  # 避免重复
                corr_value = corr_matrix.loc[stock1, stock2]
                if abs(corr_value) >= threshold:
                    sector1 = STOCK_SECTOR_MAPPING.get(stock1, {}).get('sector', 'unknown')
                    sector2 = STOCK_SECTOR_MAPPING.get(stock2, {}).get('sector', 'unknown')
                    name1 = STOCK_SECTOR_MAPPING.get(stock1, {}).get('name', stock1)
                    name2 = STOCK_SECTOR_MAPPING.get(stock2, {}).get('name', stock2)

                    high_corr_pairs.append({
                        'stock1': stock1,
                        'stock1_name': name1,
                        'stock1_sector': sector1,
                        'stock2': stock2,
                        'stock2_name': name2,
                        'stock2_sector': sector2,
                        'correlation': corr_value,
                        'same_sector': sector1 == sector2
                    })

    # 按相关性排序
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return high_corr_pairs


def calculate_sector_correlation(stock_data):
    """计算板块间相关性"""
    print("\n📊 计算板块间相关性...")

    # 按板块分组计算平均收益率
    sector_returns = {}
    for stock_code, df in stock_data.items():
        sector = STOCK_SECTOR_MAPPING.get(stock_code, {}).get('sector', 'unknown')
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(df['Return'])

    # 计算板块平均收益率
    sector_avg_returns = {}
    for sector, returns_list in sector_returns.items():
        if returns_list:
            combined = pd.concat(returns_list, axis=1)
            sector_avg_returns[sector] = combined.mean(axis=1)

    # 构建板块收益率 DataFrame
    sector_df = pd.DataFrame(sector_avg_returns)
    sector_df = sector_df.dropna()

    # 计算板块间相关性
    sector_corr = sector_df.corr(method='pearson')

    print(f"  ✅ 板块相关性矩阵计算完成（{len(sector_corr)} 个板块）")

    return sector_corr, sector_avg_returns


def analyze_lead_lag_relationship(stock_data, max_lag=5):
    """使用 Granger 因果检验分析领先滞后关系"""
    print("\n📊 分析领先滞后关系（Granger 因果检验）...")

    # 构建收益率 DataFrame
    returns_dict = {}
    for stock_code, df in stock_data.items():
        returns_dict[stock_code] = df['Return']

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    # 选择代表性股票进行 Granger 检验（避免计算量过大）
    # 取每板块代表性股票
    sector_rep_stocks = {}
    for stock_code, info in STOCK_SECTOR_MAPPING.items():
        sector = info['sector']
        if sector not in sector_rep_stocks:
            sector_rep_stocks[sector] = stock_code

    rep_stocks = list(sector_rep_stocks.values())
    print(f"  使用 {len(rep_stocks)} 只代表性股票进行 Granger 检验...")

    lead_lag_results = []
    lead_count = {}  # 统计每只股票领先其他股票的次数
    lag_count = {}   # 统计每只股票滞后其他股票的次数

    for i, stock1 in enumerate(rep_stocks):
        for j, stock2 in enumerate(rep_stocks):
            if i != j and stock1 in returns_df.columns and stock2 in returns_df.columns:
                try:
                    # 准备数据（stock1 是否 Granger 导致 stock2）
                    test_data = returns_df[[stock2, stock1]].values

                    # Granger 因果检验
                    result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                    # 检查各 lag 的 p 值
                    for lag in range(1, max_lag + 1):
                        p_value = result[lag][0]['ssr_ftest'][1]
                        if p_value < 0.05:  # 显著的 Granger 因果关系
                            name1 = STOCK_SECTOR_MAPPING.get(stock1, {}).get('name', stock1)
                            name2 = STOCK_SECTOR_MAPPING.get(stock2, {}).get('name', stock2)

                            lead_lag_results.append({
                                'leader': stock1,
                                'leader_name': name1,
                                'follower': stock2,
                                'follower_name': name2,
                                'lag': lag,
                                'p_value': p_value,
                                'significant': True
                            })

                            # 更新计数
                            lead_count[stock1] = lead_count.get(stock1, 0) + 1
                            lag_count[stock2] = lag_count.get(stock2, 0) + 1

                except Exception as e:
                    continue

    # 排序结果
    lead_lag_results.sort(key=lambda x: x['p_value'])

    # 找出领头羊股票
    leader_stocks = sorted(lead_count.items(), key=lambda x: x[1], reverse=True)

    print(f"  ✅ Granger 检验完成，发现 {len(lead_lag_results)} 个显著领先滞后关系")

    return lead_lag_results, leader_stocks, lead_count, lag_count


def analyze_cyclical_defensive_rotation(sector_corr, sector_avg_returns):
    """分析周期/防御板块轮动"""
    print("\n📊 分析周期/防御板块轮动...")

    # 定义周期性和防御性板块
    cyclical_sectors = ['tech', 'semiconductor', 'new_energy', 'energy', 'shipping',
                        'real_estate', 'consumer', 'auto', 'ai', 'biotech']
    defensive_sectors = ['bank', 'insurance', 'utility', 'index']

    # 计算周期板块平均相关性
    cyclical_corr_values = []
    defensive_corr_values = []
    cross_corr_values = []

    sectors = list(sector_corr.columns)
    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            if i < j:
                corr = sector_corr.loc[s1, s2]
                if s1 in cyclical_sectors and s2 in cyclical_sectors:
                    cyclical_corr_values.append(corr)
                elif s1 in defensive_sectors and s2 in defensive_sectors:
                    defensive_corr_values.append(corr)
                else:
                    cross_corr_values.append(corr)

    rotation_analysis = {
        'cyclical_internal_corr': np.mean(cyclical_corr_values) if cyclical_corr_values else 0,
        'defensive_internal_corr': np.mean(defensive_corr_values) if defensive_corr_values else 0,
        'cross_corr': np.mean(cross_corr_values) if cross_corr_values else 0,
        'cyclical_sectors': cyclical_sectors,
        'defensive_sectors': defensive_sectors
    }

    print(f"  周期板块内部相关性: {rotation_analysis['cyclical_internal_corr']:.4f}")
    print(f"  防御板块内部相关性: {rotation_analysis['defensive_internal_corr']:.4f}")
    print(f"  周期-防御交叉相关性: {rotation_analysis['cross_corr']:.4f}")

    return rotation_analysis


def generate_report(pearson_corr, spearman_corr, high_corr_pairs,
                    sector_corr, lead_lag_results, leader_stocks,
                    rotation_analysis, lead_count=None):
    """生成分析报告"""
    print("\n📝 生成分析报告...")

    report_lines = []
    report_lines.append("# 股票关联与因果关系分析报告")
    report_lines.append(f"\n**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**分析范围**: {len(pearson_corr)} 只股票，{len(sector_corr)} 个板块")
    report_lines.append("\n---")

    # ========== 1. 价格相关性分析 ==========
    report_lines.append("\n## 一、价格相关性分析")

    report_lines.append("\n### 1.1 高相关性股票组合（r > 0.7）")
    report_lines.append("\n| 排名 | 股票A | 股票B | 相关性 | 同板块 |")
    report_lines.append("|------|-------|-------|--------|--------|")

    for i, pair in enumerate(high_corr_pairs[:20]):  # 只显示前20个
        same_sector = "✅" if pair['same_sector'] else "⚠️"
        report_lines.append(f"| {i+1} | {pair['stock1_name']} | {pair['stock2_name']} | "
                           f"{pair['correlation']:.4f} | {same_sector} |")

    # 统计同板块 vs 跨板块
    same_sector_count = sum(1 for p in high_corr_pairs if p['same_sector'])
    cross_sector_count = len(high_corr_pairs) - same_sector_count

    report_lines.append(f"\n**统计**：")
    report_lines.append(f"- 同板块高相关性组合：{same_sector_count} 个")
    report_lines.append(f"- 跨板块高相关性组合：{cross_sector_count} 个")

    # ========== 1.2 跨板块高相关性组合详细列表 ==========
    cross_high = [p for p in high_corr_pairs if not p['same_sector']]
    if cross_high:
        report_lines.append("\n### 1.2 跨板块高相关性组合详细列表")
        report_lines.append("\n| 排名 | 股票A | 板块A | 股票B | 板块B | 相关性 |")
        report_lines.append("|------|-------|--------|-------|--------|--------|")
        for i, pair in enumerate(cross_high):
            report_lines.append(f"| {i+1} | {pair['stock1_name']} | {pair['stock1_sector']} | "
                               f"{pair['stock2_name']} | {pair['stock2_sector']} | "
                               f"{pair['correlation']:.4f} |")

    # ========== 2. 板块间相关性 ==========
    report_lines.append("\n## 二、板块间相关性分析")

    report_lines.append("\n### 2.1 板块相关性矩阵（Top 10）")
    report_lines.append("\n| 板块A | 板块B | 相关性 |")
    report_lines.append("|-------|-------|--------|")

    # 找出板块间高相关性组合
    sector_pairs = []
    sectors = list(sector_corr.columns)
    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            if i < j:
                corr = sector_corr.loc[s1, s2]
                sector_pairs.append({'sector1': s1, 'sector2': s2, 'correlation': corr})

    sector_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    for pair in sector_pairs[:10]:
        report_lines.append(f"| {pair['sector1']} | {pair['sector2']} | {pair['correlation']:.4f} |")

    # ========== 3. 领先滞后关系 ==========
    report_lines.append("\n## 三、领先滞后关系分析（Granger 因果检验）")

    report_lines.append("\n### 3.1 领头羊股票（领先次数最多）")
    report_lines.append("\n| 排名 | 股票 | 领先次数 | 说明 |")
    report_lines.append("|------|------|----------|------|")

    for i, (stock, count) in enumerate(leader_stocks[:10]):
        name = STOCK_SECTOR_MAPPING.get(stock, {}).get('name', stock)
        sector = STOCK_SECTOR_MAPPING.get(stock, {}).get('sector', 'unknown')
        report_lines.append(f"| {i+1} | {name} | {count} | {sector}板块领头羊 |")

    # ========== 3.2 领头羊详细领先关系 ==========
    report_lines.append("\n### 3.2 领头羊详细领先关系")

    for rank, (leader_stock, count) in enumerate(leader_stocks[:5], 1):  # 前5名领头羊
        leader_name = STOCK_SECTOR_MAPPING.get(leader_stock, {}).get('name', leader_stock)
        leader_sector = STOCK_SECTOR_MAPPING.get(leader_stock, {}).get('sector', 'unknown')

        # 找出该领头羊领先的所有股票
        led_stocks = [r for r in lead_lag_results if r['leader'] == leader_stock]
        led_stocks.sort(key=lambda x: x['p_value'])  # 按p值排序

        if led_stocks:
            report_lines.append(f"\n#### {rank}. {leader_name}（{leader_sector}板块）- 领先 {len(led_stocks)} 只股票")
            report_lines.append("\n| 滞后股票 | 所属板块 | 滞后天数 | p值 | 显著性 |")
            report_lines.append("|----------|----------|----------|-----|--------|")

            for result in led_stocks[:10]:  # 每个领头羊最多显示10个
                follower_sector = STOCK_SECTOR_MAPPING.get(result['follower'], {}).get('sector', 'unknown')
                significance = "⭐⭐⭐" if result['p_value'] < 0.001 else ("⭐⭐" if result['p_value'] < 0.01 else "⭐")
                report_lines.append(f"| {result['follower_name']} | {follower_sector} | "
                                   f"{result['lag']}天 | {result['p_value']:.4f} | {significance} |")

    report_lines.append("\n### 3.3 显著领先滞后关系汇总（p < 0.05）")
    report_lines.append("\n| 领先股票 | 滞后股票 | 滞后天数 | p值 |")
    report_lines.append("|----------|----------|----------|-----|")

    for result in lead_lag_results[:15]:
        report_lines.append(f"| {result['leader_name']} | {result['follower_name']} | "
                           f"{result['lag']}天 | {result['p_value']:.4f} |")

    # ========== 4. 周期/防御轮动 ==========
    report_lines.append("\n## 四、周期/防御板块轮动分析")

    report_lines.append("\n| 类型 | 内部相关性 | 说明 |")
    report_lines.append("|------|-----------|------|")
    report_lines.append(f"| 周期板块 | {rotation_analysis['cyclical_internal_corr']:.4f} | "
                       f"科技、半导体、新能源等 |")
    report_lines.append(f"| 防御板块 | {rotation_analysis['defensive_internal_corr']:.4f} | "
                       f"银行、保险、公用事业等 |")
    report_lines.append(f"| 周期-防御 | {rotation_analysis['cross_corr']:.4f} | "
                       f"轮动相关性 |")

    # ========== 5. 关键发现 ==========
    report_lines.append("\n## 五、关键发现")

    findings = []

    # 发现1：领头羊股票
    if leader_stocks:
        top_leader = leader_stocks[0]
        leader_name = STOCK_SECTOR_MAPPING.get(top_leader[0], {}).get('name', top_leader[0])
        findings.append(f"- **领头羊股票**：{leader_name} 领先 {top_leader[1]} 只股票，可作为市场风向标")

    # 发现2：高相关性组合
    if high_corr_pairs:
        top_pair = high_corr_pairs[0]
        findings.append(f"- **最高相关性**：{top_pair['stock1_name']} vs {top_pair['stock2_name']} "
                       f"(r={top_pair['correlation']:.4f})")

    # 发现3：跨板块异常相关性
    cross_high = [p for p in high_corr_pairs if not p['same_sector']]
    if cross_high:
        findings.append(f"- **跨板块异常相关性**：发现 {len(cross_high)} 个跨板块高相关性组合，需关注联动效应")

    # 发现4：周期防御轮动
    if rotation_analysis['cross_corr'] < 0:
        findings.append(f"- **轮动效应**：周期-防御板块相关性为负值，存在明显轮动效应")
    else:
        findings.append(f"- **同步效应**：周期-防御板块相关性为正值，倾向于同步涨跌")

    for finding in findings:
        report_lines.append(finding)

    report_lines.append("\n---")
    report_lines.append(f"\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    report_content = "\n".join(report_lines)

    # 保存报告
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'stock_correlation_analysis.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"  ✅ 报告已保存到: {report_path}")

    return report_content


def save_json_results(pearson_corr, spearman_corr, high_corr_pairs,
                      sector_corr, lead_lag_results, leader_stocks,
                      rotation_analysis):
    """保存 JSON 格式结果"""

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_count': len(pearson_corr),
        'sector_count': len(sector_corr),
        'price_correlation': {
            'high_correlation_pairs': high_corr_pairs,
            'pearson_matrix': pearson_corr.to_dict(),
            'spearman_matrix': spearman_corr.to_dict()
        },
        'sector_correlation': {
            'matrix': sector_corr.to_dict(),
            'high_pairs': []
        },
        'lead_lag_analysis': {
            'significant_relations': lead_lag_results,
            'leader_stocks': [{'stock': s, 'count': c} for s, c in leader_stocks]
        },
        'rotation_analysis': rotation_analysis
    }

    # 添加板块高相关性组合
    sectors = list(sector_corr.columns)
    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            if i < j:
                corr = sector_corr.loc[s1, s2]
                if abs(corr) > 0.5:
                    results['sector_correlation']['high_pairs'].append({
                        'sector1': s1,
                        'sector2': s2,
                        'correlation': float(corr)
                    })

    # 保存 JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'stock_correlation_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ JSON 结果已保存到: {json_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("股票关联与因果关系分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 获取股票列表
    stock_list = get_stock_list()
    print(f"股票列表: {len(stock_list)} 只股票")

    # 2. 获取股票数据
    stock_data = fetch_all_stock_data(stock_list, period="2y")

    if len(stock_data) < 10:
        print("❌ 数据不足，无法分析")
        return

    # 3. 计算价格相关性
    pearson_corr, spearman_corr = calculate_price_correlation(stock_data)

    # 4. 找出高相关性组合
    high_corr_pairs = find_high_correlation_pairs(pearson_corr, threshold=0.7)

    # 5. 计算板块间相关性
    sector_corr, sector_avg_returns = calculate_sector_correlation(stock_data)

    # 6. 分析领先滞后关系
    lead_lag_results, leader_stocks, lead_count, lag_count = \
        analyze_lead_lag_relationship(stock_data, max_lag=5)

    # 7. 分析周期/防御轮动
    rotation_analysis = analyze_cyclical_defensive_rotation(sector_corr, sector_avg_returns)

    # 8. 生成报告
    report = generate_report(pearson_corr, spearman_corr, high_corr_pairs,
                             sector_corr, lead_lag_results, leader_stocks,
                             rotation_analysis, lead_count)

    # 9. 保存 JSON
    save_json_results(pearson_corr, spearman_corr, high_corr_pairs,
                      sector_corr, lead_lag_results, leader_stocks,
                      rotation_analysis)

    # 10. 打印摘要
    print("\n" + "=" * 80)
    print("📊 分析摘要")
    print("=" * 80)
    print(f"  高相关性组合: {len(high_corr_pairs)} 个")
    print(f"  显著领先滞后关系: {len(lead_lag_results)} 个")
    if leader_stocks:
        top_leader = leader_stocks[0]
        leader_name = STOCK_SECTOR_MAPPING.get(top_leader[0], {}).get('name', top_leader[0])
        print(f"  领头羊股票: {leader_name}（领先 {top_leader[1]} 只股票）")

    print("\n" + "=" * 80)
    print(f"分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()