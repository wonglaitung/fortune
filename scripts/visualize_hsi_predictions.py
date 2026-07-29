#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数预测历史可视化

将多天的恒指预测结果叠加在图表上，并与实际情况对比

图表类型：
1. 价格走势 + 预测标记图
2. 累计收益曲线图
3. 准确率统计图
4. 置信度与准确率关系图

创建时间：2026-06-09
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入 yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ yfinance 未安装，将无法获取恒指价格数据")


# ============== 配置 ==============
PREDICTION_HISTORY_FILE = "data/hsi_prediction_history.json"
OUTPUT_DIR = "output/hsi_prediction_visualization"

# 颜色配置
COLORS = {
    'correct_up': '#2ca02c',      # 正确预测上涨 - 绿色
    'correct_down': '#1f77b4',    # 正确预测下跌 - 蓝色
    'incorrect_up': '#ff7f0e',    # 错误预测上涨 - 橙色
    'incorrect_down': '#d62728',  # 错误预测下跌 - 红色
    'pending': '#7f7f7f',         # 待验证 - 灰色
    'price_line': '#1f77b4',      # 价格线 - 蓝色
    'strategy': '#2ca02c',        # 策略收益 - 绿色
    'benchmark': '#ff7f0e',       # 基准收益 - 橙色
}

HORIZON_COLORS = {
    1: '#d62728',   # 1天 - 红色
    5: '#ff7f0e',   # 5天 - 橙色
    20: '#2ca02c',  # 20天 - 绿色
}


def load_prediction_history():
    """加载预测历史数据"""
    with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['predictions']


def get_hsi_price_data(start_date, end_date):
    """获取恒指价格数据"""
    if not HAS_YFINANCE:
        return None

    try:
        hsi = yf.Ticker("^HSI")
        df = hsi.history(start=start_date, end=end_date)
        if df.empty:
            print("⚠️ 无法获取恒指价格数据")
            return None
        return df
    except Exception as e:
        print(f"⚠️ 获取恒指价格数据失败: {e}")
        return None


def prepare_prediction_dataframe(predictions):
    """将预测数据转换为 DataFrame"""
    records = []
    for p in predictions:
        record = {
            'prediction_id': p['prediction_id'],
            'data_date': pd.to_datetime(p['data_date']),
            'target_date': pd.to_datetime(p['target_date']),
            'horizon': p['horizon'],
            'predicted_direction': p['predicted_direction'],
            'probability': p['prediction_probability'],
            'confidence_level': p['confidence_level'],
            'entry_price': p['entry_price'],
            'outcome': p.get('outcome'),
            'actual_return': p.get('actual_return'),
            'actual_direction': p.get('actual_direction'),
            'verified': p.get('verified', False)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('data_date')
    return df


def plot_price_with_predictions(hsi_df, pred_df, horizon, output_path):
    """
    绘制价格走势 + 预测标记图

    参数:
    - hsi_df: 恒指价格 DataFrame
    - pred_df: 预测数据 DataFrame
    - horizon: 预测周期 (1, 5, 20)
    - output_path: 输出文件路径
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # 筛选指定周期的预测
    pred_horizon = pred_df[pred_df['horizon'] == horizon].copy()

    # 绘制价格走势
    ax.plot(hsi_df.index, hsi_df['Close'], color=COLORS['price_line'],
            linewidth=1.5, label='HSI Close Price', alpha=0.8)

    # 标记预测点
    for _, row in pred_horizon.iterrows():
        data_date = row['data_date']
        target_date = row['target_date']

        # 获取预测点的价格
        try:
            price = hsi_df.loc[data_date, 'Close']
            if pd.isna(price):
                continue
        except KeyError:
            continue

        # 确定颜色和标记
        if row['outcome'] is None:
            color = COLORS['pending']
            marker = 'o'
            alpha = 0.7
        elif row['outcome'] == 'correct':
            if row['predicted_direction'] == 'up':
                color = COLORS['correct_up']
                marker = '^'  # 上三角
            else:
                color = COLORS['correct_down']
                marker = 'v'  # 下三角
            alpha = 1.0
        else:
            if row['predicted_direction'] == 'up':
                color = COLORS['incorrect_up']
                marker = '^'
            else:
                color = COLORS['incorrect_down']
                marker = 'v'
            alpha = 0.7

        # 绘制预测点
        ax.scatter(data_date, price, c=color, marker=marker, s=100, alpha=alpha, zorder=5)

        # 绘制目标日期的垂直线
        ax.axvline(x=target_date, color=color, linestyle='--', alpha=0.3, linewidth=0.8)

    # 设置标题和标签
    ax.set_title(f'HSI Price with {horizon}d Predictions ({pred_horizon["data_date"].min().strftime("%Y-%m-%d")} ~ {pred_horizon["data_date"].max().strftime("%Y-%m-%d")})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('HSI Close Price', fontsize=12)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['price_line'], linewidth=2, label='HSI Price'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['correct_up'],
               markersize=10, label='Correct Up'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['correct_down'],
               markersize=10, label='Correct Down'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['incorrect_up'],
               markersize=10, label='Incorrect Up'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['incorrect_down'],
               markersize=10, label='Incorrect Down'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pending'],
               markersize=10, label='Pending'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def plot_cumulative_returns(pred_df, output_path):
    """
    绘制累计收益曲线图

    模拟按预测执行的累计收益，与买入持有策略对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, horizon in enumerate([1, 5, 20]):
        ax = axes[idx]

        # 筛选已验证的预测
        pred_horizon = pred_df[(pred_df['horizon'] == horizon) &
                               (pred_df['verified'] == True)].copy()
        pred_horizon = pred_horizon.sort_values('data_date')

        if pred_horizon.empty:
            ax.text(0.5, 0.5, 'No verified predictions', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{horizon}d Horizon')
            continue

        # 计算策略收益
        # 每次预测正确则获得该周期的实际收益，错误则损失
        cumulative_return = 1.0
        returns_list = [1.0]
        dates_list = [pred_horizon['data_date'].iloc[0]]

        for _, row in pred_horizon.iterrows():
            if row['actual_return'] is not None:
                # 如果预测上涨且正确，获得正收益；如果预测下跌且正确，获得正收益（做空）
                # 简化处理：正确则获得 |actual_return|，错误则损失 |actual_return|
                if row['outcome'] == 'correct':
                    strategy_return = abs(row['actual_return'])
                else:
                    strategy_return = -abs(row['actual_return'])

                cumulative_return *= (1 + strategy_return)
                returns_list.append(cumulative_return)
                dates_list.append(row['target_date'])

        # 绘制策略收益
        ax.plot(dates_list, returns_list, color=COLORS['strategy'],
                linewidth=2, label='Strategy', marker='o', markersize=4)

        # 计算买入持有基准（从第一个预测到最后一）
        first_date = pred_horizon['data_date'].min()
        last_date = pred_horizon['target_date'].max()

        # 标记收益
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(dates_list, 1.0, returns_list,
                        where=[r >= 1.0 for r in returns_list],
                        alpha=0.3, color=COLORS['strategy'])

        # 设置标题和标签
        final_return = (returns_list[-1] - 1) * 100
        ax.set_title(f'{horizon}d Horizon\nFinal Return: {final_return:+.1f}%', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Cumulative Return', fontsize=10)

        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cumulative Strategy Returns by Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def plot_accuracy_statistics(pred_df, output_path):
    """
    绘制准确率统计图

    按预测周期统计准确率、正确/错误数量
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 只统计已验证的预测
    verified = pred_df[pred_df['verified'] == True]

    # 1. 按周期统计准确率
    ax1 = axes[0]
    horizons = [1, 5, 20]
    accuracies = []
    for h in horizons:
        h_pred = verified[verified['horizon'] == h]
        if len(h_pred) > 0:
            correct = (h_pred['outcome'] == 'correct').sum()
            total = len(h_pred)
            accuracies.append(correct / total * 100)
        else:
            accuracies.append(0)

    bars = ax1.bar([f'{h}d' for h in horizons], accuracies, color=[HORIZON_COLORS[h] for h in horizons])
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Prediction Accuracy by Horizon', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 正确/错误数量堆叠柱状图
    ax2 = axes[1]
    correct_counts = []
    incorrect_counts = []
    pending_counts = []

    for h in horizons:
        h_pred = pred_df[pred_df['horizon'] == h]
        correct_counts.append((h_pred['outcome'] == 'correct').sum())
        incorrect_counts.append((h_pred['outcome'] == 'incorrect').sum())
        pending_counts.append(h_pred['outcome'].isna().sum())

    x = np.arange(len(horizons))
    width = 0.6

    ax2.bar(x, correct_counts, width, label='Correct', color='#2ca02c')
    ax2.bar(x, incorrect_counts, width, bottom=correct_counts, label='Incorrect', color='#d62728')
    ax2.bar(x, pending_counts, width, bottom=np.array(correct_counts)+np.array(incorrect_counts),
            label='Pending', color='#7f7f7f')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h}d' for h in horizons])
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Outcomes by Horizon', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 按置信度统计准确率
    ax3 = axes[2]
    confidence_levels = ['低', '中', '高']
    conf_accuracies = []
    conf_counts = []

    for conf in confidence_levels:
        conf_pred = verified[verified['confidence_level'] == conf]
        if len(conf_pred) > 0:
            correct = (conf_pred['outcome'] == 'correct').sum()
            total = len(conf_pred)
            conf_accuracies.append(correct / total * 100)
            conf_counts.append(total)
        else:
            conf_accuracies.append(0)
            conf_counts.append(0)

    bars = ax3.bar(confidence_levels, conf_accuracies, color=['#ffbb78', '#ff7f0e', '#d62728'])
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_xlabel('Confidence Level', fontsize=12)
    ax3.set_title('Accuracy by Confidence Level', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)

    # 添加数值标签
    for bar, acc, count in zip(bars, conf_accuracies, conf_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def plot_probability_distribution(pred_df, output_path):
    """
    绘制预测概率分布图

    展示不同概率区间的准确率
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 只统计已验证的预测
    verified = pred_df[pred_df['verified'] == True].copy()

    # 1. 概率分布直方图
    ax1 = axes[0]

    # 分离正确和错误预测
    correct = verified[verified['outcome'] == 'correct']
    incorrect = verified[verified['outcome'] == 'incorrect']

    bins = np.arange(0, 1.05, 0.1)

    ax1.hist(correct['probability'], bins=bins, alpha=0.7, label='Correct', color='#2ca02c')
    ax1.hist(incorrect['probability'], bins=bins, alpha=0.7, label='Incorrect', color='#d62728')

    ax1.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Prediction Probabilities', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 概率区间准确率
    ax2 = axes[1]

    prob_bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    prob_labels = ['<40%', '40-50%', '50-60%', '60-70%', '>70%']
    bin_accuracies = []
    bin_counts = []

    for low, high in prob_bins:
        bin_pred = verified[(verified['probability'] >= low) & (verified['probability'] < high)]
        if len(bin_pred) > 0:
            correct = (bin_pred['outcome'] == 'correct').sum()
            total = len(bin_pred)
            bin_accuracies.append(correct / total * 100)
            bin_counts.append(total)
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    bars = ax2.bar(prob_labels, bin_accuracies, color='#1f77b4')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax2.set_xlabel('Probability Range', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy by Probability Range', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)

    # 添加数值标签
    for bar, acc, count in zip(bars, bin_accuracies, bin_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def generate_summary_report(pred_df, output_path):
    """生成统计摘要报告"""
    verified = pred_df[pred_df['verified'] == True]

    report_lines = [
        "# 恒指预测历史统计报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## 总体统计",
        f"- 总预测数: {len(pred_df)}",
        f"- 已验证数: {len(verified)}",
        f"- 待验证数: {len(pred_df) - len(verified)}",
        f"\n## 按周期统计",
    ]

    for h in [1, 5, 20]:
        h_pred = verified[verified['horizon'] == h]
        if len(h_pred) > 0:
            correct = (h_pred['outcome'] == 'correct').sum()
            total = len(h_pred)
            acc = correct / total * 100
            report_lines.append(f"\n### {h}天预测")
            report_lines.append(f"- 预测数: {total}")
            report_lines.append(f"- 正确: {correct}")
            report_lines.append(f"- 错误: {total - correct}")
            report_lines.append(f"- 准确率: {acc:.1f}%")

            # 平均置信度
            avg_prob = h_pred['probability'].mean()
            report_lines.append(f"- 平均概率: {avg_prob:.1%}")

    report_lines.append(f"\n## 按置信度统计")

    for conf in ['低', '中', '高']:
        conf_pred = verified[verified['confidence_level'] == conf]
        if len(conf_pred) > 0:
            correct = (conf_pred['outcome'] == 'correct').sum()
            total = len(conf_pred)
            acc = correct / total * 100
            report_lines.append(f"\n### {conf}置信度")
            report_lines.append(f"- 预测数: {total}")
            report_lines.append(f"- 准确率: {acc:.1f}%")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ 已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='恒指预测历史可视化')
    parser.add_argument('--no-price', action='store_true', help='不绘制价格走势图')
    args = parser.parse_args()

    print("=" * 60)
    print("恒指预测历史可视化")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载预测历史
    print("\n📂 加载预测历史数据...")
    predictions = load_prediction_history()
    pred_df = prepare_prediction_dataframe(predictions)
    print(f"   总预测数: {len(pred_df)}")
    print(f"   已验证: {(pred_df['verified'] == True).sum()}")
    print(f"   时间范围: {pred_df['data_date'].min().strftime('%Y-%m-%d')} ~ {pred_df['data_date'].max().strftime('%Y-%m-%d')}")

    # 获取恒指价格数据
    hsi_df = None
    if not args.no_price:
        print("\n📈 获取恒指价格数据...")
        start_date = pred_df['data_date'].min() - timedelta(days=5)
        end_date = pred_df['target_date'].max() + timedelta(days=5)
        hsi_df = get_hsi_price_data(start_date, end_date)
        if hsi_df is not None:
            print(f"   价格数据: {hsi_df.index[0].strftime('%Y-%m-%d')} ~ {hsi_df.index[-1].strftime('%Y-%m-%d')}")

    # 生成图表
    print("\n📊 生成可视化图表...")

    # 1. 价格走势 + 预测标记图
    if hsi_df is not None:
        for horizon in [1, 5, 20]:
            output_path = os.path.join(OUTPUT_DIR, f'hsi_predictions_{horizon}d.png')
            plot_price_with_predictions(hsi_df, pred_df, horizon, output_path)

    # 2. 累计收益曲线图
    output_path = os.path.join(OUTPUT_DIR, 'hsi_cumulative_returns.png')
    plot_cumulative_returns(pred_df, output_path)

    # 3. 准确率统计图
    output_path = os.path.join(OUTPUT_DIR, 'hsi_accuracy_statistics.png')
    plot_accuracy_statistics(pred_df, output_path)

    # 4. 概率分布图
    output_path = os.path.join(OUTPUT_DIR, 'hsi_probability_distribution.png')
    plot_probability_distribution(pred_df, output_path)

    # 5. 生成统计报告
    output_path = os.path.join(OUTPUT_DIR, 'hsi_prediction_summary.md')
    generate_summary_report(pred_df, output_path)

    print("\n" + "=" * 60)
    print(f"✅ 所有图表已保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
