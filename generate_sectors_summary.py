#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成所有板块Walk-forward验证汇总报告

读取所有已生成的板块报告文件，生成汇总对比报告
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SECTOR_NAME_MAPPING


def generate_summary_report(output_dir='output'):
    """生成所有板块的汇总报告"""
    
    # 查找所有板块报告文件
    json_files = list(Path(output_dir).glob('walk_forward_sector_*_catboost_20d_*.json'))
    
    if not json_files:
        print("没有找到板块报告文件")
        return None
    
    print(f"找到 {len(json_files)} 个板块报告文件")
    
    # 读取所有报告
    all_sector_reports = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                all_sector_reports.append(report)
                print(f"✅ 已读取: {json_file.name} ({report.get('sector_name', 'Unknown')})")
        except Exception as e:
            print(f"❌ 读取失败 {json_file.name}: {e}")
            continue
    
    if not all_sector_reports:
        print("没有成功读取任何板块报告")
        return None
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_file = os.path.join(output_dir, f'walk_forward_sectors_summary_{timestamp}.md')
    
    # 提取所有板块的整体指标
    sector_summary = []
    for report in all_sector_reports:
        if report.get('overall_metrics'):
            metrics = report['overall_metrics']
            sector_summary.append({
                'sector_code': report['sector_code'],
                'sector_name': report['sector_name'],
                'num_stocks': report['num_stocks'],
                'num_folds': metrics.get('num_folds', 0),
                'avg_return': metrics.get('avg_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'avg_win_rate': metrics.get('avg_win_rate', 0),
                'avg_accuracy': metrics.get('avg_accuracy', 0),
                'avg_correct_decision_ratio': metrics.get('avg_correct_decision_ratio', 0),
                'avg_sharpe_ratio': metrics.get('avg_sharpe_ratio', 0),
                'avg_max_drawdown': metrics.get('avg_max_drawdown', 0),
                'return_std': metrics.get('return_std', 0),
                'stability_rating': metrics.get('stability_rating', '未知')
            })
    
    # 排序
    sector_summary_df = pd.DataFrame(sector_summary)
    sector_summary_df = sector_summary_df.sort_values('avg_sharpe_ratio', ascending=False).reset_index(drop=True)
    
    with open(md_file, 'w', encoding='utf-8') as f:
        # 标题
        f.write(f"# 所有板块Walk-forward验证汇总报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**板块数量**: {len(sector_summary)}\n\n")
        
        # 配置信息
        if all_sector_reports:
            first_config = all_sector_reports[0]['validation_config']
            f.write("## 🔬 验证配置\n\n")
            f.write(f"- **模型类型**: {first_config['model_type'].upper()}\n")
            f.write(f"- **训练窗口**: {first_config['train_window_months']} 个月\n")
            f.write(f"- **测试窗口**: {first_config['test_window_months']} 个月\n")
            f.write(f"- **滚动步长**: {first_config['step_window_months']} 个月\n")
            f.write(f"- **预测周期**: {first_config['horizon']} 天\n")
            f.write(f"- **置信度阈值**: {first_config['confidence_threshold']}\n")
            f.write(f"- **特征选择**: {'是' if first_config['use_feature_selection'] else '否'}\n")
            f.write(f"- **验证日期**: {first_config['start_date']} 至 {first_config['end_date']}\n")
            f.write(f"- **指标说明**: 收益率为年化，胜率仅统计买入信号，正确决策比例 = (盈利买入 + 正确不买入) / 总决策\n\n")
        
        # 板块对比表格
        f.write("## 📊 板块性能对比\n\n")
        f.write("| 排名 | 板块 | 股票数 | Fold数 | 年化收益率 | 买入胜率 | 准确率 | 正确决策率 | 夏普比率 | 最大回撤 | 稳定性 |\n")
        f.write("|------|------|-------|-------|-----------|---------|-------|-----------|---------|--------|--------|\n")
        
        for idx, row in sector_summary_df.iterrows():
            f.write(f"| {idx + 1} | {row['sector_name']} ({row['sector_code']}) | {row['num_stocks']} | "
                   f"{row['num_folds']} | {row['annualized_return']:.2%} | {row['avg_win_rate']:.2%} | "
                   f"{row['avg_accuracy']:.2%} | {row['avg_correct_decision_ratio']:.2%} | "
                   f"{row['avg_sharpe_ratio']:.4f} | {row['avg_max_drawdown']:.2%} | {row['stability_rating']} |\n")
        
        f.write("\n")
        
        # 排名分析
        f.write("## 🏆 板块排名分析\n\n")
        
        # TOP 5 夏普比率
        top_sharpe = sector_summary_df.head(5)
        f.write("### 夏普比率 TOP 5\n\n")
        for idx, row in top_sharpe.iterrows():
            f.write(f"{idx + 1}. **{row['sector_name']}**: 夏普比率 {row['avg_sharpe_ratio']:.4f}, "
                   f"收益率 {row['avg_return']:.2%}, 年化收益率 {row['annualized_return']:.2%}, 胜率 {row['avg_win_rate']:.2%}\n")
        f.write("\n")
        
        # TOP 5 收益率
        top_return = sector_summary_df.nlargest(5, 'avg_return')
        f.write("### 平均收益率 TOP 5\n\n")
        for idx, (_, row) in enumerate(top_return.iterrows(), 1):
            f.write(f"{idx}. **{row['sector_name']}**: 收益率 {row['avg_return']:.2%}, 年化收益率 {row['annualized_return']:.2%}, "
                   f"夏普比率 {row['avg_sharpe_ratio']:.4f}, 胜率 {row['avg_win_rate']:.2%}\n")
        f.write("\n")
        
        # 稳定性分析
        high_stability = sector_summary_df[sector_summary_df['stability_rating'] == "高（优秀）"]
        f.write("### 稳定性分析\n\n")
        f.write(f"- **高稳定性板块**: {len(high_stability)} 个\n")
        if len(high_stability) > 0:
            f.write(f"  - {', '.join(high_stability['sector_name'].tolist())}\n")
        
        medium_stability = sector_summary_df[sector_summary_df['stability_rating'] == "中（良好）"]
        f.write(f"- **中稳定性板块**: {len(medium_stability)} 个\n")
        if len(medium_stability) > 0:
            f.write(f"  - {', '.join(medium_stability['sector_name'].tolist())}\n")
        f.write("\n")
        
        # 投资建议
        f.write("## 💡 投资建议\n\n")
        if len(top_sharpe) > 0:
            best_sector = top_sharpe.iloc[0]
            f.write(f"### 推荐板块\n\n")
            f.write(f"**{best_sector['sector_name']}** ({best_sector['sector_code']})\n\n")
            f.write(f"- 夏普比率: {best_sector['avg_sharpe_ratio']:.4f}\n")
            f.write(f"- 平均收益率: {best_sector['avg_return']:.2%}\n")
            f.write(f"- 年化收益率: {best_sector['annualized_return']:.2%}\n")
            f.write(f"- 胜率: {best_sector['avg_win_rate']:.2%}\n")
            f.write(f"- 准确率: {best_sector['avg_accuracy']:.2%}\n")
            f.write(f"- 稳定性: {best_sector['stability_rating']}\n\n")
            f.write(f"建议：该板块模型表现优秀，稳定性高，适合重点配置。\n\n")
            
            # 次优推荐
            if len(top_sharpe) > 1:
                second_sector = top_sharpe.iloc[1]
                f.write(f"### 次优板块\n\n")
                f.write(f"**{second_sector['sector_name']}** ({second_sector['sector_code']})\n\n")
                f.write(f"- 夏普比率: {second_sector['avg_sharpe_ratio']:.4f}\n")
                f.write(f"- 平均收益率: {second_sector['avg_return']:.2%}\n")
                f.write(f"- 年化收益率: {second_sector['annualized_return']:.2%}\n")
                f.write(f"- 胜率: {second_sector['avg_win_rate']:.2%}\n")
                f.write(f"- 准确率: {second_sector['avg_accuracy']:.2%}\n")
                f.write(f"- 稳定性: {second_sector['stability_rating']}\n\n")
                f.write(f"建议：该板块模型表现良好，可作为辅助配置。\n\n")
        
        # 板块详细数据
        f.write("## 📈 板块详细数据\n\n")
        for idx, row in sector_summary_df.iterrows():
            f.write(f"### {idx + 1}. {row['sector_name']} ({row['sector_code']})\n\n")
            f.write(f"- **股票数量**: {row['num_stocks']} 只\n")
            f.write(f"- **Fold数量**: {row['num_folds']}\n")
            f.write(f"- **平均收益率**: {row['avg_return']:.2%}\n")
            f.write(f"- **年化收益率**: {row['annualized_return']:.2%}\n")
            f.write(f"- **买入胜率**: {row['avg_win_rate']:.2%}\n")
            f.write(f"- **准确率**: {row['avg_accuracy']:.2%}\n")
            f.write(f"- **正确决策率**: {row['avg_correct_decision_ratio']:.2%}\n")
            f.write(f"- **夏普比率**: {row['avg_sharpe_ratio']:.4f}\n")
            f.write(f"- **最大回撤**: {row['avg_max_drawdown']:.2%}\n")
            f.write(f"- **收益率标准差**: {row['return_std']:.2%}\n")
            f.write(f"- **稳定性评级**: {row['stability_rating']}\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"\n✅ 汇总报告已保存: {md_file}")
    return md_file


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    summary_file = generate_summary_report(output_dir)
    
    if summary_file:
        print(f"\n{'='*80}")
        print("✅ 汇总报告生成完成")
        print(f"{'='*80}")
        print(f"报告文件: {summary_file}")
        print(f"{'='*80}\n")
    else:
        print("\n❌ 汇总报告生成失败")
