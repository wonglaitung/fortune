#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习预测结果邮件发送脚本
读取预测结果CSV文件，生成格式化的邮件并发送
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate


class MLPredictionEmailSender:
    """机器学习预测邮件发送器"""

    def __init__(self):
        self.email = os.getenv('YAHOO_EMAIL')
        self.app_password = os.getenv('YAHOO_APP_PASSWORD')
        self.smtp_server = os.getenv('YAHOO_SMTP', 'smtp.163.com')
        self.recipients = os.getenv('RECIPIENT_EMAIL', '').split(',')

    def load_predictions(self, horizon):
        """加载指定周期的预测结果

        Args:
            horizon: 预测周期（1=次日，5=一周，20=一个月）

        Returns:
            tuple: (lgbm_df, gbdt_lr_df)
        """
        try:
            lgbm_file = f'data/ml_trading_model_lgbm_predictions_{horizon}d.csv'
            gbdt_lr_file = f'data/ml_trading_model_gbdt_lr_predictions_{horizon}d.csv'

            if not os.path.exists(lgbm_file) or not os.path.exists(gbdt_lr_file):
                print(f"⚠️  预测文件不存在: {lgbm_file} 或 {gbdt_lr_file}")
                return None, None

            lgbm_df = pd.read_csv(lgbm_file)
            gbdt_lr_df = pd.read_csv(gbdt_lr_file)

            return lgbm_df, gbdt_lr_df
        except Exception as e:
            print(f"❌ 加载预测结果失败: {e}")
            return None, None

    def generate_comparison_table(self, lgbm_df, gbdt_lr_df, horizon):
        """生成对比表格

        Args:
            lgbm_df: LightGBM预测结果
            gbdt_lr_df: GBDT+LR预测结果
            horizon: 预测周期

        Returns:
            str: 格式化的表格字符串
        """
        # 合并数据
        comparison = lgbm_df.merge(
            gbdt_lr_df,
            on='code',
            suffixes=('_lgbm', '_gbdt_lr')
        )

        # 重命名列
        comparison.columns = ['code', 'name_lgbm', 'prediction_lgbm', 'probability_lgbm',
                             'current_price', 'date_lgbm', 'target_lgbm',
                             'name_gbdt_lr', 'prediction_gbdt_lr', 'probability_gbdt_lr',
                             'current_price_gbdt_lr', 'date_gbdt_lr', 'target_gbdt_lr']

        # 计算预测一致性
        comparison['consistent'] = comparison['prediction_lgbm'] == comparison['prediction_gbdt_lr']

        # 计算概率差异
        comparison['probability_diff'] = abs(comparison['probability_lgbm'] - comparison['probability_gbdt_lr'])

        # 排序
        comparison = comparison.sort_values('probability_diff', ascending=False)

        # 生成表格
        horizon_text = {1: '次日', 5: '一周', 20: '一个月'}[horizon]
        table = f"""
{'=' * 136}
📊 两种模型预测结果对比 - {horizon_text}涨跌预测
{'=' * 136}

{'代码':<12} {'股票名称':<14} {'LGBM预测':<10} {'LGBM概率':<10} {'GBDT+LR预测':<12} {'GBDT+LR概率':<12} {'是否一致':<8} {'概率差异':<10} {'当前价格':<10} {'预测目标'}
{'-' * 136}
"""

        for _, row in comparison.iterrows():
            lgbm_pred_label = "上涨" if row['prediction_lgbm'] == 1 else "下跌"
            gbdt_lr_pred_label = "上涨" if row['prediction_gbdt_lr'] == 1 else "下跌"
            consistent = "✓" if row['consistent'] else "✗"

            table += f"{row['code']:<12} {row['name_lgbm']:<14} {lgbm_pred_label:<10} {row['probability_lgbm']:<10.4f} {gbdt_lr_pred_label:<12} {row['probability_gbdt_lr']:<12.4f} {consistent:<8} {row['probability_diff']:<10.4f} {row['current_price']:<10} {row['target_lgbm']}\n"

        # 统计摘要
        consistent_count = len(comparison[comparison['consistent']])
        total_count = len(comparison)
        consistency_rate = (consistent_count / total_count * 100) if total_count > 0 else 0

        lgbm_up = len(comparison[comparison['prediction_lgbm'] == 1])
        lgbm_down = len(comparison[comparison['prediction_lgbm'] == 0])
        gbdt_lr_up = len(comparison[comparison['prediction_gbdt_lr'] == 1])
        gbdt_lr_down = len(comparison[comparison['prediction_gbdt_lr'] == 0])

        avg_prob_diff = comparison['probability_diff'].mean()

        table += f"""
{'=' * 136}
📈 统计摘要
{'=' * 136}

预测一致性: {consistent_count}/{total_count} ({consistency_rate:.1f}%)

LGBM 模型: 上涨 {lgbm_up} 只, 下跌 {lgbm_down} 只
GBDT+LR 模型: 上涨 {gbdt_lr_up} 只, 下跌 {gbdt_lr_down} 只

平均概率差异: {avg_prob_diff:.4f}
"""

        # 预测不一致的股票
        inconsistent = comparison[~comparison['consistent']]
        if not inconsistent.empty:
            table += f"""
{'=' * 136}
⚠️  预测不一致的股票
{'=' * 136}
"""
            for _, row in inconsistent.iterrows():
                lgbm_pred_label = "上涨" if row['prediction_lgbm'] == 1 else "下跌"
                gbdt_lr_pred_label = "上涨" if row['prediction_gbdt_lr'] == 1 else "下跌"
                table += f"{row['code']:<12} {row['name_lgbm']:<14} LGBM: {lgbm_pred_label} ({row['probability_lgbm']:.4f})  vs  GBDT+LR: {gbdt_lr_pred_label} ({row['probability_gbdt_lr']:.4f})\n"

        table += f"\n{'=' * 136}\n"

        return table

    def send_email(self, subject, content):
        """发送邮件

        Args:
            subject: 邮件主题
            content: 邮件内容（纯文本）
        """
        if not all([self.email, self.app_password, self.recipients]):
            print("❌ 邮件配置不完整，跳过发送")
            return False

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            msg['Date'] = formatdate(localtime=True)

            # 添加内容（使用纯文本格式）
            msg.attach(MIMEText(content, 'plain', 'utf-8'))

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, 587) as server:
                server.starttls()
                server.login(self.email, self.app_password)
                server.send_message(msg)

            print(f"✅ 邮件发送成功: {subject}")
            return True
        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")
            return False

    def send_prediction_alert(self, horizons=[1, 5, 20]):
        """发送预测结果邮件

        Args:
            horizons: 要发送的预测周期列表
        """
        content = f"🤖 机器学习交易模型预测报告\n"
        content += f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"\n"

        for horizon in horizons:
            print(f"\n📊 处理 {horizon} 天预测...")
            lgbm_df, gbdt_lr_df = self.load_predictions(horizon)

            if lgbm_df is not None and gbdt_lr_df is not None:
                table = self.generate_comparison_table(lgbm_df, gbdt_lr_df, horizon)
                content += table
            else:
                content += f"⚠️  {horizon} 天预测数据加载失败\n\n"

        # 发送邮件
        subject = f"🤖 机器学习预测报告 - {datetime.now().strftime('%Y-%m-%d')}"
        self.send_email(subject, content)


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 机器学习预测结果邮件发送")
    print("=" * 60)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    sender = MLPredictionEmailSender()
    sender.send_prediction_alert(horizons=[1, 5, 20])

    print()
    print("=" * 60)
    print("✅ 任务完成！")
    print("=" * 60)
    print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()