#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估特定板块模型的买入胜率

使用方法：
  python3 ml_services/evaluate_sector_model.py --sector bank
  python3 ml_services/evaluate_sector_model.py --sector index
  python3 ml_services/evaluate_sector_model.py --sector exchange
"""

import warnings
import os
import sys
import argparse
from datetime import datetime
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml_services.ml_trading_model import CatBoostModel
from ml_services.logger_config import get_logger
from config import STOCK_SECTOR_MAPPING

# 获取日志记录器
logger = get_logger('evaluate_sector_model')


def get_stocks_by_sector(sector_type):
    """根据板块类型获取股票代码列表"""
    stocks = []
    for code, info in STOCK_SECTOR_MAPPING.items():
        if info['type'] == sector_type:
            stocks.append(code)
    return stocks


def evaluate_buy_win_rate(model, codes, horizon=20, confidence_threshold=0.55):
    """
    评估模型的买入胜率

    Args:
        model: 训练好的模型
        codes: 股票代码列表
        horizon: 预测周期
        confidence_threshold: 置信度阈值

    Returns:
        评估结果字典
    """
    # 准备测试数据
    df = model.prepare_data(codes, horizon=horizon)

    print(f"数据准备完成，共 {len(df)} 条记录")
    print(f"数据集特征数量: {len(df.columns)}")

    # 只保留模型中存在的特征
    available_features = [col for col in model.feature_columns if col in df.columns]
    missing_features = [col for col in model.feature_columns if col not in df.columns]

    if missing_features:
        print(f"⚠️  模型中有 {len(missing_features)} 个特征在当前数据中不存在")
        print(f"   前10个缺失特征: {missing_features[:10]}")

    df = df[available_features + ['Label']].copy()

    # 删除 NaN
    df_before_drop = len(df)
    df = df.dropna()
    df_after_drop = len(df)

    print(f"删除 NaN 前: {df_before_drop} 条记录")
    print(f"删除 NaN 后: {df_after_drop} 条记录")

    if df_after_drop < 100:
        print(f"⚠️  数据不足：{df_after_drop} 条记录 < 100 条最小要求")
        return None

    df = df.sort_index()

    # 获取特征
    X = df[model.feature_columns].values
    y = df['Label'].values

    # 预测
    y_pred = model.catboost_model.predict(X)
    y_pred_proba = model.catboost_model.predict_proba(X)[:, 1]

    # 计算基础指标
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)

    # 计算买入胜率
    # 买入信号：预测概率 > 置信度阈值
    buy_signals = y_pred_proba > confidence_threshold
    buy_count = np.sum(buy_signals)

    if buy_count > 0:
        # 买入信号的准确率（胜率）
        buy_accuracy = accuracy_score(y[buy_signals], y_pred[buy_signals])
        # 买入信号的正确决策比例（考虑不买的情况）
        true_positives = np.sum((y == 1) & (y_pred == 1))
        false_positives = np.sum((y == 0) & (y_pred == 1))
        true_negatives = np.sum((y == 0) & (y_pred == 0))
        false_negatives = np.sum((y == 1) & (y_pred == 0))

        # 正确决策比例 = (买入且正确 + 不买且正确) / 总决策数
        correct_decisions = true_positives + true_negatives
        total_decisions = true_positives + false_positives + true_negatives + false_negatives
        correct_decision_rate = correct_decisions / total_decisions if total_decisions > 0 else 0

        # 买入信号比例
        buy_signal_rate = buy_count / len(y)
    else:
        buy_accuracy = 0
        correct_decision_rate = 0
        buy_signal_rate = 0

    return {
        'total_samples': len(y),
        'buy_signals': int(buy_count),
        'buy_signal_rate': buy_signal_rate,
        'accuracy': accuracy,
        'buy_accuracy': buy_accuracy,  # 买入胜率
        'correct_decision_rate': correct_decision_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confidence_threshold': confidence_threshold
    }


def main():
    parser = argparse.ArgumentParser(description='评估特定板块模型的买入胜率')
    parser.add_argument('--sector', type=str, required=True,
                       help='板块类型: bank, tech, semiconductor, ai, consumer, index, exchange 等')
    parser.add_argument('--horizon', type=int, default=20, choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月')
    parser.add_argument('--confidence-threshold', type=float, default=0.55,
                       help='置信度阈值')

    args = parser.parse_args()

    # 获取板块股票代码
    stock_codes = get_stocks_by_sector(args.sector)

    if not stock_codes:
        print(f"❌ 未找到板块 '{args.sector}' 的股票")
        return

    print(f"🎯 评估板块: {args.sector}")
    print(f"📊 股票数量: {len(stock_codes)}")
    print(f"⏱️  预测周期: {args.horizon} 天")
    print("=" * 70)

    # 加载模型
    model = CatBoostModel()
    # 尝试加载板块特定的模型
    model_path = f'data/ml_trading_model_catboost_{args.sector}_{args.horizon}d.pkl'
    # 如果不存在，尝试加载通用模型
    if not os.path.exists(model_path):
        model_path = f'data/ml_trading_model_catboost_{args.horizon}d.pkl'

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先训练模型: python3 ml_services/train_sector_model.py --sector {args.sector}")
        return

    print(f"📂 加载模型: {model_path}")
    model.load_model(model_path)
    print(f"✅ 模型加载成功")
    print(f"   模型特征数量: {len(model.feature_columns)}")
    print("\n")

    # 评估买入胜率
    print("📈 评估买入胜率...")
    results = evaluate_buy_win_rate(model, stock_codes, args.horizon, args.confidence_threshold)

    if results is None:
        print("❌ 评估失败：数据不足")
        return

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    print(f"总样本数: {results['total_samples']}")
    print(f"买入信号数: {results['buy_signals']} ({results['buy_signal_rate']:.2%})")
    print(f"置信度阈值: {results['confidence_threshold']:.2f}")
    print()
    print(f"🎯 整体准确率: {results['accuracy']:.2%}")
    print(f"💰 买入胜率: {results['buy_accuracy']:.2%}")
    print(f"✅ 正确决策比例: {results['correct_decision_rate']:.2%}")
    print()
    print(f"📌 精确率: {results['precision']:.2%}")
    print(f"📌 召回率: {results['recall']:.2%}")
    print(f"📌 F1分数: {results['f1_score']:.4f}")
    print("=" * 70)

    # 保存结果
    output_file = f'output/sector_model_evaluation_{args.sector}_{args.horizon}d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 结果已保存到: {output_file}")


if __name__ == '__main__':
    main()