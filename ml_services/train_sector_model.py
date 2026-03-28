#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练特定板块的 CatBoost 模型

使用方法：
  python3 ml_services/train_sector_model.py --sector bank
  python3 ml_services/train_sector_model.py --sector consumer
  python3 ml_services/train_sector_model.py --sector index
  python3 ml_services/train_sector_model.py --sector exchange
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

from ml_services.ml_trading_model import CatBoostModel
from ml_services.logger_config import get_logger
from config import STOCK_SECTOR_MAPPING

# 获取日志记录器
logger = get_logger('train_sector_model')


def get_stocks_by_sector(sector_type):
    """根据板块类型获取股票代码列表"""
    stocks = []
    for code, info in STOCK_SECTOR_MAPPING.items():
        if info['type'] == sector_type:
            stocks.append(code)
    return stocks


def main():
    parser = argparse.ArgumentParser(description='训练特定板块的 CatBoost 模型')
    parser.add_argument('--sector', type=str, required=True,
                       help='板块类型: bank, tech, semiconductor, ai, consumer, index, exchange 等')
    parser.add_argument('--horizon', type=int, default=20, choices=[1, 5, 20],
                       help='预测周期: 1=次日, 5=一周, 20=一个月')

    args = parser.parse_args()

    # 获取板块股票代码
    stock_codes = get_stocks_by_sector(args.sector)

    if not stock_codes:
        print(f"❌ 未找到板块 '{args.sector}' 的股票")
        return

    print(f"🎯 训练板块: {args.sector}")
    print(f"📊 股票数量: {len(stock_codes)}")
    print(f"📋 股票代码: {', '.join(stock_codes)}")
    print(f"⏱️  预测周期: {args.horizon} 天")
    print("=" * 70)

    # 初始化 CatBoost 模型（方案B：Balanced + Fixed 0.55）
    model = CatBoostModel()

    # 训练模型
    print("\n开始训练...")
    start_time = datetime.now()

    try:
        feature_importance = model.train(
            stock_codes,
            horizon=args.horizon,
            use_feature_selection=False  # 默认使用全量特征
        )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print(f"\n✅ 训练完成！")
        print(f"⏱️  训练耗时: {training_time:.2f} 秒")
        print(f"📈 模型保存路径: data/ml_trading_model_catboost_{args.horizon}d_*.pkl")

        # 保存模型准确率到独立文件
        accuracy_file = f'data/model_accuracy_{args.sector}_{args.horizon}d.json'

        # 加载已有的准确率数据
        if os.path.exists('data/model_accuracy.json'):
            with open('data/model_accuracy.json', 'r', encoding='utf-8') as f:
                all_accuracy = json.load(f)
        else:
            all_accuracy = {}

        # 提取当前模型的准确率
        if hasattr(model, 'accuracy'):
            all_accuracy[f'{args.sector}_{args.horizon}d'] = {
                'accuracy': model.accuracy,
                'f1_score': getattr(model, 'f1_score', None),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_count': len(stock_codes)
            }

            # 保存准确率
            with open(accuracy_file, 'w', encoding='utf-8') as f:
                json.dump(all_accuracy, f, indent=2, ensure_ascii=False)

            print(f"📊 准确率已保存到: {accuracy_file}")
            print(f"   准确率: {model.accuracy:.2%}")

    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()