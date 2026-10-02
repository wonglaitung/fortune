#!/usr/bin/env python3
"""生成 A 股 PIT（时点还原）网络特征

背景：A股的 prepare_data / predict 两条路径都把静态网络特征
（network_features_for_ml.json，每股一个标量）广播到全部历史行，
构成 AGENTS.md 明令禁止的「静态快照穿越」。港股已通过
network_features_pit.json + merge_pit_features 修复，A股缺失同源机制。

本脚本复用 stock_network_analysis.export_pit_network_features（港股同一实现），
输出 data/a_stock_network_features/network_features_pit.json。

窗口/步长与港股 CLI 默认一致（window_days=120, step_days=20），
避免两个市场口径不一致导致特征不可比。
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_stock_config import A_STOCK_TRAINING_LIST, A_STOCK_CACHE_DIR
from data_services.a_stock_data import get_a_stock_data
from ml_services.stock_network_analysis import (
    export_pit_network_features,
    save_pit_network_features,
)

A_NET_DIR = 'data/a_stock_network_features'


def build_stock_data(codes, period_days):
    """取数并构造 {'Return': Series}，DateTimeIndex"""
    stock_data = {}
    for i, code in enumerate(codes, 1):
        try:
            df = get_a_stock_data(code, period_days=period_days)
        except Exception as e:
            print(f'  ⚠️ {code} 取数失败: {e}')
            continue
        if df is None or df.empty or 'Close' not in df.columns:
            print(f'  ⚠️ {code} 数据为空，跳过')
            continue
        df = df.sort_index()
        stock_data[code] = pd.DataFrame({'Return': df['Close'].pct_change()})
        print(f'  [{i}/{len(codes)}] {code}: {len(df)} 行 -> {df.index[0].date()} ~ {df.index[-1].date()}')
    return stock_data


def main():
    parser = argparse.ArgumentParser(description='生成 A股 PIT 网络特征')
    parser.add_argument('--period-days', type=int, default=1460,
                        help='取数窗口（腾讯A股接口约 640 行上限，见 lessons 三.26）')
    parser.add_argument('--window-days', type=int, default=120,
                        help='PIT 滚动窗口（与港股 CLI 默认一致）')
    parser.add_argument('--step-days', type=int, default=20,
                        help='PIT 步长（与港股 CLI 默认一致）')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='阈值网络相关系数阈值')
    parser.add_argument('--limit', type=int, default=0,
                        help='仅取前 N 只（冒烟测试用，0=全部）')
    args = parser.parse_args()

    codes = list(A_STOCK_TRAINING_LIST)
    if args.limit:
        codes = codes[:args.limit]

    print(f'📊 A股 PIT 网络特征生成: {len(codes)} 只, 窗口={args.window_days}, 步长={args.step_days}')
    stock_data = build_stock_data(codes, args.period_days)
    if not stock_data:
        print('❌ 无可用数据，退出')
        return 1

    pit_features = export_pit_network_features(
        stock_data, list(stock_data.keys()),
        window_days=args.window_days, step_days=args.step_days,
        threshold=args.threshold)
    if not pit_features:
        print('❌ PIT 特征为空（数据不足），退出')
        return 1

    save_pit_network_features(pit_features, A_NET_DIR)

    sample = sorted(pit_features[next(iter(pit_features))])
    print(f'✅ 完成: {len(pit_features)} 只 × {len(sample)} 个时点')
    print(f'   覆盖 {sample[0]} ~ {sample[-1]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
