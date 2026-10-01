#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结宏观数据快照（恢复回测可复现性，lessons 三.29）

背景：us_market_data 缓存"仅当天有效"，而宏观特征（US/CN 收益率、VIX、利差）是
walk-forward 特征矩阵的唯一来源 → 同代码跨日重跑必换输入快照（实测 abs20d
净IR 0.34/0.42、PBO 0.41/0.64）。本脚本把当前宏观数据复制为 PIT 快照，之后
用 US_MARKET_SNAPSHOT_DIR 指向它即可获得可复现的回测。

用法：
    python3 scripts/pin_macro_snapshot.py                      # 建立/刷新快照
    US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot \
        bash scripts/run_walk_forward.sh ...                    # 用快照跑（可复现）
"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'data', 'us_market_cache')
DST = os.path.join(BASE, 'data', 'us_market_snapshot')

def main():
    if not os.path.isdir(SRC):
        print(f"ERROR: 源缓存目录不存在 {SRC}")
        return 1
    os.makedirs(DST, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(SRC)):
        if not f.endswith('.pkl'):
            continue
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
        n += 1
    print(f"✅ 已冻结 {n} 个宏观缓存文件 -> {os.path.relpath(DST, BASE)}")
    print("   复现方式: 前置 US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot")
    print("   快照模式下 _load_cache 忽略日期、永不刷新，_save_cache 不写（保护 PIT）")
    return 0

if __name__ == '__main__':
    sys.exit(main())
