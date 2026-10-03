#!/bin/bash
# A股 Walk-forward 确定性封装（对齐港股 scripts/run_walk_forward.sh）
# 固化进程级环境：线程数恒定（模型侧确定性前提）+ PYTHONHASHSEED
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"
export LGBM_N_JOBS="${LGBM_N_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONHASHSEED=42

# A股外部数据冻结（2026-10-03）：对齐港股 US_MARKET_SNAPSHOT_DIR 机制。
# 未冻结时 A股三样本双跑 md5 全不相同（净IR 2.14~2.72、PBO 0.30~0.61 跨门槛、
# 判定🟢/🟡翻转），根因是主力资金/融资融券/商品/汇率等实时取数源
# 「TTL过期 + 接口不稳 + 失败静默填默认值」三者叠加。
export A_STOCK_SNAPSHOT_DIR="${A_STOCK_SNAPSHOT_DIR:-data/a_stock_snapshot}"
echo "[run_a_stock_walk_forward] A_STOCK_SNAPSHOT_DIR=$A_STOCK_SNAPSHOT_DIR"
echo "[run_a_stock_walk_forward] PYTHONHASHSEED=$PYTHONHASHSEED OMP=$OMP_NUM_THREADS LGBM_N_JOBS=$LGBM_N_JOBS CATBOOST_THREAD_COUNT=$CATBOOST_THREAD_COUNT"
exec python3 a_stock_walk_forward.py "$@"
