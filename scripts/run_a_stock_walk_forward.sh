#!/bin/bash
# A股 Walk-forward 确定性封装（对齐港股 scripts/run_walk_forward.sh）
# 固化进程级环境：线程数恒定（模型侧确定性前提）+ PYTHONHASHSEED
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"
export LGBM_N_JOBS="${LGBM_N_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONHASHSEED=42
echo "[run_a_stock_walk_forward] PYTHONHASHSEED=$PYTHONHASHSEED OMP=$OMP_NUM_THREADS LGBM_N_JOBS=$LGBM_N_JOBS CATBOOST_THREAD_COUNT=$CATBOOST_THREAD_COUNT"
exec python3 a_stock_walk_forward.py "$@"
