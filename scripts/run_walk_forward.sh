#!/bin/bash
# 确定性 Walk-forward 运行器（2026-09-28 去噪声 A方案）
#
# 固化对复现性有影响的进程级环境，参数原样透传给 walk_forward_validation.py：
#   PYTHONHASHSEED=0        哈希种子（set/dict 迭代序跨进程一致）
#   OMP/MKL/OPENBLAS_THREADS BLAS/OpenMP 线程数恒定（浮点归约顺序稳定）
#   OMP_DYNAMIC=FALSE       禁用 OpenMP 动态线程团队（负载自适应会破坏确定性）
#   LGBM_N_JOBS/CATBOOST_THREAD_COUNT  学习器线程数恒定（模型侧确定性前提）
#
# 配合代码内 LGBM deterministic=True / CatBoost thread_count 固定，
# 目标：同配置双跑 prediction_analysis.csv DataFrame.equals() == True。
#
# 用法（与直接 python3 调用完全一致）：
#   scripts/run_walk_forward.sh --model-type catboost --horizon 20 --use-feature-selection \
#       --learner lightgbm --start-date 2020-06-01 --end-date 2026-07-31 --no-commit
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# 输入冻结：三者缺一不可，否则双跑输入不同 → md5 必不一致
#   HK_MARKET_SNAPSHOT_DIR  港股原始行情（2026-10-07 实证：数据源日内更新 →
#                           stock_df 末日漂移 → 特征缓存键全变 → 全量重算）
#   US_MARKET_SNAPSHOT_DIR  宏观特征（lessons 三.29）
#   GATE_SOURCE_CSV         分位门槛基准（否则 output/ 新增目录即改变基准）
# 设了 HK_MARKET_SNAPSHOT_DIR 就先校验快照完整（缺一股即非零退出，防止运行中
# 静默跳股产出残缺 CSV，lessons 三.20）。不设置时行为与从前完全一致（现状）。
if [ -n "${HK_MARKET_SNAPSHOT_DIR:-}" ]; then
    python3 scripts/pin_hk_snapshot.py --check --snap-dir "$HK_MARKET_SNAPSHOT_DIR"
fi

export OMP_NUM_THREADS="${LGBM_N_JOBS:-8}"

export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS="${LGBM_N_JOBS:-8}"
export OPENBLAS_NUM_THREADS="${LGBM_N_JOBS:-8}"
export LGBM_N_JOBS="${LGBM_N_JOBS:-8}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"

echo "[run_walk_forward] PYTHONHASHSEED=$PYTHONHASHSEED OMP/MKL/OPENBLAS=$OMP_NUM_THREADS LGBM_N_JOBS=$LGBM_N_JOBS CATBOOST_THREAD_COUNT=$CATBOOST_THREAD_COUNT"
exec python3 ml_services/walk_forward_validation.py "$@"
