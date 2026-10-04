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

# 数据末日冻结（2026-10-04，消除「末日变→缓存键变→重算→指标漂移」）
# 配套 ml_services/ml_trading_model.py：_get_feature_cache_key 去末日 +
# prepare_data 按此变量截断。**两者必须同时生效** —— 只做其一会造成
# 数据混用（键不区分内容）或假冻结（键固定但内容仍变）。
# 不设置时行为与从前完全一致（现状）。
export WALKFORWARD_DATA_END="${WALKFORWARD_DATA_END:-}"
export OMP_NUM_THREADS="${LGBM_N_JOBS:-8}"
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS="${LGBM_N_JOBS:-8}"
export OPENBLAS_NUM_THREADS="${LGBM_N_JOBS:-8}"
export LGBM_N_JOBS="${LGBM_N_JOBS:-8}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-8}"

echo "[run_walk_forward] PYTHONHASHSEED=$PYTHONHASHSEED OMP/MKL/OPENBLAS=$OMP_NUM_THREADS LGBM_N_JOBS=$LGBM_N_JOBS CATBOOST_THREAD_COUNT=$CATBOOST_THREAD_COUNT"
exec python3 ml_services/walk_forward_validation.py "$@"
