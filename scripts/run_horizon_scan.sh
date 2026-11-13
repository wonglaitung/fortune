#!/usr/bin/env bash
# A 阶段预测周期扫描（2026-10-14）：只改 horizon，其余固定。
# 三冻结 + 确定性 env。预注册见 docs/HORIZON_SCAN_PREREG.md。
#
# 设计（2026-10-14 修）：每跑 ~2.8h（每折重做统计特征选择），8 跑串行不可行。
# 故默认 **4 周期并行、单跑**，每跑 2 线程（4×2=8 核）。达标候选再单独补第二跑做
# md5（见 run_horizon_doublerun.sh）。单跑不取「最好一轮」——只跑一轮无所谓挑选。
#
# 用法：HFIX=... bash scripts/run_horizon_scan.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export US_MARKET_SNAPSHOT_DIR="${US_MARKET_SNAPSHOT_DIR:-data/us_market_snapshot}"
export HK_MARKET_SNAPSHOT_DIR="${HK_MARKET_SNAPSHOT_DIR:-data/hk_market_snapshot}"
export GATE_SOURCE_CSV="${GATE_SOURCE_CSV:-output/20261013_192444_catboost_20d/prediction_analysis.csv}"

HORIZONS="${HORIZONS:-3 10 20 30}"
THREADS="${THREADS:-2}"
L=output/horizon_scan
mkdir -p "$L" logs
echo "[hscan] $(date '+%F %T') 并行启动 horizon=[$HORIZONS] threads=$THREADS GATE=$GATE_SOURCE_CSV" | tee "$L/run.log"

pids=()
for h in $HORIZONS; do
  (
    export LGBM_N_JOBS="$THREADS" CATBOOST_THREAD_COUNT="$THREADS"
    scripts/run_walk_forward.sh \
        --model-type catboost --horizon "$h" \
        --start-date 2020-06-01 --end-date 2026-07-31 \
        --use-feature-selection --no-commit > "logs/hscan_${h}d_run1.log" 2>&1
    echo "[hscan] horizon=${h}d run1 done rc=$? $(date '+%F %T')" >> "$L/run.log"
  ) &
  pids+=($!)
  echo "[hscan] horizon=${h}d 启动 pid=$!" | tee -a "$L/run.log"
done

for p in "${pids[@]}"; do wait "$p" || true; done
echo "[hscan] 全部完成 $(date '+%F %T')" | tee -a "$L/run.log"
ls -d output/*_catboost_{3,10,20,30}d 2>/dev/null | tee -a "$L/run.log"
echo "[hscan] HSCAN_DONE" | tee -a "$L/run.log"
