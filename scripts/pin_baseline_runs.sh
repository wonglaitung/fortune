#!/bin/bash
# 双冻结（三周期）+ 双跑一致性验证，产出可引用的最终基线（严格串行，避 OOM）
#
# 冻结项：
#   1) US_MARKET_SNAPSHOT_DIR  宏观特征输入（US/CN 收益率、VIX、利差）
#   2) GATE_SOURCE_CSV         门槛 PIT 分位基准（否则回测期间 output/ 新增目录即改变）
#   两个都冻结 → 模型层与门槛层均 bit 级可复现（lessons 三.29）
#
# 用法：bash scripts/pin_baseline_runs.sh
cd /data/fortune
export US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot
# 2026-10-10 修：勿写死路径（旧值 20261001_000048 已过期）——动态取 git 入库生产基线，
# 与 comprehensive_analysis.py 取源三道闸之②同口径（排 _a_stock_/_lightgbm_/VOIDED）
export GATE_SOURCE_CSV=$(git ls-files -- 'output/*_catboost_20d/prediction_analysis.csv' | python3 -c "
import sys, os
from scripts.commit_backtest_result import VOIDED_DIRS
c = [l.strip() for l in sys.stdin
     if l.strip() and '_a_stock_' not in l and '_lightgbm_' not in l
     and os.path.basename(os.path.dirname(l.strip())) not in VOIDED_DIRS]
print(max(c) if c else '')
")
[ -n "$GATE_SOURCE_CSV" ] && [ -f "$GATE_SOURCE_CSV" ] || unset GATE_SOURCE_CSV
L=output/train_retrain
mkdir -p "$L"

echo "[$(date '+%F %T')] 双冻结基线队列启动 快照=${US_MARKET_SNAPSHOT_DIR} 门槛基准=${GATE_SOURCE_CSV:-自动}" >> $L/pin_baseline.log

run_wf () {   # $1=horizon $2=tag
  local h=$1 tag=$2 extra=""
  [ "$h" = "20" ] && extra="--learner lightgbm"
  echo "[$(date '+%F %T')] ${tag} abs${h}d 开始" >> $L/pin_baseline.log
  bash scripts/run_walk_forward.sh --model-type catboost --horizon "$h" \
       --use-feature-selection $extra --label-mode absolute \
       --start-date 2019-06-01 --end-date 2026-07-31 \
       > "output/${tag}${h}d_abs.log" 2>&1
  echo "[$(date '+%F %T')] ${tag} abs${h}d 完成 rc=$?" >> $L/pin_baseline.log
}

# 阶段1：三周期完整基线（run1）
for h in 20 5 1; do run_wf $h base_r1; done

# 阶段2：20d 再跑一次，验证双冻结下是否与 run1 逐字节一致
run_wf 20 base_r2

echo "[$(date '+%F %T')] 全部完成" >> $L/pin_baseline.log
echo PINBASE_DONE
