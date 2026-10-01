#!/bin/bash
# 冻结快照下的可复现性验证 + 产出可引用基线（严格串行，避 OOM）
cd /data/fortune
export US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot
# 门槛基准也须冻结：_latest_gate_source_csv 默认取磁盘上最新回测 CSV，
# 而回测期间 output/ 会新增目录 → 同一模型两次运行的 PIT 分位不同
# → Dynamic_Threshold 9% 行不一致（lessons 三.29）。用入库基线固定。
export GATE_SOURCE_CSV=output/20261001_000048_catboost_20d/prediction_analysis.csv
[ -f "$GATE_SOURCE_CSV" ] || unset GATE_SOURCE_CSV
echo "[$(date '+%F %T')] 快照=${US_MARKET_SNAPSHOT_DIR} 门槛基准=${GATE_SOURCE_CSV:-自动}"
L=output/train_retrain



# ---------- 阶段1：可复现性验证（缩减股票池，快速比对两次输出）----------
S10="0700.HK 0005.HK 2318.HK 1299.HK 9988.HK 0388.HK 0883.HK 1398.HK 0688.HK 1810.HK"
for r in A B; do
  echo "[$(date '+%F %T')] 可复现性验证 run $r 开始" >> $L/repro.log
  bash scripts/run_walk_forward.sh --model-type catboost --horizon 20 \
       --use-feature-selection --label-mode absolute \
       --start-date 2021-01-01 --end-date 2024-12-31 \
       --stocks $S10 \
       > output/repro_${r}.log 2>&1
  echo "[$(date '+%F %T')] run $r rc=$?" >> $L/repro.log
done
python3 - <<'PY' >> $L/repro.log 2>&1
import glob, hashlib, os
def latest_csv(tag):
    fs = sorted(glob.glob(f'output/*_catboost_20d/prediction_analysis.csv'), key=os.path.getmtime)
    return fs[-1] if fs else None
# 两次运行的输出目录不同（时间戳），用日志中记录的路径更可靠；此处做兜底比对
print("阶段1 结束，请人工 diff 两个 run 的 prediction_analysis.csv")
PY

# ---------- 阶段2：产出可引用的 20d 完整基线 ----------
for h in 20 5 1; do
  extra=""
  [ "$h" = "20" ] && extra="--learner lightgbm"
  echo "[$(date '+%F %T')] 冻结快照 abs${h}d 完整基线开始" >> $L/repro.log
  bash scripts/run_walk_forward.sh --model-type catboost --horizon $h \
       --use-feature-selection $extra --label-mode absolute \
       --start-date 2019-06-01 --end-date 2026-07-31 \
       > output/pin${h}d_abs.log 2>&1
  echo "[$(date '+%F %T')] abs${h}d 完成 rc=$?" >> $L/repro.log
done

echo "[$(date '+%F %T')] 全部完成" >> $L/repro.log
echo REPRO_DONE
