#!/bin/bash
# B′ 扩池 A/B 编排守护进程
#   1. 等 56 池 WF 结束 → 校验产物（A 闸）
#   2. 自动启动 300 池 WF
#   3. 每 3600 秒向 /tmp/opencode/ab_daemon.log 写一条状态
# 不写入仓库、不改动生产配置。
set -uo pipefail
cd /data/fortune || exit 1

LOG=/tmp/opencode/ab_daemon.log
WF56=/tmp/opencode/wf56.log
WF300=/tmp/opencode/wf300.log
SNAP=data/hk_broad/snapshot
ENVBASE=(HK_MARKET_SNAPSHOT_DIR=$SNAP US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot F1_FEATURES=0 F1_FORCE=0)

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
folds() { grep -c "✅ Fold" "$1" 2>/dev/null || echo 0; }

echo "[$(stamp)] daemon start: 等待 56 池 WF" >> "$LOG"

# ---- 阶段 1：等 56 池结束 ----
while pgrep -f "walk_forward_validation.py.*--horizon 20" > /dev/null; do
  sleep 60
done
echo "[$(stamp)] 56 池 WF 进程结束（折数 $(folds $WF56)）" >> "$LOG"

CSV56=$(find output -maxdepth 2 -name prediction_analysis.csv -newermt '-4 hours' | sort | tail -1)
echo "[$(stamp)] 56 池产物: ${CSV56:-<未找到>}" >> "$LOG"
if [ -n "${CSV56:-}" ]; then
  if python3 scripts/presentation_gate.py --input "$CSV56" --horizon 20 >> "$LOG" 2>&1; then
    echo "[$(stamp)] A 闸通过（56 池）" >> "$LOG"
  else
    echo "[$(stamp)] ⚠️ A 闸未通过（56 池），请人工判读" >> "$LOG"
  fi
fi

# ---- 阶段 2：启动 300 池 ----
if pgrep -f "walk_forward_validation.py.*--horizon 20" > /dev/null; then
  echo "[$(stamp)] 已有 WF 在跑，跳过 300 池启动" >> "$LOG"
else
  echo "[$(stamp)] 启动 300 池 WF" >> "$LOG"
  setsid env "${ENVBASE[@]}" HK_UNIVERSE_FILE=data/hk_broad/universe_2020-06-30.json \
    nohup scripts/run_walk_forward.sh --model-type catboost --horizon 20 \
    --use-feature-selection --start-date 2020-06-01 --end-date 2026-09-30 --no-commit \
    >> "$WF300" 2>&1 < /dev/null &
fi

# ---- 阶段 3：每小时状态 ----
while true; do
  sleep 3600
  {
    echo "===== [$(stamp)] 小时状态 ====="
    echo "56池折数: $(folds $WF56) / 38"
    echo "300池折数: $(folds $WF300) / 38"
    echo "运行中: $(pgrep -fc 'walk_forward_validation.py' || echo 0)"
    echo "最新产物: $(find output -maxdepth 2 -name fold_metrics_detail.json -newermt '-8 hours' 2>/dev/null | sort | tail -1)"
  } >> "$LOG"
done
