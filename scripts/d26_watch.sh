#!/usr/bin/env bash
# D26 双跑状态快照（2026-10-13）：由 cron 每 15 分钟调一次，只追加一行状态，不做任何判断。
# 真正的收尾（A闸/护栏）在 d26_postrun.sh 里；此处只是心跳留痕，便于事后回溯「哪一刻停了」。
set -uo pipefail
cd /data/fortune

LOG="logs/d26_watch.log"

if pgrep -f "run_d26_doublerun.sh" > /dev/null; then
  STATE="运行中"
else
  STATE="已退出"
fi

ROUND=$(grep -cE "===== 第 [12]/2 轮" logs/d26_doublerun.log 2>/dev/null || echo 0)
FOLD=$(grep -E "📊 Fold" logs/d26_doublerun.log 2>/dev/null | tail -1 | tr -d ' ')
CACHE=$(ls data/feature_cache/*.pkl 2>/dev/null | wc -l)
LAST=$(ls -dt output/*_catboost_20d 2>/dev/null | head -1)

echo "$(date '+%F %T') | $STATE | 轮次=$ROUND | ${FOLD:-无} | 缓存=$CACHE | 最新产物=${LAST:-无}" >> "$LOG"