#!/usr/bin/env bash
# 项目 15 分钟进度心跳（cron 驱动）。
# 只读、只追加一行，不做任何写操作或判断 —— 目的是让「哪一刻停在哪」事后可回溯。
set -uo pipefail
cd /data/fortune

LOG="logs/progress_watch.log"

# 未提交只算代码/文档/脚本；缓存与大体积产物不进这个数，否则数字没有意义
DIRTY=$(git status --porcelain 2>/dev/null \
         | grep -vE '^\?\? (data/|output/|logs/)' \
         | grep -cE '^ ?M|^\?\?' || true)
DIRTY=${DIRTY:-0}

LAST=$(git log -1 --format='%h|%s' 2>/dev/null || echo none)
GATE=$(ls -t output/monthly_guardrail_20d_*.md 2>/dev/null | head -1 | xargs -r basename)
BASE=$(ls -dt output/*_catboost_20d 2>/dev/null | head -1 | xargs -r basename)
TMUX=$(tmux ls 2>/dev/null | cut -d' ' -f1 | paste -sd, - )
PROC=$(pgrep -f "walk_forward_validation.py" 2>/dev/null | wc -l)

echo "$(date '+%F %T') | 代码待提交=$DIRTY | 基线=${BASE:-无} | 护栏=${GATE:-无} | wf进程=$PROC | 会话=${TMUX:-无} | HEAD=$LAST" >> "$LOG"