#!/bin/bash
# 自动评估 watcher：检测 walk-forward 运行完成后自动跑评估链，结果追加到汇总日志。
# 由 jwatcher 定时调用（也可手动单次执行）。
cd /data/fortune

# 防重入锁：避免上一轮评估未跑完又被定时触发
exec 9>/tmp/.eval_watcher.lock
flock -n 9 || { echo "[$(date '+%T')] watcher busy, skip"; exit 0; }

STATE=output/.eval_state
SUMMARY=output/eval_summary.log
mkdir -p output
touch "$STATE" "$SUMMARY"

run_eval () {           # $1=tag $2=horizon $3=logfile $4=is_rel
  local tag=$1 hz=$2 logf=$3 is_rel=$4
  local key="$tag"
  # 已评估过则跳过
  grep -q "^$key " "$STATE" && return 0
  # 完成判定：验证完成 + 产出 prediction_analysis.csv
  grep -q "✅ 验证完成" "$logf" 2>/dev/null || return 0
  # 取本次运行实际产出的 CSV。日志里除"预测分析已保存"外还含 GATE 快照来源引用等旧路径，
  # 故收集所有匹配后按 mtime 取最新存在者（勿用 tail -1，会抓到过期 CSV）
  local csv
  csv=$(for c in $(grep -oE "output/[0-9]{8}_[0-9]{6}_catboost_[0-9]+d/prediction_analysis.csv" "$logf" | sort -u); do
          [ -f "$c" ] && stat -c '%Y %n' "$c"
        done 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [ -z "$csv" ] || [ ! -f "$csv" ] && { echo "$key NO_CSV" >> "$SUMMARY"; return 0; }

  {
    echo "##### $key  $(date '+%F %T')  csv=$csv"
    echo "--- backtest_eval ---"
    timeout 900 python3 ml_services/backtest_eval.py --input "$csv" --horizon "$hz" 2>&1 | tail -18
    echo "--- monthly_guardrail ---"
    timeout 900 python3 ml_services/monthly_guardrail.py --horizon "$hz" --pred "$csv" 2>&1 | tail -22
    if [ "$hz" != "1" ]; then
      echo "--- portfolio_backtest ---"
      timeout 900 python3 ml_services/portfolio_backtest.py --horizon "$hz" --topk 10 --pred "$csv" 2>&1 | tail -20
    fi
    if [ "$is_rel" = "1" ]; then
      echo "--- rel_alpha_check ---"
      timeout 600 python3 scripts/rel_alpha_check.py --input "$csv" --horizon "$hz" 2>&1 | tail -16
    fi
    echo ""
  } >> "$SUMMARY" 2>&1
  echo "$key done $(date '+%T')" >> "$STATE"
}

# 每次调用：先输出进度快照，再评估任何新完成的运行
{
  echo "=== $(date '+%F %T') ==="
  for f in wf20d_rel wf5d_rel wf1d_rel wf20d_abs wf5d_abs wf1d_abs; do
    p=$(grep -oiE "fold [0-9]+/[0-9]+" output/$f.log 2>/dev/null | tail -1)
    err=$(grep -cE "Traceback|KeyError" output/$f.log 2>/dev/null)
    v=$(grep -c "✅ 验证完成" output/$f.log 2>/dev/null)
    dg=$(grep -cE "退化/样本不足跳过" output/$f.log 2>/dev/null)
    echo "[$f] ${p:-starting} err=$err degenSkip=$dg verified=$v"
  done
} >> output/wf_all_monitor.log

run_eval rel20d 20 output/wf20d_rel.log 1
run_eval rel5d   5 output/wf5d_rel.log  1
run_eval rel1d   1 output/wf1d_rel.log  1
run_eval abs20d 20 output/wf20d_abs.log 0
run_eval abs5d   5 output/wf5d_abs.log  0
run_eval abs1d   1 output/wf1d_abs.log  0
