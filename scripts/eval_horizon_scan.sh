#!/usr/bin/env bash
# A 阶段评估：对每个 horizon 的最新 WF 产物跑四道评估，产物存 output/horizon_scan/。
# 用法：bash scripts/eval_horizon_scan.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=output/horizon_scan
mkdir -p "$OUT"

for h in 3 10 20 30; do
  csv=$(ls -t output/*_catboost_${h}d/prediction_analysis.csv 2>/dev/null | head -1)
  if [ -z "$csv" ]; then echo "[eval] ❌ ${h}d 无产物"; continue; fi
  echo "=============== horizon=${h}d  csv=$csv ==============="
  echo "--- A 闸 presentation_gate ---"
  python3 scripts/presentation_gate.py --input "$csv" --horizon "$h" > "$OUT/gate_${h}d.txt" 2>&1
  echo "    rc=$? (0=通过)"
  echo "--- backtest_eval ---"
  python3 ml_services/backtest_eval.py --input "$csv" --horizon "$h" \
      --output "$OUT/backtest_eval_${h}d.md" > /dev/null 2>&1 && echo "    ok"
  echo "--- monthly_guardrail ---"
  python3 ml_services/monthly_guardrail.py --horizon "$h" --pred "$csv" \
      --output "$OUT/guardrail_${h}d.md" > /dev/null 2>&1 && echo "    ok"
  echo "--- portfolio_backtest ---"
  python3 - "$h" "$csv" "$OUT/portfolio_${h}d.md" <<'PY'
import sys
from ml_services.portfolio_backtest import run
run(int(sys.argv[1]), sys.argv[2], 10, sys.argv[3], market='hk')
PY
  echo
done
