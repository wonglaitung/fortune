#!/usr/bin/env bash
# D26 双跑收尾（2026-10-13）：等双跑结束 → 验 md5 → 跑 A 闸（新/旧基线各一次）→ 月度护栏。
#
# 等待条件用 `run_d26_doublerun.sh` 本身而不是 walk_forward 的 python 进程：
# 两轮之间存在 python 已退、脚本未退的间隙，用 pgrep 认 python 会提前误判结束。
#
# md5 不一致 ⇒ 只报告、不跑闸（产物不可复现时任何指标都无意义，lessons 三.30）。
# 本脚本不做任何升级/部署动作：DEPLOYMENT.md 已写「⓪ 不通过 ⇒ 升级动作一律不得执行」。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OLD_BASELINE="output/20261013_141729_catboost_20d"   # D25 基线（新口径、显式化之前）
STAMP() { date '+%F %T'; }

echo "[post] 开始守候 D26 双跑 $(STAMP)"

LAST_FOLD=""
# 等待条件认「脚本自己写下的完成信号」，不认进程：
# 起因是 tmux server 的 argv 继承自首次 new-session 调用，带着 run_d26_doublerun.sh 字样，
# 于是 `pgrep -f` 在双跑结束后仍恒命中 server ⇒ 死等（lessons 三.59 同族：靠名字匹配判活会骗人）。
# 完成信号 = 日志里出现 md5 比对段（双跑脚本最后一步必然打印），再加超时兜底。
MAX_WAIT=$((6 * 60 * 60))
WAITED=0
while :; do
  grep -q "===== md5 比对" logs/d26_doublerun.log 2>/dev/null && break
  [ "$WAITED" -ge "$MAX_WAIT" ] && { echo "[post] ⚠️ 等待超时 ${MAX_WAIT}s，日志无 md5 比对段"; break; }
  FOLD=$(grep -E "📊 Fold" logs/d26_doublerun.log 2>/dev/null | tail -1)
  ROUND=$(grep -cE "===== 第 [12]/2 轮" logs/d26_doublerun.log 2>/dev/null)
  if [ "$FOLD" != "$LAST_FOLD" ]; then
    echo "[watch $(STAMP)] ${ROUND:-?} ${FOLD:-初始化}"
    LAST_FOLD="$FOLD"
  fi
  sleep 900
  WAITED=$((WAITED + 900))
done

echo; echo "[post] 双跑进程已退出 $(STAMP)"
echo; echo "===== ① md5 比对（双跑日志末尾）====="
MD5_BLOCK=$(grep -E "^\[d26\] .*md5|^\[d26\] ✅|^\[d26\] ❌" logs/d26_doublerun.log | tail -5)
echo "${MD5_BLOCK:-'(未取到比对输出 —— 脚本可能被中断)'}"

MD5_OK=0
echo "$MD5_BLOCK" | grep -q "✅ md5 一致" && MD5_OK=1

NEW=$(ls -dt output/*_catboost_20d 2>/dev/null | head -1)
echo; echo "新基线(D26): $NEW"
echo "旧基线(D25): $OLD_BASELINE"
echo

if [ "$MD5_OK" != "1" ]; then
  echo "🚫 md5 未确认一致 —— 按纪律不跑 A 闸/护栏，人工核对后再执行："
  echo "   python3 scripts/presentation_gate.py --input $NEW/prediction_analysis.csv --horizon 20"
  echo "   python3 ml_services/monthly_guardrail.py --horizon 20 --pred $NEW/prediction_analysis.csv"
  exit 2
fi

echo "===== ② A 闸：新基线 ====="
python3 scripts/presentation_gate.py --input "$NEW/prediction_analysis.csv" --horizon 20 2>&1 | tail -40
echo "[post] A闸(新) exit=$?"

echo; echo "===== ③ A 闸：旧基线（D25，作对照）====="
if [ -f "$OLD_BASELINE/prediction_analysis.csv" ]; then
  python3 scripts/presentation_gate.py --input "$OLD_BASELINE/prediction_analysis.csv" --horizon 20 2>&1 | tail -40
  echo "[post] A闸(旧) exit=$?"
else
  echo "⚠️ 缺 $OLD_BASELINE/prediction_analysis.csv，跳过"
fi

echo; echo "===== ④ 月度护栏（净IR/PBO/DSR/lift）====="
python3 ml_services/monthly_guardrail.py --horizon 20 --pred "$NEW/prediction_analysis.csv" 2>&1 | tail -30
echo "[post] 护栏 exit=$?"

echo; echo "[post] 全部完成 $(STAMP)"