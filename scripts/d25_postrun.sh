#!/usr/bin/env bash
# D25 双跑收尾（2026-10-13）：等双跑结束 → 验 md5 → 重算 A 闸 H3 参考基准。
#
# 为什么必须重算：`presentation_gate.py` 的 H3「符号退化」用
# Relative_Return = Actual_Return − HSI 未来收益。D25 把标签窗口从 [T, T+h]
# 改成 [T+1, T+1+h]，两段都变了 ⇒ 旧基准（合法值 +0.4209，旧窗口、1d）失效。
# 拿旧基准判新数据 ⇒ A 闸结论不可信（B 闸第 2 项：文档↔代码一致）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[post] 等待双跑结束 $(date '+%F %T')"
while pgrep -f "walk_forward_validation.py --model-type lgbm --horizon 20" > /dev/null; do
  sleep 60
done
echo "[post] 双跑进程已退出 $(date '+%F %T')"

echo; echo "===== ① 双跑 md5 比对（来自 run_d25_doublerun.log 末尾）====="
tail -14 logs/d25_doublerun.log | grep -E "^\[d25\]|md5" || echo "(未取到比对输出)"

NEW=$(ls -dt output/*_catboost_20d 2>/dev/null | head -1)
OLD="output/20261012_234236_catboost_20d"
echo; echo "新基线: $NEW"; echo "旧基线(旧口径参照): $OLD"

for D in "$NEW" "$OLD"; do
  F="$D/prediction_analysis.csv"
  [ -f "$F" ] || { echo "⚠️ 缺 $F，跳过"; continue; }
  echo; echo "===== A 闸：$(basename "$D") ====="
  python3 scripts/presentation_gate.py --input "$F" --horizon 20 2>&1 \
    | grep -E "H3|符号|Relative|GATE|结论|PASS|FAIL|阻断" | head -20
  echo "--- exit=$? ---"
done

echo
echo "===== ② 护栏与组合层（新口径）====="
if [ -f "$NEW/prediction_analysis.csv" ]; then
  python3 ml_services/monthly_guardrail.py --horizon 20 2>&1 | tail -25
fi

echo "[post] 完成 $(date '+%F %T')"
