#!/usr/bin/env bash
# D26 双跑验收（2026-10-13）：社区集合显式化后重跑：新口径（T 收盘特征 + T+1 成交标签）跑两轮，比 md5。
# 三冻结缺一不可（AGENTS 常用命令）；GATE_SOURCE_CSV 指向**旧基线**入库 CSV，
# 避免「跑完自动入库 → 第二轮的门槛源变成第一轮产物」的污染（lessons 三.30）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 门槛源固定到**跑前已存在**的一份 CSV；第二轮跑完不会把第一轮产物当门槛源（lessons 三.30）。
# 注：三.61 已证真实分布上分位守卫恒触发 ⇒ 门槛实际生效值仍是绝对 0.70/0.65，
#     故此处固定门槛源对本次数字无影响，但**保持流程同源**以便将来分位不再退化时可复现。
GATE_SOURCE_CSV="${GATE_SOURCE_CSV:-output/20261013_141729_catboost_20d/prediction_analysis.csv}"
export US_MARKET_SNAPSHOT_DIR="${US_MARKET_SNAPSHOT_DIR:-data/us_market_snapshot}"
export HK_MARKET_SNAPSHOT_DIR="${HK_MARKET_SNAPSHOT_DIR:-data/hk_market_snapshot}"
export GATE_SOURCE_CSV

echo "[d26] GATE_SOURCE_CSV=$GATE_SOURCE_CSV"
ls -la "$GATE_SOURCE_CSV" 2>/dev/null || echo "[d26] ⚠️ 门槛源不存在，用回退绝对阈值（须在报告里注明）"

for RUN in 1 2; do
  echo "[d26] ===== 第 ${RUN}/2 轮 $(date '+%F %T') ====="
  if [ "$RUN" = "2" ]; then
    # 冷/热差异：第二轮保留缓存（同配置双跑，验证 bit 级复现）
    echo "[d26] 第2轮保留特征缓存（热跑）"
  fi
  scripts/run_walk_forward.sh \
      --model-type lgbm --horizon 20 \
      --start-date 2020-06-01 --end-date 2026-07-31 \
      --use-feature-selection --no-commit 2>&1 | tee "logs/d26_run${RUN}.log"
done

echo "[d26] ===== md5 比对 $(date '+%F %T') ====="
python3 - << 'PY'
import glob, hashlib, os, sys
dirs = sorted(glob.glob('output/*_catboost_20d'), key=os.path.getmtime)[-2:]
if len(dirs) < 2:
    print('[d26] ❌ 产物不足 2 个，无法比对'); sys.exit(1)
h = []
for d in dirs:
    f = os.path.join(d, 'prediction_analysis.csv')
    if not os.path.exists(f):
        print(f'[d26] ❌ 缺 {f}'); sys.exit(1)
    m = hashlib.md5(open(f,'rb').read()).hexdigest()
    rows = sum(1 for _ in open(f, encoding='utf-8')) - 1
    print(f'[d26] {d}\n       md5={m}  行数={rows}')
    h.append(m)
print('[d26] ✅ md5 一致 —— 双跑可复现' if h[0] == h[1] else '[d26] ❌ md5 不一致 —— 禁止引用')
sys.exit(0 if h[0] == h[1] else 2)
PY
