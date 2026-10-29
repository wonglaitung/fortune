#!/bin/bash
# 港股三周期模型月度重训（生产日常只加载，不重训）
#
# 用法：
#   bash scripts/monthly_retrain.sh
#
# 设计：
# - 串行训练 5/10/20（310 池 / rank / Top500+FORCE），每周期单独日志
# - 每个模型保存后立即校验（类型/horizon/label_mode/特征数）
# - 任一周期失败即非零退出；save_model 为原子写，失败不会破坏旧模型
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export LABEL_MODE=rank
STAMP="$(date '+%Y%m%d')"
mkdir -p output/_logs

# 防并发：同一时间只允许一个重训任务
LOCK="output/_logs/.monthly_retrain.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "❌ 已有月度重训在运行（$LOCK）"
    exit 2
fi

echo "[$(date '+%F %T')] 月度重训开始"
for horizon in 5 10 20; do
    LOG="output/_logs/monthly_retrain_${horizon}d_${STAMP}.log"
    echo "[$(date '+%F %T')] 训练 ${horizon}d → $LOG"
    python3 -u ml_services/ml_trading_model.py \
        --mode train --horizon "$horizon" --model-type lgbm \
        --use-feature-selection > "$LOG" 2>&1

    python3 - "$horizon" <<'PY'
import pickle
import sys

h = int(sys.argv[1])
path = f'data/ml_trading_model_lgbm_{h}d.pkl'
with open(path, 'rb') as f:
    d = pickle.load(f)
m = d['model']
assert type(m).__name__ == 'LGBMRegressor', type(m).__name__
assert d.get('horizon') == h, (d.get('horizon'), h)
assert d.get('label_mode') == 'rank', d.get('label_mode')
assert len(d.get('feature_columns') or []) > 100, len(d.get('feature_columns') or [])
from ml_services.ml_trading_model import LGBM_BAG_SEEDS_BY_HORIZON
exp = len([x for x in LGBM_BAG_SEEDS_BY_HORIZON.get(h, '42').split(',') if x.strip()])
assert len(d.get('models') or [m]) == exp, (len(d.get('models') or [m]), exp)
print(f"✅ 校验通过: {path}（{type(m).__name__}, n={m.n_estimators}, "
      f"features={len(d['feature_columns'])}）")
PY
done

echo "[$(date '+%F %T')] 月度重训完成：5/10/20 均通过校验"
