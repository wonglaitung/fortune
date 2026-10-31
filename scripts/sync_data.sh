#!/usr/bin/env bash
# 本地同步：代码/模型/事件表/报告 + 可选 CCASS 私有仓
# 用法：
#   bash scripts/sync_data.sh
#   CCASS_DATA_REPO=owner/repo CCASS_DATA_TOKEN=... bash scripts/sync_data.sh --ccass
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[1/2] git pull --rebase --autostash"
git pull --rebase --autostash

if [ "${1:-}" = "--ccass" ]; then
  : "${CCASS_DATA_REPO:?请设置 CCASS_DATA_REPO=owner/repo}"
  : "${CCASS_DATA_TOKEN:?请设置 CCASS_DATA_TOKEN}"
  dir="data/ccass_private"
  if [ -d "$dir/.git" ]; then
    echo "[2/2] 更新 CCASS 私有仓"
    git -C "$dir" pull --ff-only
  else
    echo "[2/2] 克隆 CCASS 私有仓（token 不写入 remote）"
    git clone "https://x-access-token:${CCASS_DATA_TOKEN}@github.com/${CCASS_DATA_REPO}.git" "$dir"
    git -C "$dir" remote set-url origin "https://github.com/${CCASS_DATA_REPO}.git"
  fi
else
  echo "[2/2] 跳过 CCASS（需要时加 --ccass）"
fi

python3 scripts/data_freshness.py --out output/_logs/data_freshness.md
