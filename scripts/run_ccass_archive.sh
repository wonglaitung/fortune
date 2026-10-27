#!/bin/bash
# CCASS 日归档（本地 cron 用；数据不入库——HKEX 仅限个人非商业使用）
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p data/ccass output/_logs
echo "[$(date '+%F %T')] CCASS 归档开始" >> output/_logs/ccass_archive.log
python3 scripts/fetch_ccass.py --workers 4 >> output/_logs/ccass_archive.log 2>&1
echo "[$(date '+%F %T')] CCASS 归档结束" >> output/_logs/ccass_archive.log
