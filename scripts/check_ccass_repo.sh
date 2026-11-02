#!/usr/bin/env bash
# 诊断 CCASS 私有数据仓访问（不打印完整 token）
# 用法：
#   CCASS_DATA_REPO=owner/repo CCASS_DATA_TOKEN=xxx bash scripts/check_ccass_repo.sh
set -u
: "${CCASS_DATA_REPO:?请设置 CCASS_DATA_REPO}"
: "${CCASS_DATA_TOKEN:?请设置 CCASS_DATA_TOKEN}"

echo "repo=[$CCASS_DATA_REPO]"
echo "token_len=${#CCASS_DATA_TOKEN} prefix=${CCASS_DATA_TOKEN:0:11} suffix=${CCASS_DATA_TOKEN: -4}"

api=$(curl -s -o /tmp/ccass_api.json -w '%{http_code}' \
  -H "Authorization: Bearer $CCASS_DATA_TOKEN" \
  "https://api.github.com/repos/$CCASS_DATA_REPO")
priv=$(python3 -c "import json;print(json.load(open('/tmp/ccass_api.json')).get('private'))" 2>/dev/null || echo parse_error)
echo "api_http=$api private=$priv"

git_http=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $CCASS_DATA_TOKEN" \
  "https://github.com/$CCASS_DATA_REPO.git/info/refs?service=git-upload-pack")
echo "git_http=$git_http   # 200=ok, 401/403=token 无 git 权限, 404=仓库/权限"

echo "--- git ls-remote ---"
GIT_TERMINAL_PROMPT=0 git ls-remote --heads \
  "https://x-access-token:${CCASS_DATA_TOKEN}@github.com/${CCASS_DATA_REPO}.git" 2>&1 \
  | sed "s|${CCASS_DATA_TOKEN}|***|g" | head -5
