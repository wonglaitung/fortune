#!/bin/bash

# 脚本功能：将 data 目录下的文件更新到 GitHub
# 预定在每天下午五点运行
# 0 17 * * * /path/to/fortune/update_data.sh

# 设置重试次数和延迟时间
MAX_RETRIES=3
RETRY_DELAY=10

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 进入项目根目录（使用相对路径）
cd "$SCRIPT_DIR"

retry_count=0
while [ $retry_count -lt $MAX_RETRIES ]; do
    git pull
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Files updated from GitHub successfully at $(date '+%Y-%m-%d %H:%M:%S')"
	break
    else
        retry_count=$((retry_count + 1))
        echo "Git pull failed (attempt $retry_count/$MAX_RETRIES), retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    fi
done

# 添加 data 目录下的所有更改到暂存区
git add data/

# 检查是否有任何更改需要提交
if git diff --cached --quiet --exit-code; then
    echo "No changes staged for commit, skipping commit and push" 
    exit 0
fi

# 提交更改
git commit -m "Update data files on $(date '+%Y-%m-%d %H:%M:%S')"
if [ $? -ne 0 ]; then
    echo "Git commit failed at $(date '+%Y-%m-%d %H:%M:%S')"
    exit 1
fi

# 推送到远程仓库，带重试机制
retry_count=0
while [ $retry_count -lt $MAX_RETRIES ]; do
    # 设置 git 超时时间
    GIT_TERMINAL_PROMPT=0 git push --progress origin main -o timeout=60
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Data files updated to GitHub successfully at $(date '+%Y-%m-%d %H:%M:%S')"
        exit 0
    else
        retry_count=$((retry_count + 1))
        echo "Git push failed (attempt $retry_count/$MAX_RETRIES), retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    fi
done

echo "Failed to update data files to GitHub after $MAX_RETRIES attempts at $(date '+%Y-%m-%d %H:%M:%S')"
exit 1
