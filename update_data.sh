#!/bin/bash

# 脚本功能：将 data 目录下的文件更新到 GitHub
# 预定在每天下午五点运行
# 0 17 * * * /data/fortune/update_data.sh

# 设置重试次数和延迟时间
MAX_RETRIES=3
RETRY_DELAY=10

# 进入项目根目录
cd /data/fortune

# 添加 data 目录下的所有更改到暂存区
git add data/

# 提交更改
git commit -m "Update data files on $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || echo "No changes to commit"

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
