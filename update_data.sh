#!/bin/bash

# 脚本功能：将 data 目录下的文件更新到 GitHub
# 预定在每天下午五点运行
# 0 17 * * * /data/fortune/update_data.sh

# 进入项目根目录
cd /data/fortune

# 添加 data 目录下的所有更改到暂存区
git add data/

# 提交更改
git commit -m "Update data files on $(date '+%Y-%m-%d %H:%M:%S')"

# 推送到远程仓库
git push origin main

echo "Data files updated to GitHub at $(date '+%Y-%m-%d %H:%M:%S')"
