#!/bin/bash

# 脚本功能：将 data 目录下的文件更新到 GitHub
# 预定在每天下午五点运行
# 0 6 * * * /data/fortune/send_alert.sh

# 进入项目根目录
cd /data/fortune
source set_key.sh
python hk_smart_money_tracker.py
python gold_analyzer.py

