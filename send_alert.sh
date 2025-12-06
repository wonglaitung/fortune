#!/bin/bash

# 脚本功能：将 data 目录下的文件更新到 GitHub
# 预定在每天下午五点运行
# 0 6 * * * /data/fortune/send_alert.sh

# 进入项目根目录
cd /data/fortune
source set_key.sh

# 获取股票新闻(需yahoo finance)
python batch_stock_news_fetcher.py

# 分析市场环境(需yahoo finance)
python hsi_llm_strategy.py

# 获取昨天的日期
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)

# 调用 hk_smart_money_tracker.py 并传入昨天的日期
python hk_smart_money_tracker.py --date $YESTERDAY

# 获取黄金信息
python gold_analyzer.py

