#!/bin/bash
# 后台运行去年4月到今年4月的异常检测

echo "开始检测: $(date)" > output/april_to_april_detection.log

python3 detect_april_to_april_anomalies.py >> output/april_to_april_detection.log 2>&1

echo "检测完成: $(date)" >> output/april_to_april_detection.log
