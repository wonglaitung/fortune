#!/usr/bin/env python3
# 简单修复 Isolation Forest 的异常评分阈值

# 读取文件
with open('anomaly_detector/isolation_forest_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并替换
new_content = content.replace(
    "        if anomaly_score < -0.7:",
    "        if anomaly_score <= -0.5:"
)

# 写回文件
with open('anomaly_detector/isolation_forest_detector.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('已更新异常评分阈值：-0.7分界从 high 调整为 medium')
print('现在 0.5 < anomaly_score <= -0.7 将返回 medium')
