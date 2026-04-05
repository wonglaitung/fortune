#!/usr/bin/env python3
# 修复 Isolation Forest 的异常评分阈值

import re

# 读取文件
with open('anomaly_detector/isolation_forest_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找 _get_severity 方法
pattern = r"def _get_severity\(self, anomaly_score: float\) -> str:"
match = re.search(pattern, content)

if match:
    # 获取原方法
    original_method = match.group(0)
    
    # 获取整个方法体（包括后续代码）
    method_start = content.find(original_method)
    method_end = content.find('    def ', method_start + 1)
    
    if method_end != -1:
        method_body = content[method_start:method_end]
        
        # 替换阈值逻辑
        new_method = '''    def _get_severity(self, anomaly_score: float) -> str:
        """
        Get severity level based on anomaly score.
        
        Args:
            anomaly_score: Anomaly score from Isolation Forest (lower = more anomalous)
        
        Returns:
            Severity level ('high', 'medium', 'low')
        """
        if anomaly_score < -0.5:
            return 'high'
        elif anomaly_score < -0.3:
            return 'medium'
        else:
            return 'low'
        '''
        
        # 构造新内容
        new_content = content[:method_start] + new_method + '    def ' + content[method_end:]
        
        # 写回文件
        with open('anomaly_detector/isolation_forest_detector.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print('已更新异常评分阈值')
        print('  - < -0.5.0 high 级别')
        print('  - -0.5.0 <= anomaly_score < -0.3.0 medium 级别')
        print('  - anomaly_score >= -0.3.0 low 级别')
else:
    print('未找到 _get_severity 方法')