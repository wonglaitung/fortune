#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的 print 到 logger 替换脚本
正确处理 f-string 中的变量
"""

import re

def main():
    file_path = '/data/fortune/ml_services/ml_trading_model.py'

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换规则 - 正确保留 f-string 的变量
    replacements = [
        # 替换成功信息
        (r'print\(f"✅ ([^"]+)"\)', r'logger.info(f"\1")', 'info'),
        (r'print\("✅ ([^"]+)"\)', r'logger.info(r"\1")', 'info'),

        # 替换错误信息
        (r'print\(f"❌ ([^"]+)"\)', r'logger.error(f"\1")', 'error'),
        (r'print\("❌ ([^"]+)"\)', r'logger.error(r"\1")', 'error'),

        # 替换警告信息
        (r'print\(f"⚠️ ([^"]+)"\)', r'logger.warning(f"\1")', 'warning'),
        (r'print\("⚠️ ([^"]+)"\)', r'logger.warning(r"\1")', 'warning'),

        # 替换调试信息
        (r'print\(f"🔧 ([^"]+)"\)', r'logger.debug(f"\1")', 'debug'),
        (r'print\(f"🔍 ([^"]+)"\)', r'logger.debug(f"\1")', 'debug'),

        # 替换信息输出
        (r'print\(f"🚀 ([^"]+)"\)', r'logger.info(f"\1")', 'info'),
        (r'print\(f"📊 ([^"]+)"\)', r'logger.info(f"\1")', 'info'),
        (r'print\(f"📈 ([^"]+)"\)', r'logger.info(f"\1")', 'info'),
        (r'print\(f"📂 ([^"]+)"\)', r'logger.debug(f"\1")', 'debug'),

        # 替换简单的字符串（非 f-string）
        (r'print\("🚀 ([^"]+)"\)', r'logger.info("\1")', 'info'),
        (r'print\("📊 ([^"]+)"\)', r'logger.info("\1")', 'info'),
        (r'print\("=" \* (\d+)\)', r'logger.info("=" * \1)', 'info'),
    ]

    modified = False
    total_replacements = 0

    # 应用替换规则
    for pattern, replacement, log_level in replacements:
        matches = list(re.finditer(pattern, content))
        if matches:
            print(f"替换规则 [{log_level}]: 找到 {len(matches)} 个匹配")
            # 从后往前替换，避免位置问题
            for match in reversed(matches):
                old_text = match.group(0)
                new_text = re.sub(pattern, replacement, old_text)
                content = content[:match.start()] + new_text + content[match.end():]
                total_replacements += 1
                modified = True

    if modified:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✅ 替换完成！共替换 {total_replacements} 处")
    else:
        print("没有找到需要替换的 print 语句")

if __name__ == '__main__':
    main()
