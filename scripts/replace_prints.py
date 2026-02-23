#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本：自动将 print 语句替换为 logger 调用
"""

import re
import os

# 文件路径
FILE_PATH = '/data/fortune/ml_services/ml_trading_model.py'

# print 语句到 logger 调用的映射规则
# 格式: (pattern, replacement, log_level)
REPLACEMENT_RULES = [
    # 成功/完成信息 -> INFO
    (r'print\(f"✅ ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
    (r'print\(f"\\n✅ ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
    (r'print\("✅ ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
    (r'print\("🚀 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
    (r'print\("📂 ([^"]+)"\)', r'logger.debug(r"\1")', 'debug'),

    # 进度信息 -> INFO
    (r'print\("🔧 ([^"]+)"\)', r'logger.debug(r"\1")', 'debug'),
    (r'print\("🔍 ([^"]+)"\)', r'logger.debug(r"\1")', 'debug'),
    (r'print\("📊 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
    (r'print\("📈 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),

    # 警告信息 -> WARNING
    (r'print\(f"⚠️ ([^"]+)"\)', r'logger.warning(r"\1")', 'warning'),
    (r'print\("⚠️ ([^"]+)"\)', r'logger.warning(r"\1")', 'warning'),

    # 错误信息 -> ERROR
    (r'print\("❌ ([^"]+)"\)', r'logger.error(r"\1")', 'error'),
    (r'print\(f"❌ ([^"]+)"\)', r'logger.error(r"\1")', 'error'),

    # 分隔线和标题 -> INFO
    (r'print\(("=" \* \d+)\)', r'logger.info(r"\1")', 'info'),
    (r'print\("-" \* \d+', r'logger.debug("-" * ', 'debug'),
]


def replace_prints_in_file(file_path):
    """替换文件中的 print 语句"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    replacements_made = 0

    # 应用替换规则
    for pattern, replacement, log_level in REPLACEMENT_RULES:
        matches = re.findall(pattern, content)
        if matches:
            print(f"找到 {len(matches)} 个匹配: {pattern} -> logger.{log_level}()")
            content = re.sub(pattern, replacement, content)
            replacements_made += len(matches)

    # 特殊处理：处理多行 print 语句和复杂的 f-string
    # 例如: print(f"{'代码':<10} {'股票名称':<12} ...")
    # 这种情况下保持 print 不变（因为是格式化输出）

    if content != original_content:
        # 备份原文件
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"\n已创建备份文件: {backup_path}")

        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✅ 替换完成！")
        print(f"共替换 {replacements_made} 处 print 语句")
        print(f"文件: {file_path}")

        # 显示统计信息
        print("\n" + "="*70)
        print("替换统计：")
        print("="*70)

        # 统计剩余的 print 语句
        remaining_prints = re.findall(r'print\(', content)
        if remaining_prints:
            print(f"⚠️  还有 {len(remaining_prints)} 处 print 语句未替换")
            print("\n可能的原因：")
            print("- 复杂的 f-string 格式化输出（建议保持原样）")
            print("- 多行 print 语句（需要手动处理）")
            print("- 特殊的 print 格式（不在匹配规则中）")
        else:
            print("✅ 所有 print 语句已替换完成！")

    else:
        print("未找到需要替换的 print 语句")


if __name__ == '__main__':
    print("="*70)
    print("Print 语句替换工具")
    print("="*70)
    print(f"\n处理文件: {FILE_PATH}")
    print()

    replace_prints_in_file(FILE_PATH)
