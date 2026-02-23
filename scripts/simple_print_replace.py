#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版的 print 到 logger 替换脚本
仅处理最常见的模式
"""

import re

def main():
    file_path = '/data/fortune/ml_services/ml_trading_model.py'

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified_lines = []
    replacements = []

    # 替换规则
    rules = [
        (r'print\(f"✅ ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
        (r'print\(f"❌ ([^"]+)"\)', r'logger.error(r"\1")', 'error'),
        (r'print\(f"⚠️ ([^"]+)"\)', r'logger.warning(r"\1")', 'warning'),
        (r'print\(f"🚀 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
        (r'print\(f"📊 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
        (r'print\(f"🔧 ([^"]+)"\)', r'logger.debug(r"\1")', 'debug'),
        (r'print\(f"🔍 ([^"]+)"\)', r'logger.debug(r"\1")', 'debug'),
        (r'print\("🚀 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
        (r'print\("📊 ([^"]+)"\)', r'logger.info(r"\1")', 'info'),
        (r'print\("=" \* \d+\)', r'logger.info("=" * 50)', 'info'),
    ]

    for i, line in enumerate(lines):
        original_line = line
        modified_line = line

        # 跳过已经是 logger 的行
        if 'logger.' in line:
            modified_lines.append(line)
            continue

        # 尝试应用替换规则
        for pattern, replacement, log_level in rules:
            match = re.search(pattern, line)
            if match:
                # 提取匹配内容
                content = match.group(1) if match.groups() else ''
                # 生成新的 logger 调用
                new_call = f'logger.{log_level}(r"{content}")'
                modified_line = re.sub(pattern, new_call, line)

                if modified_line != original_line:
                    replacements.append({
                        'line': i + 1,
                        'original': original_line.strip(),
                        'modified': modified_line.strip(),
                        'level': log_level
                    })
                    break

        modified_lines.append(modified_line)

    # 如果有替换，备份并写入
    if replacements:
        # 备份
        backup_path = file_path + '.print_backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())
        print(f"✅ 已备份到: {backup_path}")

        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)

        print(f"\n✅ 共替换 {len(replacements)} 处\n")

        # 显示替换详情
        print("替换详情（前20条）：")
        print("-" * 100)
        for r in replacements[:20]:
            print(f"行 {r['line']:4d} [{r['level']:7s}]: {r['original'][:60]}")
        print("-" * 100)

        if len(replacements) > 20:
            print(f"\n... 还有 {len(replacements) - 20} 处替换")
    else:
        print("没有找到可替换的 print 语句")

if __name__ == '__main__':
    main()
