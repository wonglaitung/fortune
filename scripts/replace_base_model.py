#!/usr/bin/env python3
import re

file_path = '/data/fortune/ml_services/base_model_processor.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换规则
replacements = [
    (r'print\(f"⚠️ ([^"]+)"\)', r'logger.warning(f"\1")'),
    (r'print\(f"✅ ([^"]+)"\)', r'logger.info(f"\1")'),
    (r'print\(f"❌ ([^"]+)"\)', r'logger.error(f"\1")'),
    (r'print\("🧠 ([^"]+)"\)', r'logger.info("\1")'),
    (r'print\("📊 ([^"]+)"\)', r'logger.info("\1")'),
    (r'print\("💡 ([^"]+)"\)', r'logger.info("\1")'),
    (r'print\("=" \* \d+\)', r'logger.info("=" * 50)'),
]

total = 0
for pattern, replacement in replacements:
    matches = len(re.findall(pattern, content))
    if matches:
        content = re.sub(pattern, replacement, content)
        total += matches
        print(f'替换 {matches} 处: {pattern}')

if total > 0:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'✅ 共替换 {total} 处')
else:
    print('没有找到需要替换的 print 语句')
