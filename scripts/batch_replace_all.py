#!/usr/bin/env python3
import re
import sys

def replace_prints_in_file(file_path, module_name):
    """在指定文件中替换 print 语句"""

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    total_replacements = 0

    # 替换规则
    replacements = [
        (r'print\(f"⚠️ ([^"]+)"\)', f'logger.warning(f"\\1")'),
        (r'print\("⚠️ ([^"]+)"\)', f'logger.warning("\\1")'),
        (r'print\(f"✅ ([^"]+)"\)', f'logger.info(f"\\1")'),
        (r'print\("✅ ([^"]+)"\)', f'logger.info("\\1")'),
        (r'print\(f"❌ ([^"]+)"\)', f'logger.error(f"\\1")'),
        (r'print\("❌ ([^"]+)"\)', f'logger.error("\\1")'),
        (r'print\(f"🚀 ([^"]+)"\)', f'logger.info(f"\\1")'),
        (r'print\(f"📊 ([^"]+)"\)', f'logger.info(f"\\1")'),
        (r'print\(f"📈 ([^"]+)"\)', f'logger.info(f"\\1")'),
        (r'print\(f"🔧 ([^"]+)"\)', f'logger.debug(f"\\1")'),
        (r'print\(f"🔍 ([^"]+)"\)', f'logger.debug(f"\\1")'),
        (r'print\("=" \* \d+\)', 'logger.info("=" * 50)'),
        (r'print\("-" \* \d+\)', 'logger.debug("-" * 80)'),
    ]

    for i, line in enumerate(lines):
        original_line = line

        # 跳过已经是 logger 的行
        if 'logger.' in line:
            continue

        # 尝试应用替换规则
        for pattern, replacement in replacements:
            if re.search(pattern, line):
                lines[i] = re.sub(pattern, replacement, line)
                if lines[i] != original_line:
                    total_replacements += 1
                    modified = True
                    break

    if modified:
        # 备份
        backup_path = file_path + '.print_backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())

        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"✅ {module_name}: 替换 {total_replacements} 处")
        return total_replacements
    else:
        print(f"⚠️  {module_name}: 无需替换")
        return 0


def add_logger_import(file_path, module_name):
    """添加 logger 导入语句"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经有 logger 导入
    if 'from ml_services.logger_config import get_logger' in content:
        print(f"⚠️  {module_name}: logger 已导入")
        return False

    # 找到导入部分的位置
    import_pattern = r'(import.*?\n)+'

    # 查找最后一个 import 语句
    import_match = None
    for match in re.finditer(import_pattern, content):
        import_match = match

    if import_match:
        # 在最后一个 import 语句后添加 logger 导入
        end_pos = import_match.end()
        import_line = f'from ml_services.logger_config import get_logger\n'
        logger_line = f'\nlogger = get_logger("{module_name}")\n'

        content = content[:end_pos] + import_line + logger_line + content[end_pos:]

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ {module_name}: 已添加 logger 导入")
        return True
    else:
        print(f"⚠️  {module_name}: 未找到导入部分")
        return False


def main():
    # 需要处理的文件列表
    files_to_process = [
        ('/data/fortune/ml_services/feature_selection.py', 'feature_selection'),
        ('/data/fortune/ml_services/batch_backtest.py', 'batch_backtest'),
        ('/data/fortune/ml_services/backtest_evaluator.py', 'backtest_evaluator'),
        ('/data/fortune/ml_services/topic_modeling.py', 'topic_modeling'),
    ]

    print("="*70)
    print("日志系统升级 - 批量替换脚本")
    print("="*70)
    print()

    # 第一步：添加 logger 导入
    print("步骤 1: 添加 logger 导入")
    print("-"*70)
    for file_path, module_name in files_to_process:
        try:
            add_logger_import(file_path, module_name)
        except Exception as e:
            print(f"❌ {module_name}: 添加导入失败 - {e}")

    print()

    # 第二步：替换 print 语句
    print("步骤 2: 替换 print 语句")
    print("-"*70)
    total = 0
    for file_path, module_name in files_to_process:
        try:
            count = replace_prints_in_file(file_path, module_name)
            total += count
        except Exception as e:
            print(f"❌ {module_name}: 替换失败 - {e}")

    print()
    print("="*70)
    print(f"✅ 完成！共替换 {total} 处 print 语句")
    print("="*70)


if __name__ == '__main__':
    main()
