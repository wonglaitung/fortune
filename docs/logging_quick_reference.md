# 日志系统快速参考

## 快速开始

### 在代码中使用

```python
from ml_services.logger_config import get_logger

logger = get_logger('my_module')

logger.info("这是一条信息")
logger.warning("这是一条警告")
logger.error("这是一条错误")
logger.debug("这是一条调试信息")
```

### 配置日志级别

```bash
# 环境变量方式（推荐）
export LOG_LEVEL=DEBUG
python your_script.py

# 或在代码中
from ml_services.logger_config import set_log_level
set_log_level('DEBUG')
```

### 可用级别

- `DEBUG`: 详细的调试信息
- `INFO`: 一般信息（默认）
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 日志位置

- **控制台**: 彩色输出（INFO及以上）
- **文件**: `logs/ml_services_YYYY-MM-DD.log`（DEBUG及以上）

## 日志格式

```
时间戳 | 级别 | 模块名 | 文件:行号 | 消息
```

示例：
```
2026-02-23 21:52:02,462 | INFO | ml_trading_model | ml_trading_model.py:123 | 处理完成
```

## 装饰器

### 记录执行时间

```python
from ml_services.logger_config import log_execution_time

@log_execution_time()
def my_function():
    # 自动记录开始、完成和执行时间
    pass
```

### 记录函数调用

```python
from ml_services.logger_config import log_function_call

@log_function_call()
def my_function(arg1, arg2):
    # 自动记录函数调用和参数
    pass
```

## 颜色映射

- 🔵 DEBUG: 青色
- 🟢 INFO: 绿色
- 🟡 WARNING: 黄色
- 🔴 ERROR: 红色
- 🟣 CRITICAL: 紫色

## 常见用法

### 格式化消息

```python
logger.info(f"处理了 {count} 条记录")
logger.warning(f"⚠️  警告: 缺少 {missing} 个参数")
logger.error(f"❌ 错误: 文件 {filepath} 不存在")
```

### 记录异常

```python
try:
    # 你的代码
    pass
except Exception as e:
    logger.error(f"操作失败: {e}")
    logger.debug(traceback.format_exc())  # 详细错误信息
```

### 记录进度

```python
for i, item in enumerate(items):
    logger.debug(f"处理 [{i+1}/{len(items)}]")

logger.info(f"✅ 处理完成: {len(items)} 项")
```

## 最佳实践

1. **使用合适的级别**
   - DEBUG: 开发调试用
   - INFO: 重要的流程节点、结果
   - WARNING: 潜在问题、可恢复错误
   - ERROR: 需要处理的错误
   - CRITICAL: 系统级严重错误

2. **避免过度日志**
   - 不在循环中频繁记录 INFO
   - 大量调试信息使用 DEBUG 级别

3. **保持一致性**
   - 统一使用 `logger` 对象
   - 遵循项目日志格式约定

## 故障排查

### 日志不显示
- 检查日志级别是否正确
- 确认是否正确导入 `get_logger`

### 日志文件过大
- 日志会自动轮转（10MB/文件）
- 可调整 `logger_config.py` 中的 `MAX_LOG_SIZE`

### 格式化问题
- 确保 f-string 使用正确
- 变量在作用域内可用

## 完整示例

```python
#!/usr/bin/env python3
from ml_services.logger_config import get_logger, log_execution_time

logger = get_logger(__name__)

@log_execution_time()
def main():
    logger.info("程序启动")

    try:
        # 执行一些操作
        result = process_data()
        logger.info(f"✅ 操作成功: {result}")

    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info("程序结束")

if __name__ == '__main__':
    main()
```

## 相关文件

- `ml_services/logger_config.py`: 日志配置模块
- `logs/ml_services_*.log`: 日志文件
- `scripts/test_logger.py`: 测试脚本
- `docs/logging_system_upgrade.md`: 完整文档
