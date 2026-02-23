#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日志系统功能
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.logger_config import get_logger, set_log_level

def test_logger():
    """测试日志系统"""

    print("=" * 70)
    print("日志系统测试")
    print("=" * 70)
    print()

    # 获取日志记录器
    logger = get_logger('test_logger')

    # 测试不同日志级别
    logger.debug("这是一条 DEBUG 日志")
    logger.info("这是一条 INFO 日志")
    logger.warning("这是一条 WARNING 日志")
    logger.error("这是一条 ERROR 日志")
    logger.critical("这是一条 CRITICAL 日志")

    print()
    print("-" * 70)
    print("测试 f-string 格式化:")
    logger.info(f"处理了 {42} 条记录，耗时 {1.23:.2f} 秒")
    logger.warning(f"⚠️  警告: 缺少 {3} 个必要参数")
    logger.error(f"❌ 错误: 文件 {__file__} 不存在")

    print()
    print("-" * 70)
    print("测试日志级别切换:")
    print("(当前: INFO)")
    set_log_level('DEBUG')
    logger.debug("现在可以看到 DEBUG 日志了")

    set_log_level('WARNING')
    logger.debug("这条 DEBUG 日志不会显示")
    logger.info("这条 INFO 日志也不会显示")
    logger.warning("这条 WARNING 日志会显示")

    # 恢复默认级别
    set_log_level('INFO')

    print()
    print("-" * 70)
    print("测试多模块日志:")
    logger1 = get_logger('module1')
    logger2 = get_logger('module2')
    logger1.info("模块 1 的消息")
    logger2.info("模块 2 的消息")

    print()
    print("=" * 70)
    print("测试完成！")
    print("=" * 70)
    print()
    print("💡 日志文件位置:")
    print(f"   - logs/ml_services_*.log")
    print()
    print("💡 环境变量配置:")
    print("   - LOG_LEVEL=DEBUG/INFO/WARNING/ERROR/CRITICAL")


def test_integration():
    """测试与现有模块的集成"""

    print()
    print("=" * 70)
    print("测试与现有模块的集成")
    print("=" * 70)
    print()

    try:
        # 测试导入 ml_trading_model
        from ml_services import ml_trading_model

        logger = get_logger('integration_test')
        logger.info("✅ ml_trading_model 模块导入成功")

        # 测试导入其他模块
        from ml_services import feature_selection
        logger.info("✅ feature_selection 模块导入成功")

        from ml_services import batch_backtest
        logger.info("✅ batch_backtest 模块导入成功")

        from ml_services import backtest_evaluator
        logger.info("✅ backtest_evaluator 模块导入成功")

        from ml_services import topic_modeling
        logger.info("✅ topic_modeling 模块导入成功")

        from ml_services import base_model_processor
        logger.info("✅ base_model_processor 模块导入成功")

        print()
        print("✅ 所有模块导入成功，日志系统集成正常！")

    except Exception as e:
        logger = get_logger('integration_test')
        logger.error(f"❌ 模块导入失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == '__main__':
    test_logger()
    test_integration()
