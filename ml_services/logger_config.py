#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的日志配置模块
为 ml_services 模块提供标准化的日志功能
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 日志目录配置
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 默认日志级别（可通过环境变量覆盖）
DEFAULT_LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_LEVEL = LOG_LEVELS.get(DEFAULT_LOG_LEVEL.upper(), logging.INFO)

# 日志格式
CONSOLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(message)s'
FILE_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
SIMPLE_FORMAT = '%(message)s'

# 日志文件配置
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # 保留5个备份


class MLLogger:
    """机器学习模块专用日志记录器"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name='ml_services', level=None):
        """
        获取或创建日志记录器

        参数:
        - name: 日志记录器名称（通常为模块名）
        - level: 日志级别（None 则使用全局配置）

        返回:
        - logging.Logger 实例
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level if level else LOG_LEVEL)

            # 避免重复添加 handler
            if not logger.handlers:
                # 控制台处理器（彩色输出）
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_formatter = ColoredFormatter(CONSOLE_FORMAT)
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

                # 文件处理器（详细日志）
                today = datetime.now().strftime('%Y-%m-%d')
                log_file = os.path.join(LOG_DIR, f'ml_services_{today}.log')
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(FILE_FORMAT)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

            cls._loggers[name] = logger

        return cls._loggers[name]


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        # 获取日志级别对应的颜色
        level_color = self.COLORS.get(record.levelname, '')
        levelname = f"{level_color}{record.levelname}{self.RESET}"

        # 替换格式中的levelname
        record.levelname = levelname

        # 调用父类的format方法
        result = super().format(record)

        # 恢复levelname
        record.levelname = record.levelname.replace(level_color, '').replace(self.RESET, '')

        return result


def get_logger(name='ml_services'):
    """
    便捷函数：获取日志记录器

    用法:
    >>> from ml_services.logger_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("这是一条信息")
    >>> logger.warning("这是一条警告")
    >>> logger.error("这是一条错误")
    """
    return MLLogger.get_logger(name)


def set_log_level(level):
    """
    设置全局日志级别

    参数:
    - level: 日志级别字符串（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    """
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    for logger in MLLogger._loggers.values():
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)


# ========== 便捷装饰器 ==========

def log_execution_time(logger_name='ml_services'):
    """
    记录函数执行时间的装饰器

    用法:
    >>> from ml_services.logger_config import log_execution_time
    >>> @log_execution_time()
    >>> def my_function():
    >>>     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = datetime.now()
            logger.debug(f"开始执行: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"完成执行: {func.__name__} (耗时: {execution_time:.2f}秒)")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"执行失败: {func.__name__} (耗时: {execution_time:.2f}秒) - {str(e)}")
                raise

        return wrapper
    return decorator


def log_function_call(logger_name='ml_services'):
    """
    记录函数调用的装饰器

    用法:
    >>> from ml_services.logger_config import log_function_call
    >>> @log_function_call()
    >>> def my_function(arg1, arg2):
    >>>     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            args_str = ', '.join([str(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            params = ', '.join(filter(None, [args_str, kwargs_str]))
            logger.debug(f"调用函数: {func.__name__}({params})")

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"函数调用异常: {func.__name__}({params}) - {str(e)}")
                raise

        return wrapper
    return decorator


if __name__ == '__main__':
    # 测试代码
    logger = get_logger('test')
    logger.debug("这是 DEBUG 级别日志")
    logger.info("这是 INFO 级别日志")
    logger.warning("这是 WARNING 级别日志")
    logger.error("这是 ERROR 级别日志")
    logger.critical("这是 CRITICAL 级别日志")
