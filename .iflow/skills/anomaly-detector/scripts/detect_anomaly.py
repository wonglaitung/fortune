#!/usr/bin/env python3
"""
时间序列异常检测工具
支持 Z-Score 和 Isolation Forest 两种检测方法

用法：
    python detect_anomaly.py data.csv --column price
    python detect_anomaly.py data.csv --column price --method isolation-forest
    python detect_anomaly.py data.xlsx --column value --sheet Sheet1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 添加技能目录到路径，从技能内部导入 anomaly_detector 模块
# 路径: scripts -> anomaly-detector (技能根目录)
SCRIPT_DIR = Path(__file__).parent.parent  # skills/anomaly-detector/
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from anomaly_detector.zscore_detector import ZScoreDetector, TimeInterval
    from anomaly_detector.isolation_forest_detector import IsolationForestDetector
    from anomaly_detector.feature_extractor import FeatureExtractor
except ImportError:
    print("错误：无法导入 anomaly_detector 模块")
    print("请确保 anomaly_detector 模块在技能目录下")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('anomaly_detector_cli')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_data(
    file_path: str, 
    column: str, 
    sheet: Optional[str] = None,
    timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        file_path: 文件路径
        column: 要检测的列名
        sheet: Excel 工作表名称
        timestamp_column: 时间戳列名（可选，不指定则自动检测）
    
    Returns:
        DataFrame 包含时间戳和目标列
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 根据文件扩展名读取
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        if sheet:
            df = pd.read_excel(file_path, sheet_name=sheet)
        else:
            df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    # 检查列是否存在
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中。可用列: {list(df.columns)}")
    
    # 处理时间戳列
    timestamp_col = None
    
    if timestamp_column:
        # 用户手动指定时间戳列名
        if timestamp_column not in df.columns:
            raise ValueError(f"时间戳列 '{timestamp_column}' 不存在于数据中。可用列: {list(df.columns)}")
        timestamp_col = timestamp_column
    else:
        # 自动检测时间戳列（常见英文列名）
        common_timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        for col in common_timestamp_cols:
            # 不区分大小写匹配
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    timestamp_col = actual_col
                    break
            if timestamp_col:
                break
    
    if timestamp_col:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        except Exception as e:
            logger.warning(f"无法解析时间戳列 '{timestamp_col}': {e}")
    
    return df


def detect_zscore(
    df: pd.DataFrame,
    column: str,
    window_size: int = 30,
    threshold: float = 3.0,
    time_interval: str = 'day'
) -> List[Dict]:
    """
    使用 Z-Score 方法检测异常
    
    Args:
        df: 数据 DataFrame
        column: 要检测的列名
        window_size: 窗口大小
        threshold: 阈值
        time_interval: 时间间隔
    
    Returns:
        异常列表
    """
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中。可用列: {list(df.columns)}")
    
    detector = ZScoreDetector(
        window_size=window_size,
        threshold=threshold,
        time_interval=time_interval
    )
    
    history = df[column].dropna()
    anomalies = []
    
    # 检测每个时间点
    for i in range(window_size, len(history)):
        current_value = history.iloc[i]
        timestamp = history.index[i]
        history_window = history.iloc[:i]
        
        anomaly = detector.detect_anomaly(
            metric_name=column,
            current_value=current_value,
            history=history_window,
            timestamp=timestamp,
            time_interval=time_interval
        )
        
        if anomaly:
            anomalies.append(anomaly)
    
    return anomalies


def detect_isolation_forest(
    df: pd.DataFrame,
    column: str,
    contamination: float = 0.03,
    lookback_days: int = 7,
    use_all_columns: bool = False
) -> List[Dict]:
    """
    使用 Isolation Forest 方法检测异常
    
    Args:
        df: 数据 DataFrame
        column: 要检测的列名
        contamination: 异常比例
        lookback_days: 回溯天数
        use_all_columns: 是否使用所有数值列进行多维特征提取（默认 False，仅使用指定列）
    
    Returns:
        异常列表
    """
    # 检查列是否存在
    if column not in df.columns:
        raise ValueError(f"列 '{column}' 不存在于数据中。可用列: {list(df.columns)}")
    
    # 使用 FeatureExtractor 提取特征
    # 会自动检测 OHLCV 列或通用数值列
    extractor = FeatureExtractor()
    
    if use_all_columns:
        # 使用所有数值列进行多维特征提取
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"INFO: 使用多维特征提取，列: {numeric_cols}")
        features, timestamps = extractor.extract_features(df, columns=numeric_cols)
    else:
        # 仅使用指定列
        features, timestamps = extractor.extract_features(df, columns=[column])
    
    if features.empty:
        raise ValueError(f"无法从列 '{column}' 提取特征，请检查数据")
    
    # 训练 Isolation Forest
    detector = IsolationForestDetector(
        contamination=contamination,
        anomaly_type='time_series'
    )
    detector.train(features)
    
    # 检测异常
    anomalies = detector.detect_anomalies(
        features=features,
        timestamps=timestamps,
        lookback_days=lookback_days
    )
    
    return anomalies


def format_output(
    anomalies: List[Dict],
    method: str,
    column: str = None,
    window_size: int = None,
    threshold: float = None,
    contamination: float = None
) -> str:
    """
    格式化输出结果
    
    Args:
        anomalies: 异常列表
        method: 检测方法
        column: 列名
        window_size: 窗口大小
        threshold: 阈值
        contamination: 异常比例
    
    Returns:
        格式化的输出字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append("异常检测结果")
    lines.append("=" * 50)
    lines.append(f"检测方法: {method.upper()}")
    
    if column:
        lines.append(f"检测指标: {column}")
    if window_size:
        lines.append(f"窗口大小: {window_size}")
    if threshold:
        lines.append(f"阈值: {threshold}")
    if contamination:
        lines.append(f"异常比例: {contamination}")
    
    lines.append("")
    
    if not anomalies:
        lines.append("未发现异常")
        return "\n".join(lines)
    
    lines.append(f"发现 {len(anomalies)} 个异常:")
    lines.append("")
    
    for i, anomaly in enumerate(anomalies, 1):
        lines.append(f"[{i}] {anomaly['timestamp']}")
        lines.append(f"    类型: {anomaly['type']}")
        lines.append(f"    严重程度: {anomaly['severity']}")
        
        if method == 'zscore':
            lines.append(f"    Z-Score: {anomaly['z_score']:.2f}")
            lines.append(f"    当前值: {anomaly['value']:.2f}")
            lines.append(f"    均值: {anomaly['mean']:.2f}")
            lines.append(f"    标准差: {anomaly['std']:.2f}")
        else:
            lines.append(f"    异常分数: {anomaly['anomaly_score']:.2f}")
            if 'feature_count' in anomaly:
                lines.append(f"    特征数: {anomaly['feature_count']}")
        
        lines.append("")
    
    return "\n".join(lines)


def save_output(anomalies: List[Dict], output_path: str):
    """
    保存结果到文件
    
    Args:
        anomalies: 异常列表
        output_path: 输出文件路径
    """
    path = Path(output_path)
    
    # 转换异常为可序列化格式
    serializable_anomalies = []
    for anomaly in anomalies:
        a = anomaly.copy()
        if 'timestamp' in a:
            a['timestamp'] = a['timestamp'].isoformat()
        if 'features' in a and isinstance(a['features'], dict):
            # 转换 numpy 类型
            a['features'] = {k: float(v) if hasattr(v, 'item') else v 
                           for k, v in a['features'].items()}
        serializable_anomalies.append(a)
    
    if path.suffix.lower() == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_anomalies, f, indent=2, ensure_ascii=False)
    elif path.suffix.lower() == '.csv':
        df = pd.DataFrame(serializable_anomalies)
        df.to_csv(output_path, index=False)
    else:
        # 默认 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_anomalies, f, indent=2, ensure_ascii=False)


def get_optimal_params(time_interval: str) -> dict:
    """
    根据时间间隔获取最佳检测参数
    
    Args:
        time_interval: 时间间隔类型 (minute, hour, day, week)
    
    Returns:
        包含最佳参数的字典
    """
    # 基于时间间隔的最佳参数配置
    params = {
        'minute': {
            'window_size': 60,      # 1小时窗口
            'threshold': 3.0,
            'contamination': 0.02,
            'description': '分钟级数据：60分钟窗口，阈值3.0'
        },
        'hour': {
            'window_size': 24,      # 1天窗口
            'threshold': 3.0,
            'contamination': 0.03,
            'description': '小时级数据：24小时窗口，阈值3.0'
        },
        'day': {
            'window_size': 30,      # 1个月窗口
            'threshold': 3.0,
            'contamination': 0.03,
            'description': '日级数据：30天窗口，阈值3.0'
        },
        'week': {
            'window_size': 12,      # 3个月窗口
            'threshold': 3.0,
            'contamination': 0.05,
            'description': '周级数据：12周窗口，阈值3.0'
        }
    }
    
    return params.get(time_interval, params['day'])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='时间序列异常检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测（根据时间间隔自动设置最佳参数）
  python detect_anomaly.py data.csv --column price --time-interval hour
  
  # 分钟级数据检测
  python detect_anomaly.py data.csv --column price --time-interval minute
  
  # 日级数据检测（默认）
  python detect_anomaly.py data.csv --column price
  
  # 指定时间戳列名（当自动检测失败时）
  python detect_anomaly.py data.csv --column price --timestamp-column 日期
  
  # 多维特征提取（使用所有数值列）
  python detect_anomaly.py data.csv --column sales --multi-column --method isolation-forest
  
  # 手动指定参数（覆盖自动设置）
  python detect_anomaly.py data.csv --column price --window-size 60 --threshold 2.5
  
  # 仅使用 Z-Score 检测
  python detect_anomaly.py data.csv --column price --method zscore
  
  # 处理 Excel 文件
  python detect_anomaly.py data.xlsx --column value --sheet Sheet1
  
  # 导出结果
  python detect_anomaly.py data.csv --column price --output anomalies.json
        """
    )
    
    parser.add_argument(
        'input_file',
        help='输入数据文件路径（CSV 或 Excel）'
    )
    
    parser.add_argument(
        '--column', '-c',
        required=True,
        help='要检测的列名'
    )
    
    parser.add_argument(
        '--method', '-m',
        choices=['zscore', 'isolation-forest', 'both'],
        default='both',
        help='检测方法：zscore、isolation-forest 或 both（默认：both 同时使用两种方法）'
    )
    
    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=None,
        help='Z-Score 窗口大小（默认根据时间间隔自动设置：minute=60, hour=24, day=30, week=12）'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Z-Score 阈值（默认 3.0）'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=None,
        help='Isolation Forest 异常比例（默认根据时间间隔自动设置：minute=0.02, hour=0.03, day=0.03, week=0.05）'
    )
    
    parser.add_argument(
        '--time-interval',
        choices=['minute', 'hour', 'day', 'week'],
        default='day',
        help='时间间隔（默认 day）'
    )
    
    parser.add_argument(
        '--timestamp-column',
        help='时间戳列名（可选，不指定则自动检测）'
    )
    
    parser.add_argument(
        '--sheet', '-s',
        help='Excel 工作表名称'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出文件路径（JSON 或 CSV）'
    )
    
    parser.add_argument(
        '--multi-column',
        action='store_true',
        help='使用所有数值列进行多维特征提取（仅 Isolation Forest 有效）'
    )
    
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=30,
        help='Isolation Forest 回溯天数（默认 30 天，设为 0 检测全部数据）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    args = parser.parse_args()
    
    # 根据时间间隔自动设置最佳参数
    optimal_params = get_optimal_params(args.time_interval)
    
    if args.window_size is None:
        args.window_size = optimal_params['window_size']
    
    if args.threshold is None:
        args.threshold = optimal_params['threshold']
    
    if args.contamination is None:
        args.contamination = optimal_params['contamination']
    
    # 设置日志
    logger = setup_logging(args.verbose)
    logger.info("异常检测工具启动")
    logger.info(f"时间间隔: {args.time_interval} - {optimal_params['description']}")
    
    try:
        # 加载数据
        logger.info(f"加载数据: {args.input_file}")
        df = load_data(args.input_file, args.column, args.sheet, args.timestamp_column)
        logger.info(f"数据行数: {len(df)}")
        
        # 检测异常
        all_anomalies = []
        
        # Z-Score 检测
        if args.method in ['zscore', 'both']:
            logger.info(f"使用 Z-Score 方法检测 (窗口: {args.window_size}, 阈值: {args.threshold})")
            zscore_anomalies = detect_zscore(
                df=df,
                column=args.column,
                window_size=args.window_size,
                threshold=args.threshold,
                time_interval=args.time_interval
            )
            
            output = format_output(
                anomalies=zscore_anomalies,
                method='zscore',
                column=args.column,
                window_size=args.window_size,
                threshold=args.threshold
            )
            print(output)
            all_anomalies.extend(zscore_anomalies)
            logger.info(f"Z-Score 检测完成，发现 {len(zscore_anomalies)} 个异常")
        
        # Isolation Forest 检测
        if args.method in ['isolation-forest', 'both']:
            logger.info(f"使用 Isolation Forest 方法检测 (异常比例: {args.contamination}, 回溯: {args.lookback_days} 天)")
            if args.multi_column:
                logger.info("多维特征提取模式：使用所有数值列")
            if_anomalies = detect_isolation_forest(
                df=df,
                column=args.column,
                contamination=args.contamination,
                lookback_days=args.lookback_days,
                use_all_columns=args.multi_column
            )
            
            output = format_output(
                anomalies=if_anomalies,
                method='isolation-forest',
                column=args.column,
                contamination=args.contamination
            )
            print(output)
            all_anomalies.extend(if_anomalies)
            logger.info(f"Isolation Forest 检测完成，发现 {len(if_anomalies)} 个异常")
        
        # 保存到文件
        if args.output:
            save_output(all_anomalies, args.output)
            logger.info(f"结果已保存到: {args.output}")
        
        logger.info(f"检测完成，共发现 {len(all_anomalies)} 个异常")
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"检测失败: {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
