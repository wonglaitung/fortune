#!/bin/bash

# 机器学习交易模型 - 20天预测模型训练和预测脚本
# 用于训练20天后的涨跌预测模型，并进行预测
# 支持历史回测功能

echo "=========================================="
echo "🚀 机器学习交易模型 - 20天预测"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 解析命令行参数
MODE="current"  # 默认模式：当前日期
PREDICT_DATE=""
START_DATE=""
END_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --backtest)
            MODE="backtest"
            shift
            ;;
        --predict-date)
            PREDICT_DATE="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--backtest] [--predict-date YYYY-MM-DD] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]"
            exit 1
            ;;
    esac
done

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 显示运行模式
if [ "$MODE" = "backtest" ]; then
    echo "📊 运行模式: 历史回测"
    if [ -n "$PREDICT_DATE" ]; then
        echo "📅 预测日期: $PREDICT_DATE"
    fi
    if [ -n "$START_DATE" ]; then
        echo "📅 训练起始日期: $START_DATE"
    fi
    if [ -n "$END_DATE" ]; then
        echo "📅 训练结束日期: $END_DATE"
    fi
else
    echo "📊 运行模式: 当前日期"
fi
echo ""

# 构建训练参数
TRAIN_PARAMS=""
if [ -n "$START_DATE" ]; then
    TRAIN_PARAMS="$TRAIN_PARAMS --start-date $START_DATE"
fi
if [ -n "$END_DATE" ]; then
    TRAIN_PARAMS="$TRAIN_PARAMS --end-date $END_DATE"
fi

# 构建预测参数
PREDICT_PARAMS=""
if [ -n "$PREDICT_DATE" ]; then
    PREDICT_PARAMS="$PREDICT_PARAMS --predict-date $PREDICT_DATE"
fi

echo "=========================================="
echo "📊 第一阶段: 训练20天预测模型"
echo "=========================================="
echo ""

# 训练一个月涨跌模型（预测20天后）
echo "🌳 训练20天涨跌模型 (horizon=20)..."
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type both --model-path data/ml_trading_model.pkl $TRAIN_PARAMS
if [ $? -ne 0 ]; then
    echo "❌ 训练20天涨跌模型失败"
    exit 1
fi
echo "✅ 20天涨跌模型训练完成"
echo ""

echo "=========================================="
echo "🔮 第二阶段: 预测20天涨跌"
echo "=========================================="
echo ""

# 预测一个月涨跌（基于指定日期或今天的数据预测20天后）
echo "📈 预测20天涨跌 (horizon=20)..."
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type both --model-path data/ml_trading_model.pkl $PREDICT_PARAMS
if [ $? -ne 0 ]; then
    echo "❌ 预测20天涨跌失败"
    exit 1
fi
echo "✅ 20天涨跌预测完成"
echo ""

echo "=========================================="
echo "✅ 所有任务完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - 模型文件: data/ml_trading_model_*_20d.pkl"
echo "  - 特征重要性: data/ml_trading_model_*_20d_importance.csv"
echo "  - 预测结果: data/ml_trading_model_*_20d_predictions_*.csv"
echo "  - 模型对比: data/ml_trading_model_comparison.csv"
echo "  - 预测结果文本: data/ml_predictions_20d_YYYY-MM-DD.txt"
echo "  - 可解释性报告: output/gbdt_feature_importance.csv"
echo "  - LR系数: output/lr_leaf_coefficients.csv"
echo "  - ROC曲线: output/roc_curve.png"
echo ""
echo "💡 使用提示:"
echo "  - 当前日期预测: ./train_and_predict_all.sh"
echo "  - 历史日期预测: ./train_and_predict_all.sh --predict-date 2026-01-15"
echo "  - 限制训练数据: ./train_and_predict_all.sh --start-date 2024-01-01 --end-date 2024-12-31"
echo "  - 完整历史回测: ./train_and_predict_all.sh --backtest --start-date 2024-01-01 --end-date 2024-12-31 --predict-date 2024-12-31"
echo "=========================================="