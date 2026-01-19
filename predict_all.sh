#!/bin/bash

# 机器学习交易模型 - 仅预测脚本
# 用于预测1天、5天、20天后的涨跌（假设模型已经训练好）
# 支持历史回测功能

echo "=========================================="
echo "🔮 机器学习交易模型 - 预测涨跌"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 解析命令行参数
PREDICT_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --predict-date)
            PREDICT_DATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--predict-date YYYY-MM-DD]"
            exit 1
            ;;
    esac
done

# 检查模型文件是否存在（检查任一周期模型）
if [ ! -f "data/ml_trading_model_lgbm_1d.pkl" ] && [ ! -f "data/ml_trading_model_lgbm_5d.pkl" ] && [ ! -f "data/ml_trading_model_lgbm_20d.pkl" ]; then
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "💡 请先运行 train_and_predict_all.sh 训练模型"
    exit 1
fi

echo "✅ 找到训练好的模型文件"
echo ""

# 显示预测模式
if [ -n "$PREDICT_DATE" ]; then
    echo "📊 预测模式: 历史日期回测"
    echo "📅 预测日期: $PREDICT_DATE"
else
    echo "📊 预测模式: 当前日期"
fi
echo ""

# 构建预测参数
PREDICT_PARAMS=""
if [ -n "$PREDICT_DATE" ]; then
    PREDICT_PARAMS="$PREDICT_PARAMS --predict-date $PREDICT_DATE"
fi

echo "=========================================="
echo "📈 预测涨跌"
echo "=========================================="
echo ""

# 预测次日涨跌（基于指定日期或今天的数据预测1天后）
echo "📈 [1/3] 预测次日涨跌 (horizon=1)..."
python3 ml_trading_model.py --mode predict --horizon 1 --model-type both --model-path data/ml_trading_model.pkl $PREDICT_PARAMS
if [ $? -ne 0 ]; then
    echo "❌ 预测次日涨跌失败"
    exit 1
fi
echo "✅ 次日涨跌预测完成"
echo ""

# 预测一周涨跌（基于指定日期或今天的数据预测5天后）
echo "📈 [2/3] 预测一周涨跌 (horizon=5)..."
python3 ml_trading_model.py --mode predict --horizon 5 --model-type both --model-path data/ml_trading_model.pkl $PREDICT_PARAMS
if [ $? -ne 0 ]; then
    echo "❌ 预测一周涨跌失败"
    exit 1
fi
echo "✅ 一周涨跌预测完成"
echo ""

# 预测一个月涨跌（基于指定日期或今天的数据预测20天后）
echo "📈 [3/3] 预测一个月涨跌 (horizon=20)..."
python3 ml_trading_model.py --mode predict --horizon 20 --model-type both --model-path data/ml_trading_model.pkl $PREDICT_PARAMS
if [ $? -ne 0 ]; then
    echo "❌ 预测一个月涨跌失败"
    exit 1
fi
echo "✅ 一个月涨跌预测完成"
echo ""

echo "=========================================="
echo "✅ 预测完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的预测结果文件:"
if [ -n "$PREDICT_DATE" ]; then
    echo "  - data/ml_trading_model_lgbm_predictions_1d.csv (预测日期: $PREDICT_DATE)"
    echo "  - data/ml_trading_model_lgbm_predictions_5d.csv (预测日期: $PREDICT_DATE)"
    echo "  - data/ml_trading_model_lgbm_predictions_20d.csv (预测日期: $PREDICT_DATE)"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_1d.csv (预测日期: $PREDICT_DATE)"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_5d.csv (预测日期: $PREDICT_DATE)"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_20d.csv (预测日期: $PREDICT_DATE)"
else
    echo "  - data/ml_trading_model_lgbm_predictions_1d.csv"
    echo "  - data/ml_trading_model_lgbm_predictions_5d.csv"
    echo "  - data/ml_trading_model_lgbm_predictions_20d.csv"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_1d.csv"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_5d.csv"
    echo "  - data/ml_trading_model_gbdt_lr_predictions_20d.csv"
fi
echo ""
echo "💡 使用提示:"
echo "  - 当前日期预测: ./predict_all.sh"
echo "  - 历史日期预测: ./predict_all.sh --predict-date 2026-01-15"
echo "  - 批量历史回测: ./backtest_batch.sh (需单独创建)"
echo "=========================================="