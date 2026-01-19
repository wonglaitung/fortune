#!/bin/bash

# 机器学习交易模型 - 仅预测脚本
# 用于预测1天、5天、20天后的涨跌（假设模型已经训练好）

echo "=========================================="
echo "🔮 机器学习交易模型 - 预测涨跌"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查模型文件是否存在
if [ ! -f "data/ml_trading_model_lgbm.pkl" ]; then
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "💡 请先运行 train_and_predict_all.sh 训练模型"
    exit 1
fi

echo "✅ 找到训练好的模型文件"
echo ""

echo "=========================================="
echo "📈 预测涨跌"
echo "=========================================="
echo ""

# 预测次日涨跌（基于今天的数据预测1天后）
echo "📈 [1/3] 预测次日涨跌 (horizon=1)..."
python3 ml_trading_model.py --mode predict --horizon 1 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 预测次日涨跌失败"
    exit 1
fi
echo "✅ 次日涨跌预测完成"
echo ""

# 预测一周涨跌（基于今天的数据预测5天后）
echo "📈 [2/3] 预测一周涨跌 (horizon=5)..."
python3 ml_trading_model.py --mode predict --horizon 5 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 预测一周涨跌失败"
    exit 1
fi
echo "✅ 一周涨跌预测完成"
echo ""

# 预测一个月涨跌（基于今天的数据预测20天后）
echo "📈 [3/3] 预测一个月涨跌 (horizon=20)..."
python3 ml_trading_model.py --mode predict --horizon 20 --model-type both --model-path data/ml_trading_model.pkl
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
echo "  - data/ml_trading_model_lgbm_predictions.csv"
echo "  - data/ml_trading_model_gbdt_lr_predictions.csv"
echo "  - data/ml_trading_model_comparison.csv"
echo ""
echo "💡 提示: 查看预测结果文件以获取详细的股票涨跌预测"
echo "=========================================="