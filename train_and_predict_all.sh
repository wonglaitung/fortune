#!/bin/bash

# 机器学习交易模型 - 完整训练和预测脚本
# 用于训练1天、5天、20天后的涨跌预测模型，并进行预测

echo "=========================================="
echo "🚀 机器学习交易模型 - 完整训练和预测"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "=========================================="
echo "📊 第一阶段: 训练模型"
echo "=========================================="
echo ""

# 训练次日涨跌模型（预测1天后）
echo "🌳 [1/3] 训练次日涨跌模型 (horizon=1)..."
python3 ml_trading_model.py --mode train --horizon 1 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 训练次日涨跌模型失败"
    exit 1
fi
echo "✅ 次日涨跌模型训练完成"
echo ""

# 训练一周涨跌模型（预测5天后）
echo "🌳 [2/3] 训练一周涨跌模型 (horizon=5)..."
python3 ml_trading_model.py --mode train --horizon 5 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 训练一周涨跌模型失败"
    exit 1
fi
echo "✅ 一周涨跌模型训练完成"
echo ""

# 训练一个月涨跌模型（预测20天后）
echo "🌳 [3/3] 训练一个月涨跌模型 (horizon=20)..."
python3 ml_trading_model.py --mode train --horizon 20 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 训练一个月涨跌模型失败"
    exit 1
fi
echo "✅ 一个月涨跌模型训练完成"
echo ""

echo "=========================================="
echo "🔮 第二阶段: 预测涨跌"
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
echo "✅ 所有任务完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - 模型文件: data/ml_trading_model_*.pkl"
echo "  - 特征重要性: data/ml_trading_model_*_importance.csv"
echo "  - 预测结果: data/ml_trading_model_*_predictions.csv"
echo "  - 模型对比: data/ml_trading_model_comparison.csv"
echo "  - 可解释性报告: output/gbdt_feature_importance.csv"
echo "  - LR系数: output/lr_leaf_coefficients.csv"
echo "  - ROC曲线: output/roc_curve.png"
echo ""
echo "💡 提示: 查看预测结果文件以获取详细的股票涨跌预测"
echo "=========================================="