#!/bin/bash

# A股综合分析自动化脚本
# ⚠️ 建议在A股收市后（15:00 CST）运行

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 切换到项目目录
cd "$PROJECT_DIR"

echo "=========================================="
echo "🚀 A股综合分析自动化流程"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📍 项目根目录: $PROJECT_DIR"
echo ""

# 步骤1: 训练 CatBoost 三周期模型
echo "=========================================="
echo "📊 步骤 1/3: 训练 CatBoost 模型"
echo "=========================================="
echo ""

TRAIN_SUCCESS=0
TRAIN_FAILED=0
for horizon in 20; do
    echo "  🔄 训练 ${horizon}d 模型..."
    python3 a_stock_ml_model.py --mode train --horizon $horizon --use-feature-selection
    if [ $? -ne 0 ]; then
        echo "  ⚠️ 训练 ${horizon}d 模型失败"
        TRAIN_FAILED=$((TRAIN_FAILED + 1))
    else
        echo "  ✅ ${horizon}d 模型训练完成"
        TRAIN_SUCCESS=$((TRAIN_SUCCESS + 1))
    fi
done

echo ""
echo "✅ 步骤1完成: $TRAIN_SUCCESS 个模型训练成功"
echo ""

# 步骤2: 生成预测
echo "=========================================="
echo "📊 步骤 2/3: 生成预测"
echo "=========================================="
echo ""

# 清除特征缓存
echo "🗑️ 清除特征缓存..."
rm -rf data/a_stock_feature_cache/*.pkl 2>/dev/null
echo "✅ 特征缓存已清除"
echo ""

PREDICT_SUCCESS=0
for horizon in 20; do
    echo "  🔄 生成 ${horizon}d 预测..."
    python3 a_stock_ml_model.py --mode predict --horizon $horizon --use-feature-selection
    if [ $? -ne 0 ]; then
        echo "  ⚠️ 生成 ${horizon}d 预测失败"
    else
        echo "  ✅ ${horizon}d 预测完成"
        PREDICT_SUCCESS=$((PREDICT_SUCCESS + 1))
    fi
done

echo ""
echo "✅ 步骤2完成: $PREDICT_SUCCESS 个预测成功"
echo ""

# 步骤3: 生成AI报告
echo "=========================================="
echo "📊 步骤 3/4: 生成AI买卖建议"
echo "=========================================="
echo ""

python3 a_stock_email.py --force
if [ $? -ne 0 ]; then
    echo "⚠️ AI报告生成失败（继续执行）"
else
    echo "✅ AI报告生成完成"
fi
echo ""

# 步骤4: 生成预测报告
echo "=========================================="
echo "📊 步骤 4/4: 生成预测报告"
echo "=========================================="
echo ""

python3 a_stock_prediction.py --horizon 20 --no-email
if [ $? -ne 0 ]; then
    echo "⚠️ 报告生成失败"
else
    echo "✅ 报告生成完成"
fi
echo ""

echo "=========================================="
echo "✅ 所有步骤完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - AI买卖建议: $(ls -t data/a_stock_llm_recommendations_*.txt 2>/dev/null | head -1)"
echo "  - 预测报告: $(ls -t data/a_stock_prediction_*.json 2>/dev/null | head -1)"
echo ""
echo "💡 提示: 查看AI买卖建议了解详细投资建议"
echo "=========================================="
