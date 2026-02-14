#!/bin/bash

# 综合分析自动化脚本
# 1. 调用hsi_email.py生成大模型建议（使用force参数）
# 2. 调用ml_trading_model.py生成20天预测
# 3. 调用comprehensive_analysis.py进行综合分析

echo "=========================================="
echo "🚀 综合分析自动化流程"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 步骤1: 调用hsi_email.py生成大模型建议（使用force参数）
echo "=========================================="
echo "📊 步骤 1/4: 生成大模型建议"
echo "=========================================="
echo ""
python3 hsi_email.py --force
if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败: 生成大模型建议失败"
    exit 1
fi
echo "✅ 步骤1完成: 大模型建议已生成"
echo ""

# 步骤2: 训练20天模型
echo "=========================================="
echo "📊 步骤 2/4: 训练20天模型"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败: 训练20天模型失败"
    exit 1
fi
echo "✅ 步骤2完成: 20天模型训练完成"
echo ""

# 步骤3: 调用ml_trading_model.py生成20天预测
echo "=========================================="
echo "📊 步骤 3/4: 生成20天预测"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type both --model-path data/ml_trading_model.pkl
if [ $? -ne 0 ]; then
    echo "❌ 步骤3失败: 生成20天预测失败"
    exit 1
fi
echo "✅ 步骤3完成: 20天预测已生成"
echo ""

# 步骤4: 调用comprehensive_analysis.py进行综合分析
echo "=========================================="
echo "📊 步骤 4/4: 综合分析"
echo "=========================================="
echo ""
python3 comprehensive_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤4失败: 综合分析失败"
    exit 1
fi
echo "✅ 步骤4完成: 综合分析已生成"
echo ""

echo "=========================================="
echo "✅ 所有步骤完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - 大模型建议: data/llm_recommendations_YYYY-MM-DD.txt"
echo "  - ML预测结果: data/ml_predictions_20d_YYYY-MM-DD.txt"
echo "  - 综合买卖建议: data/comprehensive_recommendations_YYYY-MM-DD.txt"
echo ""
echo "💡 提示: 查看综合买卖建议了解最终投资建议"
echo "=========================================="