#!/bin/bash

# 综合分析自动化脚本
# 1. 调用hsi_email.py生成大模型建议（使用force参数）
# 2. 训练CatBoost 20天模型
# 3. 生成CatBoost单模型预测
# 4. 调用comprehensive_analysis.py进行综合分析

# 获取项目根目录（脚本所在目录的父目录）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 切换到项目目录
cd "$PROJECT_DIR"

echo "=========================================="
echo "🚀 综合分析自动化流程（使用 CatBoost 单模型）"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📍 项目根目录: $PROJECT_DIR"
echo "📍 当前工作目录: $(pwd)"
echo ""

# 步骤1: 训练 CatBoost 20天模型
echo "=========================================="
echo "📊 步骤 1/4: 训练 CatBoost 20天模型（使用全量特征）"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败: 训练20天CatBoost模型失败"
    exit 1
fi
echo "✅ CatBoost模型训练完成"
echo ""

echo "✅ 步骤1完成: CatBoost模型训练完成（使用全量特征）"
echo ""

# 步骤2: 生成 CatBoost 单模型预测
echo "=========================================="
echo "📊 步骤 2/4: 生成 CatBoost 单模型预测"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost
if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败: 生成 CatBoost 预测失败"
    exit 1
fi
echo "✅ CatBoost 预测完成"
echo ""

# 步骤3: 调用hsi_email.py生成大模型建议（使用force参数，不发送邮件）
echo "=========================================="
echo "📊 步骤 3/4: 生成大模型建议"
echo "=========================================="
echo ""
python3 hsi_email.py --force --no-email
if [ $? -ne 0 ]; then
    echo "❌ 步骤3失败: 生成大模型建议失败"
    exit 1
fi
echo "✅ 步骤3完成: 大模型建议已生成"
echo ""

# 步骤4: 调用comprehensive_analysis.py进行综合分析
echo "=========================================="
echo "📊 步骤 4/4: 综合分析"
echo "=========================================="
echo ""
# 获取步骤3生成的大模型建议文件（使用最新日期）
LLM_FILE=$(ls -t data/llm_recommendations_*.txt 2>/dev/null | head -1)
if [ -z "$LLM_FILE" ]; then
    echo "⚠️  警告: 未找到大模型建议文件，跳过综合分析"
else
    echo "📊 使用大模型建议文件: $LLM_FILE"
    python3 comprehensive_analysis.py --llm-file "$LLM_FILE"
    if [ $? -ne 0 ]; then
        echo "❌ 步骤4失败: 综合分析失败"
        exit 1
    fi
    echo "✅ 步骤4完成: 综合分析已生成"
fi
echo ""

echo "=========================================="
echo "✅ 所有步骤完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - 大模型建议: $(ls -t data/llm_recommendations_*.txt 2>/dev/null | head -1)"
echo "  - ML预测结果: $(ls -t data/ml_trading_model_catboost_predictions_20d.csv 2>/dev/null | head -1)"
echo "  - 综合买卖建议: $(ls -t data/comprehensive_recommendations_*.txt 2>/dev/null | head -1)"
echo ""
echo "💡 提示: 查看综合买卖建议了解最终投资建议"
echo "=========================================="
