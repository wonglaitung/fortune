#!/bin/bash

# A股综合分析自动化脚本
# ⚠️ 建议在A股收市后（15:00 CST）运行
# 1. 训练 CatBoost 三周期模型（1d, 5d, 20d）
# 2. 生成 CatBoost 三周期预测
# 3. 调用 a_stock_email.py 生成大模型建议
# 4. 调用 a_stock_comprehensive_analysis.py 进行综合分析

# 获取项目根目录（脚本所在目录的父目录）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 切换到项目目录
cd "$PROJECT_DIR"

echo "=========================================="
echo "🚀 A股综合分析自动化流程"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📍 项目根目录: $PROJECT_DIR"
echo "📍 当前工作目录: $(pwd)"
echo ""

# 加载环境变量（密钥）
if [ -f "set_key.sh" ]; then
    source set_key.sh
    echo "✅ 环境变量已加载"
else
    echo "⚠️ 未找到 set_key.sh，请确保 QWEN_API_KEY 已配置"
fi
echo ""

# 步骤1: 训练 CatBoost 三周期模型（1d, 5d, 20d）
echo "=========================================="
echo "📊 步骤 1/4: 训练 CatBoost 三周期模型（1d, 5d, 20d）"
echo "=========================================="
echo ""

TRAIN_SUCCESS=0
TRAIN_FAILED=0
#for horizon in 1 5 20; do
#    echo "  🔄 训练 ${horizon}d 模型..."
#    python3 a_stock_ml_model.py --mode train --horizon $horizon
#    if [ $? -ne 0 ]; then
#        echo "  ⚠️ 训练 ${horizon}d 模型失败（继续执行其他周期）"
#        TRAIN_FAILED=$((TRAIN_FAILED + 1))
#    else
#        echo "  ✅ ${horizon}d 模型训练完成"
#        TRAIN_SUCCESS=$((TRAIN_SUCCESS + 1))
#    fi
#done

#if [ $TRAIN_SUCCESS -eq 0 ]; then
#    echo "❌ 步骤1失败: 所有模型训练失败"
#    exit 1
#fi
echo ""
echo "✅ 步骤1完成: $TRAIN_SUCCESS/3 个模型训练成功（$TRAIN_FAILED 个失败）"
echo ""

# 步骤2: 生成 CatBoost 三周期预测
echo "=========================================="
echo "📊 步骤 2/4: 生成 CatBoost 三周期预测"
echo "=========================================="
echo ""

# 清除特征缓存
echo "🗑️ 清除特征缓存..."
rm -rf data/feature_cache/*.pkl
rm -rf data/a_stock_feature_cache/*.pkl 2>/dev/null
echo "✅ 特征缓存已清除"
echo ""

PREDICT_SUCCESS=0
PREDICT_FAILED=0
for horizon in 1 5 20; do
    echo "  🔄 生成 ${horizon}d 预测..."
    # 只预测核心股票（4只），用于邮件展示
    python3 a_stock_ml_model.py --mode predict --horizon $horizon --core-only
    if [ $? -ne 0 ]; then
        echo "  ⚠️ 生成 ${horizon}d 预测失败（继续执行其他周期）"
        PREDICT_FAILED=$((PREDICT_FAILED + 1))
    else
        echo "  ✅ ${horizon}d 预测完成"
        PREDICT_SUCCESS=$((PREDICT_SUCCESS + 1))
    fi
done

if [ $PREDICT_SUCCESS -eq 0 ]; then
    echo "❌ 步骤2失败: 所有预测生成失败"
    exit 1
fi
echo ""
echo "✅ 步骤2完成: $PREDICT_SUCCESS/3 个周期预测成功（$PREDICT_FAILED 个失败）"
echo ""

# 步骤3: 调用 a_stock_email.py 生成大模型建议
echo "=========================================="
echo "📊 步骤 3/4: 生成大模型建议"
echo "=========================================="
echo ""
python3 a_stock_email.py --force --no-email
if [ $? -ne 0 ]; then
    echo "⚠️ 大模型建议生成失败（继续执行）"
else
    echo "✅ 大模型建议已生成"
fi
echo ""

# 步骤4: 调用 a_stock_comprehensive_analysis.py 进行综合分析
echo "=========================================="
echo "📊 步骤 4/4: 综合分析"
echo "=========================================="
echo ""
# 获取步骤3生成的大模型建议文件（使用最新日期）
LLM_FILE=$(ls -t data/a_stock_llm_recommendations_*.txt 2>/dev/null | head -1)
if [ -z "$LLM_FILE" ]; then
    echo "⚠️ 警告: 未找到大模型建议文件，跳过综合分析"
else
    echo "📊 使用大模型建议文件: $LLM_FILE"
    # 默认发送邮件，如需禁用可添加 --no-email 参数
    python3 a_stock_comprehensive_analysis.py --llm-file "$LLM_FILE" --use-cached-predictions
    if [ $? -ne 0 ]; then
        echo "❌ 步骤4失败: 综合分析失败"
        exit 1
    fi
    echo "✅ 步骤4完成: 综合分析已生成（邮件已发送）"
fi
echo ""

echo "=========================================="
echo "✅ 所有步骤完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 生成的文件:"
echo "  - 大模型建议: $(ls -t data/a_stock_llm_recommendations_*.txt 2>/dev/null | head -1)"
echo "  - ML预测结果:"
echo "    - 1d: $(ls -t data/ml_trading_model_catboost_predictions_1d.csv 2>/dev/null | head -1)"
echo "    - 5d: $(ls -t data/ml_trading_model_catboost_predictions_5d.csv 2>/dev/null | head -1)"
echo "    - 20d: $(ls -t data/ml_trading_model_catboost_predictions_20d.csv 2>/dev/null | head -1)"
echo "  - 综合买卖建议: $(ls -t data/a_stock_comprehensive_recommendations_*.txt 2>/dev/null | head -1)"
echo ""
echo "💡 提示: 查看综合买卖建议了解最终投资建议"
echo "=========================================="
