#!/bin/bash

# 机器学习交易模型 - 批量历史回测脚本
# 用于批量预测多个历史日期，进行历史回测验证

echo "=========================================="
echo "📊 机器学习交易模型 - 批量历史回测"
echo "=========================================="
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 解析命令行参数
DAYS_BACK=10  # 默认回测最近10个交易日
HORIZON=1     # 默认预测周期（1=次日，5=一周，20=一个月）
SINGLE_DATE=""  # 单个日期回测

while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS_BACK="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --date)
            SINGLE_DATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--days N] [--horizon 1|5|20] [--date YYYY-MM-DD]"
            echo ""
            echo "参数说明:"
            echo "  --days N       回测最近N个交易日（默认：10）"
            echo "  --horizon 1|5|20 预测周期（默认：1）"
            echo "  --date YYYY-MM-DD 指定单个日期回测"
            echo ""
            echo "示例:"
            echo "  $0                    # 回测最近10个交易日的次日涨跌"
            echo "  $0 --days 20          # 回测最近20个交易日"
            echo "  $0 --horizon 5        # 回测最近10个交易日的一周涨跌"
            echo "  $0 --date 2026-01-15  # 回测2026-01-15的次日涨跌"
            exit 1
            ;;
    esac
done

# 检查模型文件是否存在
MODEL_FILE="data/ml_trading_model_lgbm_${HORIZON}d.pkl"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "💡 请先运行 train_and_predict_all.sh 训练模型"
    echo "   需要的模型文件: $MODEL_FILE"
    exit 1
fi

echo "✅ 找到训练好的模型文件: $MODEL_FILE"
echo ""

# 显示回测配置
if [ -n "$SINGLE_DATE" ]; then
    echo "📊 回测配置:"
    echo "   - 模式: 单日期回测"
    echo "   - 日期: $SINGLE_DATE"
    echo "   - 预测周期: ${HORIZON}天"
else
    echo "📊 回测配置:"
    echo "   - 模式: 批量回测"
    echo "   - 回测天数: 最近${DAYS_BACK}个交易日"
    echo "   - 预测周期: ${HORIZON}天"
fi
echo ""

# 创建回测结果目录
BACKTEST_DIR="backtest_results"
mkdir -p "$BACKTEST_DIR"

echo "=========================================="
echo "📈 开始历史回测"
echo "=========================================="
echo ""

# 单日期回测
if [ -n "$SINGLE_DATE" ]; then
    echo "📅 回测日期: $SINGLE_DATE"
    
    python3 ml_trading_model.py --mode predict --horizon $HORIZON --model-type both --model-path data/ml_trading_model.pkl --predict-date "$SINGLE_DATE"
    
    if [ $? -eq 0 ]; then
        # 复制预测结果到回测目录
        cp data/ml_trading_model_lgbm_predictions_${HORIZON}d.csv "$BACKTEST_DIR/prediction_${SINGLE_DATE}_lgbm.csv"
        cp data/ml_trading_model_gbdt_lr_predictions_${HORIZON}d.csv "$BACKTEST_DIR/prediction_${SINGLE_DATE}_gbdt_lr.csv"
        
        echo "✅ 回测完成: $SINGLE_DATE"
        echo "   结果已保存到: $BACKTEST_DIR/"
    else
        echo "❌ 回测失败: $SINGLE_DATE"
        exit 1
    fi
    
    echo ""
    
# 批量回测
else
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for i in $(seq 1 $DAYS_BACK); do
        # 计算日期（跳过周末）
        date_str=$(date -d "$i days ago" '+%Y-%m-%d' 2>/dev/null)
        
        if [ $? -ne 0 ]; then
            echo "⚠️  跳过无效日期: 第${i}天"
            continue
        fi
        
        # 检查是否为周末
        day_of_week=$(date -d "$date_str" +%u 2>/dev/null)
        if [ "$day_of_week" -eq 6 ] || [ "$day_of_week" -eq 7 ]; then
            echo "⏭️  跳过周末: $date_str"
            continue
        fi
        
        echo "📅 [$i/$DAYS_BACK] 回测日期: $date_str"
        
        python3 ml_trading_model.py --mode predict --horizon $HORIZON --model-type both --model-path data/ml_trading_model.pkl --predict-date "$date_str"
        
        if [ $? -eq 0 ]; then
            # 复制预测结果到回测目录
            cp data/ml_trading_model_lgbm_predictions_${HORIZON}d.csv "$BACKTEST_DIR/prediction_${date_str}_lgbm.csv"
            cp data/ml_trading_model_gbdt_lr_predictions_${HORIZON}d.csv "$BACKTEST_DIR/prediction_${date_str}_gbdt_lr.csv"
            
            echo "✅ 回测成功: $date_str"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ 回测失败: $date_str"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
        echo ""
    done
    
    echo "=========================================="
    echo "📊 回测统计"
    echo "=========================================="
    echo "   总尝试: $((SUCCESS_COUNT + FAIL_COUNT))"
    echo "   成功: $SUCCESS_COUNT"
    echo "   失败: $FAIL_COUNT"
    echo "   成功率: $(awk "BEGIN {printf \"%.1f\", ($SUCCESS_COUNT/($SUCCESS_COUNT+$FAIL_COUNT))*100}")%"
    echo ""
fi

echo "=========================================="
echo "✅ 历史回测完成！"
echo "=========================================="
echo "📅 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 回测结果文件:"
echo "   目录: $BACKTEST_DIR/"
echo "   文件: prediction_YYYY-MM-DD_lgbm.csv"
echo "   文件: prediction_YYYY-MM-DD_gbdt_lr.csv"
echo ""
echo "💡 下一步:"
echo "   1. 查看回测结果: ls -lh $BACKTEST_DIR/"
echo "   2. 分析预测准确性: 对比预测结果与实际价格"
echo "   3. 生成回测报告: 使用 Python 脚本分析预测准确率"
echo "=========================================="