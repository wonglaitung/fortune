#!/bin/bash
# 股票表现TOP 10排名分析自动化脚本
# 用法：
#   ./run_ranking_analysis.sh                              # 使用默认参数（上个月之前的一年）
#   ./run_ranking_analysis.sh 2024-01-01 2025-12-31         # 自定义日期范围
#   ./run_ranking_analysis.sh 2024-01-01 2025-12-31 markdown  # 自定义日期范围和输出格式

set -e  # 遇到错误立即退出

# 获取项目根目录（脚本所在目录的父目录）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认参数（自动计算上个月之前的一年）
if [ -z "$1" ]; then
    # 未提供日期参数，自动计算上个月之前的一年
    # 获取上个月的最后一天
    END_DATE=$(date -d "$(date +%Y-%m-01) -1 day" +%Y-%m-%d)
    # 获取一年前的同一天
    START_DATE=$(date -d "$END_DATE -1 year" +%Y-%m-%d)
else
    # 提供了日期参数，使用用户指定的日期
    START_DATE=$1
    END_DATE=${2:-$(date -d "$(date +%Y-%m-01) -1 day" +%Y-%m-%d)}
    OUTPUT_FORMAT=${3:-all}
fi

OUTPUT_FORMAT=${OUTPUT_FORMAT:-all}
OUTPUT_DIR="${PROJECT_DIR}/output"

echo "=========================================="
echo "股票表现TOP 10排名分析自动化脚本"
echo "=========================================="
echo "开始日期: $START_DATE"
echo "结束日期: $END_DATE"
echo "输出格式: $OUTPUT_FORMAT"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 切换到项目目录
cd "$PROJECT_DIR"

# 步骤1：运行回测（生成新的交易记录）
echo ""
echo "=========================================="
echo "步骤 1/2: 运行20天持有期回测"
echo "=========================================="
echo "回测日期范围: $START_DATE 至 $END_DATE"

# 运行回测（使用CatBoost 20天模型）
python3 ml_services/backtest_20d_horizon.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --horizon 20 \
    --confidence-threshold 0.55 \
    --use-feature-selection \
    --skip-feature-selection \
    --enable-dynamic-risk-control

echo "✅ 回测完成"

# 步骤2：查找最新生成的回测交易记录文件
echo ""
echo "=========================================="
echo "步骤 2/2: 查找回测交易记录文件"
echo "=========================================="

# 查找最新的交易记录文件
LATEST_TRADES_FILE=$(find output -name "backtest_20d_trades_*.csv" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_TRADES_FILE" ]; then
    echo "❌ 错误：未找到回测交易记录文件"
    exit 1
fi

echo "✅ 找到最新交易记录文件: $LATEST_TRADES_FILE"

# 步骤3：运行排名分析
echo ""
echo "=========================================="
echo "运行股票表现TOP 10排名分析"
echo "=========================================="
echo "使用交易记录: $LATEST_TRADES_FILE"

python3 ml_services/ranking_analysis.py \
    --trades-file "$LATEST_TRADES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --output-format "$OUTPUT_FORMAT"

echo ""
echo "=========================================="
echo "✅ 完整流程已完成！"
echo "=========================================="
echo ""
echo "执行摘要："
echo "  1. ✅ 回测已运行: $START_DATE 至 $END_DATE"
echo "  2. ✅ 交易记录文件: $LATEST_TRADES_FILE"
echo "  3. ✅ 排名分析已生成"
echo ""
echo "报告已保存到: $OUTPUT_DIR"
echo ""
echo "查看报告："
echo "  CSV: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.csv 2>/dev/null | head -1)"
echo "  JSON: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.json 2>/dev/null | head -1)"
echo "  Markdown: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.md 2>/dev/null | head -1)"
