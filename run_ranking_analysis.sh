#!/bin/bash
# 股票表现TOP 10排名分析自动化脚本
# 用法：
#   ./run_ranking_analysis.sh                    # 使用默认参数（最新回测）
#   ./run_ranking_analysis.sh backtest_20d_trades_20260305_115839.csv    # 指定回测文件
#   ./run_ranking_analysis.sh backtest_20d_trades_20260305_115839.csv markdown    # 指定输出格式

set -e  # 遇到错误立即退出

# 获取脚本所在目录（项目根目录）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
if [ -z "$1" ]; then
    # 未提供回测文件，查找最新的回测交易记录文件
    echo "未指定回测文件，查找最新的回测交易记录..."
    TRADES_FILE=$(find "${PROJECT_DIR}/output" -name "backtest_20d_trades_*.csv" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$TRADES_FILE" ]; then
        echo "❌ 错误：未找到回测交易记录文件"
        echo "请先运行回测或指定回测文件路径"
        exit 1
    fi
else
    # 提供了回测文件，使用用户指定的文件
    TRADES_FILE="${PROJECT_DIR}/output/$1"
fi

OUTPUT_FORMAT=${2:-all}
OUTPUT_DIR="${PROJECT_DIR}/output"

echo "=========================================="
echo "股票表现TOP 10排名分析自动化脚本"
echo "=========================================="
echo "回测文件: $TRADES_FILE"
echo "输出格式: $OUTPUT_FORMAT"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 检查回测文件是否存在
if [ ! -f "$TRADES_FILE" ]; then
    echo "❌ 错误：回测文件不存在: $TRADES_FILE"
    exit 1
fi

# 切换到项目目录
cd "$PROJECT_DIR"

# 运行排名分析
echo ""
echo "=========================================="
echo "运行股票表现TOP 10排名分析"
echo "=========================================="
echo "使用回测文件: $TRADES_FILE"

python3 ml_services/ranking_analysis.py \
    --trades-file "$TRADES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --output-format "$OUTPUT_FORMAT"

echo ""
echo "=========================================="
echo "✅ 排名分析完成！"
echo "=========================================="
echo ""
echo "报告已保存到: $OUTPUT_DIR"
echo ""
echo "查看报告："
echo "  CSV: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.csv 2>/dev/null | head -1)"
echo "  JSON: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.json 2>/dev/null | head -1)"
echo "  Markdown: $(ls -t ${OUTPUT_DIR}/ranking_analysis_*.md 2>/dev/null | head -1)"
