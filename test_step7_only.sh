#!/bin/bash

# 测试步骤7：只运行汇总对比报告生成

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 输出目录
OUTPUT_DIR="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 回测配置
HORIZON=20
CONFIDENCE_THRESHOLD=0.55

# 模型列表
BASE_MODELS=("lgbm" "gbdt" "catboost")
FUSION_MODELS=("ensemble_weighted" "ensemble_average" "ensemble_voting" "ensemble_dynamic-market" "ensemble_advanced-dynamic")

print_header "步骤7: 生成汇总对比报告"

REPORT_FILE="$OUTPUT_DIR/model_comparison_report_${TIMESTAMP}.txt"

echo "================================================================================" > "$REPORT_FILE"
echo "模型对比回测报告" >> "$REPORT_FILE"
echo "================================================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "预测周期: ${HORIZON}天" >> "$REPORT_FILE"
echo "置信度阈值: ${CONFIDENCE_THRESHOLD}" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "================================================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 提取所有回测结果
print_info "提取回测结果..."

echo "## 一、基本模型性能对比" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 模型 | 平均总收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 | 优秀股票数量 |" >> "$REPORT_FILE"
echo "|------|-------------|-----------|---------|---------|------|-------------|" >> "$REPORT_FILE"

for model in "${BASE_MODELS[@]}"; do
    print_info "处理 ${model^^} 模型..."
    # 查找最新的回测结果文件
    LATEST_SUMMARY=$(ls -t "$OUTPUT_DIR/batch_backtest_summary_${model}_${HORIZON}d_"*.txt 2>/dev/null | head -1)

    if [ -n "$LATEST_SUMMARY" ]; then
        print_info "找到文件: $LATEST_SUMMARY"
        
        # 提取关键指标
        TOTAL_RETURN=$(grep "平均总收益率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 平均总收益率: ${TOTAL_RETURN}%"

        # 从"收益分布"部分提取年化收益率中位数
        ANNUAL_RETURN=$(grep "收益率中位数" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 年化收益率: ${ANNUAL_RETURN}%"

        SHARPE=$(grep "平均夏普比率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+' | head -1)
        print_info "  - 夏普比率: ${SHARPE}"
        
        MAX_DRAWDOWN=$(grep "平均最大回撤" "$LATEST_SUMMARY" | grep -oP '[-+]?\d+\.\d+(?=%)' | head -1)
        print_info "  - 最大回撤: ${MAX_DRAWDOWN}"
        
        WIN_RATE=$(grep "平均胜率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 胜率: ${WIN_RATE}%"

        # 统计优秀股票数量（收益率 > 50%）
        EXCELLENT=$(grep "总收益率:" "$LATEST_SUMMARY" | awk -F': ' '{if ($2+0 > 50) count++} END {print count+0}')
        print_info "  - 优秀股票数量: ${EXCELLENT} 只"

        echo "| ${model^^} | ${TOTAL_RETURN}% | ${ANNUAL_RETURN}% | ${SHARPE} | ${MAX_DRAWDOWN} | ${WIN_RATE}% | ${EXCELLENT} 只 |" >> "$REPORT_FILE"
        print_success "${model^^} 模型数据提取完成"
    else
        print_warning "未找到 ${model^^} 模型回测结果文件"
        echo "| ${model^^} | N/A | N/A | N/A | N/A | N/A | N/A |" >> "$REPORT_FILE"
    fi
done

echo "" >> "$REPORT_FILE"
echo "## 二、融合模型性能对比" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 模型 | 平均总收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 | 优秀股票数量 |" >> "$REPORT_FILE"
echo "|------|-------------|-----------|---------|---------|------|-------------|" >> "$REPORT_FILE"

# 获取所有融合模型文件（按时间倒序）
print_info "查找融合模型文件..."
FUSION_FILES=$(ls -t "$OUTPUT_DIR/batch_backtest_summary_ensemble_${HORIZON}d_"*.txt 2>/dev/null)
FUSION_ARRAY=($FUSION_FILES)
print_info "找到 ${#FUSION_ARRAY[@]} 个融合模型文件"

# 为每个融合模型找到对应的summary文件
declare -A FUSION_FILE_MAP
for fusion in "${FUSION_MODELS[@]}"; do
    # 从文件名中提取融合方法（ensemble_weighted -> weighted）
    FUSION_METHOD=$(echo "$fusion" | sed 's/ensemble_//')
    
    # 查找包含该融合方法的最新JSON文件
    LATEST_FUSION_JSON=$(ls -t "$OUTPUT_DIR/batch_backtest_${fusion}_${HORIZON}d_"*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_FUSION_JSON" ]; then
        # 从JSON文件名中提取时间戳
        TIMESTAMP=$(basename "$LATEST_FUSION_JSON" | sed "s/batch_backtest_${fusion}_${HORIZON}d_//" | sed 's/\.json//')
        
        # 构造对应的summary文件名
        SUMMARY_FILE="$OUTPUT_DIR/batch_backtest_summary_ensemble_${HORIZON}d_${TIMESTAMP}.txt"
        
        if [ -f "$SUMMARY_FILE" ]; then
            FUSION_FILE_MAP[$fusion]=$SUMMARY_FILE
            print_info "  - ${fusion^^}: $SUMMARY_FILE"
        else
            print_warning "  - ${fusion^^}: 找不到对应的summary文件 ($SUMMARY_FILE)"
        fi
    fi
done

# 按顺序处理融合模型
for fusion in "${FUSION_MODELS[@]}"; do
    print_info "处理 ${fusion^^} 模型..."
    
    if [[ -v FUSION_FILE_MAP[$fusion] ]]; then
        LATEST_SUMMARY="${FUSION_FILE_MAP[$fusion]}"
        print_info "  - 使用文件: $LATEST_SUMMARY"

        # 提取关键指标
        TOTAL_RETURN=$(grep "平均总收益率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 平均总收益率: ${TOTAL_RETURN}%"

        # 从"收益分布"部分提取年化收益率中位数
        ANNUAL_RETURN=$(grep "收益率中位数" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 年化收益率: ${ANNUAL_RETURN}%"

        # 夏普比率没有百分号
        SHARPE=$(grep "平均夏普比率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+' | head -1)
        print_info "  - 夏普比率: ${SHARPE}"
        
        MAX_DRAWDOWN=$(grep "平均最大回撤" "$LATEST_SUMMARY" | grep -oP '[-+]?\d+\.\d+(?=%)' | head -1)
        print_info "  - 最大回撤: ${MAX_DRAWDOWN}"
        
        WIN_RATE=$(grep "平均胜率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        print_info "  - 胜率: ${WIN_RATE}%"

        # 统计优秀股票数量（收益率 > 50%）
        EXCELLENT=$(grep "总收益率:" "$LATEST_SUMMARY" | awk -F': ' '{if ($2+0 > 50) count++} END {print count+0}')
        print_info "  - 优秀股票数量: ${EXCELLENT} 只"

        echo "| ${fusion^^} | ${TOTAL_RETURN}% | ${ANNUAL_RETURN}% | ${SHARPE} | ${MAX_DRAWDOWN} | ${WIN_RATE}% | ${EXCELLENT} 只 |" >> "$REPORT_FILE"
        print_success "${fusion^^} 模型数据提取完成"
    else
        print_warning "未找到 ${fusion^^} 模型回测结果文件"
        echo "| ${fusion^^} | N/A | N/A | N/A | N/A | N/A | N/A |" >> "$REPORT_FILE"
    fi
done

echo "" >> "$REPORT_FILE"
echo "## 三、综合排名（按年化收益率）" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 排名 | 模型 | 年化收益率 | 夏普比率 | 优秀股票数量 |" >> "$REPORT_FILE"
echo "|------|------|-----------|---------|-------------|" >> "$REPORT_FILE"

print_info "收集所有模型数据进行排名..."

# 收集所有模型数据
ALL_MODELS=("${BASE_MODELS[@]}" "${FUSION_MODELS[@]}")
RANK_DATA=()

for model in "${ALL_MODELS[@]}"; do
    print_info "收集 ${model^^} 模型数据..."
    if [[ "$model" == ensemble_* ]]; then
        # 融合模型：跳过，单独处理
        continue
    else
        # 基本模型文件查找
        LATEST_SUMMARY=$(ls -t "$OUTPUT_DIR/batch_backtest_summary_${model}_${HORIZON}d_"*.txt 2>/dev/null | head -1)
    fi

    if [ -n "$LATEST_SUMMARY" ]; then
        # 从"收益分布"部分提取年化收益率中位数
        ANNUAL_RETURN=$(grep "收益率中位数" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        SHARPE=$(grep "平均夏普比率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)

        # 统计优秀股票数量（收益率 > 50%）
        EXCELLENT=$(grep "总收益率:" "$LATEST_SUMMARY" | awk -F': ' '{if ($2+0 > 50) count++} END {print count+0}')

        if [ -n "$ANNUAL_RETURN" ]; then
            print_info "  - 添加到排名数据: ${ANNUAL_RETURN}% | ${SHARPE} | ${EXCELLENT} | ${model}"
            RANK_DATA+=("$ANNUAL_RETURN|$SHARPE|$EXCELLENT|$model")
        fi
    else
        print_warning "未找到 ${model^^} 模型数据"
    fi
done

print_info "基本模型数据收集完成，共 ${#RANK_DATA[@]} 条"

# 添加融合模型数据
print_info "添加融合模型数据..."
for fusion in "${FUSION_MODELS[@]}"; do
    print_info "收集 ${fusion^^} 模型数据..."
    
    if [[ -v FUSION_FILE_MAP[$fusion] ]]; then
        LATEST_SUMMARY="${FUSION_FILE_MAP[$fusion]}"

        # 从"收益分布"部分提取年化收益率中位数
        ANNUAL_RETURN=$(grep "收益率中位数" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
        SHARPE=$(grep "平均夏普比率" "$LATEST_SUMMARY" | grep -oP '\d+\.\d+' | head -1)

        # 统计优秀股票数量（收益率 > 50%）
        EXCELLENT=$(grep "总收益率:" "$LATEST_SUMMARY" | awk -F': ' '{if ($2+0 > 50) count++} END {print count+0}')

        if [ -n "$ANNUAL_RETURN" ]; then
            print_info "  - 添加到排名数据: ${ANNUAL_RETURN}% | ${SHARPE} | ${EXCELLENT} | ${fusion}"
            RANK_DATA+=("$ANNUAL_RETURN|$SHARPE|$EXCELLENT|$fusion")
        fi
    else
        print_warning "未找到 ${fusion^^} 模型数据"
    fi
done

print_info "融合模型数据收集完成，共 ${#RANK_DATA[@]} 条"

# 按年化收益率排序
print_info "开始排序排名数据..."
if [ ${#RANK_DATA[@]} -gt 0 ]; then
    print_info "找到 ${#RANK_DATA[@]} 条排名数据，开始排序..."
    
    # 使用临时文件进行排序
    TEMP_SORT_FILE=$(mktemp)
    print_info "创建临时文件: $TEMP_SORT_FILE"
    
    printf '%s\n' "${RANK_DATA[@]}" | sort -t'|' -k1 -nr > "$TEMP_SORT_FILE"
    print_info "排序完成"
    
    print_info "生成排名表格..."
    RANK=1
    while IFS='|' read -r annual_return sharpe excellent model; do
        echo "| $RANK | ${model^^} | ${annual_return}% | ${sharpe} | ${excellent} 只 |" >> "$REPORT_FILE"
        print_info "  - 排名 $RANK: ${model^^} | ${annual_return}% | ${sharpe} | ${excellent} 只"
        ((RANK++))
    done < "$TEMP_SORT_FILE"
    
    # 清理临时文件
    rm -f "$TEMP_SORT_FILE"
    print_info "临时文件已清理"
    print_success "排名表格生成完成"
else
    print_warning "没有找到有效的回测数据"
    echo "| - | 无有效数据 | - | - | - |" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## 四、推荐模型" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 找出最佳模型
print_info "分析最佳模型..."
if [ ${#RANK_DATA[@]} -gt 0 ]; then
    BEST_MODEL=$(printf '%s\n' "${RANK_DATA[@]}" | sort -t'|' -k1 -nr | head -1 | cut -d'|' -f4)
    BEST_RETURN=$(printf '%s\n' "${RANK_DATA[@]}" | sort -t'|' -k1 -nr | head -1 | cut -d'|' -f1)
    BEST_SHARPE=$(printf '%s\n' "${RANK_DATA[@]}" | sort -t'|' -k1 -nr | head -1 | cut -d'|' -f2)
    print_success "最佳模型分析完成: ${BEST_MODEL^^}"
else
    BEST_MODEL=""
    BEST_RETURN=""
    BEST_SHARPE=""
    print_warning "没有找到有效的回测数据，无法确定最佳模型"
fi

echo "**最佳模型**: ${BEST_MODEL^^}" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- 年化收益率: ${BEST_RETURN}%" >> "$REPORT_FILE"
echo "- 夏普比率: ${BEST_SHARPE}" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**推荐使用**: ${BEST_MODEL^^}模型作为主要预测模型" >> "$REPORT_FILE"

echo "" >> "$REPORT_FILE"
echo "================================================================================" >> "$REPORT_FILE"
echo "报告生成完成: $REPORT_FILE" >> "$REPORT_FILE"
echo "================================================================================" >> "$REPORT_FILE"

print_success "汇总对比报告已生成: $REPORT_FILE"
print_info "报告路径: $REPORT_FILE"

print_header "步骤7 完成"
print_info "可以查看报告文件以验证结果"