#!/bin/bash

# 测试脚本：验证 pre-test hook 是否正确加载环境变量

echo "=========================================="
echo "测试 pre-test hook"
echo "=========================================="

# 保存当前的环境变量状态（如果存在）
OLD_YAHOO_EMAIL="$YAHOO_EMAIL"
OLD_QWEN_API_KEY="$QWEN_API_KEY"

# 清空环境变量以测试 hook 的加载功能
unset YAHOO_EMAIL
unset QWEN_API_KEY

echo ""
echo "1. 清空环境变量后："
echo "   YAHOO_EMAIL: ${YAHOO_EMAIL:-未设置}"
echo "   QWEN_API_KEY: ${QWEN_API_KEY:-未设置}"
echo ""

# 执行 pre-test hook（假设 iFlow 会 source 这个脚本）
echo "2. 执行 pre-test hook..."
source /data/fortune/.iflow/hooks/pre-test.sh
echo ""

# 检查环境变量是否被正确设置
echo "3. 执行 hook 后的环境变量："
echo "   YAHOO_EMAIL: ${YAHOO_EMAIL:-未设置}"
echo "   QWEN_API_KEY: ${QWEN_API_KEY:0:10}...${QWEN_API_KEY:+已设置}"
echo ""

# 验证
if [ -n "$YAHOO_EMAIL" ] && [ -n "$QWEN_API_KEY" ]; then
    echo "✅ 测试通过：环境变量已正确加载"
    EXIT_CODE=0
else
    echo "❌ 测试失败：环境变量未正确加载"
    EXIT_CODE=1
fi

echo ""
echo "=========================================="

# 恢复之前的环境变量
if [ -n "$OLD_YAHOO_EMAIL" ]; then
    export YAHOO_EMAIL="$OLD_YAHOO_EMAIL"
fi
if [ -n "$OLD_QWEN_API_KEY" ]; then
    export QWEN_API_KEY="$OLD_QWEN_API_KEY"
fi

exit $EXIT_CODE