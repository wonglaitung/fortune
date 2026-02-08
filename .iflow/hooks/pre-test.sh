#!/bin/bash

# iFlow Pre-Test Hook
# 在执行测试前自动加载环境变量

# 获取项目根目录（从脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 设置环境变量
if [ -f "$PROJECT_ROOT/set_key.sh" ]; then
    echo "[iFlow Pre-Test] Loading environment variables from set_key.sh..."
    source "$PROJECT_ROOT/set_key.sh"
    echo "[iFlow Pre-Test] Environment variables loaded successfully"
else
    echo "[iFlow Pre-Test] Warning: set_key.sh not found at $PROJECT_ROOT/set_key.sh"
    exit 1
fi

# 验证关键环境变量是否已设置
if [ -z "$YAHOO_EMAIL" ]; then
    echo "[iFlow Pre-Test] Error: YAHOO_EMAIL environment variable not set"
    exit 1
fi

if [ -z "$QWEN_API_KEY" ]; then
    echo "[iFlow Pre-Test] Error: QWEN_API_KEY environment variable not set"
    exit 1
fi

echo "[iFlow Pre-Test] All required environment variables are set"