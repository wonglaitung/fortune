# iFlow Hooks 配置

## 概述

iFlow hooks 允许在特定事件发生时自动执行自定义脚本。本项目配置了以下 hooks，用于自动化环境配置和验证。

## 可用的 Hooks

### Pre-Test Hook (`pre-test.sh`)

**触发时机**：在执行任何测试之前，iFlow 会自动 source 此脚本

**功能**：
- 自动加载 `set_key.sh` 中的环境变量
- 验证关键环境变量是否正确设置
- 确保测试环境配置正确

**环境变量验证**：
- `YAHOO_EMAIL`：163邮箱地址
- `YAHOO_APP_PASSWORD`：163邮箱应用专用密码
- `QWEN_API_KEY`：大模型API密钥
- `RECIPIENT_EMAIL`：收件人邮箱地址

**工作原理**：
iFlow 在执行测试前会通过 `source` 命令加载此脚本，这意味着：
- 脚本中设置的环境变量会保留在 iFlow 的 shell 进程中
- 后续的测试脚本可以访问这些环境变量
- hook 执行失败（返回非零退出码）会阻止测试执行

## Hook 文件位置

```
/data/fortune/.iflow/hooks/
├── pre-test.sh       # 测试前环境变量加载
├── test_hook.sh      # Hook 功能测试脚本
└── README.md         # 本文档
```

## 如何添加新的 Hook

1. 在 `.iflow/hooks/` 目录下创建新的 shell 脚本文件
2. 给脚本添加执行权限：`chmod +x <hook文件名>`
3. 按照 iFlow 的 hook 命名约定命名文件（如 `pre-test.sh`, `post-test.sh` 等）
4. 脚本应该通过 `source` 方式被调用，确保环境变量可以传递

**命名约定**：
- `pre-<event>.sh`：在事件发生前执行
- `post-<event>.sh`：在事件发生后执行
- 常见事件：`test`, `build`, `deploy`, `commit`

## 测试 Hook

项目提供了测试脚本来验证 hook 的功能：

```bash
# 运行测试脚本
./.iflow/hooks/test_hook.sh
```

测试脚本会：
1. 清空当前环境变量
2. 执行 pre-test hook
3. 验证环境变量是否被正确加载
4. 输出测试结果

## 注意事项

- **必须使用 source**：Hook 脚本应该通过 `source` 命令加载，而不是作为独立进程执行
- **返回值很重要**：Hook 脚本应该返回 0 表示成功，非 0 表示失败
- **失败会阻止执行**：如果 hook 失败，后续的测试将不会执行
- **错误处理**：确保脚本具有适当的错误处理和日志输出
- **路径处理**：使用相对于脚本位置的路径，避免硬编码绝对路径

## 调试 Hook

如果 hook 没有按预期工作，可以：

1. **检查权限**：
   ```bash
   ls -l .iflow/hooks/
   ```

2. **手动测试（source 方式）**：
   ```bash
   source ./.iflow/hooks/pre-test.sh
   echo $YAHOO_EMAIL
   echo $QWEN_API_KEY
   ```

3. **运行自动测试脚本**：
   ```bash
   ./.iflow/hooks/test_hook.sh
   ```

4. **查看输出**：检查脚本输出中的错误信息和警告

5. **验证文件存在**：
   ```bash
   ls -l set_key.sh
   ```

6. **检查环境变量格式**：确认 `set_key.sh` 中的环境变量格式正确

## iFlow 集成

iFlow 会自动检测并执行 `.iflow/hooks/` 目录下的 hook 脚本。集成方式：

1. **自动发现**：iFlow 在启动时会扫描 hooks 目录
2. **事件触发**：在特定事件（如测试）发生前，自动 source 对应的 hook
3. **环境传递**：Hook 中设置的环境变量会传递给后续的测试脚本
4. **错误处理**：如果 hook 返回非零退出码，iFlow 会停止执行并显示错误

## 示例：添加 Post-Test Hook

创建一个在测试后运行的 hook，用于清理临时文件：

```bash
#!/bin/bash
# .iflow/hooks/post-test.sh

echo "[iFlow Post-Test] Cleaning up temporary files..."

# 清理临时文件
rm -rf /tmp/test_*.log

# 清理缓存
python3 -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"

echo "[iFlow Post-Test] Cleanup completed"
```

添加执行权限：
```bash
chmod +x .iflow/hooks/post-test.sh
```

## 相关文件

- `set_key.sh`：项目根目录下的环境变量配置文件
- `.iflow/commands/programmer_skill.md`：编程规范和开发流程
- `requirements.txt`：Python 依赖包列表

## 更新日志

- **2026-02-08**：创建 pre-test hook，自动加载环境变量
- **2026-02-08**：添加测试脚本 test_hook.sh
- **2026-02-08**：完善文档说明和调试指南