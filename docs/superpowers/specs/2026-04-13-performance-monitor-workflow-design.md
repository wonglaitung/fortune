---
name: performance-monitor-workflow-improvement
description: 改进 performance-monitor.yml 工作流的 Cron 定时和 Git 提交逻辑
type: spec
---

# performance-monitor.yml 工作流改进设计

**日期**: 2026-04-13
**状态**: 待实现

## 一、背景

`.github/workflows/performance-monitor.yml` 是预测性能月度报告的工作流，存在以下问题：

1. **Cron 配置错误**: `0 20 28-31 * *` 导致每月 28、29、30、31 日各运行一次，共 4 次
2. **Git 逻辑过于复杂**: `stash → rebase → merge → retry` 分支过多，难以维护

## 二、改进方案

### 2.1 Cron 定时修复

**修改前**:
```yaml
- cron: '0 20 28-31 * *'
```

**修改后**:
```yaml
- cron: '0 20 28 * *'
```

**原因**: 28 日是所有月份都存在的最大日期，确保每月仅运行一次。

### 2.2 Git 逻辑简化

**修改前** (复杂的 stash/rebase/merge 逻辑，约 30 行):

```yaml
- name: Commit updated history
  run: |
    git config --local user.email "github-actions[bot]@users.noreply.github.com"
    git config --local user.name "github-actions[bot]"
    # 先拉取远程更新
    git fetch origin main
    # 如果本地有未提交的更改，先暂存
    if ! git diff --quiet; then
      git stash push -m "Temporarily stash changes before rebase"
    fi
    # 尝试rebase
    if git rebase origin/main; then
      # ... 更多分支逻辑
```

**修改后** (简洁的 pull --rebase + 重试，约 25 行):

```yaml
- name: Commit updated history
  run: |
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git config user.name "github-actions[bot]"

    # 添加文件
    git add data/prediction_history.json output/performance_report_*.md

    # 检查是否有更改
    if git diff --staged --quiet; then
      echo "没有新的性能监控数据需要提交"
      exit 0
    fi

    # 提交
    git commit -m "Update performance monitoring data [skip ci]"

    # 拉取并变基（最多重试3次）
    for i in 1 2 3; do
      if git pull --rebase origin main; then
        break
      fi
      echo "重试 $i/3: 拉取失败，等待5秒后重试..."
      sleep 5
    done

    # 推送（最多重试3次）
    for i in 1 2 3; do
      if git push origin main; then
        echo "推送成功"
        exit 0
      fi
      echo "重试 $i/3: 推送失败，等待5秒后重试..."
      sleep 5
    done

    echo "推送失败，请手动检查"
    exit 1
```

**改进点**:
1. 移除复杂的 stash/rebase/merge 逻辑
2. 使用 `git pull --rebase` 自动处理远程更新
3. 添加明确的重试机制（最多3次）
4. 添加失败时的明确退出码

## 三、影响范围

| 影响项 | 说明 |
|--------|------|
| 执行频率 | 从每月 4 次减少到 1 次 |
| 代码行数 | 从 ~30 行减少到 ~25 行 |
| 可维护性 | 逻辑更清晰，分支更少 |
| 兼容性 | 无破坏性变更 |

## 四、实施步骤

1. 修改 `.github/workflows/performance-monitor.yml`
2. 提交并推送更改
3. 手动触发工作流验证
