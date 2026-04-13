# performance-monitor 工作流改进实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 performance-monitor.yml 工作流的 Cron 定时和 Git 提交逻辑

**Architecture:** 简化 GitHub Actions 工作流配置，将每月4次执行改为1次，将复杂的 stash/rebase/merge 逻辑简化为 pull --rebase + 重试

**Tech Stack:** GitHub Actions, YAML, Bash

**Spec:** `docs/superpowers/specs/2026-04-13-performance-monitor-workflow-design.md`

---

## Task 1: 修复 Cron 定时配置

**Files:**
- Modify: `.github/workflows/performance-monitor.yml:5-6`

- [ ] **Step 1: 修改注释和 Cron 表达式**

将第 5-6 行修改为：

```yaml
    # 每月28日 UTC 20:00 (次日 HK 时间凌晨4:00)
    - cron: '0 20 28 * *'
```

**注意**: 原注释"每月1号"是错误的，需要一并更正。

- [ ] **Step 2: 验证 YAML 语法**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/performance-monitor.yml'))"`
Expected: 无输出（语法正确）

- [ ] **Step 3: 提交更改**

```bash
git add .github/workflows/performance-monitor.yml
git commit -m "fix: 修复 performance-monitor 月度报告 cron 配置为每月28日运行一次"
```

---

## Task 2: 简化 Git 提交逻辑

**Files:**
- Modify: `.github/workflows/performance-monitor.yml:38-78`

- [ ] **Step 1: 替换整个 "Commit updated history" 步骤**

将第 38-78 行的整个步骤替换为以下内容：

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

- [ ] **Step 2: 验证 YAML 语法**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/performance-monitor.yml'))"`
Expected: 无输出（语法正确）

- [ ] **Step 3: 提交更改**

```bash
git add .github/workflows/performance-monitor.yml
git commit -m "refactor: 简化 performance-monitor Git 提交逻辑为 pull --rebase + 重试机制"
```

---

## Task 3: 推送并验证

- [ ] **Step 1: 推送所有提交**

```bash
git push origin main
```

- [ ] **Step 2: 手动触发工作流验证**

在 GitHub Actions 页面手动触发 `performance-monitor` 工作流（使用 `workflow_dispatch`），确认：
1. 工作流能正常启动
2. Git 提交逻辑能正确执行
3. 推送成功后检查提交记录

- [ ] **Step 3: 更新进度文档**

在 `progress.txt` 中记录本次更新。
