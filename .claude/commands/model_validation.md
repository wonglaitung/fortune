---
description: 模型更新后的标准验证流程 - Walk-forward测试、三周期验证、文档更新
allowed-tools: read_file, write_file, edit_file, bash, grep, glob
---

## Context

你是港股智能分析系统的模型验证专家，负责在模型更新后执行标准化的验证流程，确保模型效果提升并更新相关文档。

## Your Task

当恒指或个股模型发生更改时，按照以下 SOP 流程执行验证：

## ⚠️ 重要：首先确认模型类型

在开始验证前，必须明确本次更新的是哪个模型：

| 模型类型 | 验证脚本 | 数据泄漏阈值 | 特征数量 |
|---------|---------|-------------|---------|
| **恒指模型** | `hsi_walk_forward.py` | >80% 可疑 | 33个（增强模型） |
| **个股模型** | `walk_forward_validation.py` | >65% 可疑 | 730个（完整模型） |

**两个模型的验证流程独立但并行**，都需要完整执行。

---

## 验证流程

### 阶段 0：特征选择（可选但推荐）

**目的**：减少特征数量，提高模型泛化能力

**⚠️ 重要**：如果要在 Walk-forward 验证中使用特征选择，**必须先运行特征选择**！

#### 0A. 特征选择命令

```bash
# 模型重要性法（推荐，Top 500 特征）
python3 ml_services/feature_selection.py --method model --top-k 500 --horizon 20

# 统计方法（Top 300 特征）
python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20

# 累积重要性法（自动选择特征数量，覆盖95%重要性）
python3 ml_services/feature_selection.py --method cumulative_importance --horizon 20 --target-importance 0.95
```

#### 0B. 特征选择方法对比

| 方法 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **model** | 效果稳定，计算快 | 依赖模型 | ⭐⭐⭐⭐⭐ 推荐 |
| statistical | 不依赖模型，独立性强 | 可能遗漏非线性关系 | 特征较少时 |
| cumulative_importance | 自动选择数量 | 可能选择过多特征 | 不确定特征数量时 |

#### 0C. 输出文件位置

特征选择后会生成以下文件：

| 文件格式 | 路径示例 | 用途 |
|---------|---------|------|
| TXT 特征名称 | `output/model_importance_features_20260509.txt` | 直接读取特征名 |
| CSV 选择结果 | `output/model_importance_selected_20260509.csv` | 包含重要性得分 |
| 最新特征文件 | `output/model_importance_features_latest.txt` | 软链接到最新 |

**验证特征选择文件是否存在**：

```bash
ls -la output/model_importance_features_*.txt output/statistical_features_*.txt
```

#### 0D. 特征选择检查清单

- [ ] 已运行特征选择脚本（如 `--method model --top-k 500`）
- [ ] 已确认输出文件存在（`output/model_importance_features_*.txt`）
- [ ] 已确认预测周期与验证周期一致（如 `--horizon 20`）

---

### 阶段 1：Walk-forward 测试

**目的**：验证模型效果是否有提升

#### 1A. 恒指模型测试

```bash
# 20天周期（推荐）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20

# 5天周期（趋势确认）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 5

# 1天周期（仅供参考，噪音大）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 1
```

**恒指判断标准**：

| 指标 | 正常范围 | 优秀 | 数据泄漏信号 |
|------|---------|------|-------------|
| 20天准确率 | 60-80% | >80% | **>85%** |
| 5天准确率 | 55-65% | >65% | >70% |
| 1天准确率 | ~50% | - | >60% |

#### 1B. 个股模型测试

**⚠️ 重要：需要先决定是否使用特征选择！**

```bash
# 【推荐】使用特征选择（Top 500 特征，需先运行阶段 0）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20 --use-feature-selection

# 使用全量特征（约 1132 个特征）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 带置信度阈值
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55

# 板块验证（可选）
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20
```

**个股判断标准**：

| 指标 | 全量特征 (~1132) | Top 500 特征 | 数据泄漏信号 |
|------|-----------------|--------------|-------------|
| 准确率 | 50-55% | 50-55% | **>65%** |
| 夏普比率 | 5.0-6.0 | 4.8-5.5 | - |
| 最大回撤 | -0.8%~-1.0% | -0.7%~-0.9% | - |
| 平均收益率 | +5%~+6% | +5%~+6% | - |
| 总体盈亏比 | ~1.5 | ~1.5 | - |

**使用特征选择的优势**：
- 特征数量减少 55.8%（1132 → 500）
- 性能基本持平，训练速度更快
- 降低过拟合风险

#### 1C. 两个模型对比验证

**如果同时更新两个模型**，需要对比验证结果：

| 对比项 | 恒指 | 个股 | 说明 |
|--------|------|------|------|
| 准确率上限 | 80% | 55% | 恒指噪声低，准确率更高 |
| 预测概率与实际方向相关性 | 正向（r=+0.35） | 弱正向（r=+0.03） | 个股相关性弱 |
| 最优策略 | 假突破(101) | 反弹失败(010) | 策略不同 |

### 阶段 2：三周期验证（如果阶段 1 有效）

**目的**：检查各种交易策略的准确度

**⚠️ 重要**：三周期验证必须分别对恒指和个股执行，个股必须使用完整模型（59只股票），禁止使用快速模式。

#### 2A. 恒指三周期验证

```bash
# 恒指三周期关系分析（增强模型，33特征）
python3 ml_services/analyze_three_horizon_relationships.py
```

**恒指策略评估标准**：

| 策略代码 | 策略名称 | 目标胜率 | 当前胜率（2026-04-29） |
|---------|---------|---------|----------------------|
| 101 | 假突破做多 | >85% | **95.00%** ⭐⭐⭐⭐⭐ |
| 010 | 反弹失败做空 | >85% | **85.98%** ⭐⭐⭐⭐⭐ |
| 001 | 下跌中继做多 | >80% | **84.00%** ⭐⭐⭐⭐ |
| 000 | 一致看跌做空 | >75% | **79.57%** ⭐⭐⭐⭐ |
| 111 | 一致看涨买入 | >75% | **80.62%** ⭐⭐⭐⭐ |

**输出文件**：`docs/THREE_HORIZON_ANALYSIS.md`

#### 2B. 个股三周期验证（完整模型）

```bash
# 个股三周期关系分析（完整模型，730特征，59只股票）
# ⚠️ 禁止使用快速模式（--quick 或 5只代表性股票）
python3 ml_services/analyze_stock_causal_chain.py --full
```

**个股验证要求**：
- ✅ 必须使用完整模型（730特征）
- ✅ 必须使用59只股票
- ❌ 禁止使用 `--quick` 参数
- ❌ 禁止使用5只代表性股票

**个股策略评估标准**：

| 策略代码 | 策略名称 | 目标胜率 | 说明 |
|---------|---------|---------|------|
| 010 | 反弹失败 | >60% | 个股效果一般 |
| 000 | 一致看跌 | >55% | 接近随机 |
| 101 | 假突破 | ~55% | 个股效果差（随机水平） |

**输出文件**：`output/stock_causal_chain_analysis.json`

#### 2C. 恒指 vs 个股关键差异

| 对比项 | 恒指 | 个股 | 说明 |
|--------|------|------|------|
| 预测概率与实际方向相关性 | 正向（r=+0.35） | 弱正向（r=+0.03） | 个股相关性弱 |
| 假突破(101)胜率 | **95.00%** | ~55% | 恒指最优，个股随机 |
| 最优策略 | 假突破(101) | 一致看涨(111) | 策略效果都一般 |
| 准确率上限 | 81% | 57% | 恒指噪声低 |

**关键教训**：
- 恒指假突破(101)胜率 95%，个股仅 55%
- 个股所有模式准确率接近随机水平（50%左右）
- **不能将恒指策略直接套用于个股**

### 阶段 3：文档更新（如果阶段 2 有提升）

**目的**：同步更新所有相关文档的指标数字

**⚠️ 重要**：
1. 文档更新范围包括 `CLAUDE.md` 和 `docs/` 目录下的所有相关文档，确保信息一致性
2. **恒指和个股数据必须分别更新**，不能只更新其中一个

#### 3A. 恒指模型文档更新

**必须更新的文件**：

1. **CLAUDE.md** - 项目主文档
   - 恒指模型可信度表格（准确率：1d/5d/20d）
   - 可用策略表格（假突破、下跌中继胜率）
   - 最后更新日期

2. **docs/THREE_HORIZON_ANALYSIS.md** - 三周期分析（恒指部分）
   - 第一部分：恒指验证摘要（1d/5d/20d 准确率）
   - 八大模式胜率表格（假突破 95%、反弹失败 86% 等）
   - 因果关系分析数据
   - 附录对比表中的恒指数据

3. **docs/VALIDATION_GUIDE.md** - 验证指南
   - 最新验证结果（如有恒指相关内容）

4. **progress.txt** - 项目进度
   - 记录恒指模型更新的内容和效果

#### 3B. 个股模型文档更新

**必须更新的文件**：

1. **CLAUDE.md** - 项目主文档
   - 个股模型可信度表格（准确率、夏普比率、最大回撤）
   - CatBoost 配置参数
   - 特征重要性排名

2. **docs/THREE_HORIZON_ANALYSIS.md** - 三周期分析（个股部分）
   - 第二部分：个股验证概述（准确率、因果链数据）
   - 各股票准确率排名表
   - 个股与恒指核心差异对比表
   - 附录对比表中的个股数据

3. **docs/FEATURE_IMPORTANCE_ANALYSIS.md** - 特征重要性
   - Top 10 特征排名
   - 特征类别分布

4. **docs/VALIDATION_GUIDE.md** - 验证指南
   - Walk-forward 验证结果（准确率、夏普比率、最大回撤）
   - 稳定性分析数据

5. **progress.txt** - 项目进度
   - 记录个股模型更新的内容和效果

6. **lessons.md** - 经验教训
   - 如有新的发现或警告，添加到对应章节

#### 3C. 文档更新检查清单

**恒指数据更新检查**：
- [ ] CLAUDE.md - 恒指模型可信度表格（1d/5d/20d 准确率）
- [ ] CLAUDE.md - 可用策略表格（假突破、下跌中继等胜率）
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 第一部分恒指验证摘要
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 八大模式胜率表格
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 附录恒指数据

**个股数据更新检查**：
- [ ] CLAUDE.md - 个股模型可信度表格（准确率、夏普比率、最大回撤）
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 第二部分个股验证概述
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 个股与恒指核心差异对比表
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 附录个股数据
- [ ] docs/VALIDATION_GUIDE.md - Walk-forward 验证结果

**更新后必须确认**：
- [ ] 所有相关文档的数值已同步更新
- [ ] 恒指和个股数据都已更新（不能只更新其中一个）
- [ ] 更新日期已修改
- [ ] 无遗漏的文档

#### 更新格式示例

```markdown
**恒指增强模型**（2026-04-29，33特征）：

| 周期 | 准确率 | 推荐度 |
|------|--------|--------|
| **20天** | **81.24%** | ⭐⭐⭐⭐⭐ 推荐 |
```

### 阶段 4：代码更新（必须执行，不可跳过）

**⚠️ 重要：阶段 4 是必须执行的步骤，不是可选步骤！**

**目的**：将最新发现应用到生产代码

**常见问题**：文档更新后忘记同步代码，导致邮件报告显示旧数据。

#### 4A. 恒指模型代码更新

1. **hsi_prediction.py** - 恒生指数预测主程序
   - 更新三周期策略胜率（假突破、下跌中继等）
   - 更新 `historical_accuracy` 字典（1d/5d/20d 准确率）
   - 更新 `pattern_accuracy` 字典（八大模式胜率）
   - 更新邮件内容中的 `win_rate` 和 `description` 字段
   - 同步邮件报告格式

2. **ml_services/hsi_ml_model.py** - 恒指模型
   - 特征配置
   - 模型参数
   - 预测阈值

#### 4B. 个股模型代码更新

1. **comprehensive_analysis.py** - 综合分析主程序
   - 整合大模型建议 + CatBoost预测 + 异常检测
   - 更新策略判断逻辑
   - 调整信号权重配置
   - 更新报告生成格式

2. **ml_services/ml_trading_model.py** - 个股模型
   - 特征工程逻辑
   - CatBoost 参数
   - 分类特征处理

#### 4C. 公共配置更新

1. **config.py** - 配置文件
   - 策略阈值
   - 特征开关

#### 代码更新检查点

**hsi_prediction.py 关键位置**（必须逐一检查并更新）：

| 位置 | 搜索关键词 | 说明 |
|------|-----------|------|
| 第 2571-2575 行 | `historical_accuracy` | 1d/5d/20d 准确率 |
| 第 2585-2594 行 | `pattern_accuracy` | 八大模式胜率 |
| 第 1277-1400 行 | `win_rate` | 邮件中的策略胜率 |
| 第 2700-2720 行 | `suggestion` | 控制台建议文字 |

**comprehensive_analysis.py 关键位置**：
- 恒指三周期策略配置（搜索 `THREE_HORIZON_PATTERNS_HSI`）
- 个股策略配置（搜索 `THREE_HORIZON_PATTERNS_STOCK`）
- 报告输出格式

#### 4D. 代码与文档一致性验证（新增）

**目的**：确保代码中的数据与文档一致，避免遗漏。

**验证方法**：

```bash
# 检查 hsi_prediction.py 中的准确率是否与文档一致
grep -E "81\.24|62\.36|49\.67|95\.00|85\.98" hsi_prediction.py

# 检查 CLAUDE.md 中的准确率
grep -E "81\.24|62\.36|49\.67|95\.00|85\.98" CLAUDE.md

# 对比两者是否一致
```

**如果不一致**：
1. 以文档（CLAUDE.md 和 docs/THREE_HORIZON_ANALYSIS.md）为准
2. 更新代码中的硬编码数据
3. 重新运行语法检查和测试

---

### 阶段 5：Fold 详细分析报告（必须执行）

**⚠️ 重要：阶段 5 是必须执行的步骤，用于深度诊断模型表现！**

**目的**：分析每个 Fold 的详细表现，识别问题 Fold，评估盈亏比和风险控制

#### 5A. 运行 Fold 分析脚本

在 Walk-forward 验证完成后，运行以下 Python 脚本生成详细分析：

```python
import pandas as pd
import numpy as np

# 读取预测分析数据（路径格式：output/YYYYMMDD_HHMMSS_catboost_20d/prediction_analysis.csv）
# 找到最新的输出目录
df = pd.read_csv('output/[最新目录]/prediction_analysis.csv')

print("=" * 140)
print("📊 各 Fold 详细交易分析（含盈亏比）")
print("=" * 140)

# 按 Fold 分组计算详细统计
results = []
for fold in range(1, 13):
    fold_df = df[df['Fold'] == fold]
    all_returns = fold_df['Actual_Return'].values
    profit_trades = fold_df[fold_df['Actual_Return'] > 0]
    loss_trades = fold_df[fold_df['Actual_Return'] < 0]
    
    avg_profit = np.mean(profit_trades['Actual_Return']) * 100 if len(profit_trades) > 0 else 0
    avg_loss = np.mean(loss_trades['Actual_Return']) * 100 if len(loss_trades) > 0 else 0
    profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
    
    # 评级计算
    if profit_loss_ratio > 2.0:
        rating = "⭐⭐⭐⭐⭐"
    elif profit_loss_ratio >= 1.5:
        rating = "⭐⭐⭐⭐"
    elif profit_loss_ratio >= 1.0:
        rating = "⭐⭐⭐"
    else:
        rating = "⚠️ 警告"
    
    results.append({
        'Fold': fold, '测试期间': fold_df['Date'].values[0][:7] if len(fold_df) > 0 else 'N/A',
        '总交易': len(fold_df), '准确率': len(fold_df[fold_df['Is_Correct'] == True]) / len(fold_df) * 100,
        '平均收益': np.mean(all_returns) * 100, '最大收益': np.max(all_returns) * 100,
        '最大亏损': np.min(all_returns) * 100,
        '盈利次数': len(profit_trades), '亏损次数': len(loss_trades),
        '盈利平均': avg_profit, '亏损平均': avg_loss,
        '盈亏比': profit_loss_ratio, '评级': rating
    })

# 打印详细表格（必须使用 13 列格式）
print("\n| Fold | 测试期间 | 总交易 | 准确率 | 平均收益 | 最大收益 | 最大亏损 | 盈利次数 | 亏损次数 | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |")
print("|------|---------|--------|--------|---------|---------|---------|---------|---------|---------|---------|--------|------|")
for r in results:
    print(f"| {r['Fold']} | {r['测试期间']} | {r['总交易']} | {r['准确率']:.2f}% | {r['平均收益']:+.2f}% | {r['最大收益']:+.2f}% | {r['最大亏损']:+.2f}% | {r['盈利次数']} | {r['亏损次数']} | {r['盈利平均']:+.2f}% | {r['亏损平均']:+.2f}% | {r['盈亏比']:.2f} | {r['评级']} |")

# 盈亏比排名表
print("\n| 排名 | Fold | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |")
# ... 继续生成其他表格
```

#### 5B. 必须输出的报告内容

**总体统计表格**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 总交易次数 | XX,XXX | 12个Fold总计 |
| 准确率 | XX.XX% | 正确预测比例 |
| 平均收益率 | +X.XX% | 每次交易平均收益 |
| 最大单次收益 | +XX.XX% | 单次最大盈利幅度 |
| 最大单次亏损 | -XX.XX% | 单次最大亏损幅度 |
| 盈利交易平均 | +XX.XX% | 盈利时的平均幅度 |
| 亏损交易平均 | -X.XX% | 亏损时的平均幅度 |
| **盈亏比** | X.XX | 平均盈利/平均亏损 |

**各 Fold 详细表格**：

> ⚠️ **格式要求**：必须严格使用以下 **13 列格式**，不可增减列！

| Fold | 测试期间 | 总交易 | 准确率 | 平均收益 | 最大收益 | 最大亏损 | 盈利次数 | 亏损次数 | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
|------|---------|--------|--------|---------|---------|---------|---------|---------|---------|---------|--------|------|
| 1 | 2025-01 | XXXX | XX.XX% | +X.XX% | +XX.XX% | -X.XX% | XXX | XXX | +X.XX% | -X.XX% | X.XX | ⭐⭐⭐⭐⭐ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**盈亏比排名表**：

| 排名 | Fold | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
|------|------|---------|---------|--------|------|
| 1 | Fold X | +XX.XX% | -X.XX% | X.XX | ⭐⭐⭐⭐⭐ 最优 |
| ... | ... | ... | ... | ... | ... |

**问题 Fold 分析**：

| Fold | 问题 | 根本原因 |
|------|------|---------|
| Fold X | 盈亏比 < 1，收益为负 | 亏损幅度大于盈利幅度 |
| ... | ... | ... |

#### 5C. Fold 分析判断标准

**盈亏比评级**：

| 盈亏比范围 | 评级 | 说明 |
|-----------|------|------|
| > 2.0 | ⭐⭐⭐⭐⭐ 优秀 | 风险控制极好，盈利覆盖亏损 |
| 1.5 - 2.0 | ⭐⭐⭐⭐ 良好 | 风险控制良好 |
| 1.0 - 1.5 | ⭐⭐⭐ 一般 | 需要改进风险控制 |
| < 1.0 | ⚠️ 警告 | 盈亏比倒挂，必须改进 |

**问题 Fold 定义**：
- 平均收益 < 0%
- 盈亏比 < 1.5
- 亏损次数 > 盈利次数 × 2

#### 5D. Fold 分析检查清单

- [ ] 已读取 prediction_analysis.csv 文件
- [ ] 已生成总体统计表格（总交易、准确率、平均收益、最大收益、最大亏损、盈亏比）
- [ ] 已生成各 Fold 详细表格（12个Fold）
- [ ] 已生成盈亏比排名表
- [ ] 已识别问题 Fold（盈亏比 < 1.5 或收益 < 0）
- [ ] 已分析问题 Fold 的根本原因

## 执行检查清单

在完成验证流程前，请确认：

### 阶段 0 检查（特征选择）

- [ ] 已决定是否使用特征选择
- [ ] 如使用特征选择，已运行 `feature_selection.py`
- [ ] 已确认特征选择文件存在（`output/model_importance_features_*.txt`）
- [ ] 已确认预测周期与验证周期一致（如 `--horizon 20`）

### 阶段 1 检查（两个模型独立检查）

#### 恒指模型
- [ ] 已执行 `hsi_walk_forward.py` 测试
- [ ] 已记录 1d/5d/20d 准确率
- [ ] 已对比更新前后的效果
- [ ] 确认无数据泄漏（20天准确率 < 85%）

#### 个股模型
- [ ] 已执行 `walk_forward_validation.py` 测试
- [ ] 已记录准确率、夏普比率、最大回撤
- [ ] 已对比更新前后的效果
- [ ] 确认无数据泄漏（准确率 < 65%）

### 阶段 2 检查

#### 恒指三周期验证
- [ ] 已执行 `analyze_three_horizon_relationships.py`
- [ ] 已记录各策略胜率（假突破、下跌中继等）
- [ ] 已对比策略效果变化

#### 个股三周期验证
- [ ] 已执行 `analyze_stock_causal_chain.py --full`（完整模型）
- [ ] 确认未使用快速模式（无 `--quick`，非5只股票）
- [ ] 已确认预测概率与实际方向相关性（应为负值 r≈-0.17）
- [ ] 已记录个股最优策略（反弹失败010）

### 阶段 3 检查

**⚠️ 重要：恒指和个股数据必须分别更新，不能只更新其中一个**

#### 恒指文档（docs/ 目录下相关文档）
- [ ] 已更新 CLAUDE.md 的恒指模型指标（1d/5d/20d 准确率）
- [ ] 已更新 CLAUDE.md 的可用策略表格（假突破、下跌中继等胜率）
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 第一部分（恒指验证摘要）
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 八大模式胜率表格
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 附录恒指数据
- [ ] 已更新 docs/VALIDATION_GUIDE.md（如有恒指相关内容）
- [ ] 已更新 progress.txt 恒指部分

#### 个股文档（docs/ 目录下相关文档）
- [ ] 已更新 CLAUDE.md 的个股模型指标（准确率、夏普比率、最大回撤）
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 第二部分（个股验证概述）
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 个股与恒指核心差异对比表
- [ ] 已更新 docs/THREE_HORIZON_ANALYSIS.md 附录个股数据
- [ ] 已更新 docs/FEATURE_IMPORTANCE_ANALYSIS.md（如有变化）
- [ ] 已更新 docs/VALIDATION_GUIDE.md Walk-forward 验证结果
- [ ] 已更新 progress.txt 个股部分
- [ ] 已更新 lessons.md（如有新发现）

#### 文档一致性检查
- [ ] 恒指数据已完整更新（不能只更新个股）
- [ ] 个股数据已完整更新（不能只更新恒指）
- [ ] 所有文档的数值已同步
- [ ] 更新日期已修改
- [ ] 无遗漏的文档

### 阶段 4 检查（必须完成，不可跳过）

**⚠️ 重要：这是最容易遗漏的步骤！文档更新后必须同步更新代码！**

#### 恒指代码
- [ ] 已更新 hsi_prediction.py 的 `historical_accuracy` 字典（1d/5d/20d 准确率）
- [ ] 已更新 hsi_prediction.py 的 `pattern_accuracy` 字典（八大模式胜率）
- [ ] 已更新 hsi_prediction.py 的邮件内容 `win_rate` 字段
- [ ] 已更新 hsi_prediction.py 的控制台 `suggestion` 文字
- [ ] 已更新 ml_services/hsi_ml_model.py（如需）

#### 个股代码
- [ ] 已更新 comprehensive_analysis.py 的策略配置
- [ ] 已更新 ml_services/ml_trading_model.py（如需）

#### 代码与文档一致性验证（新增）
- [ ] 已对比 hsi_prediction.py 和 CLAUDE.md 中的准确率数据
- [ ] 已对比 hsi_prediction.py 和 docs/THREE_HORIZON_ANALYSIS.md 中的胜率数据
- [ ] 确认代码中的硬编码数据与文档完全一致
- [ ] 无遗漏的代码位置

#### 验证
- [ ] 已执行语法检查 `python3 -m py_compile <文件>`
- [ ] 已运行测试 `python3 -m pytest tests/ -v`
- [ ] 已运行 `python3 hsi_prediction.py --no-email` 验证输出正确

### 阶段 5 检查（必须完成）

**⚠️ 重要：此阶段用于深度诊断模型表现，识别问题 Fold！**

#### Fold 分析报告
- [ ] 已读取 prediction_analysis.csv 文件
- [ ] 已生成总体统计表格（总交易、准确率、平均收益、最大收益、最大亏损、盈亏比）
- [ ] 已生成各 Fold 详细表格（12个Fold）
- [ ] 已生成盈亏比排名表
- [ ] 已识别问题 Fold（盈亏比 < 1.5 或收益 < 0）
- [ ] 已分析问题 Fold 的根本原因

#### 关键指标检查
- [ ] 总体盈亏比 > 1.5（风险控制达标）
- [ ] 问题 Fold 数量 < 3（稳定性达标）
- [ ] 最大单次亏损 < -30%（极端风险可控）

## 数据泄漏警告

**关键检查点**：

1. **准确率异常高**
   - 恒指 20 天 > 85%：需检查特征泄漏
   - 个股 > 65%：需检查特征泄漏

2. **常见泄漏来源**
   - 使用当日数据计算特征（应使用 `.shift(1)`）
   - `future_return` 计算错误（应使用 `.shift(-N)`）
   - BB_Position、Price_Percentile 等特征未延迟

3. **验证方法**
   - 检查特征工程代码中的 `.shift()` 调用
   - 确认训练数据和预测数据的预处理一致

## 输出要求

### 验证报告格式

```markdown
# 模型更新验证报告

## 更新类型
- 模型类型：[恒指/个股/两者]
- 更新日期：YYYY-MM-DD
- 更新内容：简要描述

---

## 恒指模型验证结果

### Walk-forward 测试

| 周期 | 更新前 | 更新后 | 变化 | 评估 |
|------|--------|--------|------|------|
| 1天 | XX.XX% | XX.XX% | +X.XX% | 噪音大 |
| 5天 | XX.XX% | XX.XX% | +X.XX% | [有效/无效] |
| 20天 | XX.XX% | XX.XX% | +X.XX% | [有效/无效] |

### 三周期策略验证

| 策略 | 更新前胜率 | 更新后胜率 | 变化 | 评估 |
|------|-----------|-----------|------|------|
| 101 假突破做多 | XX.XX% | XX.XX% | +X.XX% | ⭐⭐⭐⭐⭐ |
| 001 下跌中继做多 | XX.XX% | XX.XX% | +X.XX% | ⭐⭐⭐⭐⭐ |
| 000 一致看跌做空 | XX.XX% | XX.XX% | +X.XX% | ⭐⭐⭐⭐ |
| 111 一致看涨买入 | XX.XX% | XX.XX% | +X.XX% | ⭐⭐⭐⭐ |

### 数据泄漏检查
- 20天准确率：XX.XX% [✅ 正常 / ⚠️ 可疑]
- 阈值：>85% 为数据泄漏信号

---

## 个股模型验证结果

### Walk-forward 测试（12 folds，59只股票）

| 指标 | 更新前 | 更新后 | 变化 | 评估 |
|------|--------|--------|------|------|
| 准确率 | XX.XX% | XX.XX% | +X.XX% | [正常/异常] |
| 夏普比率 | X.XX | X.XX | +X.XX | [达标/未达标] |
| 最大回撤 | -X.XX% | -X.XX% | +X.XX% | [良好/需改进] |
| 综合评分 | XX | XX | +XX | [优秀/良好] |

### 三周期策略验证（完整模型，730特征，59只股票）

| 策略 | 更新前胜率 | 更新后胜率 | 变化 | 评估 |
|------|-----------|-----------|------|------|
| 010 反弹失败 | XX.XX% | XX.XX% | +X.XX% | 个股最优 |
| 000 一致看跌 | XX.XX% | XX.XX% | +X.XX% | 次优 |
| 101 假突破 | XX.XX% | XX.XX% | +X.XX% | 随机水平 |

### 因果链验证
- 5d预测概率与20d实际方向相关性 r：X.XX [✅ 负值正常 / ⚠️ 异常]
- 预期：r ≈ +0.03（弱正向，个股预测信号不可靠）

### 数据泄漏检查
- 准确率：XX.XX% [✅ 正常 / ⚠️ 可疑]
- 阈值：>65% 为数据泄漏信号

---

## Fold 详细分析报告

### 总体统计

| 指标 | 数值 | 说明 |
|------|------|------|
| 总交易次数 | XX,XXX | 12个Fold总计 |
| 准确率 | XX.XX% | 正确预测比例 |
| 平均收益率 | +X.XX% | 每次交易平均收益 |
| 最大单次收益 | +XX.XX% | 单次最大盈利幅度 |
| 最大单次亏损 | -XX.XX% | 单次最大亏损幅度 |
| 盈利交易平均 | +XX.XX% | 盈利时的平均幅度 |
| 亏损交易平均 | -X.XX% | 亏损时的平均幅度 |
| **盈亏比** | X.XX | 平均盈利/平均亏损 |

### 各 Fold 详细表格

| Fold | 测试期间 | 总交易 | 准确率 | 平均收益 | 最大收益 | 最大亏损 | 盈利次数 | 亏损次数 | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
|------|---------|--------|--------|---------|---------|---------|---------|---------|---------|---------|--------|------|
| 1 | 2025-01 | XXXX | XX.XX% | +X.XX% | +XX.XX% | -X.XX% | XXX | XXX | +X.XX% | -X.XX% | X.XX | ⭐⭐⭐⭐⭐ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 盈亏比排名

| 排名 | Fold | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
|------|------|---------|---------|--------|------|
| 1 | Fold X | +XX.XX% | -X.XX% | X.XX | ⭐⭐⭐⭐⭐ 最优 |
| ... | ... | ... | ... | ... | ... |

### 问题 Fold 分析

| Fold | 问题 | 根本原因 |
|------|------|---------|
| Fold X | 盈亏比 < 1，收益为负 | 亏损幅度大于盈利幅度 |
| ... | ... | ... |

### 关键发现
1. 盈亏比与收益率的关系：盈亏比 > 1.5 的 Fold 全部正收益
2. 最佳表现：Fold X，盈亏比 X.XX，收益 +X.XX%
3. 最差表现：Fold X，盈亏比 X.XX，收益 -X.XX%
4. 总体盈亏比：X.XX（[达标/需改进]）

---

## 文档更新清单

**⚠️ 重要：恒指和个股数据必须分别更新**

### 恒指文档（docs/ 目录下相关文档）
- [ ] CLAUDE.md - 恒指模型可信度表格（1d/5d/20d 准确率）
- [ ] CLAUDE.md - 可用策略表格（假突破、下跌中继等胜率）
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 第一部分恒指验证摘要
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 八大模式胜率表格
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 附录恒指数据
- [ ] docs/VALIDATION_GUIDE.md - 恒指相关内容（如有）
- [ ] progress.txt - 恒指部分

### 个股文档（docs/ 目录下相关文档）
- [ ] CLAUDE.md - 个股模型可信度表格（准确率、夏普比率、最大回撤）
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 第二部分个股验证概述
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 个股与恒指核心差异对比表
- [ ] docs/THREE_HORIZON_ANALYSIS.md - 附录个股数据
- [ ] docs/FEATURE_IMPORTANCE_ANALYSIS.md - Top 10 特征排名
- [ ] docs/VALIDATION_GUIDE.md - Walk-forward 验证结果
- [ ] progress.txt - 个股部分
- [ ] lessons.md - 新发现或警告

### 文档一致性检查
- [ ] 恒指数据已完整更新（不能只更新个股）
- [ ] 个股数据已完整更新（不能只更新恒指）
- [ ] 所有 docs/ 目录下相关文档数值已同步
- [ ] 更新日期已修改

---

## 代码更新清单

### 恒指代码
- [ ] hsi_prediction.py：更新 `historical_accuracy` 字典
- [ ] hsi_prediction.py：更新 `pattern_accuracy` 字典
- [ ] hsi_prediction.py：更新邮件内容 `win_rate` 字段
- [ ] hsi_prediction.py：更新控制台 `suggestion` 文字
- [ ] ml_services/hsi_ml_model.py：更新参数（如需）

### 个股代码
- [ ] comprehensive_analysis.py：更新策略逻辑
- [ ] ml_services/ml_trading_model.py：更新参数（如需）

### 代码与文档一致性验证
- [ ] 已对比代码和文档中的准确率数据
- [ ] 确认代码中的硬编码数据与文档完全一致
- [ ] 已运行 `python3 hsi_prediction.py --no-email` 验证输出正确

---

## 下一步建议
- 建议内容...
```

## 注意事项

1. **特征选择是可选但推荐的**：
   - 如果要在 Walk-forward 验证中使用特征选择，**必须先运行特征选择**（阶段 0）
   - 特征选择文件路径：`output/model_importance_features_*.txt`
   - Walk-forward 验证时添加 `--use-feature-selection` 参数
2. **两个模型独立验证**：恒指和个股模型有各自的验证脚本、判断标准、数据泄漏阈值
3. **三周期验证必须分别执行**：
   - 恒指：`analyze_three_horizon_relationships.py`
   - 个股：`analyze_stock_causal_chain.py --full`（完整模型，禁止快速模式）
4. **个股验证禁止快速模式**：
   - ❌ 禁止使用 `--quick` 参数
   - ❌ 禁止使用5只代表性股票
   - ✅ 必须使用完整模型（推荐 Top 500 特征或全量特征 ~1132）
5. **顺序执行**：必须按阶段 0→1→2→3→4→5 顺序执行，前一阶段有效才进入下一阶段
6. **记录完整**：每个阶段的测试结果必须完整记录
7. **对比验证**：必须与更新前的指标对比，确认提升幅度
8. **⚠️ 阶段 4 不可跳过**：
   - 文档更新后**必须**同步更新代码
   - 常见问题：文档更新了，但代码中的硬编码数据没有更新
   - 导致：邮件报告显示旧数据，用户收到错误信息
   - **必须执行代码与文档一致性验证**
9. **⚠️ 阶段 5 必须执行**：
   - Fold 分析报告用于深度诊断模型表现
   - 识别问题 Fold（盈亏比 < 1.5 或收益 < 0）
   - 分析问题根因，指导后续优化
   - **输出格式必须包含：总体统计、各Fold详细表格、盈亏比排名、问题Fold分析**
   - **⚠️ 格式要求**：各 Fold 详细表格必须严格使用 **13 列格式**，不可增减列：
     ```
     | Fold | 测试期间 | 总交易 | 准确率 | 平均收益 | 最大收益 | 最大亏损 | 盈利次数 | 亏损次数 | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
     ```
   - 盈亏比排名表使用 **6 列格式**：
     ```
     | 排名 | Fold | 盈利平均 | 亏损平均 | 盈亏比 | 评级 |
     ```
10. **文档同步**：
   - 代码更新后立即同步文档，避免信息不一致
   - **文档更新范围包括 `docs/` 目录下所有相关文档**
   - **恒指和个股数据必须分别更新，不能只更新其中一个**
11. **语法检查**：每次代码修改后必须执行 `python3 -m py_compile`
12. **核心文件优先**：hsi_prediction.py（恒指）和 comprehensive_analysis.py（个股）是主要入口

## 快速参考

### 核心命令速查

| 任务 | 命令 |
|------|------|
| **特征选择** | `python3 ml_services/feature_selection.py --method model --top-k 500 --horizon 20` |
| **恒指 Walk-forward** | `python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20` |
| **个股 Walk-forward（推荐）** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20 --use-feature-selection` |
| **个股 Walk-forward（全量特征）** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |

### 模型对比

| 项目 | 恒指模型 | 个股模型 |
|------|---------|---------|
| 验证脚本 | `hsi_walk_forward.py` | `walk_forward_validation.py` |
| 特征数量 | 33个（增强模型） | ~1132个（全量）/ 500个（推荐） |
| 数据泄漏阈值 | >85% | >65% |
| 最优策略 | 假突破(101) 95% | 一致看涨(111) 56% |
| 预测概率与实际方向相关性 | 正向 r=+0.35 | 弱正向 r=+0.03 |

### 特征选择相关文件

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| 特征选择输出 | `output/model_importance_features_*.txt` | 模型重要性法结果 |
| 特征选择输出 | `output/statistical_features_*.txt` | 统计方法结果 |
| 特征选择缓存 | `data/feature_selection_*.json` | Top 300 特征（JSON格式） |

### 模型文件位置
- 恒指模型：`data/hsi_models/hsi_catboost_*.cbm`
- 个股模型：`data/models/catboost_*.cbm`
- 特征配置：`data/hsi_models/hsi_feature_config_*.json`

### 输出文件位置
- 恒指 Walk-forward 报告：`output/walk_forward_catboost_*.md`
- 个股 Walk-forward 报告：`output/walk_forward_catboost_20d_*.md`
- 三周期分析：`docs/THREE_HORIZON_ANALYSIS.md`
- 预测历史：`data/prediction_history.json`

### 核心入口文件
- 恒指预测：`hsi_prediction.py`
- 综合分析：`comprehensive_analysis.py`

记住：恒指和个股是两个不同的模型，需要分别验证、分别更新。
