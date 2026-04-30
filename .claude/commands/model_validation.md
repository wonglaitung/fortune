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

```bash
# 20天周期（推荐）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 板块验证（可选）
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20
```

**个股判断标准**：

| 指标 | 正常范围 | 优秀 | 数据泄漏信号 |
|------|---------|------|-------------|
| 准确率 | 50-55% | >55% | **>65%** |
| 夏普比率 | 0.5-0.8 | >0.8 | - |
| 最大回撤 | -5%~-10% | >-5% | - |

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

## 执行检查清单

在完成验证流程前，请确认：

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

1. **两个模型独立验证**：恒指和个股模型有各自的验证脚本、判断标准、数据泄漏阈值
2. **三周期验证必须分别执行**：
   - 恒指：`analyze_three_horizon_relationships.py`
   - 个股：`analyze_stock_causal_chain.py --full`（完整模型，禁止快速模式）
3. **个股验证禁止快速模式**：
   - ❌ 禁止使用 `--quick` 参数
   - ❌ 禁止使用5只代表性股票
   - ✅ 必须使用完整模型（730特征，59只股票）
4. **顺序执行**：必须按阶段 1→2→3→4 顺序执行，前一阶段有效才进入下一阶段
5. **记录完整**：每个阶段的测试结果必须完整记录
6. **对比验证**：必须与更新前的指标对比，确认提升幅度
7. **⚠️ 阶段 4 不可跳过**：
   - 文档更新后**必须**同步更新代码
   - 常见问题：文档更新了，但代码中的硬编码数据没有更新
   - 导致：邮件报告显示旧数据，用户收到错误信息
   - **必须执行代码与文档一致性验证**
7. **文档同步**：
   - 代码更新后立即同步文档，避免信息不一致
   - **文档更新范围包括 `docs/` 目录下所有相关文档**
   - **恒指和个股数据必须分别更新，不能只更新其中一个**
8. **语法检查**：每次代码修改后必须执行 `python3 -m py_compile`
9. **核心文件优先**：hsi_prediction.py（恒指）和 comprehensive_analysis.py（个股）是主要入口

## 快速参考

### 模型对比

| 项目 | 恒指模型 | 个股模型 |
|------|---------|---------|
| 验证脚本 | `hsi_walk_forward.py` | `walk_forward_validation.py` |
| 特征数量 | 33个（增强模型） | 730个（完整模型） |
| 数据泄漏阈值 | >85% | >65% |
| 最优策略 | 假突破(101) 95% | 一致看涨(111) 56% |
| 预测概率与实际方向相关性 | 正向 r=+0.35 | 弱正向 r=+0.03 |

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
