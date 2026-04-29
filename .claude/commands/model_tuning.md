---
description: 模型特征修改后的标准调优流程 - 特征残差化、超参调优、Walk-forward测试
allowed-tools: read_file, write_file, edit_file, bash, grep, glob
---

## Context

你是港股智能分析系统的模型调优专家，负责在特征修改后执行标准化的调优流程，确保模型效果提升并避免过拟合。

## Your Task

当模型特征发生修改时，按照以下 SOP 流程执行调优：

## ⚠️ 重要：首先确认修改类型

在开始调优前，必须明确本次修改的类型：

| 修改类型 | 需要的步骤 | 说明 |
|---------|-----------|------|
| **新增宏观特征** | 残差化 → 超参调优 → Walk-forward | 宏观特征会导致"特征坍缩" |
| **新增微观特征** | 超参调优 → Walk-forward | 无需残差化 |
| **删除特征** | 超参调优 → Walk-forward | 特征减少需重新调参 |
| **参数调整** | Walk-forward | 仅验证效果 |

---

## 调优流程

### 阶段 1：特征残差化（仅宏观特征）

**目的**：解决全局特征（美债利率、VIX等）导致的"特征坍缩"问题

**触发条件**：新增以下类型的特征时必须执行
- 宏观经济指标（美债利率、VIX、美元指数等）
- 市场整体指标（恒指收益率、市场情绪等）
- 行业/板块指标（板块轮动信号等）

#### 1A. 理解残差化原理

**问题**：全局特征导致所有股票预测结果"全涨全跌"，无法选股

**原因**：
```
预测 = f(宏观特征) + f(微观特征)
      ↑
      主导作用，导致所有股票预测方向一致
```

**解决方案**：
```
微观特征残差 = 微观特征 - α * 宏观特征
预测 = f(宏观特征) + f(微观特征残差)
                        ↑
                        剔除宏观影响后的特异性信息
```

#### 1B. 执行残差化

**脚本**：`data_services/feature_residualizer.py`

**配置**：
```python
# 宏观特征列表（用于回归）
MACRO_FEATURES = [
    'US_10Y_Yield',      # 美债利率
    'VIX_Level',         # VIX指数
    'HSI_Return_60d',    # 恒指60日收益率
    'Dollar_Index',      # 美元指数
]

# 微观特征列表（需要残差化）
MICRO_FEATURES = [
    'Momentum_20d',      # 动量
    'Volume_Ratio_5d',   # 成交量比率
    'RSI_14',            # RSI
    'MACD',              # MACD
    'BB_Position',       # 布林带位置
    'ATR_14',            # ATR
    # ... 更多微观特征
]
```

**执行命令**：
```bash
# 检查残差化模块
python3 -c "from data_services.feature_residualizer import FeatureResidualizer; print('OK')"

# 残差化已集成到 ml_trading_model.py 的特征工程流程中
# 无需单独执行，模型训练时自动应用
```

#### 1C. 验证残差化效果

**检查点**：
1. 残差特征数量 = 微观特征数量
2. 残差特征与原始特征相关性 < 0.5
3. 残差特征均值 ≈ 0

**验证脚本**：
```python
# 在 ml_trading_model.py 中检查
import pandas as pd

# 加载特征数据
df = pd.read_pickle('data/feature_cache/stock_features.pkl')

# 检查残差特征
residual_cols = [col for col in df.columns if '_Residual' in col]
print(f"残差特征数量: {len(residual_cols)}")

# 检查相关性
for col in residual_cols:
    orig_col = col.replace('_Residual', '')
    if orig_col in df.columns:
        corr = df[col].corr(df[orig_col])
        print(f"{col} vs {orig_col}: r={corr:.3f}")
```

#### 1D. 残差化检查清单

- [ ] 已识别所有宏观特征
- [ ] 已识别需要残差化的微观特征
- [ ] 残差化模块已集成到特征工程流程
- [ ] 残差特征数量正确
- [ ] 残差特征相关性检查通过

---

### 阶段 2：超参数调优

**目的**：找到最优模型参数，避免过拟合

**触发条件**：特征数量变化（增加或减少）时必须执行

#### 2A. 理解调优原理

**特征数量与参数关系**：

| 特征数量 | 参数调整方向 | 原因 |
|---------|-------------|------|
| **增加** | 增加正则化 | 防止过拟合 |
| **减少** | 降低正则化 | 补偿信息损失 |

**关键参数**：

| 参数 | 作用 | 特征增加时 | 特征减少时 |
|------|------|----------|----------|
| n_estimators | 树数量 | 减少 | 增加 |
| depth | 树深度 | 减少 | 增加 |
| learning_rate | 学习率 | 降低 | 提高 |
| l2_leaf_reg | L2正则化 | 增加 | 减少 |
| subsample | 行采样 | 减少 | 增加 |
| colsample_bylevel | 列采样 | 减少 | 增加 |

#### 2B. 执行超参数调优

**脚本**：`ml_services/hyperparameter_tuner.py`

**快速模式**（初步探索）：
```bash
# 快速调优（20只股票，4个fold）
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 15 --quick
```

**标准模式**（推荐）：
```bash
# 标准调优（59只股票，8个fold）
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30
```

**完整模式**（最终确认）：
```bash
# 完整调优（59只股票，18个fold）
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 50 --full
```

#### 2C. 参数搜索空间

**当前配置**（`ml_services/hyperparameter_tuner.py`）：

```python
param_grid = {
    'n_estimators': [400, 500, 600, 700, 800],
    'depth': [5, 6, 7, 8],
    'learning_rate': [0.02, 0.03, 0.04, 0.05],
    'l2_leaf_reg': [1, 2, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.75, 0.8],
    'colsample_bylevel': [0.6, 0.7, 0.75, 0.8],
}
```

**调优策略**：
1. **粗搜索**：大范围搜索，找到大致最优区域
2. **细搜索**：在最优区域附近精细搜索
3. **验证**：用完整 Walk-forward 验证最终参数

#### 2D. 评估调优结果

**输出文件**：`output/hyperparameter_tuning_*.json`

**关键指标**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 准确率 | >55% | 方向预测准确率 |
| 夏普比率 | >0.8 | 风险调整收益 |
| 最大回撤 | >-5% | 风险控制 |
| 稳定性 | <0.5 | 夏普比率标准差 |

**判断标准**：

| 结果 | 行动 |
|------|------|
| 所有指标达标 | 采用新参数 |
| 准确率提升但夏普下降 | 权衡选择，优先夏普 |
| 准确率下降 | 放弃，使用旧参数 |
| 过拟合迹象（训练CV >> Walk-forward） | 增加正则化 |

#### 2E. 超参调优检查清单

- [ ] 已根据特征数量变化确定调优方向
- [ ] 已执行超参数调优脚本
- [ ] 已记录最优参数组合
- [ ] 已对比新旧参数效果
- [ ] 已检查过拟合迹象

---

### 阶段 3：Walk-forward 测试

**目的**：验证模型真实预测能力，检测数据泄漏

**触发条件**：所有模型修改都必须执行

#### 3A. 理解 Walk-forward 原理

**与训练 CV 的区别**：

| 验证方法 | 数据使用 | 准确率 | 可信度 |
|---------|---------|--------|--------|
| 训练 CV | TimeSeriesSplit，数据来自相似时期 | 较高（62-71%） | 中等 |
| **Walk-forward** | 滚动训练，跨越不同市场环境 | 较低（50-65%） | **高** |

**Walk-forward 机制**：
```
Fold 1: 训练[1-252] → 测试[253-272]
Fold 2: 训练[41-292] → 测试[293-312]
Fold 3: 训练[81-332] → 测试[333-352]
...
```

#### 3B. 执行 Walk-forward 测试

**个股模型**：
```bash
# 20天周期（推荐）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 5天周期（趋势确认）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 5

# 1天周期（仅供参考，噪音大）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 1
```

**恒指模型**：
```bash
# 20天周期（推荐）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20

# 5天周期（趋势确认）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 5

# 1天周期（仅供参考，噪音大）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 1
```

#### 3C. 数据泄漏检测

**阈值标准**：

| 模型类型 | 正常范围 | 数据泄漏信号 |
|---------|---------|-------------|
| 个股 | 50-65% | **>65%** |
| 恒指 | 60-82% | **>85%** |

**常见泄漏来源**：

| 泄漏来源 | 检查方法 | 修复方法 |
|---------|---------|---------|
| 使用当日数据 | 检查 `.shift(1)` | 添加延迟 |
| 未来收益计算错误 | 检查 `.shift(-N)` | 修正计算 |
| BB_Position 未延迟 | 检查特征工程 | 添加 `.shift(1)` |
| Price_Percentile 未延迟 | 检查特征工程 | 添加 `.shift(1)` |

**验证脚本**：
```bash
# 检查特征工程中的 shift 调用
grep -n "\.shift(" ml_services/ml_trading_model.py | head -20

# 检查未来收益计算
grep -n "future_return" ml_services/ml_trading_model.py
```

#### 3D. 评估 Walk-forward 结果

**输出文件**：`output/walk_forward_catboost_20d_*.md`

**关键指标**：

| 指标 | 个股目标 | 恒指目标 | 说明 |
|------|---------|---------|------|
| 准确率 | 50-65% | 60-82% | 方向预测 |
| 夏普比率 | >0.8 | >1.0 | 风险调整收益 |
| 最大回撤 | >-5% | >-1% | 风险控制 |
| IC | >0.02 | >0.05 | 选股能力 |
| 预测分散度 | >0.1 | >0.1 | 避免"全涨全跌" |

**判断标准**：

| 结果 | 行动 |
|------|------|
| 所有指标达标 | 模型可用 |
| 准确率 >65%（个股）或 >85%（恒指） | 检查数据泄漏 |
| IC < 0 | 选股能力有限，需结合恒指择时 |
| 预测分散度 < 0.1 | 检查特征坍缩，考虑残差化 |

#### 3E. Walk-forward 检查清单

- [ ] 已执行 Walk-forward 测试
- [ ] 已记录所有关键指标
- [ ] 已对比更新前后效果
- [ ] 已检查数据泄漏（准确率 < 阈值）
- [ ] 已检查 IC 和预测分散度

---

## 完整调优流程示例

### 示例：新增宏观特征（美债利率）

#### 阶段 1：特征残差化

```bash
# 1. 识别宏观特征
# US_10Y_Yield 是宏观特征，需要对微观特征进行残差化

# 2. 检查残差化配置
grep -A 20 "MACRO_FEATURES" data_services/feature_residualizer.py

# 3. 验证残差化效果
python3 -c "
from data_services.feature_residualizer import FeatureResidualizer
print('残差化模块正常')
"
```

#### 阶段 2：超参数调优

```bash
# 1. 特征数量增加，需要增加正则化
# 2. 执行快速调优探索
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 15 --quick

# 3. 根据快速调优结果，执行标准调优
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30

# 4. 记录最优参数
# 假设结果：n_estimators=600, depth=7, learning_rate=0.03, l2_leaf_reg=2
```

#### 阶段 3：Walk-forward 测试

```bash
# 1. 执行 Walk-forward 测试
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 2. 检查输出报告
cat output/walk_forward_catboost_20d_*.md

# 3. 验证数据泄漏
# 准确率 60.77% < 65%，无数据泄漏 ✅

# 4. 验证 IC
# IC = -0.0181，选股能力有限 ⚠️

# 5. 验证预测分散度
# 预测分散度 = 0.2393 > 0.1，有区分度 ✅
```

---

## 调优检查清单总结

### 阶段 1：特征残差化（仅宏观特征）

- [ ] 已识别所有宏观特征
- [ ] 已识别需要残差化的微观特征
- [ ] 残差化模块已集成到特征工程流程
- [ ] 残差特征数量正确
- [ ] 残差特征相关性检查通过

### 阶段 2：超参数调优

- [ ] 已根据特征数量变化确定调优方向
- [ ] 已执行超参数调优脚本
- [ ] 已记录最优参数组合
- [ ] 已对比新旧参数效果
- [ ] 已检查过拟合迹象

### 阶段 3：Walk-forward 测试

- [ ] 已执行 Walk-forward 测试
- [ ] 已记录所有关键指标
- [ ] 已对比更新前后效果
- [ ] 已检查数据泄漏（准确率 < 阈值）
- [ ] 已检查 IC 和预测分散度

### 阶段 4：文档更新（参考 model_validation.md）

- [ ] 已更新 CLAUDE.md 模型指标
- [ ] 已更新 docs/ 相关文档
- [ ] 已更新 progress.txt
- [ ] 已更新 lessons.md（如有新发现）

---

## 快速参考

### 参数调整方向

| 特征变化 | n_estimators | depth | learning_rate | l2_leaf_reg | subsample | colsample |
|---------|-------------|-------|--------------|-------------|-----------|-----------|
| 增加 | ↓ | ↓ | ↓ | ↑ | ↓ | ↓ |
| 减少 | ↑ | ↑ | ↑ | ↓ | ↑ | ↑ |

### 数据泄漏阈值

| 模型类型 | 正常范围 | 数据泄漏信号 |
|---------|---------|-------------|
| 个股 | 50-65% | **>65%** |
| 恒指 | 60-82% | **>85%** |

### 关键指标目标

| 指标 | 个股目标 | 恒指目标 |
|------|---------|---------|
| 准确率 | 50-65% | 60-82% |
| 夏普比率 | >0.8 | >1.0 |
| 最大回撤 | >-5% | >-1% |
| IC | >0.02 | >0.05 |
| 预测分散度 | >0.1 | >0.1 |

### 常用命令

```bash
# 超参数调优（标准模式）
python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30

# Walk-forward 测试（个股）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# Walk-forward 测试（恒指）
python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20

# 语法检查
python3 -m py_compile ml_services/ml_trading_model.py
python3 -m py_compile ml_services/hyperparameter_tuner.py
python3 -m py_compile ml_services/walk_forward_validation.py
```

---

## 注意事项

1. **顺序执行**：必须按阶段 1→2→3→4 顺序执行
2. **残差化触发条件**：仅新增宏观特征时需要
3. **超参调优必要性**：特征数量变化时必须执行
4. **Walk-forward 必须执行**：所有模型修改都必须验证
5. **数据泄漏检查**：准确率超过阈值必须排查
6. **IC 负值处理**：IC < 0 表示选股能力有限，需结合恒指择时
7. **文档同步**：调优完成后必须更新相关文档

---

*文档版本：v1.0*
*创建日期：2026-04-29*
*参考文档：model_validation.md、lessons.md*
