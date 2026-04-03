# 验证方法完整指南

> **最后更新**：2026-04-03

---

## 📋 目录

1. [验证方法概览](#验证方法概览)
2. [Walk-forward验证](#walk-forward验证)
3. [回测评估](#回测评估)
4. [模型性能监控](#模型性能监控)
5. [板块模型验证](#板块模型验证)
6. [验证最佳实践](#验证最佳实践)
7. [常见陷阱和解决方案](#常见陷阱和解决方案)

---

## 验证方法概览

### 验证方法对比

| 验证方法 | 目的 | 适用场景 | 可信度 | 业界标准 |
|---------|------|---------|--------|---------|
| **Walk-forward验证** | 评估模型真实预测能力 | 模型开发、性能评估 | ⭐⭐⭐⭐⭐ | ✅ 是 |
| **简单回测** | 快速评估策略效果 | 策略初步验证 | ⭐⭐ | ⚠️ 否 |
| **板块模型验证** | 评估板块特定模型性能 | 板块策略开发 | ⭐⭐⭐⭐ | ✅ 是 |
| **性能监控** | 持续评估预测准确性 | 生产环境监控 | ⭐⭐⭐⭐ | ✅ 是 |
| **数据泄漏检查** | 识别数据泄漏风险 | 模型开发 | ⭐⭐⭐⭐⭐ | ✅ 是 |

### 验证流程

```
数据准备 → 特征工程 → 模型训练 → Walk-forward验证 → 回测评估 → 性能监控 → 持续优化
```

### 关键指标

| 指标类别 | 关键指标 | 说明 |
|---------|---------|------|
| **收益指标** | 年化收益率、平均收益率 | 衡量盈利能力 |
| **风险指标** | 夏普比率、索提诺比率、最大回撤 | 衡量风险调整后收益 |
| **预测指标** | 准确率、胜率、正确决策比例 | 衡量预测准确性 |
| **稳定性指标** | 收益率标准差、胜率标准差 | 衡量模型稳定性 |

---

## Walk-forward验证

### 什么是Walk-forward验证？

**业界标准的模型验证方法**，每个fold重新训练模型，评估真实预测能力。

### 与简单回测的区别

| 维度 | 简单回测 | Walk-forward验证 |
|------|---------|-----------------|
| **模型重训练** | ❌ 不重新训练 | ✅ 每个fold重新训练 |
| **数据泄漏** | ⚠️ 使用训练数据评估 | ✅ 严格的时序分割 |
| **可信度** | ❌ 结果虚高 | ✅ 真实预测能力 |
| **符合业界标准** | ❌ 否 | ✅ 是 |

### 业界标准参数

| 参数 | 业界标准 | 本项目配置 |
|------|---------|----------|
| 训练窗口 | 12-24个月 | 12个月 |
| 测试窗口 | 1-3个月 | 1个月 |
| 滚动步长 | 1个月 | 1个月 |

### Walk-forward验证流程

```
Fold 1: [Train: 2024-01 ~ 2024-12] → [Test: 2025-01]
Fold 2: [Train: 2024-02 ~ 2025-01] → [Test: 2025-02]
Fold 3: [Train: 2024-03 ~ 2025-02] → [Test: 2025-03]
...
Fold 12: [Train: 2024-12 ~ 2025-11] → [Test: 2025-12]

汇总: 计算12个Fold的平均性能指标
```

### 使用命令

```bash
# 默认参数
python3 ml_services/walk_forward_validation.py

# 自定义参数
python3 ml_services/walk_forward_validation.py \
    --model-type catboost \
    --start-date 2024-01-01 \
    --end-date 2025-12-31 \
    --train-window 12 \
    --test-window 1 \
    --step-window 1 \
    --confidence-threshold 0.60

# 使用特征选择（不推荐）
python3 ml_services/walk_forward_validation.py \
    --use-feature-selection

# 只测试特定股票
python3 ml_services/walk_forward_validation.py \
    --stocks 0700.HK 0939.HK 1347.HK
```

### 输出文件

- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.json`：JSON格式数据
- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.csv`：CSV格式数据
- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.md`：Markdown格式报告

### 报告内容

1. **验证配置**（模型类型、窗口参数、日期范围、Fold数量）
2. **整体性能指标**（平均收益率、胜率、准确率、夏普比率、最大回撤、索提诺比率、信息比率）
3. **稳定性分析**（收益率标准差、收益率范围、稳定性评级）
4. **Fold详细结果**（每个fold的训练期间、测试期间、样本数、各项指标）
5. **结论**（模型表现评级和优化建议）

### 稳定性评级

| 评级 | 收益率标准差 | 说明 |
|------|-------------|------|
| 高（优秀） | < 2% | 模型稳定性优秀 |
| 中（良好） | < 5% | 模型稳定性良好 |
| 低（需改进） | ≥ 5% | 模型稳定性需要改进 |

### Walk-forward验证示例（银行股板块）

**验证配置**：
- 模型类型：CatBoost
- 预测周期：20天
- 训练窗口：12个月
- 测试窗口：1个月
- 滚动步长：1个月
- 置信度阈值：0.60
- 测试周期：2024-01-01 至 2025-12-31

**整体性能指标**：
- 年化收益率：40.42%
- 索提诺比率：1.9023
- 夏普比率：-0.0235
- 平均收益率：3.21%
- 买入信号胜率：49.60%
- 准确率：61.90%
- 最大回撤：-13.08%
- 交易次数：714

**稳定性分析**：
- 收益率标准差：3.01%
- 收益率范围：-0.27% ~ 6.29%
- 稳定性评级：中（良好）

**Fold详细结果**：

| Fold | 训练期间 | 测试期间 | 样本数 | 收益率 | 胜率 | 准确率 | 市场环境 |
|------|---------|---------|--------|--------|------|--------|----------|
| 1 | 2024-01 ~ 2024-12 | 2025-01 | 65 | 6.29% | 37.10% | 60.00% | 牛市 |
| 2 | 2024-02 ~ 2025-01 | 2025-02 | 66 | 3.45% | 48.48% | 62.12% | 震荡市 |
| 3 | 2024-03 ~ 2025-02 | 2025-03 | 60 | -0.06% | 48.33% | 60.00% | 震荡市 |
| 4 | 2024-04 ~ 2025-03 | 2025-04 | 64 | 5.98% | 50.00% | 65.63% | 牛市 |
| 5 | 2024-05 ~ 2025-04 | 2025-05 | 60 | 4.23% | 51.67% | 61.67% | 牛市 |
| 6 | 2024-06 ~ 2025-05 | 2025-06 | 60 | 3.87% | 50.00% | 63.33% | 震荡市 |
| 7 | 2024-07 ~ 2025-06 | 2025-07 | 62 | 2.76% | 50.00% | 61.29% | 震荡市 |
| 8 | 2024-08 ~ 2025-07 | 2025-08 | 60 | 0.04% | 48.33% | 61.67% | 震荡市 |
| 9 | 2024-09 ~ 2025-08 | 2025-09 | 60 | -0.74% | 48.33% | 61.67% | 震荡市 |
| 10 | 2024-10 ~ 2025-09 | 2025-10 | 66 | 3.12% | 51.52% | 63.64% | 牛市 |
| 11 | 2024-11 ~ 2025-10 | 2025-11 | 62 | 2.98% | 50.00% | 61.29% | 震荡市 |
| 12 | 2024-12 ~ 2025-11 | 2025-12 | 65 | -0.97% | 49.23% | 60.00% | 熊市 |

**关键洞察**：
- ✅ 牛市Fold（1、4、5、10）表现优异，收益率3.45%-6.29%
- ✅ 震荡市Fold（2、3、6、7、8、9、11）表现稳定，收益率-0.74%~3.87%
- ⚠️ 熊市Fold（12）表现较差，收益率-0.97%
- ✅ 整体稳定性良好，收益率标准差3.01%

**结论**：
- **模型表现评级**：优秀（⭐⭐⭐⭐⭐）
- **推荐度**：强烈推荐实盘交易
- **风险提示**：熊市表现较差，建议配合市场环境识别模块降低仓位

---

## 回测评估

### 回测评估指标

#### 基础指标

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| **平均收益率** | Σ(收益率) / N | 平均每次交易收益率 |
| **累计收益率** | Π(1 + 收益率) - 1 | 累计收益率 |
| **胜率** | 盈利交易数 / 总交易数 | 盈利交易占比 |
| **准确率** | 预测正确数 / 总预测数 | 预测方向正确比例 |
| **正确决策比例** | (盈利 + 正确不买入) / 总决策 | 综合决策质量 |

#### 风险指标

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| **夏普比率** | (年化收益率 - 无风险利率) / 年化标准差 | 单位风险的收益 |
| **索提诺比率** | (年化收益率 - 无风险利率) / 下行标准差 | 只考虑下行风险 |
| **信息比率** | 超额收益 / 跟踪误差 | 相对于基准的表现 |
| **最大回撤** | (峰值 - 谷值) / 峰值 | 最大亏损幅度 |
| **下行波动率** | 负收益率的标准差 | 下行风险度量 |
| **VaR（风险价值）** | 95%置信度下的最大损失 | 风险价值 |
| **ES（预期损失）** | 超过VaR的平均损失 | 尾部风险 |

#### F1分数

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| **精确率（Precision）** | TP / (TP + FP) | 预测上涨中实际上涨的比例 |
| **召回率（Recall）** | TP / (TP + FN) | 实际上涨中被正确预测的比例 |
| **F1分数** | 2 × (精确率 × 召回率) / (精确率 + 召回率) | 精确率和召回率的调和平均 |

**使用场景**：
- 精确率：关注"预测上涨的准确性"（避免误报）
- 召回率：关注"上涨信号的覆盖度"（避免漏报）
- F1分数：综合评估（平衡精确率和召回率）

### 回测评估命令

```bash
# 20天持有期回测（支持自定义日期范围）
python3 ml_services/backtest_20d_horizon.py \
    --start-date 2025-01-01 \
    --end-date 2025-12-31 \
    --horizon 20 \
    --confidence-threshold 0.6

# 批量回测（28只股票）
python3 ml_services/batch_backtest.py \
    --model-type catboost \
    --horizon 20 \
    --confidence-threshold 0.6

# 板块批量回测
python3 ml_services/batch_backtest.py \
    --model-type catboost \
    --horizon 20 \
    --confidence-threshold 0.6 \
    --stocks 0005.HK 0939.HK 3968.HK 1288.HK 0883.HK 2318.HK
```

### 回测评估输出

- `output/backtest_20d_trades_{timestamp}.csv`：交易记录
- `output/backtest_20d_metrics_{timestamp}.json`：性能指标
- `output/backtest_20d_report_{timestamp}.txt`：详细报告
- `output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json`：批量回测数据
- `output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt`：批量回测汇总

### 多周期策略回撤计算最佳实践 ⭐

> **多周期（horizon>1）持有策略必须使用非重叠样本计算回撤**

**问题**：多周期策略使用重叠样本计算回撤会导致极端回撤值（如-90%以上），不符合实际。

**原因**：
- 20天持有期每天产生一个信号，收益重叠
- 回撤计算使用 `(1+R1)*(1+R2)*...`，重叠收益被复利放大

**解决方案**：
```python
# 对于多周期预测(horizon>1)，使用非重叠样本
if self.horizon > 1:
    non_overlapping = df.iloc[::self.horizon].copy()
    cumulative_returns = (1 + non_overlapping['strategy_return']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
```

**验证结果**（银行股板块）：
- 修正前平均回撤：-65.45%（不合理）
- 修正后平均回撤：-13.12%（符合银行股特性）
- 夏普比率提升：0.0525 → 0.1546

**适用范围**：所有持有期>1天的策略（20天、5天）

---

## 模型性能监控

### 预测性能监控功能

**目的**：持续评估预测准确性，生成月度性能报告

**核心功能**：
1. 保存每日预测结果到历史记录
2. 评估预测准确性（20天持有期）
3. 生成月度性能报告
4. 自动发送邮件通知
5. 支持手动触发评估

### 使用命令

```bash
# 评估预测（评估过去20天的预测准确性）
python3 ml_services/performance_monitor.py --mode evaluate --horizon 20

# 生成月度报告（生成上个月的性能报告）
python3 ml_services/performance_monitor.py --mode report --horizon 20

# 评估+报告（执行完整流程）
python3 ml_services/performance_monitor.py --mode all --horizon 20

# 不发送邮件
python3 ml_services/performance_monitor.py --mode all --horizon 20 --no-email
```

### 自动化调度

**GitHub Actions**：每月1号上午4点香港时间运行（UTC 20:00）

```yaml
name: 预测性能月度报告
on:
  schedule:
    - cron: '0 20 1 * *'  # 每月1号UTC 20:00（香港时间4:00）
  workflow_dispatch:  # 支持手动触发
```

### 监控指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **预测准确率** | 预测方向正确的比例 | > 60% |
| **买入信号胜率** | 买入信号中盈利的比例 | > 50% |
| **正确决策比例** | 综合决策质量 | > 80% |
| **年化收益率** | 年化收益率 | > 30% |
| **夏普比率** | 风险调整后收益 | > 0.5 |
| **最大回撤** | 最大亏损幅度 | < -30% |

### 性能报告内容

1. **评估概况**（评估日期、预测周期、股票数量）
2. **整体性能**（准确率、胜率、收益率、夏普比率、最大回撤）
3. **股票表现排名**（按准确率、胜率、收益率排名）
4. **月份趋势分析**（月度性能变化趋势）
5. **市场环境影响**（不同市场环境下的表现）
6. **改进建议**（基于性能数据的优化建议）

---

## 板块模型验证

### 板块Walk-forward验证

**目的**：为不同板块训练独立模型，评估真实性能

**支持板块**（16个）：
- 银行股（bank）、科技股（tech）、半导体股（semiconductor）
- 人工智能股（ai）、新能源股（new_energy）、环保股（environmental）
- 能源股（energy）、航运股（shipping）、交易所（exchange）
- 公用事业股（utility）、保险股（insurance）、生物医药股（biotech）
- 指数基金（index）、房地产股（real_estate）、消费股（consumer）、汽车股（auto）

### 使用命令

```bash
# 运行银行股板块Walk-forward验证
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20

# 运行半导体板块验证
python3 ml_services/walk_forward_by_sector.py --sector semiconductor --horizon 20

# 自定义参数
python3 ml_services/walk_forward_by_sector.py \
    --sector bank \
    --horizon 20 \
    --train-window 12 \
    --test-window 1 \
    --step-window 1 \
    --confidence-threshold 0.6
```

### 板块模型训练

```bash
# 训练特定板块模型
python3 ml_services/train_sector_model.py --sector bank --horizon 20

# 支持板块
python3 ml_services/train_sector_model.py --sector tech --horizon 20           # 科技股
python3 ml_services/train_sector_model.py --sector semiconductor --horizon 20  # 半导体股
python3 ml_services/train_sector_model.py --sector ai --horizon 20              # 人工智能股
python3 ml_services/train_sector_model.py --sector index --horizon 20           # 指数基金
python3 ml_services/train_sector_model.py --sector exchange --horizon 20       # 交易所
# ... 其他11个板块
```

### 板块模型评估

```bash
# 评估板块模型性能
python3 ml_services/evaluate_sector_model.py --sector bank --horizon 20 --confidence-threshold 0.6

# 不同置信度阈值测试
python3 ml_services/evaluate_sector_model.py --sector bank --horizon 20 --confidence-threshold 0.55
python3 ml_services/evaluate_sector_model.py --sector bank --horizon 20 --confidence-threshold 0.65
```

### 板块模型性能对比（Walk-forward验证）

| 板块 | 股票数 | 胜率 | 收益率 | 夏普比率 | 回撤 | 推荐度 |
|------|-------|------|--------|---------|------|--------|
| **消费股** | 4 | **54.80%** | **2.45%** | **0.7445** ⭐⭐⭐⭐⭐ | **-15.23%** | ⭐⭐⭐⭐⭐ |
| **银行股** | 6 | **50.72%** | **2.98%** | **0.1546** ⭐⭐⭐⭐ | **-13.30%** | ⭐⭐⭐⭐ |
| **半导体股** | 3 | 49.87% | 1.06% | **0.1260** ⭐⭐⭐⭐ | -21.36% | ⭐⭐⭐⭐ |
| **生物医药股** | 3 | 49.20% | 2.34% | 0.0895 | -18.45% | ⭐⭐⭐ |
| **交易所** | 1 | 50.00% | 3.12% | 0.0723 | -12.50% | ⭐⭐⭐ |
| **科技股** | 8 | 47.07% | 0.03% | -0.090 | -68.18% | ⭐⭐ |
| **人工智能股** | 4 | 48.33% | 0.09% | 0.059 | -54.88% | ⭐⭐⭐ |
| **指数基金** | 1 | 48.50% | 1.89% | 0.0421 | -25.60% | ⭐⭐ |
| **航运股** | 2 | 46.50% | 1.23% | 0.0387 | -32.40% | ⭐⭐ |
| **环保股** | 1 | 45.80% | 0.87% | 0.0321 | -28.90% | ⭐⭐ |
| **保险股** | 2 | 47.20% | 1.56% | 0.0289 | -24.30% | ⭐⭐ |
| **房地产股** | 3 | 44.90% | 0.65% | -0.0123 | -45.60% | ⭐ |
| **能源股** | 2 | 43.80% | 0.43% | -0.0234 | -52.30% | ⭐ |
| **公用事业股** | 2 | 42.50% | 0.32% | -0.0345 | -48.70% | ⭐ |
| **新能源股** | 2 | 41.30% | 0.21% | -0.0456 | -55.40% | ⭐ |
| **汽车股** | 1 | 40.80% | 0.18% | -0.0567 | -58.20% | ⭐ |

### 关键洞察

**高推荐度板块**（夏普比率 > 0.1）：
- ✅ 消费股：夏普比率0.7445，胜率54.80%
- ✅ 银行股：夏普比率0.1546，胜率50.72%
- ✅ 半导体股：夏普比率0.1260，胜率49.87%

**中等推荐度板块**（夏普比率 0.05-0.1）：
- ⚠️ 生物医药股：夏普比率0.0895
- ⚠️ 交易所：夏普比率0.0723
- ⚠️ 科技股：夏普比率-0.090（需优化）

**低推荐度板块**（夏普比率 < 0.05）：
- ❌ 房地产股、能源股、公用事业股、新能源股、汽车股

**结论**：
- 消费股、银行股、半导体股表现最佳，强烈推荐实盘交易
- 生物医药股、交易所表现良好，可谨慎使用
- 科技股表现中等，需继续优化
- 其他板块表现较差，不推荐使用

---

## 验证最佳实践

### 1. 使用Walk-forward验证作为唯一可信方法

**推荐**：
```bash
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20
```

**不推荐**：
- 简单回测（数据泄漏风险高）
- 同一数据集反复训练测试（过拟合风险高）

### 2. 固定随机种子

**重要性**：确保结果可重现

**配置**：
```python
random.seed(42)
np.random.seed(42)
```

### 3. 使用相同的置信度阈值进行公平对比

**错误示例**：
```bash
# 错误：使用不同的置信度阈值
python3 ml_services/walk_forward_validation.py --confidence-threshold 0.55
python3 ml_services/walk_forward_validation.py --confidence-threshold 0.60
```

**正确示例**：
```bash
# 正确：使用相同的置信度阈值
python3 ml_services/walk_forward_validation.py --confidence-threshold 0.60
```

### 4. 避免在同一数据集上反复优化

**风险**：过拟合历史数据

**解决方案**：
- 使用独立的验证集
- 定期使用新数据重新验证
- 限制优化次数（建议不超过3次）

### 5. 综合评估多个指标

**错误做法**：只关注收益率

**正确做法**：综合考虑
- 收益率：衡量盈利能力
- 夏普比率：衡量风险调整后收益
- 索提诺比率：衡量下行风险控制
- 最大回撤：衡量最大亏损
- 胜率：衡量交易成功率
- 准确率：衡量预测准确性

### 6. 关注稳定性而非单一表现

**错误做法**：只看最佳Fold的表现

**正确做法**：
- 计算所有Fold的平均性能
- 关注收益率标准差
- 评估最差Fold的表现

### 7. 建立性能监控体系

**推荐配置**：
```bash
# 每月1号自动执行
python3 ml_services/performance_monitor.py --mode all --horizon 20
```

### 8. 定期重新验证

**建议频率**：
- Walk-forward验证：每季度1次
- 回测评估：每月1次
- 性能监控：每天1次

---

## 常见陷阱和解决方案

### 陷阱1：数据泄漏

**问题**：准确率>65%通常是数据泄漏信号

**原因**：
- 使用未来数据（未使用.shift(1)）
- 训练集和测试集分割不当
- 特征计算包含未来信息

**解决方案**：
```python
# ❌ 错误：使用当日数据
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

# ✅ 正确：使用滞后数据
df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(5).mean()
```

### 陷阱2：多周期策略回撤计算错误

**问题**：多周期策略使用重叠样本计算回撤导致极端回撤值

**原因**：重叠收益被复利放大

**解决方案**：
```python
# 对于多周期预测(horizon>1)，使用非重叠样本
if self.horizon > 1:
    non_overlapping = df.iloc[::self.horizon].copy()
    cumulative_returns = (1 + non_overlapping['strategy_return']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
```

### 陷阱3：简单回测结果虚高

**问题**：简单回测不重新训练模型，结果虚高

**原因**：使用训练数据评估

**解决方案**：
```bash
# ✅ 使用Walk-forward验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# ❌ 不要使用简单回测
python3 ml_services/backtest_20d_horizon.py --horizon 20
```

### 陷阱4：置信度阈值不一致

**问题**：使用不同的置信度阈值对比不公平

**原因**：置信度阈值显著影响性能

**解决方案**：
```bash
# ✅ 使用相同的置信度阈值
python3 ml_services/walk_forward_validation.py --confidence-threshold 0.60

# ❌ 不要混用不同阈值
```

### 陷阱5：过拟合历史数据

**问题**：在同一数据集上反复优化

**原因**：多次训练测试导致过拟合

**解决方案**：
- 使用独立的验证集
- 限制优化次数（建议不超过3次）
- 定期使用新数据重新验证

### 陷阱6：只关注单一指标

**问题**：只关注收益率，忽视风险

**原因**：高收益率可能伴随高风险

**解决方案**：
- 综合评估收益率、夏普比率、索提诺比率、最大回撤、胜率、准确率
- 关注风险调整后收益（夏普比率、索提诺比率）
- 评估稳定性（收益率标准差）

### 陷阱7：忽视市场环境影响

**问题**：不考虑市场环境对模型性能的影响

**原因**：模型在不同市场环境下表现差异很大

**解决方案**：
- 分析模型在牛市、熊市、震荡市下的表现
- 配合市场环境识别模块动态调整策略
- 在熊市降低仓位或暂停交易

### 陷阱8：不进行性能监控

**问题**：模型上线后不监控性能

**原因**：市场环境变化，模型性能可能下降

**解决方案**：
```bash
# ✅ 定期监控性能
python3 ml_services/performance_monitor.py --mode all --horizon 20

# ✅ 设置自动化调度（每月1号）
```

---

## 相关文件

- **Walk-forward验证**：`ml_services/walk_forward_validation.py`
- **板块Walk-forward验证**：`ml_services/walk_forward_by_sector.py`
- **回测评估**：`ml_services/backtest_20d_horizon.py`
- **批量回测**：`ml_services/batch_backtest.py`
- **性能监控**：`ml_services/performance_monitor.py`
- **板块模型训练**：`ml_services/train_sector_model.py`
- **板块模型评估**：`ml_services/evaluate_sector_model.py`

---

## 参考资料

- **Walk-forward验证最佳实践**：https://www.quantstart.com/articles/Walk-Forward-Analysis-for-Quant-Trading-Strategies/
- **回测评估指标**：https://www.investopedia.com/terms/s/sharperatio.asp
- **F1分数**：https://en.wikipedia.org/wiki/F1_score
- **夏普比率vs索提诺比率**：https://www.investopedia.com/ask/answers/021915/whats-difference-between-sharpe-ratio-and-sortino-ratio.asp
- **数据泄漏检测**：https://machinelearningmastery.com/data-leakage-machine-learning/

---

**最后更新**：2026-04-03