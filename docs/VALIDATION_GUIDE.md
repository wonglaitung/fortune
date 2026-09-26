# 验证方法完整指南

> **最后更新**：2026-09-27（删除 2026-04/05 绝对口径历史存档约 700 行——银行股示例、个股90分示例、
> 板块推荐表、已推翻的三周期结论、04-29 Fold 分析；被推翻内容一律不留档，指路即可）

---

## 📋 目录

1. [最新验证结果](#最新验证结果)
2. [验证方法概览](#验证方法概览)
3. [Walk-forward验证](#walk-forward验证)
4. [回测评估](#回测评估)
5. [月度护栏（Guardrail）](#月度护栏guardrail)
6. [模型性能监控](#模型性能监控)
7. [板块模型验证](#板块模型验证)
8. [验证最佳实践](#验证最佳实践)
9. [常见陷阱和解决方案](#常见陷阱和解决方案)

---

## 最新验证结果

> **2026-09-26 全量复测**：恒指 1/5/20d + 个股 1/5/20d，全部 PIT/embargo walk-forward。
> 完整表格见 [AGENTS.md](../AGENTS.md#-机器学习模型) 与 [THREE_HORIZON_ANALYSIS.md](THREE_HORIZON_ANALYSIS.md)。

| 对象 | 周期/学习器 | 准确率 | 超额 lift | 月度护栏 | 结论 |
|------|------------|--------|----------|---------|------|
| 恒指 | 1d | 51.3% [47.7,55.0] p=0.50 | +1.4pp p=0.64 | — | 无边缘（上轮"显著"未复现） |
| 恒指 | 5d | 54.8% [46.6,62.7] p=0.29 | +5.7pp p=0.45 | — | 样本不足（n_eff≈143） |
| 恒指 | 20d | 59.1% [42.7,73.7] p=0.36 | +8.5pp p=0.63 | — | 样本不足（n_eff≈35） |
| 个股 | 1d CatBoost | 51.0% [50.5,51.4] p=0.0001 | +1.5pp p=0.0003 | 净IR −1.96 🔴 | **停用**（D10/D2） |
| 个股 | 5d CatBoost | 51.7% [50.6,52.7] p=0.0018 | +1.2pp p=0.149 | 净IR 0.74 🟡 | 保留低配 |
| 个股 | 20d LightGBM | 51.8% [49.7,53.9] p=0.105 | +1.9pp p=0.240 | 净IR 1.41 [0.32,2.62] PBO 0.19 DSR 0.982 🟢；**超额IR 1.24 [0.18,2.41]、逐年超额全正** | **统计全过 → 可升配**（升配幅度待定：lift 不显著、2024 打平、n=39 期） |

**三周期八大模式**（697 恒指样本 / 43,610 个股样本）：相对**同向 20d 基准**的净贡献，
恒指 Bonferroni（α=0.00625）后全部不显著；个股 101(+2.7pp)/001(−2.3pp) 名义过线但效应 ≤±2.7pp。
**两端均不构成交易信号**，详见 [THREE_HORIZON_ANALYSIS.md](THREE_HORIZON_ANALYSIS.md)。

**评估铁律**：不看绝对准确率/胜率，只看 **超额 lift** 与 **方向技能**；每次 walk-forward 必跑
`monthly_guardrail.py`（净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 才可升级）；
**真正要加仓前**再跑 `portfolio_backtest.py` 补两道检验——**超额 bootstrap CI + 逐年稳健**（见下方"月度护栏"章）。

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
| **超额指标** | **方向技能**（准确率−永远看涨）、**超额 lift**（胜率−无条件买入） | 剔除市场 beta 后衡量真实技能 ⭐ |
| **稳定性指标** | 收益率标准差、胜率标准差、跨周期一致性 | 衡量模型稳定性 |

> ⚠️ **只看绝对准确率/胜率会被市场趋势误导**：趋势向上板块的"永远看涨"准确率本就 >50%，
> 买入基准胜率也高。评估必须减去基准，详见 [回测评估](#严谨评估工具backtest_evalpy) 与
> [lessons.md 0.5/0.6](../lessons.md)。

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
| 训练窗口 | 12-24个月 | 个股 36 个月（CLI 默认）/ 恒指 12 个月 |
| 测试窗口 | 1-3个月 | 1个月 |
| 滚动步长 | 1个月 | 1个月 |

### Walk-forward验证流程

```
Fold  1: [Train: 过去36个月] → [Test: 第1个月]
Fold  2: [Train: 向前滚动1个月] → [Test: 第2个月]
...
Fold 38: [Train: 向前滚动37个月] → [Test: 第38个月]

汇总: 全部 Fold 预测合并后做 backtest_eval（全样本池化，非逐折平均）
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

- `output/{timestamp}_{model_type}_{horizon}d/`：**详细结果目录**
  - `prediction_analysis.csv`：全部预测明细（回测评估 `backtest_eval.py`、市场门槛分位数据源）
  - `fold_metrics_detail.json`：每 Fold 指标 + Top 100 特征重要性
  - `validation_summary.json`：总体验证结果
- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.json`：JSON格式数据
- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.csv`：CSV格式数据
- `output/walk_forward_{model_type}_{horizon}d_{timestamp}.md`：Markdown格式报告

### 回测产物自动入库

验证成功结束后自动执行（`scripts/commit_backtest_result.py`；加 `--no-commit` 关闭）：

1. 提交本目录 `prediction_analysis.csv`（CI/本地分位数据源须同源）
2. 港股 20d 额外同步 `ml_services/market_regime.py` 的 `GATE_SNAPSHOT` 分位常量
3. `git rm --cached` 仓库中其它港股 20d CSV（工作树保留，只留最新防膨胀）
4. `commit [skip ci]` + `push`（所有 workflow 为 schedule 触发，push 不触发流水线）

任何失败仅打印 WARNING、exit 0，**不影响回测结果**。A股 `a_stock_walk_forward.py`
同样集成（只提交 CSV，不碰 20d 快照）；恒指输出在 `data/`、独立体系，不集成。

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

---

## 回测评估

### 回测评估指标

#### 基础指标

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| **平均收益率** | Σ(收益率) / N | 平均每次交易收益率 |
| **累计收益率** | Π(1 + 收益率) - 1 | 累计收益率 |
| **胜率** | 净收益>0的交易数 / 总交易数 | 扣除双边成本(约0.5%)后盈利交易占比 |
| **准确率** | 预测正确数 / 总预测数 | 预测方向正确比例（全部样本） |
| **正确决策比例** | (盈利 + 正确不买入) / 总决策 | 综合决策质量 |

> **注意**：胜率与准确率的区别
> - **准确率**：评估所有预测样本的方向正确性（包括低置信度预测）
> - **胜率**：仅评估达到置信度阈值（默认0.55）的交易，且扣除双边交易成本（佣金0.2% + 印花税0.1% + 滑点0.2% ≈ 0.5%）
> - 因此胜率可能低于准确率：方向预测正确但收益<0.5%时，净收益为负，计入亏损交易

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

### 严谨评估工具（backtest_eval.py）⭐

`ml_services/backtest_eval.py` 对 Walk-forward 输出的 `prediction_analysis.csv` 做**合并层面**评估，
核心是**把市场 beta 从模型技能里剥离出来**。

#### 核心指标

统一记号：`p`=上涨概率；`prediction = 1 if p>=0.5`；`Actual_Return`=未来 h 天实际收益；
`Label = 1 if Actual_Return>0`；成本 `TOTAL_COST=0.5%`（双边）。
**买入信号集** = 市场情绪过滤后 `p >= Dynamic_Threshold`（normal 0.50；weak/bear 分位门槛
0.6923/0.8636 = P90/P92 快照、extreme_bear 1.0，D8）；**全样本** = 所有已验证预测。

| 指标 | 定义（公式） | 意义 |
|------|-------------|------|
| **合并准确率** | `#(prediction==Label) / n`，**全样本**池化（非逐折平均） | 方向对错，不扣成本；公平基准=50%/永远看涨 |
| **有效独立样本** `n_eff` | `n / horizon` | 重叠窗口下 n 天预测≈n/h 个独立观测，CI 基于此 |
| **信号胜率** | `#(Actual_Return>0.5%) / #(买入信号)` | 交易样本、扣成本后的净盈利比例 |
| **基准胜率** | `#(Actual_Return>0.5%) / n`（**全样本**） | "无条件买入"胜率，代表市场/板块 beta |
| **超额 lift** | **信号胜率 − 基准胜率** | **剔除行情后才是选股/择时能力**（>0 且显著） |
| **信号平均收益** | `mean(Actual_Return | 买入信号)`（**毛**收益） | 平均持有信号的收益 |
| **全样本平均收益** | `mean(Actual_Return | 全样本)`（**毛**收益） | "无条件均买"收益；两者接近=无选股价值 |
| **永远看涨** | 该组 `Actual_Return>0` 占比 | 方向准确率的公平基准 |
| **方向技能** | **准确率 − 永远看涨** | **>0 才说明方向判断超越趋势** |

> **易混点**
> - 准确率=方向、全样本、不扣成本；胜率=盈亏、仅交易样本、扣成本 → 二者会背离
>   （例：5d 2024-09 准确率 47.6% 但胜率 67.8%，因当月基准高）。
> - **只信减过基准的两个数**：`超额 lift`（对胜率）与 `方向技能`（对准确率）；
>   绝对准确率/胜率会被行情/趋势误导。
> - 平均收益是毛收益，用于看选股方向；胜率是净收益，用于看实际盈亏。
> - CI 用 `n_eff=n/horizon`：1d 无重叠（n_eff=n），20d 仅约 n/20，区间宽得多。

#### 用法

```bash
# 单周期评估
python3 ml_services/backtest_eval.py \
    --input output/<dir>/prediction_analysis.csv --horizon 20 \
    --output output/backtest_eval_20d.md

# 跨周期一致性（用另一周期的 CSV 标注「一致正/一致负/混合」）
python3 ml_services/backtest_eval.py \
    --input   output/<5d_dir>/prediction_analysis.csv  --horizon 5 \
    --compare output/<20d_dir>/prediction_analysis.csv --compare-horizon 20 \
    --output output/backtest_eval_5d.md
```

**常用参数**：`--cost`（默认 0.005，与 walk-forward 的 `TOTAL_COST` 一致）、
`--reliability-threshold`（默认 30 有效样本）。

#### 报告章节

1. 合并准确率 2. 买入胜率与基准 lift（含逐年/逐 fold）3. **板块表现**
（`config.STOCK_SECTOR_MAPPING` 关联，方向技能 + lift + 可靠性）4. 逐 fold 准确率
5. 概率校准分桶 6. **逐股票表现**（名称 + 方向技能 + lift + 跨月一致性 + `--compare`）
7. 逐月表现 8. 月份×股票最佳组合（含 `n_eff` 与多重比较警告，禁用）

#### 关键判读（2026-09-22 数据）

- **银行是"高准确率陷阱"**：20d 准确率 58.3%（全场最高），但"永远看涨"基准 62.4%
  → **方向技能 −4.2pp**；买入基准 60.7%、信号胜率 61.5% → **lift 仅 +0.9pp**。
  绝对数字漂亮，实则无技能。评估必须看方向技能与 lift。
- **板块层可迁移，个股层不可**：板块 lift 跨周期 Spearman(5d,20d)=**0.73**（p=0.0009），
  个股仅 **0.14**（p=0.29，不显著）。板块可用于倾斜，个股排名基本是噪声。

### 回测评估命令

> ⚠️ **口径分工**：评估一律用 `backtest_eval.py`（上方 ⭐）；以下两个是**策略复盘工具**，
> 不作模型评估/达标依据（D3 绝对口径）。

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
- `output/backtest_eval_{horizon}d.md`：严谨评估报告（准确率/胜率 lift/板块/个股，`backtest_eval.py` 生成）

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

## 月度护栏（Guardrail）

> **作用**：防止把"过拟合/虚假信号"当成真 alpha 上线。它不是一次性的评估，而是**决策闸门**：
> 判断策略该 **🟢 升级 / 🟡 保留低配 / 🔴 停用**（依据 `docs/DECISIONS.md` D2）。
> 命令：`python3 ml_services/monthly_guardrail.py --horizon 20`（每次 Walk-forward 后必跑）。

### 判定规则（DECISIONS D2）

| 条件 | 判定 |
|------|------|
| 净IR ≥ 0.7 且 PBO < 0.5 且 DSR ≥ 0.95 | 🟢 升级 |
| 净IR > 0（未达升级） | 🟡 保留低配 |
| 净IR ≤ 0 | 🔴 停用 |

> 三者**必须同时满足**才升级；任一不过都不能上核心仓位。
> ⚠️ 三者是**必要条件，不是充分条件**——加仓前还须过下面两道检验。

### 升配前的两道额外检验（2026-09-26 起必做）

```bash
python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10 \
    --pred output/<最新回测目录>/prediction_analysis.csv \
    --output output/portfolio_20d_<日期>.md
```

| 检验 | 用白话说就是 | 做法 | 通过标准 |
|------|------------|------|---------|
| **bootstrap CI** | "把成绩单洗牌重抽 2000 次，还站得住吗？" | 对期收益有放回重抽 2000 次，看 IR 的分布 | 95%CI **下界 > 0**（连最差的一批都为正） |
| **逐年稳健** | "是不是全靠某一年撑着？" | 按年拆开算**超额**（TopK净 − 等权基准） | 多数年为正、**无单年独撑** |

**为什么两个都要过**：只过 CI → 可能整体赚钱但全靠某一年，换年份就没了；
只过逐年 → 每年赚一点但太少，可能只是运气。**两个都过才算稳。**

**三个坑**：
1. **CI 要算「超额」口径**（TopK净 − 等权基准）。基准自己的 CI 常跨 0（2026-09-26 实测基准净IR
   0.97 [−0.16, 2.04]），只看绝对净IR 会高估。
2. **DSR 本身就是选择后校正**（`deflated_sharpe(sr[best], T, N=配置族数)`），拿最优配置去算 DSR
   是标准用法；**不要**再要求"改用先验口径复核"——那是重复扣水（lessons 三.15）。
3. **逐年样本很小**（20d 每年仅 7~12 期），单年 IR（如 2023 的 4.66）不可外推，看**符号一致性**而非大小。

**2026-09-26 实测**（`output/portfolio_20d_20260926.md`）：超额IR **1.24 [0.18, 2.41]**、
P(超额IR>0)=98.9%；逐年超额 2023/2024/2025/2026 **全正**（+1.75/+0.27/+1.70/+0.64 %/期）→ **两道全过**。

### 指标（分两层）

**① 信号层（模型排得准不准）**

| 指标 | 定义 | 判读 |
|------|------|------|
| 横截面 Rank IC | 分数与未来收益的逐日秩相关均值 | 0.02–0.06 正常 |
| ICIR | mean(IC)/std(IC) | >0.3 较好 |
| 超额 lift | 信号胜率 − 无条件买入基准 | >0 才有选股能力 |
| 方向技能 | 准确率 − 永远看涨 | >0 才超越趋势 |

**② 组合层（真拿去交易赚不赚）**

| 指标 | 定义 | 判读 |
|------|------|------|
| **净IR** | 扣成本后收益均值/标准差 × √(252/持期) | ≥0.7 且 95%CI 下限>0 |
| **PBO** | 回测过拟合概率（CSCV 法，Bailey et al.） | <0.5 未过拟合，<0.25 强证据 |
| **DSR** | 收缩后的显著性（计入试验次数与收益非正态） | ≥0.95 才显著 |
| P(IR>0.5) | bootstrap 中净IR 超过 0.5 的比例 | 越高越稳 |
| 95%CI / n_eff | 置信区间 / 有效独立样本 | CI 跨 0 = 不显著 |
| 换手 / 累计净收益 | 调仓成本 / 累计 | 成本敏感性 |

### 为什么不用准确率/绝对胜率

- **净IR**：扣成本后真正落袋的收益质量，不是"猜对多少"；
- **PBO**：防"在样本内挑出最优配置其实过拟合"（实测 YetiRank 的 PBO 0.71 因此被拒）；
- **DSR**：把"试过多少次试验"算进去，防反复调参把噪音当信号（元标签 +0.55 增量即被 bootstrap 证伪）；
- **bootstrap CI**：防单一年份点估计骗人。

### 应用示例（20d 中性 TopK）

| 口径 | 净IR | PBO | DSR | 判定 |
|------|------|-----|-----|------|
| 全期 12m（2021–2026） | 0.71 | 0.30 | 0.872 | 🟡 保留 |
| 36m（2023–2026） | 1.52 | 0.14 | 0.987 | 🟢 通过 |

> 同一策略不同口径结论不同 → 护栏逼你看清它**依赖什么行情**（受 2025 主导），这正是它的价值。

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

**GitHub Actions**：工作日香港时间 00:00 运行（UTC 16:00）

```yaml
name: 预测性能监控
on:
  schedule:
    - cron: '0 16 * * 1-5'  # 工作日UTC 16:00（香港时间00:00）
  workflow_dispatch:  # 支持手动触发
```

### 监控指标（D3 口径：诚实摘要前置，基准扣除）

> `performance_monitor.py` 现行报告先给基准扣除后的诚实指标，绝对准确率仅作参考、
> **不作达标/排名依据**（DECISIONS D3）。

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **超额 lift** | 信号净胜率 − 无条件买入基准胜率 | > 0 且显著 |
| **方向技能** | 准确率 − 永远看涨占比 | > 0 |
| **信号净胜率** | 扣双边成本后信号盈利比例 | 高于基准胜率 |
| **月度护栏（20d）** | 净IR / PBO / DSR（`monthly_guardrail.py`） | 净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 |
| 准确率 / 夏普 / 回撤 | 报告附列的绝对值（参考项，D3 禁用其排名） | — |

### 性能报告内容

1. **评估概况**（评估日期、预测周期、股票数量）
2. **诚实摘要**（lift / 方向技能 / 基准，D3 口径前置）
3. **整体性能**（准确率、胜率、收益率、夏普比率、最大回撤——附列参考）
4. **股票表现排名**（按方向技能 / lift，不按绝对准确率）
5. **月份趋势分析**（月度性能变化趋势）
6. **市场环境影响**（不同市场环境下的表现）

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

**正确做法**：一切指标**先扣基准**（D3）——lift/方向技能/净IR 优先，绝对准确率/胜率仅参考；
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
# 工作日自动执行
python3 ml_services/performance_monitor.py --mode all --horizon 20
```

### 8. 定期重新验证（现行 SOP）

- **模型/特征变更后**：必跑 walk-forward（产物自动入库）；
- **每次 walk-forward 后**：必跑 `backtest_eval.py` + `monthly_guardrail.py`；
  **升配前**再跑 `portfolio_backtest.py` 两道检验（bootstrap CI + 逐年）；
- **每月**：`monthly_guardrail.py` 复核（[DEPLOYMENT.md](DEPLOYMENT.md)）；
- **每天**：`performance_monitor.py`（诚实摘要，D3 口径）。

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

#### 训练CV准确率 vs Walk-forward准确率

**正常现象**：训练时CV准确率通常高于Walk-forward准确率，这不是数据泄漏

| 验证方法 | 典型准确率 | 原因 |
|----------|-----------|------|
| **训练时 CV** | 62-71% | TimeSeriesSplit 虽保持时序，但数据来自相似时期 |
| **Walk-forward** | 54-60% | 用过去预测未来，市场环境可能变化 |

**为什么会有差距**：
1. 训练CV使用同一时期数据，市场特征相似
2. Walk-forward跨越不同市场环境，更接近真实预测场景
3. 模型"记住"了训练期间的市场特征，但不代表数据泄漏

**数据泄漏的真实信号**：
- Walk-forward准确率 > 65%（个股）或 > 80%（恒指）
- 训练CV和Walk-forward准确率差距 < 5%
- 夏普比率异常高（> 3.0）且最大回撤极小

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

# ✅ 设置自动化调度（工作日）
```

### 陷阱9：市场情绪过滤器 lookback_days 设置错误

**问题**：Walk-forward 验证和实际预测使用相同的 `lookback_days`，导致语义混乱

**原因**：两种场景的数据可用性不同

**场景对比**：

| 场景 | 运行时机 | 数据日期 | 预测目标 | 应用的上涨比例 | `lookback_days` |
|------|---------|---------|---------|---------------|----------------|
| **Walk-forward 验证** | 模拟历史预测 | Fold 内 test 日期 | test 日期 + N 天 | test 日期 - 1 天 | `1` |
| **实际预测** | 收市后运行 | 今天 | 明天及之后 | 今天（已知） | `0` |

**关键区别**：

1. **Walk-forward 验证**模拟"在历史某个时点做预测"
   - 假设今天是 2026-05-10，要预测 2026-05-10 + 20 天
   - 此时 2026-05-10 的上涨比例还未知（还没收市）
   - 只能用 2026-05-09 的上涨比例，`lookback_days=1`

2. **实际预测**在收市后运行
   - 今天 2026-05-15 已收市，上涨比例已知
   - 预测 2026-05-16 及之后
   - 可以用 2026-05-15 的上涨比例，`lookback_days=0`

**解决方案**：

```python
# Walk-forward 验证（ml_services/walk_forward_validation.py）
market_filter = MarketSentimentFilter(lookback_days=1)  # ✅ 正确

# 实际预测（comprehensive_analysis.py）
market_filter = MarketSentimentFilter(lookback_days=0)  # ✅ 正确
```

**邮件显示**：
```
数据日期: 2026-05-15
今日上涨比例: 32.1%  ← 2026-05-15 刚收市的数据
动态阈值: 0.65
```

---

## 相关文件

- **Walk-forward验证**：`ml_services/walk_forward_validation.py`（成功后自动入库，`--no-commit` 关闭）
- **回测评估 ⭐**：`ml_services/backtest_eval.py`（lift / 方向技能 / n_eff）
- **月度护栏 ⭐**：`ml_services/monthly_guardrail.py`（净IR / PBO / DSR 判定）
- **升配两道检验 ⭐**：`ml_services/portfolio_backtest.py`（超额 CI + 逐年）
- **过拟合审计**：`ml_services/eval_overfit.py`（PBO / DSR）
- **性能监控**：`ml_services/performance_monitor.py`
- **回测产物入库**：`scripts/commit_backtest_result.py`
- **板块Walk-forward验证**：`ml_services/walk_forward_by_sector.py`
- **策略复盘（非评估口径）**：`ml_services/backtest_20d_horizon.py`、`ml_services/batch_backtest.py`
- **板块模型训练/评估**：`ml_services/train_sector_model.py`、`ml_services/evaluate_sector_model.py`

---

## 参考资料

- **Walk-forward验证最佳实践**：https://www.quantstart.com/articles/Walk-Forward-Analysis-for-Quant-Trading-Strategies/
- **回测评估指标**：https://www.investopedia.com/terms/s/sharperatio.asp
- **F1分数**：https://en.wikipedia.org/wiki/F1_score
- **夏普比率vs索提诺比率**：https://www.investopedia.com/ask/answers/021915/whats-difference-between-sharpe-ratio-and-sortino-ratio.asp
- **数据泄漏检测**：https://machinelearningmastery.com/data-leakage-machine-learning/

---

## 时间序列泄漏审查（结论）

> 2026-04 全面审查（原 TIME_SERIES_LEAKAGE_ANALYSIS.md），此处只留结论——
> 具体行号与超参数已过时，不再复述。

- **结论：无致命泄漏**。交叉验证用 `TimeSeriesSplit`（训练严格早于验证），未用随机 K-Fold；
  early stopping + 正则化防过拟合；`feature_selection` 前 `sort_index()` 保证时序。
- **正确**：`TimeSeriesSplit(n_splits=5)`；**错误**：`KFFold`（训练集可含未来数据，分数虚高）。
- 时点风险主要在**特征层**：所有 `.rolling()`/`shift(1)`/未来收益标签走
  [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) 检查表；**>65% 准确率当泄漏信号排查**。
- 运行时保障：双模式（`production`/`backtest`）、PIT 还原、embargo，
  见 [QUANT_SYSTEM_METHODOLOGY.md](QUANT_SYSTEM_METHODOLOGY.md) 阶段 1。

---

## 三周期一致预测策略（历史结论已推翻）

> 2026-04 的无 embargo 日级回测（938 样本）曾得出"验证通过、建议保持策略"，
> **2026-09-26 PIT/embargo 复测已推翻**：八大模式相对同向 20d 基准的净贡献
> Bonferroni 校正后全部不显著，"至少一周期正确率 92%"属口径膨胀。
> 现行结论见[最新验证结果](#最新验证结果)与 [THREE_HORIZON_ANALYSIS.md](THREE_HORIZON_ANALYSIS.md)。

---

**最后更新**：2026-09-27