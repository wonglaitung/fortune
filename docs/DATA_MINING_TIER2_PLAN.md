# 深度数据挖掘增强方案（五维度分析）

**创建时间**: 2026-04-26
**基线指标**: Walk-forward 20d 准确率 82.23%（恒指）、60.11%（个股）
**训练窗口**: 12 个月（已验证 18/24 个月反而降低准确率）

---

## 一、维度评估与优先级

| 维度 | 核心价值 | 预期提升 | 复杂度 | 泄漏风险 | 推荐度 |
|------|---------|---------|--------|---------|--------|
| 2. Regime 增强 | 扩展现有 HMM，增加转换动力学 | +0.3-1.0% | 低(3-5h) | 中 | ⭐⭐⭐⭐⭐ Tier 1 |
| 3. 跨尺度关联 | vol-of-vol + 动量一致性 | +0.5-1.5% | 中(6-8h) | 中 | ⭐⭐⭐⭐⭐ Tier 1 |
| 1. 信息衰减 | MI 多尺度特征矩阵 | +0.5-1.5% | 高(6-8h) | 低 | ⭐⭐⭐⭐ Tier 1 |
| 4. 模式指纹 | Matrix Profile + 异常评分 | +0.2-0.8% | 高(8-10h) | 高 | ⭐⭐⭐ Tier 2 |
| 5. 拓扑结构 | 市场广度（仅最小实现） | +0.1-0.5% | 中(4-6h) | 高 | ⭐⭐ Tier 2 |

---

## 二、Tier 1 实施方案

### 2.1 Regime 增强（扩展现有模块）

**修改文件**：`data_services/regime_detector.py`

**现状**：已有 6 个特征（Market_Regime, Regime_Prob_0/1/2, Regime_Duration, Regime_Transition_Prob）。SHAP 排名：Regime_Duration 第 13 位。

**新增 4 个特征**：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `Regime_Switch_Prob_5d` | `T^5[current, other_states]`，T 为转移矩阵 | 5 天内转换到不同状态的概率 |
| `Regime_Expected_Duration` | `1 / (1 - T[i,i])` | 当前状态期望剩余持续时间 |
| `Regime_Momentum` | `Regime_Prob_1(t) - Regime_Prob_1(t-5)` | 状态概率 5 日变化（增强/减弱） |
| `Regime_Vol_Interaction` | `GARCH_Conditional_Vol × Regime_Transition_Prob` | 高波动+高转换=动荡期 |

**实现要点**：
- `Regime_Switch_Prob_5d`：用 `numpy.linalg.matrix_power(transmat, 5)` 计算
- `Regime_Expected_Duration`：从转移矩阵对角线直接计算
- `Regime_Momentum`：在 `calculate_features()` 中用 `.diff(5)` 计算
- `Regime_Vol_Interaction`：需要 GARCH_Conditional_Vol 先计算好（已有），在 `calculate_features()` 中交叉计算
- 所有新特征加入 `shift(1)` 防泄漏（与现有代码 line 309-314 一致）

**集成**：修改 `REGIME_FEATURE_CONFIG`，更新 `get_feature_names()`，无需新增文件。

---

### 2.2 跨尺度关联（新模块）

**新增文件**：`data_services/multiscale_features.py`

**新增 5 个特征**：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `Vol_of_Vol_20d` | `Volatility_20d.rolling(20).std().shift(1)` | 波动率的波动率（vol-of-vol 飙升预示趋势转折） |
| `Vol_Ratio_5d_20d` | `Volatility_5d / Volatility_20d` | 短期/长期波动率比率（>1 表示短期波动加剧） |
| `Return_1d_5d_Correlation` | `Return_1d.rolling(20).corr(Return_5d).shift(1)` | 1d/5d 收益率的滚动相关性（跨尺度耦合强度） |
| `Vol_Cluster_Signal` | `Vol_of_Vol_20d > 60日分位数90%` | 波动率聚集信号（趋势转折前兆） |
| `Momentum_Consistency` | `sign(R_1d) + sign(R_5d) + sign(R_20d)` | 动量方向一致性 [-3,+3]，编码 111/000 等模式 |

**核心洞察**：
- 短期波动加剧（即使均值没变）往往预示中长期趋势的转折
- `Momentum_Consistency` 将三周期一致/背离模式编码为连续特征，让 CatBoost 直接利用
- `Vol_of_Vol_20d` 是经典的"波动率体制转换"前导指标

**防泄漏**：所有特征使用 `shift(1)`

**集成**：
- 新增 `from data_services.multiscale_features import MultiscaleFeatureCalculator, MULTISCALE_FEATURE_CONFIG`
- `FEATURE_CONFIG` 新增 `'multiscale_features'`
- `calculate_features()` 中在 GARCH/Regime 之后调用

---

### 2.3 信息衰减分析（新模块）

**新增文件**：`data_services/info_decay_analyzer.py`

**核心方法**：计算每个特征在不同 Lag（1,5,10,20）下与目标变量的互信息（Mutual Information），识别"快变量"（短期有效）和"慢变量"（长期有效）。

**新增 5 个特征**：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `MI_Fast_Signal_Count` | 快变量中正值特征的数量 | 短期动量共识强度 |
| `MI_Slow_Signal_Count` | 慢变量中正值特征的数量 | 长期趋势共识强度 |
| `MI_Fast_Slow_Divergence` | `Fast_Count - Slow_Count` | 快慢信号背离=潜在反转 |
| `MI_Decay_Rate_RSI` | RSI 在各 lag 的 MI 归一化衰减速率 | RSI 信号衰减有多快 |
| `MI_Decay_Rate_MACD` | MACD_Hist 的 MI 归一化衰减速率 | MACD 信号衰减有多快 |

**两阶段实现**：

**阶段 1：离线 MI 分析**（不纳入生产流程）
- 使用 `sklearn.feature_selection.mutual_info_classif`
- 对每个特征计算 lag=1,5,10,20 的 MI 值
- 输出：`data/mi_lag_assignments.json`（每个特征的最优 lag 和衰减速率）
- **防泄漏**：MI 分析使用历史数据（至少 6 个月前），结果冻结使用

**阶段 2：运行时特征生成**
- 读取 `mi_lag_assignments.json`
- 对特征应用最优 lag（如 RSI 最优 lag=5 → 使用 `RSI.shift(5)`）
- 计算快/慢信号聚合特征和衰减速率特征

**防泄漏关键规则**：
- MI 分析结果在 Walk-forward 的每个 fold 开始时基于训练数据重新计算
- lag 分配在 fold 内保持冻结
- 所有特征保持 `shift(1)` 最小偏移

**依赖**：`sklearn.feature_selection.mutual_info_classif`（已有）

---

## 三、Tier 2 实施方案（Tier 1 验证后）

### 2.4 模式指纹（部分实现）

**新增文件**：`data_services/motif_anomaly_features.py`

**新增 3 个特征**：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `Matrix_Profile_Distance` | `stumpy.stump`，窗口=20天，回看252天 | 与最近邻子序列的距离（大=异常，小=模式匹配） |
| `Matrix_Profile_Index` | 最近邻的时间位置 | 近期匹配=模式重复，远期匹配=罕见模式 |
| `IF_Anomaly_Score_HSI` | 复用 Isolation Forest 逻辑应用于恒指特征向量 | 恒指多维异常评分 |

**新依赖**：`pip install stumpy`

**高泄漏风险**：必须确保 `stumpy.stump` 仅使用过去数据

---

### 2.5 市场结构（最小实现）

**新增文件**：`data_services/market_structure_features.py`

**新增 2 个特征**：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| `Market_Breadth` | 成份股中 Close > MA20 的比例 | 市场广度（与指数背离=反转信号） |
| `Sector_Momentum_Spread` | 最强3板块 vs 最弱3板块的动量差 | 板块轮动强度 |

**实现要点**：
- `Market_Breadth`：获取成份股数据（`config.py` TRAINING_STOCKS）
- `Sector_Momentum_Spread`：利用 `data_services/hk_sector_analysis.py` 的板块数据
- **需确认**：Market_Breadth 与现有 `MA_Bullish_Alignment` 相关性 < 0.7

---

## 四、特征预算

| 类别 | 当前 | Tier 1 新增 | Tier 2 新增 | 池总量 |
|------|------|-----------|-----------|--------|
| 基础/技术/宏观 | 19 | - | - | 19 |
| 日历效应 | 22 | - | - | 22 |
| GARCH | 4 | - | - | 4 |
| Regime | 6 | +4 | - | 10 |
| 跨尺度 | 0 | +5 | - | 5 |
| 信息衰减 | 0 | +5 | - | 5 |
| 模式指纹 | 0 | - | +3 | 3 |
| 市场结构 | 0 | - | +2 | 2 |
| **合计** | **51** | **+14** | **+5** | **70** |

生产模型通过 SHAP 选择 < 50 特征（当前增强模型 33 个）。

---

## 五、实施顺序

```
Step 1: Regime 增强（扩展现有模块，最快见效）
Step 2: 跨尺度关联（新模块，理论扎实）
Step 3: 信息衰减分析（最复杂，但最创新）
Step 4: Walk-forward 验证 Tier 1，对比基线 82.23%
Step 5: SHAP 特征选择，确认最终特征集 < 50
Step 6: （如果 Tier 1 验证有效）实施 Tier 2
```

---

## 六、验证协议

每个维度完成后严格执行：

1. **泄漏检查**：确认新特征 `feature[t]` 不依赖 `t` 及之后的数据
2. **Walk-forward 验证**：`python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20`
3. **SHAP 排名**：新特征应在 top 30，否则删除
4. **相关性检查**：新特征与现有特征 Pearson < 0.8，否则删除
5. **准确率阈值**：20d > 85% 需审计泄漏（当前基线 82.23%）
6. **稳定性**：3 个随机种子 (42/123/456) 准确率波动 < 2%

---

## 七、关键文件清单

| 用途 | 文件路径 |
|------|---------|
| Regime 增强 | `data_services/regime_detector.py`（修改） |
| 跨尺度关联 | `data_services/multiscale_features.py`（新建） |
| 信息衰减 | `data_services/info_decay_analyzer.py`（新建） |
| 模式指纹 | `data_services/motif_anomaly_features.py`（新建，Tier 2） |
| 市场结构 | `data_services/market_structure_features.py`（新建，Tier 2） |
| HSI 模型集成 | `ml_services/hsi_ml_model.py`（修改 FEATURE_CONFIG + calculate_features） |
| Walk-forward 验证 | `ml_services/hsi_walk_forward.py` |
| SHAP 特征选择 | `ml_services/hsi_feature_selection.py` |

---

*文档创建时间: 2026-04-26*
