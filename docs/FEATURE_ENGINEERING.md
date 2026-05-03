# 特征工程指南

> **最后更新**：2026-05-03（新增：三模型截面特征统一架构，LightGBM/GBDT 与 CatBoost 保持一致）

---

## 目录

- [一、核心原则与快速参考](#一核心原则与快速参考)
- [二、特征体系](#二特征体系)
- [三、问题诊断与解决方案](#三问题诊断与解决方案)
- [四、最佳实践与检查清单](#四最佳实践与检查清单)
- [五、相对标签模型切换指南](#五相对标签模型切换指南)
- [六、附录](#六附录)

---

## 一、核心原则与快速参考

### 1.1 标签定义决定模型学什么

**核心结论**：截面问题用相对标签学 Alpha，时序问题用绝对标签学 Beta/方向。**混用等于让模型"答非所问"**。

| 问题类型 | 标签类型 | 模型学习目标 | 适用场景 |
|----------|----------|--------------|----------|
| **截面问题** | 相对标签 | 学 Alpha（跑赢其他股票） | 选股 |
| **时序问题** | 绝对标签 | 学 Beta/方向（涨还是跌） | 择时 |

### 1.2 模型目标与标签匹配

| 模型类型 | 目标 | 标签类型 | 标签定义 |
|----------|------|----------|----------|
| **个股模型** | 选股（挑出跑赢的股票） | **相对标签** | `Return > Median_Return` |
| **恒指模型** | 择时（预测大盘涨跌） | **绝对标签** | `Return > 0` |

**禁止混用**：
- ❌ 个股模型用绝对标签 → 模型变成"宏观择时器"，无法选股
- ❌ 恒指模型用相对标签 → 无意义，恒指没有比较对象

### 1.3 港股市场的特殊性

港股是典型的**离岸市场**，其定价权在美债利率（分母端）和内地基本面（分子端）之间拉锯。

| 维度 | 说明 | 对模型的影响 |
|------|------|--------------|
| **分母端（美债利率）** | 港股定价受美债利率高度影响 | 利率上升→估值压缩→普跌 |
| **分子端（内地基本面）** | 盈利依赖内地经济 | 经济周期→行业分化 |
| **流动性敏感** | 离岸市场，资金流动剧烈 | 外资流入流出→全市场同涨同跌 |

**结论**：对于港股这种受美债利率和流动性高度敏感的市场，个股选股必须用相对标签是找回 Alpha 的第一步。

### 1.4 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward准确率 >65%（个股）或 >80%（恒指）通常是数据泄漏信号 |
| **IC 负值** | IC < 0 表示选股能力有限，需启用单调约束+时间衰减（滚动百分位已关闭） |
| **特征缓存版本** | 新增特征后必须清除缓存（`rm -rf data/feature_cache/*.pkl`） |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测的预处理必须一致 |

### 1.5 新增特征检查清单

| 步骤 | 检查项 | 标准 |
|------|--------|------|
| 1 | **泄漏检查** | 所有特征使用 `shift(1)` |
| 2 | **单调约束** | 有明确方向逻辑则添加到 `MONOTONE_CONSTRAINT_MAP` |
| 3 | **训练/预测一致性** | 特征变换在两处都执行 |
| 4 | **模型保存/加载** | 新增参数保存到 `save_model()` |

**百分位特征处理原则**：
- ❌ **滚动百分位已关闭**：消融实验证明其降低 IC（时间序列自归一化丢失绝对量级）
- ✅ **截面百分位已实施**：`use_cross_sectional_percentile=True` 启用（与相对标签匹配）
- **核心原因**：截面百分位与相对标签逻辑一致，都是"横向比较"
- **预测时注意**：单只股票无法计算截面排名，使用原始特征并输出警告

### 1.6 常用命令

```bash
# Walk-forward 验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 清除特征缓存
rm -rf data/feature_cache/*.pkl

# 语法检查
python3 -m py_compile ml_services/ml_trading_model.py
```

### 1.7 关键指标目标

| 指标 | 个股目标 | 恒指目标 |
|------|---------|---------|
| 准确率 | 50-65% | 60-82% |
| IC | >0.02 | >0.05 |
| 预测分散度 | >0.1 | >0.1 |

---

## 二、特征体系

### 2.1 特征数量统计

| 特征类别 | 数量 | 说明 |
|---------|------|------|
| **滚动统计特征** | 126 | 偏度、峰度、多周期波动率 |
| **价格形态特征** | 84 | 日内振幅、影线比例、缺口 |
| **量价关系特征** | 98 | 背离、OBV、成交量波动率 |
| **长期趋势特征** | 84 | MA120/250、长期收益率、长期RSI |
| **主题分布特征** | 10 | LDA主题建模（10个主题概率） |
| **主题情感交互特征** | 50 | 10个主题 × 5个情感指标 |
| **预期差距特征** | 5 | 新闻情感相对于市场预期的差距 |
| **市场环境自适应特征** | 8 | ADX+波动率双因子识别 |
| **风险管理特征** | 18 | ATR动态止损、连续市场状态记忆、盈亏比评估 |
| **事件驱动特征** | 9 | 分红、财报日期、财报超预期 |
| **股票类型特征** | 128 | 技术指标、基本面、市场环境等 |
| **GARCH 波动率特征** | 4 | 条件波动率、波动率比率、波动率变化、持续性参数 |
| **HSI 市场状态特征** | 6 | HMM 市场状态检测（状态标签、概率、持续时间、转换概率） |
| **日历效应特征** | 22 | 星期效应、月份效应、期权到期日、月初/月末等 |
| **交叉特征** | 360 | 8个类别 × 45个数值（已清理冗余） |
| **残差特征** | 35 | 特征相对市场残差（2026-04-29 新增） |
| **总计** | **770** | **含35个残差特征** |

### 2.2 特征分类原则

1. **时间序列特征**：基于历史价格、成交量数据计算
2. **市场环境特征**：基于市场状态、波动率等宏观指标
3. **基本面特征**：基于公司财务数据
4. **情感特征**：基于新闻情感、主题分析
5. **事件驱动特征**：基于分红、财报等公司事件
6. **残差特征**：剔除宏观因子后的个股特异性特征

### 2.3 核心特征类别详解

#### 2.3.0 三模型截面特征统一架构（2026-05-03 新增）

**背景**：此前仅 CatBoost 支持截面特征（`_CS_Pct`, `_CS_ZScore`），LightGBM 和 GBDT 仅支持逐只预测，导致三模型预测结果不一致。

**解决方案**：统一三模型的特征工程架构，所有模型均支持批量预测和截面特征计算。

**统一架构设计**：

| 组件 | CatBoost | LightGBM | GBDT | 说明 |
|------|----------|----------|------|------|
| **截面百分位** | ✅ | ✅ 新增 | ✅ 新增 | `CROSS_SECTIONAL_PERCENTILE_FEATURES`（55个） |
| **截面 Z-Score** | ✅ | ✅ 新增 | ✅ 新增 | `CROSS_SECTIONAL_ZSCORE_FEATURES`（43个） |
| **批量预测** | `predict_batch()` | `predict_batch()` 重写 | `predict_batch()` 重写 | 统一三阶段架构 |
| **单股回退** | `cs_feature_stats` | `cs_feature_stats` 新增 | `cs_feature_stats` 新增 | 训练集统计量填充 |
| **特征提取** | `_extract_raw_features_single()` | 新增 | 新增 | 原始特征提取 |

**批量预测三阶段流程**（三模型统一）：

```python
# 阶段1：逐只提取原始特征（不含截面特征）
all_features = {}
for code in codes:
    all_features[code] = self._extract_raw_features_single(code, predict_date, horizon)

# 阶段2：合并后联合计算截面特征
combined = pd.concat(all_features.values())
combined = self._calculate_cross_sectional_percentile_features(combined)
combined = self._calculate_cross_sectional_zscore_features(combined)

# 阶段3：逐只预测（使用正确的截面特征）
for code in all_features.keys():
    stock_data = combined[combined['Code'] == code]
    result = self._predict_from_features(code, stock_data.iloc[-1:], horizon)
```

**关键改进**：

| 改进项 | 之前 | 之后 | 效果 |
|--------|------|------|------|
| 截面特征计算 | 单只股票退化为 0.5/0.0 | 批量联合计算真实排名 | 特征值分布一致 |
| 单股预测回退 | 使用默认值 0.5/0.0 | 使用训练集统计量均值 | 更接近训练分布 |
| 模型间一致性 | 不一致 | 一致 | 融合预测更可靠 |

**单股预测回退机制**：

```python
# 训练时保存统计量
self.cs_feature_stats = {}
for col in df.columns:
    if col.endswith('_CS_Pct') or col.endswith('_CS_ZScore'):
        self.cs_feature_stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
        }

# 预测时回退
if cs_feat not in latest_data.columns:
    if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
        latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
    else:
        latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0
```

**实现文件**：
- `ml_services/ml_trading_model.py` - `CatBoostModel`, `LightGBMModel`, `GBDTModel`

---

#### 2.3.1 残差特征（35个，2026-04-29 新增）

对关键微观特征进行残差化处理，剔除宏观因子影响。

| 原始特征 | 残差特征 | 说明 |
|---------|---------|------|
| Momentum_20d | Momentum_20d_Residual | 剔除宏观后的动量 |
| RSI_14 | RSI_14_Residual | 剔除宏观后的相对强弱 |
| Volume_Ratio_5d | Volume_Ratio_5d_Residual | 剔除宏观后的成交量异动 |
| ... | ... | 共35个残差特征 |

**实现文件**：`ml_services/ml_trading_model.py`

#### 2.3.2 GARCH 波动率特征（4个）

使用 GARCH(1,1) 模型捕捉波动率聚类特性。

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `GARCH_Conditional_Vol` | 条件波动率 | GARCH(1,1) 模型拟合后的条件标准差 |
| `GARCH_Vol_Ratio` | 波动率比率 | 当前条件波动率 / 历史均值 |
| `GARCH_Vol_Change_5d` | 5日波动率变化 | 条件波动率的5日变化率 |
| `GARCH_Persistence` | 波动率持续性参数 | α + β（GARCH 参数之和） |

**实现文件**：`data_services/volatility_model.py`

#### 2.3.3 HSI 市场状态特征（6个）

使用 HMM（隐马尔可夫模型）识别恒生指数市场状态。

| 特征名称 | 说明 | 取值范围 |
|---------|------|---------|
| `HSI_Market_Regime` | 市场状态标签 | 0=震荡, 1=牛市, 2=熊市 |
| `HSI_Regime_Prob_0` | 震荡市概率 | 0-1 |
| `HSI_Regime_Prob_1` | 牛市概率 | 0-1 |
| `HSI_Regime_Prob_2` | 熊市概率 | 0-1 |
| `HSI_Regime_Duration` | 当前状态持续时间 | 1-100+ 天 |
| `HSI_Regime_Transition_Prob` | 状态转换概率 | 0-1 |

**特征重要性（个股20天模型）**：

| 特征 | 重要性排名 | 重要性得分 |
|------|-----------|-----------|
| HSI_Regime_Duration | 第2 | 3.95 |
| HSI_Regime_Prob_1 | 第3 | 2.44 |
| HSI_Regime_Prob_0 | 第10 | 1.44 |

**实现文件**：`data_services/regime_detector.py`

#### 2.3.4 日历效应特征（22个）

捕捉周期性市场规律。

**周期性编码特征（4个）**：
- `Month_Sin` / `Month_Cos`：月份周期性编码
- `DOW_Sin` / `DOW_Cos`：星期周期性编码

**星期效应特征（5个）**：
- `Day_of_Week`：星期几（0-4）
- `Is_Monday` / `Is_Friday`：周一/周五效应
- `Is_Week_End`：是否临近周末

**月份效应特征（4个）**：
- `Month`：月份（1-12）
- `Is_Month_Start` / `Is_Month_End`：月初/月末效应
- `Is_Quarter_End`：是否季末

**节假日效应特征（5个）**：
- `Days_to_Holiday`：距离最近假期天数
- `Is_Pre_Holiday` / `Is_Post_Holiday`：假期前后
- `Is_Typhoon_Season`：是否台风季（7-9月）
- `Is_Golden_Week`：是否黄金周前后

**期权到期特征（4个）**：
- `Days_to_Options_Expiry`：距离期权到期天数
- `Is_Options_Expiry_Week`：是否期权到期周
- `Is_Weekly_Options_Day`：是否周期权到期日
- `Is_Quarterly_Expiry`：是否季度期权到期

**实现文件**：`data_services/calendar_features.py`

#### 2.3.5 技术指标特征

**移动平均（MA系列）**：MA5、MA10、MA20、MA60、MA120、MA250，价格相对MA比率、MA交叉信号

**动量指标**：RSI（5日、10日、14日、20日）、MACD（DIF、DEA、MACD柱）、KDJ（K、D、J值）

**波动率指标**：ATR（5日、10日、14日、20日）、布林带（上轨、下轨、带宽）、波动率比率

**成交量指标**：成交量比率（5日、10日、20日）、OBV（能量潮）、量价背离信号

#### 2.3.6 基本面特征

PE、PB、ROE、ROA、股息率、EPS、净利率、毛利率、市值、流通市值

#### 2.3.7 市场环境特征

恒生指数收益率、相对表现、南向资金流向、VIX波动率水平、美国10年期国债收益率、标普500/纳斯达克收益率

---

## 三、问题诊断与解决方案

### 3.1 全局特征的"作弊金手指"效应

在二分类任务中预测股票涨跌时，美债利率等全局特征会导致"概率倾斜"问题。

#### 3.1.1 "大潮效应"：模型学会了"偷懒"

当预测多只股票涨跌时，如果美债利率大幅变动，市场往往会出现"普涨"或"普跌"。

**逻辑**：
- 假设明天因利率大涨，80%的股票都要跌
- 模型发现："只要看美债利率这一个特征，预测所有股票都'跌'，就能拿到80%准确率"
- 后果：模型放弃挖掘个股动量、成交量特征，因为"看宏观风向标"的收益成本比最高

这在机器学习中被称为**特征坍缩（Feature Collapse）**。

#### 3.1.2 标签高度相关：违反IID假设

| 问题 | 说明 |
|------|------|
| **标签同步化** | 美债利率把所有样本的标签"同步化"了 |
| **损失函数误导** | Cross-Entropy会惩罚预测错误的样本，若5000个样本标签全因利率变成0，模型会给利率特征极高权重 |
| **忽略个股信号** | 模型忽略那些可能在普跌中逆势上涨的个股信号 |

#### 3.1.3 预测"升跌"比预测收益率更易受干扰

| 维度 | 说明 |
|------|------|
| **缺乏区分度** | 收益率有"涨多涨少"的区别，升跌只有0和1 |
| **信噪比问题** | 美债利率是高信噪比宏观信号，个股Alpha信号通常是低信噪比 |
| **模型偏好** | 模型天然喜欢高信噪比特征，即便该特征对选股毫无帮助 |

### 3.2 实战问题

| 问题 | 说明 |
|------|------|
| **全仓买入或全仓空仓** | 模型预测结果往往是"全场都在升"或"全场都在跌"，导致无法选股，只能择时 |
| **宏观变盘时的"踩踏"** | 当美债利率进入震荡期，模型失去核心"拐杖"而变得像随机乱猜 |

### 3.3 解决方案

#### 方案1：相对标签（Cross-sectional Labeling）

**核心思想**：预测"跑赢/跑输"，而非"绝对涨跌"

```python
# 以当天所有股票收益率的中位数为基准
median_return = returns.median()
y = (returns > median_return).astype(int)
```

**效果**：美债利率权重下降，模型被迫学习个股特异性特征。

#### 方案2：特征端残差化

**核心思想**：剥离"宏观重力"

```python
from sklearn.linear_model import LinearRegression

def residualize_feature(X_micro, X_macro):
    """剔除宏观因子贡献"""
    model = LinearRegression()
    model.fit(X_macro, X_micro)
    return X_micro - model.predict(X_macro)
```

**效果**：显著降低特征之间的共线性。

#### 方案3：标签端残差化（Beta中性化）

**核心思想**：预测"是否跑赢宏观宿命"

```python
# 计算个股对宏观因子的敏感度（β）
beta = calculate_beta(stock_returns, macro_changes)
expected_return = beta * macro_change
y = (actual_return > expected_return).astype(int)
```

### 3.4 方案对比

| 方案 | 实施位置 | 核心思想 | 适用场景 | 复杂度 |
|------|----------|----------|----------|--------|
| **相对标签** | 标签端 | 以中位数为基准 | 选股为主 | ⭐ 简单 |
| **特征残差化** | 特征端 | 剔除宏观对微观特征的影响 | 降低共线性 | ⭐⭐ 中等 |
| **标签残差化** | 标签端 | 预测"跑赢宏观宿命" | 选股+择时分离 | ⭐⭐⭐ 复杂 |

### 3.5 对模型指标的影响

| 指标 | 变化 | 原因 |
|------|------|------|
| **总准确率** | 可能下降 | "送分题"被删除，留下的是真正的 Alpha |
| **信噪比** | 显著提高 | 准确率由个股特异性贡献，实盘中更稳健 |
| **预测一致性** | 变强 | 不再因美债利率抖动导致全场预测集体翻转 |
| **选股能力** | 提升 | 模型被迫学习个股差异 |

### 3.6 IC 负值诊断与修复

**问题**：IC < 0 表示预测概率高的股票实际收益反而低

**深层原因**：Regime Shift 导致特征方向翻转

| 问题 | 说明 |
|------|------|
| **特征方向不稳定** | 2020年"波动率↑ → 下跌"，2024年可能变成"波动率↑ → 上涨"（逼空行情） |
| **旧数据干扰** | 2020-2022 年的数据权重过高，干扰 2024-2025 年的预测 |
| **绝对量级失效** | 波动率=0.03 在2020年是"高波动"，在2024年可能是"低波动" |

**解决方案**：

| 方案 | 参数 | 说明 | 适用特征 |
|------|------|------|----------|
| **单调约束** | `use_monotone_constraints=True` | 强制特征方向不变 | RS_Ratio, RSI, MACD |
| **时间衰减** | `time_decay_lambda=0.5` | 降低旧数据权重 | 所有特征 |
| ~~滚动百分位~~ | ~~`use_rolling_percentile=True`~~ | ~~绝对值转历史百分位~~ | ❌ 已关闭（降低IC） |
| **截面百分位** | `use_cross_sectional_percentile=True` | 当日排名百分位 | 波动率、ATR、成交量、动量、RSI等（55个特征） |

**截面百分位特征列表（55个）**：

| 类别 | 特征 |
|------|------|
| 波动率 | Volatility_5d/10d/20d/60d/120d, GARCH_Conditional_Vol, GARCH_Vol_Ratio, Intraday_Range |
| ATR | ATR, ATR_Ratio, ATR_Risk_Score, ATR_Change_5d |
| 成交量 | Volume_Ratio_5d/20d, Volume_Volatility, OBV, CMF, Volume_Confirmation_Adaptive, Turnover_Change_5d |
| 动量 | Momentum_20d, Momentum_Accel_5d/10d, MACD_histogram, Price_Pct_20d, Close_Position |
| RSI | RSI, RSI_Deviation, RSI_ROC, RSI_Deviation_MA20 |
| 相对强度 | RS_Ratio_5d/20d, RS_Diff_5d/20d, Relative_Return |
| 布林带 | BB_Position, BB_Width, BB_Width_Normalized |
| 风险 | Max_Drawdown_20d/60d, Vol_Z_Score, Kurtosis_20d |
| 资金流向 | Smart_Money_Score, Accumulation_Score, Net_Flow_5d/20d |
| 基本面 | PE, PB, ROE, Market_Cap |

**截面 Z-Score 特征列表（43个）**：

在截面百分位基础上增加 `Volume`, `Turnover`, `Turnover_Mean_20` 等量级特征。

**注意事项**：
- 单只股票预测时无法计算截面排名，使用训练集统计量均值回退
- 三模型（CatBoost/LightGBM/GBDT）使用相同的特征列表和计算逻辑
- 截面特征在 `prepare_data()` 中合并所有股票后计算，在 `predict_batch()` 中批量计算

**滚动百分位 vs 截面百分位**：

| 特性 | 滚动百分位（已关闭） | 截面百分位（已启用） |
|------|---------------------|-------------------|
| **计算方式** | `df[feat].rolling(252).rank(pct=True)` | `df.groupby('Date')[feat].rank(pct=True)` |
| **比较对象** | 该股票过去252天的值 | 当日所有股票的值 |
| **信息保留** | ❌ 丢失绝对量级 | ✅ 保留相对排名 |
| **适用场景** | 时间序列预测 | 截面选股 |
| **与相对标签匹配** | ❌ 不匹配 | ✅ 完美匹配 |
| **夏普比率影响** | ↓（负面） | **↑11.6%** ✅ |
| **IC 影响** | ↓13%（负面） | 未改善（-0.0033） |
| **参数** | `use_rolling_percentile=False` | `use_cross_sectional_percentile=True` |

**截面百分位消融实验（2026-05-03）**：

| 指标 | 基线 | 截面百分位 | 变化 | 结论 |
|------|------|-----------|------|------|
| 准确率 | 60.77% | 59.51% | -1.26% | 正常范围 |
| IC | -0.0181 | -0.0214 | -0.0033 | 未改善 |
| **夏普比率** | 0.8672 | **0.9677** | **+11.6%** | ✅ 显著提升 |
| 最大回撤 | -0.27% | -0.22% | +0.05% | ✅ 改善 |

**截面百分位实现（已集成）**：
```python
# 已集成到 ml_trading_model.py
# 训练时自动计算，生成 _CS_Pct 后缀的新特征（9个）
# 预测时：单只股票无法计算截面排名，使用原始特征并输出警告
```

**验证方法**：
```bash
# 带 IC 修复参数的 Walk-forward 验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20
# 注意：单调约束、时间衰减、截面百分位已集成到模型默认配置中
```

---

## 四、最佳实践与检查清单

### 4.1 特征相关性筛选阈值（Pearson < 0.8）

**为什么是 0.8**：

| 阈值 | 问题 |
|------|------|
| 0.9 | 太松 — 81% 方差重叠，基本是同一信号 |
| 0.5 | 太严 — 0.5-0.8 之间可能包含互补信息 |
| **0.8** | **工程折中** — 保留足够独立的信号 |

### 4.2 数据泄漏防护

**原则**：所有特征必须使用滞后数据

```python
# ❌ 错误：使用当日数据
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

# ✅ 正确：使用滞后数据
df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(5).mean()
```

**高风险特征检查清单**：
- [ ] 所有 `.rolling()` 计算后是否有 `.shift(1)`
- [ ] 标签是否使用 `shift(-horizon)` 获取未来收益
- [ ] 价格距离/百分比特征是否使用滞后价格
- [ ] 宏观因子是否对齐到正确日期

### 4.3 新特征上线验证清单

| 步骤 | 验证项 | 标准（恒指） | 标准（个股） |
|------|--------|-------------|-------------|
| 1 | **泄漏检查** | 所有特征使用 `shift(1)` | 所有特征使用 `shift(1)` |
| 2 | **Walk-forward 验证** | 目标 ≥ 82%；> 85% 需审计 | 目标 ≥ 50%；> 65% 需审计 |
| 3 | **SHAP 排名** | 进入 top 30 | 进入 top 30 |
| 4 | **Pearson 相关性** | 与现有特征 < 0.8 | 与现有特征 < 0.8 |
| 5 | **随机种子稳定性** | 3 个种子波动 < 2% | 3 个种子波动 < 2% |
| 6 | **单调约束判断** | 有明确方向逻辑则添加 | 有明确方向逻辑则添加 |
| 7 | **截面百分位判断** | 与相对标签匹配的特征需添加 | 与相对标签匹配的特征需添加 |
| 8 | **训练/预测一致性** | 特征变换在两处都执行 | 特征变换在两处都执行 |
| 9 | **三模型一致性** | 三模型特征工程逻辑一致 | 三模型特征工程逻辑一致 |

**新增特征时的额外检查项**：

| 检查项 | 适用条件 | 操作 | 代码位置 |
|--------|----------|------|----------|
| **单调约束** | 特征有明确因果关系（如 RSI 高 → 超买） | 添加到 `MONOTONE_CONSTRAINT_MAP` | `ml_trading_model.py:4084-4108` |
| **训练/预测一致性** | 任何训练时的特征变换 | 在 `predict_proba()` 中重复执行 | `ml_trading_model.py:5216-5233` |
| **模型保存/加载** | 新增模型参数 | 保存到 `save_model()`，恢复于 `load_model()` | `ml_trading_model.py:5334-5356` |
| **三模型一致性** | 新增特征涉及截面特征 | 同步更新 `LightGBMModel` 和 `GBDTModel` | `ml_trading_model.py` |

**三模型一致性检查清单**：

当修改特征工程或新增特征时，必须同步更新三模型：

| 检查项 | CatBoostModel | LightGBMModel | GBDTModel | 验证方法 |
|--------|---------------|---------------|-----------|----------|
| **截面百分位列表** | `CROSS_SECTIONAL_PERCENTILE_FEATURES` | ✅ 保持一致 | ✅ 保持一致 | 列表长度和内容相同 |
| **截面 Z-Score 列表** | `CROSS_SECTIONAL_ZSCORE_FEATURES` | ✅ 保持一致 | ✅ 保持一致 | 列表长度和内容相同 |
| **批量预测方法** | `predict_batch()` | ✅ 统一架构 | ✅ 统一架构 | 三阶段流程一致 |
| **特征提取方法** | `_extract_raw_features_single()` | ✅ 统一接口 | ✅ 统一接口 | 返回相同特征集 |
| **统计量保存/加载** | `save_model()` / `load_model()` | ✅ 持久化 | ✅ 持久化 | `cs_feature_stats` 字段 |
| **单股回退机制** | `_predict_from_features()` | ✅ 统一回退 | ✅ 统一回退 | 训练集统计量填充 |

**三模型同步修改示例**：

```python
# 新增截面特征时，需在三处同步添加

# 1. CatBoostModel.CROSS_SECTIONAL_PERCENTILE_FEATURES
# 2. LightGBMModel.CROSS_SECTIONAL_PERCENTILE_FEATURES  
# 3. GBDTModel.CROSS_SECTIONAL_PERCENTILE_FEATURES

# 同时更新 Z-Score 列表（如适用）
# 1. CatBoostModel.CROSS_SECTIONAL_ZSCORE_FEATURES
# 2. LightGBMModel.CROSS_SECTIONAL_ZSCORE_FEATURES
# 3. GBDTModel.CROSS_SECTIONAL_ZSCORE_FEATURES
```

**单调约束方向参考**（2026-05-03 优化，基于业界实践）：

| 特征类别 | 约束方向 | 理论依据 |
|----------|----------|----------|
| RS_Ratio（相对强度） | `+1` 递增 | RS↑ → 跑赢恒指 → 跑赢其他股票 |
| RSI | `-1` 递减 | RSI高 → 超买 → 可能下跌 |
| MACD_histogram | `+1` 递增 | MACD柱正 → 上涨趋势 |

**已移除的约束**（原因：相对标签模型下方向不稳定）：

| 特征类别 | 原约束 | 移除原因 |
|----------|--------|----------|
| 波动率（Volatility, ATR, GARCH） | `-1` | 牛市高波动跑赢，熊市低波动抗跌，方向取决于市场状态 |
| 股息（Dividend_Yield） | `+1` | 港股股息溢价不明显，高股息股票可能是防御股 |
| 情感（sentiment_ma） | `+1` | 可能是反向指标（情感高 = 过度乐观） |

**验证命令**：
```bash
# Walk-forward 验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 特征重要性分析
python3 scripts/analyze_feature_importance_by_horizon.py
```

### 4.4 特征冗余清理记录

| 清理类别 | 删除数量 | 原因 |
|---------|---------|------|
| RS_Signal 重复 | 251 | 与 Trend 公式完全相同 |
| PE/PB 常量交互 | 18 | PE/PB 为常量，交互无信息 |
| 数学等价特征 | 15 | Returns/Volatility/Momentum 等重复 |
| 第二轮 r=1.0 冗余 | ~23 | GARCH_Persistence 等 |
| **合计** | **~307** | 特征数量 1045→730 |

---

## 五、相对标签模型切换指南

> **关键转折点**：切换到相对涨跌（Cross-sectional Relative Labeling）意味着正式从"宏观择时"转向"个股选优（Alpha 挖掘）"。目标（Label）已变，特征工程和模型逻辑必须随之调整。

### 5.1 特征端的"大扫除"：处理宏观因子

现在标签是相对的，美债利率这类"全局一致"的特征对模型来说已经变成了**纯噪声**。

| 操作 | 说明 | 理由 |
|------|------|------|
| **暂时剔除全局特征** | 美债利率、汇率、大盘指数等 | 强行保留会干扰模型对微观特征的权重分配 |
| **或转换为"变动率的变动率"** | 提取二阶变化信息 | 保留部分信息价值 |

### 5.2 特征标准化：从"绝对值"转向"截面排名"

由于标签是按天计算的相对排名，输入特征也必须进行横截面处理。

**公式**：

$$Feature_{norm} = \frac{Feature_{i,t} - \mu(Features_t)}{\sigma(Features_t)}$$

**效果**：确保模型学习的是"这只票的动量在全场排前 10%"，与相对涨跌标签逻辑对齐。

### 5.3 网络特征的价值释放

> **核心论点**：在切换到相对涨跌标签并实施截面标准化后，网络特征（Network Features）的价值才真正从宏观噪音中被"解放"出来。

**为什么网络特征在"相对模型"中更有用？**

相对标签模型的核心是"选美"，即找出在相同环境下表现更好的标的。网络特征本质上描述的是"个体与群体的关系"，这与相对标签的逻辑完美契合。

| 特征类型 | 在绝对模型中的表现 | 在相对模型中的表现 |
|---------|-------------------|-------------------|
| **中心度 (Centrality)** | 容易被大盘波动干扰，信噪比低 | 反映股票在行业/概念簇中的信息权重 |
| **邻居动量 (Spillover)** | 往往跟随大盘共振，缺乏区分度 | 捕捉领先-滞后效应（谁带涨，谁跟风） |
| **社区发现 (Clustering)** | 只能看到行业块状移动 | 识别资金流向（资金从哪个簇转移到哪个簇） |

**网络特征的三个实战增量**：

| 增量 | 说明 | 理论依据 | 港股适用性 |
|------|------|----------|----------|
| **领先-滞后效应** | 邻居跑赢中位数 → 补涨概率提升 | ✅ 强：空间相关性是时序特征无法捕捉的 | ✅ 高：港股板块联动明显 |
| **识别"伪强势"** | 区分"孤立上涨"vs"全网共振" | ✅ 强：区分有意义 | ✅ 高：港股小盘股常出现"妖股"异动 |
| **拥挤度预警** | 中心度飙升 → 反转前兆 | ⚠️ 存疑：均值回归假设在趋势市场中可能失效 | ⚠️ 中等：拥挤可能持续更久 |

**二阶网络特征建议**：

| 特征 | 计算方法 | 预测逻辑 | 建议优先级 |
|------|----------|----------|-----------|
| **节点偏离度** | $Score_{i} - \text{Average}(Score_{neighbors})$ | 捕捉"掉队补涨"或"领涨回调" | **P1 - 优先实施** |
| **动态连通性** | 计算个股在过去5天内网络邻居的变动频率 | 邻居变动剧烈说明该股正在经历逻辑切换 | **P2 - 中期目标** |

### 5.4 验证体系的升级：从 Accuracy 到分层收益

切换到相对涨跌后，IC（信息系数）应该会从负数开始转正。

**分层回测（Decile Analysis）**：

1. 按照模型输出的概率分值，将股票分成 10 组
2. 计算 Top 组（前 10%）相对于 Bottom 组（后 10%）的多空收益差

**核心观察点**：只要 Top 组的实际收益稳步高于 Bottom 组，即便总 Accuracy 只有 52%-53%，这个模型也是极其成功的。

**验证指标**：

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| **IC** | 预测值与实际收益率的Pearson相关系数 | 衡量预测能力 |
| **Rank IC** | 预测排名与实际收益排名的Spearman相关系数 | 衡量排序能力 |
| **ICIR** | IC的均值/标准差 | 衡量IC稳定性 |
| **预测分散度** | 预测结果的标准差 | 避免"全涨全跌" |

### 5.5 关键代码逻辑检查清单

请检查 `ml_trading_model.py` 是否完成以下转换：

| 检查项 | 之前（绝对涨跌） | 之后（相对涨跌） |
|--------|-----------------|-----------------|
| **Label 定义** | $Return > 0$ | $Return > Median\_Return$ |
| **特征缩放** | 全局 Min-Max / Standard | 每日截面 Rank / Z-Score |
| **主要目标** | 择时（猜对大盘） | 选股（猜对排名） |
| **核心指标** | Accuracy / F1 | Rank IC / ICIR / 分层收益 |

### 5.6 IC 负值的诊断步骤

**如果切换标签后 IC 依然为负**，请立即检查特征逻辑。

1. **检查截面标准化**：没有截面标准化的特征，在相对标签模型里往往会产生负 IC
2. **检查特征方向**：将最重要的特征反转（乘以 -1），看 IC 是否变正
   - 如果变正了，说明特征捕捉到的是"拥挤度导致的下跌"而非"动量持续"

**IC 绝对值的判断**：

| IC 绝对值 | 含义 | 建议 |
|----------|------|------|
| **绝对值大** | 离成功只差一个"方向纠偏" | 检查特征方向，可能需要反转 |
| **绝对值小** | 特征预测能力弱 | 需要重新设计特征 |

### 5.7 行动建议优先级

| 优先级 | 行动 | 预期效果 |
|--------|------|---------|
| **P0** | 全特征截面标准化（Rank/Z-Score） | 消除负 IC 的主要来源 |
| **P1** | 新增"节点偏离度"特征 | 捕捉领先-滞后效应 |
| **P1** | 实施分层回测（Decile Analysis） | 验证模型真实预测能力 |
| **P2** | 新增"动态连通性"特征 | 捕捉逻辑切换点 |

---

## 六、附录

### 6.1 缓存机制

| 缓存类型 | 位置 | 有效期 | 加速效果 |
|---------|------|--------|---------|
| 原始数据 | `data/stock_cache/` | 7天 | - |
| 特征缓存 | `data/feature_cache/` | 7天 | **170x** |

**清除缓存**：
```bash
rm -rf data/feature_cache/*.pkl
```

### 6.2 相关文件

| 文件 | 说明 |
|------|------|
| `ml_services/ml_trading_model.py` | 特征工程实现（FeatureEngineer 类），包含三模型（CatBoostModel / LightGBMModel / GBDTModel） |
| `data_services/volatility_model.py` | GARCH 波动率 |
| `data_services/regime_detector.py` | HSI 市场状态 |
| `data_services/calendar_features.py` | 日历效应 |
| `data_services/network_features.py` | 网络特征 |
| `data_services/feature_residualizer.py` | 特征残差化 |
| `docs/FEATURE_IMPORTANCE_ANALYSIS.md` | 特征重要性分析 |
| `docs/FEATURE_LABEL_COMPATIBILITY_PLAN.md` | 特征与标签兼容性修复计划 |

### 6.3 参考资料

- **CatBoost 官方文档**：https://catboost.ai/docs/
- **特征工程最佳实践**：https://www.kaggle.com/learn/feature-engineering
- **时间序列特征工程**：https://machinelearningmastery.com/feature-engineering-for-time-series/

---

*文档版本：v4.1*
*更新日期：2026-05-03*
