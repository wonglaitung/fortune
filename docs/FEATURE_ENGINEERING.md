# 特征工程指南

> **最后更新**：2026-05-09 | **版本**：v2.0

本文档是港股智能分析系统的特征工程完整指南，涵盖特征设计、实现、验证和维护的全流程。

---

## 目录

1. [快速参考](#快速参考)
2. [特征概览](#特征概览)
3. [核心特征类别](#核心特征类别)
4. [特征架构设计](#特征架构设计)
5. [特征工程最佳实践](#特征工程最佳实践)
6. [关键经验教训](#关键经验教训)
7. [案例分析：个股模型特征工程改进](#案例分析个股模型特征工程改进)
8. [附录](#附录)

---

## 快速参考

### 核心警告

| 警告 | 说明 | 后果 |
|------|------|------|
| **数据泄漏** | 特征使用了当日数据 | Walk-forward 准确率异常高（个股>65%，恒指>80%） |
| **绝对值特征** | 跨股票训练时使用未标准化的价格/成交量特征 | 模型学到无意义的"高价股"模式 |
| **缓存不一致** | 网络特征更新后未清除缓存 | 训练/预测特征不一致，性能下降 |
| **默认值混淆** | 默认值落在有效值范围内 | 模型无法区分"缺失"和"有效值" |

### 常用命令

```bash
# 模型训练（使用全量特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 特征选择
python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20

# Walk-forward 验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 清除特征缓存（新增特征后必须执行）
rm -rf data/feature_cache/*.pkl
```

### 特征数量速览

| 类别 | 数量 | 说明 |
|------|------|------|
| 完整特征 | ~1450 | 包含所有特征类别 |
| 排除的绝对值特征 | 40 | 自动标准化或排除 |
| 推荐特征数（特征选择后） | 300 | 平衡性能和效率 |

---

## 特征概览

### 特征数量统计

| 特征类别 | 数量 | 说明 | 实现文件 |
|---------|------|------|----------|
| **滚动统计特征** | 126 | 偏度、峰度、多周期波动率 | `ml_trading_model.py` |
| **价格形态特征** | 84 | 日内振幅、影线比例、缺口 | `ml_trading_model.py` |
| **量价关系特征** | 98 | 背离、OBV、成交量波动率 | `ml_trading_model.py` |
| **长期趋势特征** | 84 | MA120/250、长期收益率、长期RSI | `ml_trading_model.py` |
| **GARCH 波动率特征** | 4 | 条件波动率、波动率比率 | `data_services/volatility_model.py` |
| **HSI 市场状态特征** | 6 | HMM 市场状态检测 | `data_services/regime_detector.py` |
| **日历效应特征** | 22 | 星期、月份、节假日效应 | `data_services/calendar_features.py` |
| **网络社区特征** | 7 | 社区ID、中心性、聚类系数 | `ml_services/stock_network_analysis.py` |
| **市场-网络交叉特征** | 434 | 市场指标 × 网络特征 | `ml_trading_model.py` |
| **主题分布特征** | 10 | LDA主题建模 | `ml_trading_model.py` |
| **主题情感交互特征** | 50 | 主题 × 情感指标 | `ml_trading_model.py` |
| **事件驱动特征** | 9 | 分红、财报日期 | `ml_trading_model.py` |
| **交叉特征** | 360 | 类别 × 数值特征 | `ml_trading_model.py` |
| **总计** | **~1450** | **完整特征集** | - |

### 特征分类

按数据源分类：

```
特征体系
├── 时间序列特征（~500个）
│   ├── 价格特征：MA、布林带、价格通道
│   ├── 成交量特征：OBV、成交量比率
│   ├── 动量特征：RSI、MACD、KDJ
│   └── 波动率特征：ATR、GARCH
├── 市场环境特征（~100个）
│   ├── HSI市场状态：HMM检测
│   ├── 宏观指标：VIX、南向资金
│   └── 日历效应：节假日、期权到期
├── 网络特征（~450个）
│   ├── 拓扑特征：中心性、聚类系数
│   ├── 社区特征：社区ID、桥梁股票
│   └── 交叉特征：市场×网络
├── 基本面特征（~50个）
│   ├── 估值指标：PE、PB
│   ├── 盈利指标：ROE、ROA
│   └── 其他：股息率、市值
├── 情感特征（~65个）
│   ├── 主题分布：LDA建模
│   ├── 情感指标：正面/负面情绪
│   └── 预期差距：新闻 vs 市场
└── 交叉特征（~360个）
    └── 类别特征 × 数值特征
```

### 特征冗余清理记录

| 清理日期 | 清理内容 | 删除数量 | 原因 |
|---------|---------|---------|------|
| 2026-04-27 | RS_Signal 重复 | 251 | 与 Trend 公式完全相同 |
| 2026-04-27 | PE/PB 常量交互 | 18 | PE/PB 为常量，交互无信息 |
| 2026-04-27 | 数学等价特征 | 15 | Returns/Volatility/Momentum 等重复 |
| 2026-04-27 | r=1.0 冗余 | 23 | GARCH_Persistence 等 |
| **合计** | - | **307** | 1045→730 |

---

## 核心特征类别

### 1. GARCH 波动率特征（4个）

使用 GARCH(1,1) 模型捕捉波动率聚类特性。

| 特征名称 | 说明 | 计算方法 | 业务意义 |
|---------|------|---------|---------|
| `GARCH_Conditional_Vol` | 条件波动率 | GARCH(1,1) 条件标准差 | 当前市场波动水平 |
| `GARCH_Vol_Ratio` | 波动率比率 | 当前/历史均值 | 波动率相对高低 |
| `GARCH_Vol_Change_5d` | 5日波动率变化 | 5日变化率 | 波动率趋势 |
| `GARCH_Persistence` | 波动率持续性 | α + β | 波动率聚类程度 |

**实现示例**：
```python
from arch import arch_model

# 拟合 GARCH(1,1)
returns = df['Close'].pct_change().dropna() * 100
model = arch_model(returns, vol='Garch', p=1, q=1)
res = model.fit(disp='off')

# 提取特征
df['GARCH_Conditional_Vol'] = res.conditional_volatility
df['GARCH_Persistence'] = res.params['alpha[1]'] + res.params['beta[1]']
```

**实现文件**：`data_services/volatility_model.py`

---

### 2. HSI 市场状态特征（6个）

使用 HMM（隐马尔可夫模型）识别恒生指数市场状态。

| 特征名称 | 说明 | 取值范围 | 业务意义 |
|---------|------|---------|---------|
| `HSI_Market_Regime` | 市场状态标签 | 0=震荡, 1=牛市, 2=熊市 | 当前市场环境 |
| `HSI_Regime_Prob_0` | 震荡市概率 | 0-1 | 状态置信度 |
| `HSI_Regime_Prob_1` | 牛市概率 | 0-1 | 状态置信度 |
| `HSI_Regime_Prob_2` | 熊市概率 | 0-1 | 状态置信度 |
| `HSI_Regime_Duration` | 状态持续时间 | 1-100+ 天 | 趋势稳定性 |
| `HSI_Regime_Transition_Prob` | 状态转换概率 | 0-1 | 趋势反转风险 |

**特征重要性（个股20天模型）**：

| 特征 | 重要性排名 | 重要性得分 |
|------|-----------|-----------|
| HSI_Regime_Duration | 第2 | 3.95 |
| HSI_Regime_Prob_1 | 第3 | 2.44 |
| HSI_Regime_Prob_0 | 第10 | 1.44 |
| HSI_Market_Regime | 第20 | 1.00 |

**实现文件**：`data_services/regime_detector.py`

---

### 3. 日历效应特征（22个）

捕捉周期性市场规律，分为四大类：

#### 3.1 周期性编码特征（4个）

| 特征名称 | 说明 | 取值范围 |
|---------|------|---------|
| `Month_Sin` / `Month_Cos` | 月份周期性编码 | [-1, 1] |
| `DOW_Sin` / `DOW_Cos` | 星期周期性编码 | [-1, 1] |

**优势**：避免 12月（12）和 1月（1）的数值断裂问题。

#### 3.2 星期效应特征（5个）

| 特征名称 | 说明 |
|---------|------|
| `Day_of_Week` | 星期几（0-4） |
| `Is_Monday` | 周一效应 |
| `Is_Friday` | 周五效应 |
| `Is_Week_End` | 是否临近周末 |

#### 3.3 节假日效应特征（5个）

| 特征名称 | 说明 |
|---------|------|
| `Days_to_Holiday` | 距离最近假期天数 |
| `Is_Pre_Holiday` | 假期前 |
| `Is_Post_Holiday` | 假期后 |
| `Is_Typhoon_Season` | 台风季（7-9月） |
| `Is_Golden_Week` | 黄金周前后 |

#### 3.4 期权到期特征（4个）

| 特征名称 | 说明 |
|---------|------|
| `Days_to_Options_Expiry` | 距离期权到期天数 |
| `Is_Options_Expiry_Week` | 期权到期周 |
| `Is_Weekly_Options_Day` | 周期权到期日 |
| `Is_Quarterly_Expiry` | 季度期权到期 |

**实现文件**：`data_services/calendar_features.py`

---

### 4. 网络社区特征（7个）

基于股票网络分析提取的拓扑特征。

| 特征名称 | 说明 | 取值范围 | 业务意义 |
|---------|------|---------|---------|
| `net_community_id` | 网络社区 ID | 0-6（7个社区） | 股票所属群落 |
| `net_degree_centrality` | 度中心性 | 0-1 | 连接广度 |
| `net_betweenness_centrality` | 介数中心性 | 0-1 | 信息传递枢纽 |
| `net_eigenvector_centrality` | 特征向量中心性 | 0-1 | 连接质量 |
| `net_closeness_centrality` | 接近中心性 | 0-1 | 信息到达速度 |
| `net_clustering_coeff` | 聚类系数 | 0-1 | 局部连接密度 |
| `net_constraint` | 结构洞约束 | 0-1 | 信息控制能力 |

**默认值处理**：
- `net_community_id = -1`：股票不在网络中
- 中心性特征：用中位数填充

**实现文件**：`ml_services/stock_network_analysis.py`

---

### 5. 技术/基本面特征摘要

#### 技术指标特征

| 类别 | 特征 | 标准化替代 |
|------|------|-----------|
| 移动平均 | MA5~MA250 | MA_Ratio 系列 |
| 动量指标 | RSI(5/10/14/20), MACD, KDJ | 已是 0-100 范围，无需标准化 |
| 波动率 | ATR, 布林带 | ATR_Pct, BB_Ratio 系列 |
| 成交量 | OBV, 成交量比率 | OBV_Trend, Volume_Ratio 系列 |

#### 基本面特征

| 类别 | 特征 |
|------|------|
| 估值 | PE、PB、PS |
| 盈利 | ROE、ROA、净利率、毛利率 |
| 分红 | 股息率、EPS |
| 规模 | 市值、流通市值 |

---

## 特征架构设计

### 单一真相源原则

**问题**：特征处理逻辑分散在多个文件，维护困难且易不一致。

**解决方案**：所有特征处理逻辑集中在 `ml_trading_model.py`。

```
ml_trading_model.py（单一真相源）
├── 模块级常量
│   └── ABSOLUTE_PRICE_FEATURES（40个绝对值特征）
│
├── BaseTradingModel 类
│   ├── get_feature_columns()     # 排除绝对值，返回有效特征
│   ├── prepare_features_for_selection()  # 特征选择专用
│   └── prepare_data()            # 完整特征准备
│
└── FeatureEngineer 类
    ├── 计算技术指标
    ├── 生成交叉特征
    └── 处理 NaN 和默认值

其他模块（通过导入复用）
├── feature_selection.py
│   └── model.prepare_features_for_selection()
├── walk_forward_validation.py
│   └── model.get_feature_columns()
└── hyperparameter_tuner.py
    └── model.prepare_data()
```

### 绝对价格特征排除列表（40个）

**问题**：绝对价格特征跨股票量级差异大（腾讯 ~400元 vs 小盘股 ~5元），可能导致模型学到无意义模式。

**解决方案**：自动排除绝对值特征，使用标准化替代。

| 类别 | 绝对值特征 | 标准化替代 | 标准化方法 |
|------|-----------|-----------|-----------|
| 价格通道 | Channel_High/Low_20d | Channel_High/Low_Ratio_20d | ÷ prev_close |
| 支撑阻力 | Support/Resistance_120d | Support/Resistance_Ratio_120d | ÷ prev_close |
| 均线 | MA5~MA250 | MA_Ratio 系列 | ÷ prev_close |
| 布林带 | BB_upper/lower/middle | BB_Ratio 系列 | ÷ prev_close |
| ATR | ATR, ATR_MA | ATR_Pct | ÷ prev_close |
| 成交额 | Turnover, Turnover_Mean/Std_20 | Turnover_Z_Score | Z-score 标准化 |
| 成交量 | Volume_MA7/120/250 | Volume_Ratio 系列 | ÷ 历史均值 |
| OBV | OBV, OBV_MA5 | OBV_Trend | 差分/变化率 |
| VWAP | VWAP | VWAP_Ratio | ÷ close |
| 技术指标 | MACD, MACD_signal, TP | 比率版本 | ÷ close |

**标准化代码示例**：
```python
# 标准化：除以前一日收盘价（避免数据泄漏）
prev_close = df['Close'].shift(1)
df['Channel_High_Ratio_20d'] = df['Channel_High_20d'] / prev_close
df['MA_Ratio_20d'] = df['MA20'] / prev_close
df['ATR_Pct'] = df['ATR'] / prev_close
```

**标准化特征解读**：

| 特征 | 值 | 解读 |
|------|-----|------|
| MA_Ratio_20d = 1.02 | >1 | 当前价格比20日均线高2%（偏多） |
| Support_Ratio_120d = 0.90 | <1 | 支撑位在当前价格下方10% |
| ATR_Pct = 0.03 | - | 日波动率3% |

### 特征模块动态同步

新增特征模块时，只需修改一处：

```python
# 1. 在 ml_trading_model.py 中添加特征计算
def _build_new_features(self, df):
    df['New_Feature'] = ...
    return df

# 2. 在 get_feature_names() 中注册（自动同步到所有模块）
def get_feature_names(self):
    return [
        ...,
        'New_Feature'
    ]

# 3. 如果是绝对值，添加到排除列表
ABSOLUTE_PRICE_FEATURES = [..., 'New_Absolute_Feature']
```

---

## 特征工程最佳实践

### 1. 数据泄漏防护

**原则**：所有特征必须使用滞后数据，预测标签使用未来数据。

```python
# ❌ 错误：使用当日数据
df['Feature'] = df['Close'].rolling(5).mean()

# ✅ 正确：使用滞后数据
df['Feature'] = df['Close'].rolling(5).mean().shift(1)

# ✅ 正确：标签使用未来数据
df['Label'] = df['Close'].pct_change(horizon).shift(-horizon)
```

**高风险特征检查清单**：
- [ ] 所有 `.rolling()` 计算后是否有 `.shift(1)`
- [ ] 标签是否使用 `shift(-horizon)` 获取未来收益
- [ ] 价格距离/百分比特征是否使用滞后价格
- [ ] 宏观因子是否对齐到正确日期
- [ ] GARCH/HMM 特征是否滞后一天

**数据泄漏阈值**：

| 模型类型 | 正常范围 | 数据泄漏信号 |
|---------|---------|-------------|
| 个股 | 50-60% | >65% |
| 恒指 | 60-80% | >80% |

### 2. 特征相关性控制

**阈值**：新增特征与现有特征相关性应 **< 0.8**。

| 问题 | 说明 |
|------|------|
| 信息量不增加 | 相关性 > 0.8 本质是同一信号 |
| 特征重要性分散 | 高相关特征瓜分重要性 |
| 过拟合风险 | 不同 fold 交替选择高相关特征 |

**检查方法**：
```python
import pandas as pd

# 计算新特征与现有特征的相关性
correlations = df[existing_features].corrwith(df['New_Feature'])
max_corr = correlations.abs().max()

if max_corr > 0.8:
    print(f"警告：新特征与现有特征相关性过高 ({max_corr:.2f})")
```

### 3. NaN 处理策略

**原则**：只删除标签和关键列的 NaN，保留特征 NaN。

```python
# ❌ 错误：删除所有 NaN
df = df.dropna()  # 数据大量丢失

# ✅ 正确：只删除关键列 NaN
df = df.dropna(subset=['Label'])
critical_cols = ['Return_1d', 'Return_5d', 'Return_20d', 'Close', 'Volume']
df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
# 基本面特征的 NaN 保留，让模型自动处理
```

**技术原理**：LightGBM/XGBoost/CatBoost 原生支持 NaN，会学习最优分裂方向。

### 4. 默认值设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **分离原则** | 默认值与有效值范围完全分离 | `net_community_id = -1`（默认）vs `0-6`（有效）|
| **语义原则** | 默认值有明确语义 | `net_constraint = 1.0` 表示"高约束=无机会" |
| **中性原则** | 中性默认值用于无法判断的情况 | `sector_rising_ratio = 0.5` 表示"50%上涨" |

**常见错误**：

| 错误 | 问题 | 修正 |
|------|------|------|
| 默认值=中位数 | 无法区分"未知"和"中等" | 0.5 → -1 |
| 默认值=边界值 | 无法区分"未知"和"第一名" | 0 → -1 |
| 不合理的值 | 现实中不存在 | PE=0 → NaN |

### 5. 分类特征编码

CatBoost 原生支持分类特征，但需要一致处理 NaN：

```python
# 训练时
df[col] = df[col].fillna('unknown').astype(str)
encoder = LabelEncoder()
df[col] = encoder.fit_transform(df[col])

# 预测时（必须一致）
test_df[col] = test_df[col].fillna('unknown').astype(str)
test_df[col] = test_df[col].apply(
    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
)
```

### 6. 新特征上线验证清单

| 步骤 | 验证项 | 标准 | 检查方法 | 不通过的后果 |
|------|--------|------|----------|-------------|
| 1 | **泄漏检查** | 所有特征使用 `shift(1)` | 检查 `.rolling()` 后是否有 `.shift(1)` | 准确率虚高 |
| 2 | **绝对值特征标准化** | 跨股票训练时必须标准化 | 检查是否在 `ABSOLUTE_PRICE_FEATURES` 中或已标准化 | 模型学到无意义的"高价股"模式 |
| 3 | **市场级特征交叉** | 与股票特征交叉或作为分组因子 | 检查是否对所有股票同值；如是，必须交叉 | 无法区分个股，预测无差异 |
| 4 | **特征单调性** | 交叉特征保持逻辑单调性 | 检查交叉方式是否与特征单调性匹配 | 特征语义混乱，模型难以学习 |
| 5 | **Walk-forward 验证** | 恒指≥82%，个股≥50%；超阈值审计 | 运行验证脚本 | 特征无效或泄漏 |
| 6 | **SHAP 排名** | 进入 top 30 | 运行特征重要性分析 | 预测价值不足 |
| 7 | **Pearson 相关性** | 与现有特征 < 0.8 | 计算相关系数矩阵 | 冗余特征 |
| 8 | **随机种子稳定性** | 3 个种子波动 < 2% | 多次训练对比 | 特征不稳定 |

**验证命令**：
```bash
# Walk-forward 验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 特征重要性分析
python3 scripts/analyze_feature_importance_by_horizon.py
```

#### 检查项详解

**步骤 2：绝对值特征标准化**

```python
# 检查方法：判断是否为绝对价格/成交量特征
ABSOLUTE_PRICE_FEATURES = [
    'Channel_High_20d', 'Channel_Low_20d', 'Support_120d', 'Resistance_120d',
    'MA5', 'MA20', 'MA60', 'MA120', 'MA250',
    'BB_upper', 'BB_lower', 'ATR', 'VWAP', 'OBV', ...
]

# 新特征如果是绝对值，需要：
# 1. 添加到 ABSOLUTE_PRICE_FEATURES（自动排除）
# 2. 或创建标准化替代特征
df['New_Ratio'] = df['New_Value'] / df['Close'].shift(1)
```

**步骤 3：市场级特征交叉**

```python
# 检查方法：验证特征是否对所有股票同值
# 如果在某一天，所有股票的该特征值相同，则为市场级特征
if df.groupby('Date')['New_Feature'].nunique().max() == 1:
    print("警告：市场级特征，必须与股票特征交叉")

# 交叉示例
df['New_Feature_Cross'] = df['New_Feature'] * df['net_community_id']
df['New_Feature_Cross'] = df['New_Feature'] * df['net_degree_centrality']
```

**步骤 4：特征单调性**

```python
# 单调性定义
# POSITIVE: 值越大越好（如中心性、收益率）
# NEGATIVE: 值越大越差（如波动率、约束度）
# NEUTRAL: 无明确方向

# 交叉规则
# 正向 × 正向 → 乘法（协同效应）
# 负向 × 负向 → 风险放大（保持负向语义）
# 正向 × 负向 → 除法（风险调整）
# 涉及中性 → 乘法（保守策略）
```

---

## 关键经验教训

> 本节来自项目实践中总结的经验，详见 [lessons.md](../lessons.md)

### 1. 绝对价格特征标准化 ⭐⭐⭐

**问题**：绝对价格特征跨股票量级差异大，可能导致模型学到无意义模式。

**现象**：
- 腾讯 Channel_High_20d ≈ 400元 vs 小盘股 ≈ 5元
- CatBoost 偏向选择高方差特征，导致绝对价格特征进入 Top 20
- 模型可能学到"高价股=好"这种无意义模式

**解决方案**：除以前一日收盘价进行标准化。

**教训**：树模型虽然对特征量级不敏感，但跨股票混合训练时需要标准化。

---

### 2. 网络社区特征一致性 ⭐⭐⭐

**问题**：训练时动态提取社区 ID，预测时使用保存的社区 ID，导致特征不一致。

**现象**：
```
WARNING | 动态提取社区 ID（可能导致训练/预测不一致）: [np.int64(0)]
```

**根本原因**：
- 缓存命中时跳过交叉特征计算
- 缓存中的交叉特征是旧的网络特征文件生成的

**解决方案**：
```python
# 1. 预加载社区 ID
preloaded_community_ids = extract_community_ids_from_network_file()

# 2. 无论缓存是否命中，都重新计算交叉特征
stock_df = create_market_network_interaction_features(
    stock_df, community_ids=preloaded_community_ids)
```

**教训**：缓存命中时也需要重新计算依赖外部数据的特征。

---

### 3. 市场级特征需与股票特征交叉 ⭐⭐

**问题**：市场级特征（如 HSI_Return、VIX）对所有股票同值，无法区分个股。

**解决方案**：
- 与网络社区特征交叉：`HSI_Return_1d * net_community_id`
- 使用智能交叉：根据特征单调性选择乘法/除法

**教训**：新增特征模块时，必须在 `get_feature_names()` 中定义，确保自动同步。

---

### 4. 特征缓存版本控制 ⭐

**问题**：新增特征后，旧缓存缺少新特征列。

**解决方案**：
- 缓存验证时检查必需特征列是否存在
- 缺少新特征时标记缓存无效，重新计算

```bash
# 新增特征后必须清除缓存
rm -rf data/feature_cache/*.pkl
```

---

## 案例分析：个股模型特征工程改进

本节记录 2026-05-08 至 2026-05-09 期间，针对个股模型预测能力弱的问题，进行的一系列特征工程改进。

### 问题背景

**核心问题**：个股预测概率与实际方向相关性极弱（r=0.0186），远低于恒指（r=+0.35）。

| 指标 | 个股模型 | 恒指模型 | 差距 |
|------|---------|---------|------|
| 预测概率与实际方向相关性 | 0.0186 | +0.35 | **18倍** |
| 假突破(101)准确率 | 58% | 95% | **37%** |
| 5d→20d准确率 | 57.76% | 82.76% | **25%** |

这触发了一系列深入分析和特征工程改进。

---

### 问题发现链

```
个股预测弱 (r=0.0186)
    ↓
分析特征选择结果
    ↓
┌─────────────────────────────────────┐
│ 发现问题1: Top 20 有绝对价格特征      │
│ → 修改一: 绝对价格特征标准化          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 发现问题2: 排除列表在多个文件重复      │
│ → 修改二: 特征架构单一真相源          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 发现问题3: 训练/预测特征不一致警告     │
│ → 修改三: 网络社区特征一致性修复       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 发现问题4: 交叉特征语义混乱           │
│ → 修改四: 智能交叉特征生成            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 发现问题5: 市场级特征无法区分个股      │
│ → 修改五: 市场级特征交叉              │
└─────────────────────────────────────┘
```

---

### 修改一：绝对价格特征标准化 ⭐⭐⭐

**发现问题**：特征选择 Top 20 中出现 `Channel_High_20d`、`TP`、`MACD` 等绝对值特征。

**根因分析**：
```
腾讯 Channel_High_20d ≈ 400元
小盘股 Channel_High_20d ≈ 5元
    ↓
CatBoost 偏向选择高方差特征
    ↓
模型学到"高价股=好"的无意义模式
```

**解决方案**：
1. 排除列表从 35 → 40 个特征
2. 新增标准化替代特征

| 绝对值特征 | 标准化替代 | 标准化方法 |
|-----------|-----------|-----------|
| Channel_High_20d | Channel_High_Ratio_20d | ÷ prev_close |
| MA20 | MA_Ratio_20d | ÷ prev_close |
| ATR | ATR_Pct | ÷ prev_close |
| MACD | MACD_Ratio | ÷ close |

**验证结果**：Top 20 不再包含绝对价格特征。

---

### 修改二：特征架构单一真相源 ⭐⭐⭐

**发现问题**：`feature_selection.py` 和 `ml_trading_model.py` 各自维护排除列表。

**根因分析**：
```
修改 ml_trading_model.py 的排除列表
    ↓
忘记同步 feature_selection.py
    ↓
特征选择结果仍包含已排除特征
    ↓
训练数据不一致
```

**解决方案**：
```python
# 模块级常量作为单一真相源
ABSOLUTE_PRICE_FEATURES = [...]  # 只维护一处

# 其他模块导入使用
from ml_trading_model import ABSOLUTE_PRICE_FEATURES

# 封装方法
def prepare_features_for_selection(self, ...):
    """自动应用所有特征处理逻辑"""
    ...
```

**教训**：特征处理逻辑只维护一处，其他模块通过导入或方法调用复用。

---

### 修改三：网络社区特征一致性 ⭐⭐⭐

**发现问题**：训练日志显示警告。

```
WARNING | 动态提取社区 ID（可能导致训练/预测不一致）: [np.int64(0)]
```

**根因分析**：
```
网络特征文件更新（7个社区）
    ↓
缓存未更新（只有社区0）
    ↓
缓存命中时跳过交叉特征计算
    ↓
训练用社区0，预测用社区0-6
    ↓
特征不一致
```

**解决方案**：
```python
# 1. 预加载社区 ID（从最新网络文件）
preloaded_community_ids = extract_community_ids_from_network_file()

# 2. 缓存命中时也重新计算交叉特征
if cache_hit:
    stock_df = create_market_network_interaction_features(
        stock_df, community_ids=preloaded_community_ids)

# 3. 更新缓存
save_feature_cache(stock_df)
```

**教训**：缓存命中时也需要重新计算依赖外部数据的特征。

---

### 修改四：智能交叉特征生成 ⭐⭐

**发现问题**：交叉特征语义混乱。

```python
# 问题示例
net_constraint = 0.5  # 负向指标（值越大越差）
VIX = 20             # 负向指标（值越大越差）

# 原方案：乘法
result = 0.5 * 20 = 10  # 正值！语义错误

# 应该是"风险放大"，保持负向语义
```

**解决方案**：
```python
# 定义特征单调性
NETWORK_FEATURE_MONOTONICITY = {
    'net_constraint': 'NEGATIVE',      # 约束度高 = 信息劣势
    'net_composite_centrality': 'POSITIVE',  # 中心性高 = 影响力大
}

MARKET_FEATURE_MONOTONICITY = {
    'VIX': 'NEGATIVE',           # 波动率高 = 风险大
    'HSI_Return_1d': 'POSITIVE', # 收益率高 = 好
}

# 智能交叉
def create_monotonic_interaction(feat1, feat2, mono1, mono2):
    if mono1 == 'POSITIVE' and mono2 == 'POSITIVE':
        return feat1 * feat2  # 协同效应
    elif mono1 == 'NEGATIVE' and mono2 == 'NEGATIVE':
        return -abs(feat1) * abs(feat2)  # 风险放大
    elif mono1 == 'POSITIVE' and mono2 == 'NEGATIVE':
        return feat1 / (abs(feat2) + 1e-6)  # 风险调整
    else:
        return feat1 * feat2  # 保守策略
```

**效果**：交叉特征语义清晰，模型更容易学习。

---

### 修改五：市场级特征交叉 ⭐⭐

**发现问题**：`HSI_Return`、`VIX` 对所有股票同值。

```python
# 问题示例
date = '2026-05-09'
所有股票的 HSI_Return_1d = 0.02  # 同值
    ↓
模型无法区分个股
    ↓
预测结果无差异
```

**解决方案**：
```python
# 与网络特征交叉
df['HSI_Return_by_community'] = df['HSI_Return_1d'] * df['net_community_id']
df['VIX_by_centrality'] = df['VIX'] * df['net_composite_centrality']

# 不同社区/中心性的股票对同一市场信号反应不同
```

**教训**：市场级特征必须与股票特征交叉，才能区分个股。

---

### 改进效果

| 指标 | 修改前 | 修改后 | 变化 | 评估 |
|------|--------|--------|------|------|
| 准确率 | 53.09% | 52.40% | -0.69% | 微降但更真实 |
| 夏普比率 | 4.64 | 4.85 | +0.21 | 更稳健 |
| Top 20 绝对值特征 | 有 | 无 | - | ✅ 修复 |
| 训练/预测不一致警告 | 有 | 无 | - | ✅ 修复 |
| 交叉特征语义 | 混乱 | 清晰 | - | ✅ 改进 |

**关键结论**：准确率微降但夏普比率提升，说明特征更稳健、泛化能力更强。

---

### 经验总结

1. **问题驱动**：从模型性能问题出发，逐层深入分析特征
2. **根因优先**：不只是修复表象，要找到根本原因
3. **验证闭环**：每次修改后都要验证效果
4. **文档同步**：修改后及时更新文档，避免知识流失

---

## 附录

### A. 缓存机制

| 缓存类型 | 位置 | 有效期 | 加速效果 |
|---------|------|--------|---------|
| 原始数据 | `data/stock_cache/` | 7天 | - |
| 特征缓存 | `data/feature_cache/` | 7天 | **170x** |

**清除缓存**：
```bash
# 清除特征缓存（新增特征后）
rm -rf data/feature_cache/*.pkl

# 清除原始数据缓存
rm -rf data/stock_cache/*.pkl
```

### B. 相关文件

| 文件 | 说明 |
|------|------|
| `ml_services/ml_trading_model.py` | 特征工程核心实现 |
| `ml_services/feature_selection.py` | 特征选择 |
| `data_services/volatility_model.py` | GARCH 波动率 |
| `data_services/regime_detector.py` | HSI 市场状态 |
| `data_services/calendar_features.py` | 日历效应 |
| `ml_services/stock_network_analysis.py` | 网络社区特征 |
| `lessons.md` | 关键经验教训 |

### C. 参考资料

- **CatBoost 官方文档**：https://catboost.ai/docs/
- **特征工程最佳实践**：https://www.kaggle.com/learn/feature-engineering
- **时间序列特征工程**：https://machinelearningmastery.com/feature-engineering-for-time-series/

---

## 更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-05-09 | v2.1 | 新增：案例分析章节（个股模型特征工程改进全流程） |
| 2026-05-09 | v2.0 | 重构：完整特征工程指南，新增特征架构设计、核心特征详解 |
| 2026-05-09 | v1.5 | 新增：网络社区特征、市场-网络交叉特征、绝对值排除列表 |
| 2026-04-27 | v1.0 | 初始版本：特征冗余清理 |
