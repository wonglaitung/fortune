# 特征工程指南

> **最后更新**：2026-04-27（特征冗余清理：1045→730）

---

## 特征概览

### 特征数量统计

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
| **总计** | **730** | **精简特征（2026-04-27 清理后）** |

### 特征冗余清理记录（2026-04-27）

| 清理类别 | 删除数量 | 原因 |
|---------|---------|------|
| RS_Signal 重复 | 251 | 与 Trend 公式完全相同 |
| PE/PB 常量交互 | 18 | PE/PB 为常量，交互无信息 |
| 数学等价特征 | 15 | Returns/Volatility/Momentum 等重复 |
| 第二轮 r=1.0 冗余 | ~23 | GARCH_Persistence 等 |
| **合计** | **~307** | 特征数量 1045→730 |

### 特征分类原则

1. **时间序列特征**：基于历史价格、成交量数据计算
2. **市场环境特征**：基于市场状态、波动率等宏观指标
3. **基本面特征**：基于公司财务数据
4. **情感特征**：基于新闻情感、主题分析
5. **事件驱动特征**：基于分红、财报等公司事件

---

## 核心特征类别

### 1. GARCH 波动率特征（4个）

使用 GARCH(1,1) 模型捕捉波动率聚类特性。

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `GARCH_Conditional_Vol` | 条件波动率 | GARCH(1,1) 模型拟合后的条件标准差 |
| `GARCH_Vol_Ratio` | 波动率比率 | 当前条件波动率 / 历史均值 |
| `GARCH_Vol_Change_5d` | 5日波动率变化 | 条件波动率的5日变化率 |
| `GARCH_Persistence` | 波动率持续性参数 | α + β（GARCH 参数之和） |

**实现文件**：`data_services/volatility_model.py`

---

### 2. HSI 市场状态特征（6个）

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
| HSI_Market_Regime | 第20 | 1.00 |

**实现文件**：`data_services/regime_detector.py`

---

### 3. 日历效应特征（22个）

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

---

### 4. 技术指标特征

**移动平均（MA系列）**：
- MA5、MA10、MA20、MA60、MA120、MA250
- 价格相对MA比率、MA交叉信号

**动量指标**：
- RSI（5日、10日、14日、20日）
- MACD（DIF、DEA、MACD柱）
- KDJ（K、D、J值）

**波动率指标**：
- ATR（5日、10日、14日、20日）
- 布林带（上轨、下轨、带宽）
- 波动率比率

**成交量指标**：
- 成交量比率（5日、10日、20日）
- OBV（能量潮）
- 量价背离信号

---

### 5. 基本面特征

- PE、PB、ROE、ROA
- 股息率、EPS
- 净利率、毛利率
- 市值、流通市值

---

### 6. 市场环境特征

- 恒生指数收益率、相对表现
- 南向资金流向
- VIX波动率水平
- 美国10年期国债收益率
- 标普500/纳斯达克收益率

---

## 特征工程最佳实践

### 1. 使用全量特征

**推荐命令**：
```bash
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

CatBoost 内置 L2 正则化和自动特征重要性计算，能够自动降权不重要特征，无需预先筛选。

### 2. 特征相关性筛选阈值（Pearson < 0.8）

**阈值选择**：新增特征与现有特征相关性应 **< 0.8**，超过则视为冗余。

**原因分析**：

| 问题 | 说明 |
|------|------|
| **信息量不增加** | 相关性 > 0.8 的特征本质上是同一信号的不同表达，CatBoost 无法获得额外信息，反而增加噪音 |
| **特征重要性分散** | 同一信号由两个高相关特征瓜分重要性，导致 SHAP 排名失真，误判哪个特征真正有用 |
| **过拟合风险** | 模型可能在不同 fold 中交替选择两个高相关特征，降低稳定性 |
| **SHAP 特征选择失效** | 新特征需进 top 30、最终特征集 < 50，冗余特征挤入会排挤真正有价值但独立的特征 |

**为什么是 0.8**：

| 阈值 | 问题 |
|------|------|
| 0.9 | 太松 — 81% 方差重叠，基本是同一信号 |
| 0.5 | 太严 — 0.5-0.8 之间可能包含互补信息（如 RSI 和 MACD 都衡量动量，但捕捉不同方面） |
| **0.8** | **工程折中** — 保留足够独立的信号，同时不丢弃有部分重叠但有增量价值的特征 |

**实践建议**：
- 新特征上线前检查与现有特征的 Pearson 相关系数
- 相关性 > 0.8 时，评估是否带来增量信息；若无，则丢弃
- CatBoost 对多重共线性容忍度高于线性模型，但高相关特征仍会影响特征重要性解释

### 3. 数据泄漏防护

**原则**：所有特征必须使用滞后数据

```python
# ❌ 错误：使用当日数据
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

# ✅ 正确：使用滞后数据
df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(5).mean()
```

**新增特征的 shift 处理**：
```python
# GARCH 波动率特征
for col in ['GARCH_Conditional_Vol', 'GARCH_Vol_Ratio', 'GARCH_Vol_Change_5d']:
    df[col] = df[col].shift(1)

# HSI 市场状态特征
for col in ['Market_Regime', 'Regime_Prob_0', 'Regime_Prob_1', 'Regime_Prob_2',
            'Regime_Duration', 'Regime_Transition_Prob']:
    df[col] = df[col].shift(1)
```

**高风险特征检查清单**：
- [ ] 所有 `.rolling()` 计算后是否有 `.shift(1)`
- [ ] 标签是否使用 `shift(-horizon)` 获取未来收益
- [ ] 价格距离/百分比特征是否使用滞后价格
- [ ] 宏观因子是否对齐到正确日期

### 4. 分类特征编码

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Market_Regime_Encoded'] = le.fit_transform(df['Market_Regime'])
```

### 5. 固定随机种子

```python
random.seed(42)
np.random.seed(42)
```

### 6. 新特征上线验证清单

每个新特征维度完成后需执行以下验证：

| 步骤 | 验证项 | 标准（恒指） | 标准（个股） | 不通过的后果 |
|------|--------|-------------|-------------|-------------|
| 1 | **泄漏检查** | 所有特征使用 `shift(1)` | 所有特征使用 `shift(1)` | 数据泄漏，准确率虚高 |
| 2 | **Walk-forward 验证** | 目标 ≥ 82.23%；> 85% 需审计 | 目标 ≥ 50%；> 65% 需审计 | 准确率低于目标特征无效；超阈值需查泄漏 |
| 3 | **SHAP 排名** | 进入 top 30 | 进入 top 30 | 未进入则特征预测价值不足 |
| 4 | **Pearson 相关性** | 与现有特征 < 0.8 | 与现有特征 < 0.8 | > 0.8 为冗余特征，需评估增量价值 |
| 5 | **随机种子稳定性** | 3 个种子波动 < 2% | 3 个种子波动 < 2% | 波动过大说明特征不稳定 |

**验证命令**：
```bash
# 1. Walk-forward 验证（恒指）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# 2. Walk-forward 验证（个股）
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20 --stock-mode

# 3. 特征重要性分析
python3 scripts/analyze_feature_importance_by_horizon.py

# 4. 相关性检查（需自行实现或检查日志）
```

**注意事项**：
- 步骤 1-4 必须全部通过，步骤 5 为稳定性保障
- 恒指准确率 > 85%、个股准确率 > 65% 触发审计，需检查是否存在隐蔽的数据泄漏
- SHAP 未进 top 30 但有业务意义的特征可保留，需说明理由

---

## 缓存机制

| 缓存类型 | 位置 | 有效期 | 加速效果 |
|---------|------|--------|---------|
| 原始数据 | `data/stock_cache/` | 7天 | - |
| 特征缓存 | `data/feature_cache/` | 7天 | **170x** |

**清除缓存**：
```bash
rm -rf data/feature_cache/*.pkl
```

---

## 相关文件

- **特征工程实现**：`ml_services/ml_trading_model.py` 中的 `FeatureEngineer` 类
- **GARCH 波动率**：`data_services/volatility_model.py`
- **HSI 市场状态**：`data_services/regime_detector.py`
- **日历效应**：`data_services/calendar_features.py`
- **特征重要性分析**：`docs/FEATURE_IMPORTANCE_ANALYSIS.md`

---

## 参考资料

- **CatBoost 官方文档**：https://catboost.ai/docs/
- **特征工程最佳实践**：https://www.kaggle.com/learn/feature-engineering
- **时间序列特征工程**：https://machinelearningmastery.com/feature-engineering-for-time-series/
