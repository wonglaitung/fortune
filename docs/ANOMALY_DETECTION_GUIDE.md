# 异常检测完整指南

> **最后更新**：2026-04-03

---

## 📋 目录

1. [异常检测概览](#异常检测概览)
2. [双层检测架构](#双层检测架构)
3. [核心模块](#核心模块)
4. [Z-Score检测器](#z-score检测器)
5. [Isolation Forest检测器](#isolation-forest检测器)
6. [特征提取器](#特征提取器)
7. [异常整合器](#异常整合器)
8. [异常缓存](#异常缓存)
9. [交易信号与异常关联性验证](#交易信号与异常关联性验证)
10. [使用指南](#使用指南)
11. [最佳实践](#最佳实践)

---

## 异常检测概览

### 异常检测目的

识别加密货币市场中的异常价格和成交量模式，为交易决策提供额外的风险信号和机会识别。

### 异常检测应用场景

1. **风险预警**：识别极端市场波动，提前预警风险
2. **机会识别**：识别异常低点或高点，捕捉交易机会
3. **策略调整**：根据异常情况动态调整交易策略
4. **市场监控**：持续监控市场状态，及时发现异常

### 异常检测方法对比

| 方法 | 类型 | 执行频率 | 检测范围 | 性能 | 适用场景 |
|------|------|---------|---------|------|---------|
| **Z-Score检测器** | 统计方法 | 每小时 | 价格异常、成交量异常、价格波动异常 | 快速响应 | 实时监控 |
| **Isolation Forest** | 机器学习 | 每天凌晨2点 | 复杂异常模式、非线性行为 | 全面分析 | 深度挖掘 |

### 异常检测流程

```
数据获取 → 特征提取 → 异常检测 → 异常整合 → 去重分类 → 缓存管理 → 邮件通知
```

### 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **异常检测准确率** | 真实异常占检测异常的比例 | > 70% |
| **异常召回率** | 检测到的异常占真实异常的比例 | > 80% |
| **误报率** | 错误检测为正常的比例 | < 30% |
| **去重效果** | 重复异常减少比例 | > 90% |

---

## 双层检测架构

### 架构设计

```
Layer 1: Z-Score 实时检测（每小时）
  ├─ 检测范围：价格异常、成交量异常、价格波动异常
  ├─ 检测方法：统计方法（Z-Score > 3）
  ├─ 性能：快速响应，适合实时监控
  └─ 输出：快速异常列表（high/medium/low）

Layer 2: Isolation Forest 深度分析（凌晨2点）
  ├─ 检测范围：复杂异常模式、非线性行为
  ├─ 检测方法：无监督机器学习（Isolation Forest）
  ├─ 性能：全面分析，适合深度挖掘
  └─ 输出：深度异常分析报告

异常整合器
  ├─ 去重：避免重复通知相同异常
  ├─ 严重性分类：high/medium/low三级
  ├─ 智能合并：合并相似异常
  └─ 输出：整合后的异常列表

异常缓存
  ├─ 文件持久化：data/anomaly_cache.json
  ├─ 缓存策略：24小时有效期
  └─ 避免重复：24小时内相同异常不再通知
```

### 双层检测优势

1. **实时性**：Layer 1每小时检测，快速响应市场变化
2. **全面性**：Layer 2每天深度分析，捕捉复杂异常
3. **准确性**：两层检测互补，提高检测准确性
4. **效率性**：分层处理，平衡性能和准确性
5. **灵活性**：可独立调整两层检测参数

### 执行时间表

| 时间 | 执行内容 | 持续时间 |
|------|---------|---------|
| 每小时整点 | Z-Score实时检测（Layer 1） | ~5分钟 |
| 凌晨2:00 | Isolation Forest深度分析（Layer 2） | ~15分钟 |
| 异常时 | 异常整合器 + 邮件通知 | ~2分钟 |

### GitHub Actions工作流

```yaml
name: 加密货币异常检测
on:
  schedule:
    - cron: '0 * * * *'     # 每小时（快速模式）
    - cron: '0 2 * * *'     # 凌晨2点（深度模式）
  workflow_dispatch:        # 支持手动触发
```

---

## 核心模块

### 模块结构

```
anomaly_detector/
├── __init__.py                    # 模块入口
├── zscore_detector.py             # Z-Score检测器
├── isolation_forest_detector.py   # Isolation Forest检测器
├── feature_extractor.py           # 特征提取器
├── anomaly_integrator.py          # 异常整合器
└── cache.py                       # 异常缓存
```

### 模块依赖关系

```
crypto_email.py
    ↓
anomaly_detector/
    ├── feature_extractor.py      (提取价格和成交量特征)
    │
    ├── zscore_detector.py        (Z-Score统计检测)
    │
    ├── isolation_forest_detector.py  (Isolation Forest机器学习检测)
    │
    ├── anomaly_integrator.py     (异常整合、去重、分类)
    │
    └── cache.py                  (异常缓存管理)
```

### 模块接口

```python
# feature_extractor.py
class FeatureExtractor:
    def extract_features(self, df):  # 提取价格和成交量特征
        return features

# zscore_detector.py
class ZScoreDetector:
    def detect(self, features, threshold=3.0):  # Z-Score检测
        return anomalies

# isolation_forest_detector.py
class IsolationForestDetector:
    def detect(self, features, contamination=0.1):  # Isolation Forest检测
        return anomalies

# anomaly_integrator.py
class AnomalyIntegrator:
    def integrate(self, layer1_anomalies, layer2_anomalies):  # 异常整合
        return integrated_anomalies

# cache.py
class AnomalyCache:
    def is_cached(self, anomaly):  # 检查是否已缓存
        return is_cached
    
    def add_to_cache(self, anomaly):  # 添加到缓存
        pass
```

---

## Z-Score检测器

### 检测原理

**Z-Score**（标准分数）衡量一个数据点与均值的距离，以标准差为单位。

**计算公式**：
```
Z-Score = (x - μ) / σ
```

其中：
- x：观测值
- μ：均值
- σ：标准差

**异常判定标准**：
- |Z-Score| > 3：high严重性异常（极值）
- 2 < |Z-Score| ≤ 3：medium严重性异常（显著）
- 1.5 < |Z-Score| ≤ 2：low严重性异常（轻微）

### 检测范围

1. **价格异常**：
   - 收盘价Z-Score
   - 最高价Z-Score
   - 最低价Z-Score
   - 价格变化率Z-Score

2. **成交量异常**：
   - 成交量Z-Score
   - 成交量变化率Z-Score
   - 成交量比率Z-Score

3. **价格波动异常**：
   - 价格波动率Z-Score
   - ATR（平均真实波幅）Z-Score
   - 布林带宽度Z-Score

### 检测方法

```python
from anomaly_detector.zscore_detector import ZScoreDetector

# 初始化检测器
detector = ZScoreDetector()

# 提取特征
features = {
    'close': price_data['close'],
    'high': price_data['high'],
    'low': price_data['low'],
    'volume': price_data['volume'],
    'volume_change': volume_change,
    'price_volatility': price_volatility,
    'atr': atr
}

# 执行检测（默认阈值3.0）
anomalies = detector.detect(features, threshold=3.0)

# 自定义阈值
anomalies = detector.detect(features, threshold=2.5)
```

### 检测结果示例

```json
{
    "type": "price_anomaly",
    "severity": "high",
    "value": 35000.50,
    "zscore": 4.25,
    "mean": 32000.00,
    "std": 705.88,
    "timestamp": "2026-04-03T10:00:00Z",
    "currency": "BTC-USD"
}
```

### 优点与缺点

**优点**：
- ✅ 计算简单，快速响应
- ✅ 易于理解和解释
- ✅ 无需训练数据
- ✅ 适合实时监控

**缺点**：
- ❌ 假设数据服从正态分布（金融市场往往不满足）
- ❌ 对异常值敏感（均值和标准差受异常值影响）
- ❌ 只能检测线性异常，无法检测复杂模式

---

## Isolation Forest检测器

### 检测原理

**Isolation Forest**（孤立森林）是一种无监督机器学习算法，通过随机分割特征空间来隔离异常点。

**核心思想**：
- 异常点更容易被隔离（需要的分割次数更少）
- 正常点更难被隔离（需要的分割次数更多）

**隔离路径长度**：
- 路径越短，越可能是异常点
- 路径越长，越可能是正常点

**异常分数**：
- 异常分数接近1：异常点
- 异常分数接近0.5：正常点
- 异常分数 < 0.5：可能是噪声

### 检测范围

1. **复杂异常模式**：
   - 多维特征组合异常
   - 非线性关系异常
   - 时序模式异常

2. **非线性行为**：
   - 价格与成交量背离
   - 多个技术指标同时异常
   - 跨时间窗口异常模式

### 检测方法

```python
from anomaly_detector.isolation_forest_detector import IsolationForestDetector

# 初始化检测器
detector = IsolationForestDetector(
    n_estimators=100,      # 树的数量
    max_samples='auto',    # 每棵树的样本数
    contamination=0.1,     # 异常比例
    random_state=42        # 随机种子
)

# 提取多维特征
features = pd.DataFrame({
    'close': price_data['close'],
    'volume': price_data['volume'],
    'volume_change': volume_change,
    'price_volatility': price_volatility,
    'atr': atr,
    'rsi': rsi,
    'macd': macd,
    'bollinger_width': bollinger_width
})

# 执行检测
anomalies = detector.detect(features)

# 获取异常分数
scores = detector.decision_function(features)
```

### 检测结果示例

```json
{
    "type": "complex_anomaly",
    "severity": "high",
    "anomaly_score": 0.92,
    "features": {
        "close": 35000.50,
        "volume": 1000000,
        "price_volatility": 0.05,
        "atr": 500.0,
        "rsi": 75.5,
        "macd": 150.0
    },
    "isolation_path_length": 12.5,
    "timestamp": "2026-04-03T02:00:00Z",
    "currency": "BTC-USD"
}
```

### 优点与缺点

**优点**：
- ✅ 无需标注数据（无监督学习）
- ✅ 能检测复杂异常模式
- ✅ 对非线性数据有效
- ✅ 计算效率高（适合大规模数据）
- ✅ 鲁棒性强（对异常值不敏感）

**缺点**：
- ❌ 需要训练数据（构建森林）
- ❌ 参数选择影响性能（n_estimators、contamination）
- ❌ 解释性不如Z-Score直观
- ❌ 计算时间较长（相对于Z-Score）

---

## 特征提取器

### 特征提取目的

从价格和成交量数据中提取有意义的特征，用于异常检测。

### 特征类别

#### 1. 价格特征（10个）

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `close` | 收盘价 | 原始数据 |
| `high` | 最高价 | 原始数据 |
| `low` | 最低价 | 原始数据 |
| `open` | 开盘价 | 原始数据 |
| `price_change` | 价格变化 | Close - Open |
| `price_change_pct` | 价格变化百分比 | (Close - Open) / Open |
| `price_volatility` | 价格波动率 | Std(Price) / Mean(Price) |
| `price_range` | 价格范围 | High - Low |
| `price_range_pct` | 价格范围百分比 | (High - Low) / Close |
| `gap` | 缺口 | Open - Prev_Close |

#### 2. 成交量特征（8个）

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `volume` | 成交量 | 原始数据 |
| `volume_change` | 成交量变化 | Volume - Prev_Volume |
| `volume_change_pct` | 成交量变化百分比 | (Volume - Prev_Volume) / Prev_Volume |
| `volume_ma_5d` | 5日成交量均线 | Volume.rolling(5).mean() |
| `volume_ma_10d` | 10日成交量均线 | Volume.rolling(10).mean() |
| `volume_ratio_5d` | 5日成交量比率 | Volume / Volume_MA_5d |
| `volume_ratio_10d` | 10日成交量比率 | Volume / Volume_MA_10d |
| `volume_volatility` | 成交量波动率 | Std(Volume) / Mean(Volume) |

#### 3. 技术指标特征（10个）

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `rsi_14d` | 14日相对强弱指标 | RSI(Close, 14) |
| `macd` | MACD指标 | MACD(Close, 12, 26, 9) |
| `macd_signal` | MACD信号线 | MACD_Signal(Close, 12, 26, 9) |
| `macd_histogram` | MACD直方图 | MACD - MACD_Signal |
| `bollinger_upper` | 布林带上轨 | Bollinger_Upper(Close, 20, 2) |
| `bollinger_lower` | 布林带下轨 | Bollinger_Lower(Close, 20, 2) |
| `bollinger_width` | 布林带宽度 | (Upper - Lower) / Close |
| `atr_14d` | 14日平均真实波幅 | ATR(High, Low, Close, 14) |
| `ema_12d` | 12日指数移动平均 | EMA(Close, 12) |
| `ema_26d` | 26日指数移动平均 | EMA(Close, 26) |

#### 4. 派生特征（5个）

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `price_volume_ratio` | 价格成交量比率 | Close / Volume |
| `volume_price_change_corr` | 成交量与价格变化相关性 | Corr(Volume, Price_Change) |
| `volatility_5d` | 5日波动率 | Std(Return_5d) |
| `volatility_10d` | 10日波动率 | Std(Return_10d) |
| `trend_strength` | 趋势强度 | (Close - MA_20d) / Std(Close) |

### 特征提取方法

```python
from anomaly_detector.feature_extractor import FeatureExtractor

# 初始化特征提取器
extractor = FeatureExtractor()

# 提取特征
features = extractor.extract_features(price_data)

# 特征示例
print(features.keys())
# dict_keys(['close', 'high', 'low', 'volume', 'price_change', 
#            'price_change_pct', 'price_volatility', 'volume_change', 
#            'volume_change_pct', 'volume_ratio_5d', 'rsi_14d', 'macd', 
#            'atr_14d', 'bollinger_width', 'trend_strength', ...])
```

### 特征标准化

**Z-Score检测器**：不需要标准化（直接使用Z-Score）
**Isolation Forest检测器**：不需要标准化（基于树模型）

---

## 异常整合器

### 异常整合目的

整合两层检测的结果，去重、分类、合并相似异常。

### 整合流程

```
Layer 1异常 + Layer 2异常
    ↓
去重（24小时内相同异常）
    ↓
严重性分类（high/medium/low）
    ↓
智能合并（相似异常合并）
    ↓
输出整合后的异常列表
```

### 去重策略

**去重规则**：
- 相同货币对（如BTC-USD）
- 相同异常类型（如price_anomaly）
- 时间间隔小于24小时

**去重方法**：
```python
def is_duplicate(anomaly1, anomaly2):
    # 检查货币对
    if anomaly1['currency'] != anomaly2['currency']:
        return False
    
    # 检查异常类型
    if anomaly1['type'] != anomaly2['type']:
        return False
    
    # 检查时间间隔
    time_diff = abs(anomaly1['timestamp'] - anomaly2['timestamp'])
    if time_diff < 24 * 3600:  # 24小时
        return True
    
    return False
```

### 严重性分类

**分类标准**：

| 严重性 | 标准 | 说明 |
|--------|------|------|
| **high** | Z-Score > 3 或 异常分数 > 0.8 | 极端异常，立即关注 |
| **medium** | 2 < Z-Score ≤ 3 或 0.5 < 异常分数 ≤ 0.8 | 显著异常，密切关注 |
| **low** | 1.5 < Z-Score ≤ 2 或 0.3 < 异常分数 ≤ 0.5 | 轻微异常，适当关注 |

**Layer 1 + Layer 2整合规则**：
- 如果任一层检测为high，最终结果为high
- 如果任一层检测为medium，最终结果为medium（除非另一层为high）
- 否则，结果为low

### 智能合并

**合并规则**：
- 合并相同货币对、相同类型的连续异常
- 合并时间间隔小于6小时的异常
- 保留最高严重性级别

**合并方法**：
```python
def merge_anomalies(anomalies):
    # 按货币对和类型分组
    grouped = defaultdict(list)
    for anomaly in anomalies:
        key = (anomaly['currency'], anomaly['type'])
        grouped[key].append(anomaly)
    
    # 合并每组异常
    merged = []
    for key, group in grouped.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # 合并连续异常
            merged_anomaly = {
                'type': key[1],
                'severity': max(a['severity'] for a in group),  # 保留最高严重性
                'currency': key[0],
                'start_time': min(a['timestamp'] for a in group),
                'end_time': max(a['timestamp'] for a in group),
                'count': len(group),  # 异常次数
                'details': group
            }
            merged.append(merged_anomaly)
    
    return merged
```

### 整合器使用示例

```python
from anomaly_detector.anomaly_integrator import AnomalyIntegrator

# 初始化整合器
integrator = AnomalyIntegrator()

# 整合两层检测的结果
layer1_anomalies = zscore_detector.detect(features)
layer2_anomalies = isolation_forest_detector.detect(features)

# 整合异常
integrated_anomalies = integrator.integrate(layer1_anomalies, layer2_anomalies)

# 输出整合后的异常
print(integrated_anomalies)
```

---

## 异常缓存

### 缓存目的

避免重复通知相同的异常，减少通知频率。

### 缓存策略

**缓存内容**：
- 异常类型
- 货币对
- 检测时间
- 严重性

**缓存有效期**：24小时

**缓存存储**：`data/anomaly_cache.json`

### 缓存实现

```python
from anomaly_detector.cache import AnomalyCache

# 初始化缓存
cache = AnomalyCache(cache_file='data/anomaly_cache.json')

# 检查是否已缓存
if cache.is_cached(anomaly):
    print("异常已缓存，跳过通知")
else:
    # 发送通知
    send_notification(anomaly)
    
    # 添加到缓存
    cache.add_to_cache(anomaly)

# 清理过期缓存（24小时前）
cache.cleanup_expired()
```

### 缓存文件格式

```json
{
    "cache": {
        "BTC-USD:price_anomaly": {
            "type": "price_anomaly",
            "currency": "BTC-USD",
            "timestamp": "2026-04-03T10:00:00Z",
            "severity": "high"
        },
        "ETH-USD:volume_anomaly": {
            "type": "volume_anomaly",
            "currency": "ETH-USD",
            "timestamp": "2026-04-03T09:00:00Z",
            "severity": "medium"
        }
    },
    "last_cleanup": "2026-04-03T10:00:00Z"
}
```

---

## 交易信号与异常关联性验证

### 验证目的

分析交易信号与异常的关联性，评估异常检测对交易策略的指导价值。

### 验证方法

1. **相关性分析**：计算交易信号与异常的相关系数
2. **Granger因果检验**：测试异常是否是交易信号变化的Granger原因
3. **事件研究分析**：分析异常发生前后交易信号的变化
4. **Walk-forward策略验证**：结合异常信号进行Walk-forward回测

### 验证结果

#### 1. 相关性分析

| 关联性 | 相关系数 | 说明 |
|--------|---------|------|
| **高异常与卖出信号** | 0.68 | 正相关（高异常后卖出信号增加） |
| **低异常与买入信号** | 0.72 | 正相关（低异常后买入信号增加） |
| **异常频率与信号变化** | 0.85 | 强正相关（异常频率增加，信号变化增加） |

#### 2. Granger因果检验

| 检验项 | F统计量 | P值 | 结论 |
|--------|--------|-----|------|
| **异常 → 买入信号** | 8.45 | 0.002 | 显著因果（P < 0.05） |
| **异常 → 卖出信号** | 9.23 | 0.001 | 显著因果（P < 0.05） |
| **信号 → 异常** | 1.23 | 0.345 | 不显著因果（P > 0.05） |

**结论**：异常可以预测交易信号的变化，但交易信号不能预测异常。

#### 3. 事件研究分析

| 事件 | 异常前 | 异常后 | 变化幅度 | 统计显著性 |
|------|--------|--------|---------|-----------|
| **买入信号频率** | 0.25 | 0.38 | +52% | P < 0.01 |
| **卖出信号频率** | 0.22 | 0.35 | +59% | P < 0.01 |
| **信号波动率** | 0.15 | 0.27 | +79% | P < 0.01 |

**结论**：异常后交易信号频率显著增加，但波动率也显著增加。

#### 4. Walk-forward策略验证

| 策略 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 | 推荐度 |
|------|-----------|---------|---------|------|--------|
| **无异常信号** | 30.71% | 0.8125 | -13.08% | 50.60% | ⭐⭐⭐ |
| **有异常信号（55-60%仓位）** | **33.45%** | **0.9287** | **-12.45%** | **52.10%** | ⭐⭐⭐⭐ |
| **有异常信号（+0.03阈值调整）** | **34.12%** | **0.9391** | **-12.23%** | **52.35%** | ⭐⭐⭐⭐⭐ |

**结论**：
- 异常信号可以显著提升策略表现
- 高异常时降低仓位至55-60%
- 高异常时提高置信度阈值0.03

### 策略建议

**高异常情况**（severity = high）：
- 降低仓位至55-60%
- 提高置信度阈值0.03
- 减少交易频率
- 加强风险控制

**中等异常情况**（severity = medium）：
- 保持正常仓位
- 密切关注市场变化
- 准备风险控制措施

**无异常或低异常情况**（severity = low 或无异常）：
- 正常交易
- 按照标准策略执行

### 验证命令

```bash
# 完整验证（相关性 + Granger + 事件研究 + Walk-forward）
python3 ml_services/validate_signal_anomaly_correlation.py --mode all

# 仅相关性分析
python3 ml_services/validate_signal_anomaly_correlation.py --mode correlation

# 仅Granger因果检验
python3 ml_services/validate_signal_anomaly_correlation.py --mode granger

# 仅事件研究分析
python3 ml_services/validate_signal_anomaly_correlation.py --mode event_study

# 仅Walk-forward策略验证
python3 ml_services/validate_signal_anomaly_correlation.py --mode walk_forward
```

### 输出文件

- `output/signal_anomaly_correlation_analysis_{timestamp}.md`：完整验证报告
- `output/signal_anomaly_correlation_{timestamp}.csv`：验证数据
- `output/signal_anomaly_correlation_{timestamp}.json`：JSON格式数据

---

## 使用指南

### 快速模式（每小时）

```bash
# 快速模式（Z-Score实时检测）
python3 crypto_email.py --mode quick
```

**执行内容**：
- Layer 1：Z-Score实时检测
- 异常整合器：去重、分类
- 异常缓存：检查是否已缓存
- 邮件通知：发送high/medium级别异常

**执行时间**：~5分钟

### 深度模式（凌晨2点）

```bash
# 深度模式（Isolation Forest深度分析）
python3 crypto_email.py --mode deep
```

**执行内容**：
- Layer 1：Z-Score实时检测
- Layer 2：Isolation Forest深度分析
- 异常整合器：去重、分类、合并
- 异常缓存：检查是否已缓存
- 邮件通知：发送所有级别异常

**执行时间**：~15分钟

### 手动执行

```bash
# 手动执行快速模式
python3 crypto_email.py --mode quick

# 手动执行深度模式
python3 crypto_email.py --mode deep
```

### 自动化调度

**GitHub Actions**：自动执行

```yaml
name: 加密货币异常检测
on:
  schedule:
    - cron: '0 * * * *'     # 每小时（快速模式）
    - cron: '0 2 * * *'     # 凌晨2点（深度模式）
  workflow_dispatch:        # 支持手动触发

jobs:
  detect:
    runs-on: ubuntu-latest
    env:
      TZ: Asia/Hong_Kong
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run anomaly detection
        run: |
          if [ "$(date +%H)" = "02" ]; then
            python3 crypto_email.py --mode deep
          else
            python3 crypto_email.py --mode quick
          fi
```

**本地cron**：手动配置

```bash
# 添加到crontab
0 * * * * cd /data/fortune && python3 crypto_email.py --mode quick
0 2 * * * cd /data/fortune && python3 crypto_email.py --mode deep
```

---

## 最佳实践

### 1. 双层检测协同使用

**推荐**：
- 快速模式：每小时执行，快速响应
- 深度模式：凌晨2点执行，深度分析

**不推荐**：
- 只使用一层检测（无法兼顾实时性和全面性）
- 频繁执行深度模式（计算资源消耗大）

### 2. 异常缓存管理

**推荐**：
- 启用异常缓存（避免重复通知）
- 定期清理过期缓存（24小时前）

**不推荐**：
- 禁用异常缓存（可能重复通知）
- 缓存时间过短（可能重复通知）
- 缓存时间过长（可能错过重要异常）

### 3. 严重性分类

**推荐**：
- high级别：立即关注，调整仓位
- medium级别：密切关注，准备调整
- low级别：适当关注，正常交易

**不推荐**：
- 忽略low级别异常（可能错过早期信号）
- 对所有异常一视同仁（浪费注意力）

### 4. 交易策略调整

**推荐**：
- 高异常：降低仓位至55-60%，提高阈值0.03
- 中等异常：保持正常仓位，密切关注
- 无异常或低异常：正常交易

**不推荐**：
- 忽略异常信号（可能错过风险）
- 对所有异常一视同仁（降低策略效率）

### 5. 定期验证关联性

**推荐**：
- 每月验证一次交易信号与异常的关联性
- 根据验证结果调整策略参数

**不推荐**：
- 不验证关联性（无法评估异常检测价值）
- 验证频率过高（计算资源消耗大）

### 6. 监控异常检测性能

**推荐**：
- 定期检查异常检测准确率和召回率
- 优化检测参数（阈值、contamination等）

**不推荐**：
- 不监控性能（无法及时发现性能下降）
- 过度优化参数（可能过拟合）

---

## 相关文件

- **加密货币异常检测**：`crypto_email.py`
- **Z-Score检测器**：`anomaly_detector/zscore_detector.py`
- **Isolation Forest检测器**：`anomaly_detector/isolation_forest_detector.py`
- **特征提取器**：`anomaly_detector/feature_extractor.py`
- **异常整合器**：`anomaly_detector/anomaly_integrator.py`
- **异常缓存**：`anomaly_detector/cache.py`
- **交易信号与异常验证**：`ml_services/validate_signal_anomaly_correlation.py`

---

## 参考资料

- **Z-Score**：https://en.wikipedia.org/wiki/Standard_score
- **Isolation Forest**：https://en.wikipedia.org/wiki/Isolation_forest
- **异常检测综述**：https://ieeexplore.ieee.org/document/9098912
- **Granger因果检验**：https://en.wikipedia.org/wiki/Granger_causality
- **事件研究法**：https://en.wikipedia.org/wiki/Event_study

---

**最后更新**：2026-04-03