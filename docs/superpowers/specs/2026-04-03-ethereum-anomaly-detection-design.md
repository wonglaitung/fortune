# 以太币异常检测系统设计文档

**设计日期**: 2026-04-03  
**版本**: 1.0  
**状态**: 待审查

## 目录

1. [概述](#1-概述)
2. [总体架构](#2-总体架构)
3. [核心组件](#3-核心组件)
4. [数据流程](#4-数据流程)
5. [异常评分与警报策略](#5-异常评分与警报策略)
6. [错误处理与测试](#6-错误处理与测试)
7. [实现计划](#7-实现计划)

## 1. 概述

### 1.1 目标

在现有加密货币监控系统（crypto_email.py）基础上，增加自动化异常检测功能，提高以太币异常的及时性和准确性。

### 1.2 设计原则

- **自动化优先**：使用统计方法和机器学习算法，避免固定阈值规则
- **混合检测**：结合实时Z-Score快速筛选和Isolation Forest深度分析
- **无缝集成**：补充现有系统，统一邮件通知
- **业界最佳实践**：参考顶级量化机构的多级过滤策略

### 1.3 技术选型

| 检测方法 | 算法 | 适用场景 |
|---------|------|---------|
| 第一层 | Moving Z-Score | 实时快速检测（每小时） |
| 第二层 | Isolation Forest | 深度多指标分析（每日） |

## 2. 总体架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        数据获取层                            │
├─────────────────────────────────────────────────────────────┤
│  实时价格数据（每小时）  │  历史数据（每日）              │
│  - CoinGecko API        │  - Yahoo Finance (6个月)        │
│  - 价格、成交量         │  - OHLCV数据                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       异常检测层                            │
├─────────────────────────────────────────────────────────────┤
│  第一层：实时Z-Score检测（每小时运行）                       │
│  ├── 价格异常检测                                          │
│  ├── 成交量异常检测                                        │
│  └── 快速筛选（阈值：±3σ）                                 │
│                                                             │
│  第二层：Isolation Forest深度分析（每日运行）                │
│  ├── 多指标特征提取                                         │
│  ├── 异常评分计算                                          │
│  └── 精确识别                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       技术分析层（现有）                      │
├─────────────────────────────────────────────────────────────┤
│  ├── RSI、MACD、布林带                                      │
│  ├── TAV评分                                               │
│  └── 交易信号生成                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       异常整合器                            │
├─────────────────────────────────────────────────────────────┤
│  ├── 整合两层检测结果                                       │
│  ├── 异常严重程度分级                                       │
│  ├── 去重处理（24小时窗口）                                  │
│  └── 生成统一报告                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       邮件通知层                            │
├─────────────────────────────────────────────────────────────┤
│  ├── 整合异常检测结果                                       │
│  ├── 整合技术分析信号                                       │
│  └── 统一发送邮件                                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
crypto_email.py (主入口)
├── data_services.technical_analysis (现有)
│   └── TechnicalAnalyzer
├── anomaly_detector (新增)
│   ├── ZScoreDetector (第一层)
│   ├── IsolationForestDetector (第二层)
│   └── FeatureExtractor
└── anomaly_integrator (新增)
    ├── 整合异常结果
    └── 生成邮件内容
```

## 3. 核心组件

### 3.1 实时Z-Score检测器（ZScoreDetector）

#### 职责
每小时检测价格和成交量异常，提供快速筛选能力。

#### 输入
- 最新价格数据（每小时）
- 滚动窗口数据（30天历史）
- 阈值配置（默认：±3σ）

#### 输出
```python
{
    'timestamp': datetime,
    'anomalies': [
        {
            'type': 'price',  # 或 'volume'
            'severity': 'high' | 'medium' | 'low',
            'z_score': float,
            'value': float,
            'description': str
        }
    ]
}
```

#### 核心算法
```python
# Moving Z-Score计算
window_size = 30  # 天数
mean = data.rolling(window=window_size).mean()
std = data.rolling(window=window_size).std()
z_score = (current_value - mean) / std

# 异常判断
if abs(z_score) > 4:
    severity = 'high'
elif abs(z_score) > 3:
    severity = 'medium'
else:
    return None  # 非异常
```

#### 特性
- 自动更新滚动窗口
- 支持多维度检测（价格、成交量）
- 可配置阈值

### 3.2 Isolation Forest检测器（IsolationForestDetector）

#### 职责
每日深度分析多指标异常，提供精确识别能力。

#### 输入
- 6个月历史数据
- 特征列表（价格、成交量、技术指标等）
- 模型参数（contamination=0.05）

#### 输出
```python
{
    'anomalies': [
        {
            'timestamp': datetime,
            'anomaly_score': float,  # -1到1，越低越异常
            'severity': 'high' | 'medium' | 'low',
            'features': dict,  # 异常特征值
            'description': str
        }
    ]
}
```

#### 核心算法
```python
from sklearn.ensemble import IsolationForest

# 训练模型
model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)
model.fit(features)

# 预测
anomaly_score = model.decision_function(features)
predictions = model.predict(features)
```

#### 特性
- 自动学习异常模式
- 多维度特征融合
- 适应性强

### 3.3 特征提取器（FeatureExtractor）

#### 职责
从历史数据提取多维度特征，供Isolation Forest使用。

#### 特征列表

| 类别 | 特征 | 说明 |
|-----|------|------|
| 价格特征 | close | 收盘价 |
| | return_rate | 收益率 |
| | volatility | 波动率（20日） |
| 成交量特征 | volume | 成交量 |
| | volume_ratio | 成交量比率（相对于20日均量） |
| 技术指标 | rsi | RSI（14日） |
| | macd | MACD值 |
| | macd_signal | MACD信号线 |
| | bb_position | 布林带位置 |
| 趋势特征 | ma20_diff | 价格与MA20的差值 |
| | ma50_diff | 价格与MA50的差值 |
| | ma20_ma50_diff | MA20与MA50的差值 |

#### 输出
```python
{
    'features': pd.DataFrame,  # 标准化特征矩阵
    'feature_names': list,     # 特征名称列表
    'timestamps': list         # 时间戳列表
}
```

#### 特性
- 自动标准化
- 缺失值处理
- 异常值清理

### 3.4 异常整合器（AnomalyIntegrator）

#### 职责
整合两层检测结果，避免重复报警，生成统一报告。

#### 输入
- 第一层异常结果（Z-Score）
- 第二层异常结果（Isolation Forest）
- 技术分析信号（现有）
- 异常缓存（已报告的异常）

#### 输出
```python
{
    'has_anomaly': bool,
    'anomalies': list,
    'severity': 'high' | 'medium' | 'low',
    'email_content': str,
    'email_subject': str
}
```

#### 整合策略
```python
# 严重程度分级
if z_score > 4 or if_score < -0.7:
    severity = 'high'
elif z_score > 3 or if_score < -0.5:
    severity = 'medium'
else:
    severity = 'low'

# 去重处理
cache_key = f"{anomaly_type}_{date}"
if cache_key in reported_anomalies:
    continue  # 已报告，跳过
```

## 4. 数据流程

### 4.1 每小时流程（实时检测）

```
开始
  ↓
获取最新价格和成交量数据（CoinGecko API）
  ↓
更新滚动窗口数据（30天）
  ↓
计算Z-Score（价格、成交量）
  ↓
检测异常（|Z-Score| > 3）
  ↓
是否有异常？
  ├─ 否 → 结束
  └─ 是 → 继续
      ↓
    检查24小时内是否已报告
      ├─ 是 → 结束
      └─ 否 → 继续
          ↓
        生成警报信息
          ↓
        整合技术分析信号
          ↓
        发送邮件
          ↓
        记录到异常缓存
          ↓
        结束
```

### 4.2 每日流程（深度分析）

```
开始
  ↓
获取6个月历史数据（Yahoo Finance）
  ↓
提取多维度特征（10+特征）
  ↓
特征标准化
  ↓
训练Isolation Forest模型
  ↓
检测最近7天的异常
  ↓
生成异常报告
  ↓
整合技术分析信号
  ↓
是否有异常？
  ├─ 否 → 结束
  └─ 是 → 继续
      ↓
    发送邮件
      ↓
    清理过期异常缓存（>48小时）
      ↓
    结束
```

### 4.3 邮件整合流程

```
开始
  ↓
检查是否有异常检测结果
  ↓
检查是否有技术分析信号
  ↓
是否需要发送邮件？
  ├─ 否 → 结束
  └─ 是 → 继续
      ↓
    生成邮件内容：
    ├── 价格概览（现有）
    ├── 技术分析（现有）
    └── 异常检测结果（新增）
        ├── 第一层异常（实时）
        ├── 第二层异常（深度）
        └── 异常严重程度分级
          ↓
        发送邮件
          ↓
        结束
```

## 5. 异常评分与警报策略

### 5.1 异常严重程度分级

#### 定义
| 严重程度 | Z-Score范围 | Isolation Forest评分 | 警报样式 |
|---------|------------|-------------------|---------|
| 高度异常 | |Z-Score| > 4 或 < -4 | < -0.7 | 🔴 【高度异常】 |
| 中度异常 | 3 < |Z-Score| ≤ 4 | -0.7 ≤ 评分 < -0.5 | 🟡 【中度异常】 |
| 低度异常 | |Z-Score| ≤ 3 | 评分 ≥ -0.5 | 🟢 【低度异常】 |

#### 警报优先级
- 高度异常：立即发送邮件，标记为高优先级
- 中度异常：正常发送邮件
- 低度异常：仅当有其他信号时合并发送

### 5.2 去重策略

#### 时间窗口去重
- 同一类型异常24小时内只报告一次
- 缓存键：`{anomaly_type}_{date}`

#### 异常缓存结构
```python
{
    'price_2026-04-03': {
        'timestamp': '2026-04-03 10:00:00',
        'severity': 'high',
        'z_score': 4.5
    },
    'volume_2026-04-03': {
        'timestamp': '2026-04-03 10:00:00',
        'severity': 'medium',
        'z_score': 3.2
    }
}
```

#### 缓存清理
- 48小时后自动清理过期记录
- 启动时自动清理过期记录

### 5.3 邮件内容设计

#### 邮件主题
```
{严重程度}以太币异常检测报告 - {日期}
```

#### 邮件结构
```
===== 异常检测报告 =====

1. 异常概览
   - 异常类型：价格异常 / 成交量异常 / 多指标异常
   - 严重程度：高 / 中 / 低
   - 检测时间：YYYY-MM-DD HH:MM:SS
   - 异常评分：{分数}

2. 异常详情

2.1 价格异常（如适用）
   - 当前价格：${price} USD
   - Z-Score值：{z_score}
   - 偏离程度：{deviation_description}
   - 可能原因：{possible_cause}

2.2 成交量异常（如适用）
   - 当前成交量：{volume}
   - 成交量比率：{ratio}x
   - Z-Score值：{z_score}
   - 可能原因：{possible_cause}

2.3 多指标异常（如适用）
   - 异常特征列表：{features}
   - Isolation Forest评分：{score}
   - 异常模式：{pattern}

3. 技术分析参考

3.1 当前技术指标
   - RSI：{rsi}
   - MACD：{macd}
   - 布林带位置：{bb_position}
   - TAV评分：{tav_score}

3.2 交易信号
   - 最近买入信号：{buy_signals}
   - 最近卖出信号：{sell_signals}

3.3 建议操作
   - {recommendation}

4. 历史参考
   - 最近7天异常次数：{count}
   - 最近30天异常趋势：{trend}

===== 报告结束 =====
```

## 6. 错误处理与测试

### 6.1 错误处理策略

#### 数据获取失败
```python
try:
    data = fetch_data()
except Exception as e:
    log_error(f"数据获取失败: {e}")
    # 使用缓存数据
    cached_data = load_cache()
    if cached_data:
        use_cached_data(cached_data)
    else:
        send_alert("数据获取失败，无可用缓存")
```

#### 模型训练失败
```python
try:
    model = train_model(features)
except Exception as e:
    log_error(f"模型训练失败: {e}")
    # 回退到Z-Score检测
    fallback_to_zscore()
    # 通知管理员
    notify_admin("Isolation Forest训练失败，已回退到Z-Score")
```

#### 邮件发送失败
```python
for attempt in range(3):
    try:
        send_email()
        break
    except Exception as e:
        log_error(f"邮件发送失败（尝试{attempt+1}/3）: {e}")
        if attempt == 2:
            # 保存邮件内容到文件
            save_email_to_file()
```

#### 异常值处理
```python
# 检测极端值
is_outlier = (abs(data - median) > 3 * std)
# 使用中位数替换
data[is_outlier] = median
# 记录替换操作
log_replacement(data[is_outlier], median)
```

### 6.2 测试计划

#### 单元测试

**测试覆盖范围**：
- Z-Score计算准确性
- 特征提取正确性
- 异常去重逻辑
- 邮件内容生成
- 缓存管理

**测试用例示例**：
```python
def test_zscore_calculation():
    # 准备测试数据
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 计算Z-Score
    z_score = calculate_zscore(data, 10)
    # 验证结果
    assert abs(z_score - 0) < 0.01  # 10是均值，Z-Score应为0
```

#### 集成测试

**测试场景**：
- 端到端异常检测流程
- 与技术分析整合
- 邮件发送完整性

**测试数据**：
- 使用模拟数据
- 使用历史回测数据

#### 回测验证

**验证指标**：
- 检测准确率：真实异常被正确识别的比例
- 误报率：正常数据被误判为异常的比例
- 响应时间：从数据获取到邮件发送的耗时
- 模型性能：Isolation Forest的训练和预测时间

**验证方法**：
- 使用历史数据（2025-2026）
- 对比不同阈值的表现
- 统计误报率和漏报率

### 6.3 监控指标

#### 运行时监控
```python
metrics = {
    'detection_accuracy': 0.85,  # 检测准确率
    'false_positive_rate': 0.15,  # 误报率
    'response_time_ms': 150,     # 响应时间（毫秒）
    'model_training_time_ms': 5000  # 模型训练时间
}
```

#### 日志记录
```python
# 异常检测日志
logger.info(f"检测到异常: type={type}, severity={severity}, score={score}")

# 性能日志
logger.info(f"检测耗时: {elapsed_time}ms")

# 错误日志
logger.error(f"模型训练失败: {error}", exc_info=True)
```

## 7. 实现计划

### 7.1 实现阶段

#### 阶段1：基础框架搭建（1-2天）
- 创建anomaly_detector模块
- 实现Z-Score检测器
- 实现特征提取器
- 基础测试

#### 阶段2：Isolation Forest集成（2-3天）
- 实现Isolation Forest检测器
- 集成到现有系统
- 异常去重逻辑
- 邮件内容整合

#### 阶段3：测试与优化（2-3天）
- 单元测试
- 集成测试
- 回测验证
- 性能优化

#### 阶段4：文档与部署（1-2天）
- 编写使用文档
- GitHub Actions配置
- 生产部署

### 7.2 文件结构

```
/data/fortune/
├── crypto_email.py                    # 主入口（修改）
├── anomaly_detector/                  # 新增模块
│   ├── __init__.py
│   ├── zscore_detector.py            # Z-Score检测器
│   ├── isolation_forest_detector.py  # Isolation Forest检测器
│   ├── feature_extractor.py          # 特征提取器
│   └── anomaly_integrator.py         # 异常整合器
└── tests/
    ├── test_zscore_detector.py       # Z-Score测试
    ├── test_isolation_forest.py      # Isolation Forest测试
    └── test_integration.py           # 集成测试
```

### 7.3 依赖项

**新增依赖**：
```txt
scikit-learn>=1.3.0
```

**现有依赖**（保持不变）：
```txt
pandas
numpy
yfinance
requests
```

### 7.4 配置参数

**环境变量配置**：
```bash
# 异常检测配置
ANOMALY_DETECTION_ENABLED=true
ZSCORE_WINDOW_SIZE=30          # Z-Score窗口大小（天）
ZSCORE_THRESHOLD=3.0          # Z-Score阈值
ISOLATION_FOREST_CONTAMINATION=0.05  # Isolation Forest污染率
ANOMALY_CACHE_HOURS=24        # 异常缓存时间（小时）
```

## 8. 风险与限制

### 8.1 已知风险

1. **数据质量风险**
   - API可能返回异常数据
   - 历史数据可能缺失
   - 缓解：数据验证、异常值清理

2. **模型性能风险**
   - Isolation Forest需要足够的训练数据
   - 市场模式变化可能导致模型失效
   - 缓解：定期重新训练、回退到Z-Score

3. **误报风险**
   - 正常市场波动可能被误判为异常
   - 缓解：调整阈值、去重策略

### 8.2 系统限制

1. **实时性限制**
   - Isolation Forest训练较慢（约5-10秒）
   - 不能用于实时高频交易
   - 适用场景：中长期投资监控

2. **数据依赖**
   - 需要至少3个月历史数据
   - 新加密货币无法使用
   - 缓解：使用更短的窗口作为后备

3. **解释性限制**
   - Isolation Forest可解释性较差
   - 难以解释为什么某点异常
   - 缓解：提供特征重要性分析

## 9. 附录

### 9.1 参考文献

1. **Moving Z-Score方法**
   - QuantInsti: "Statistical Arbitrage Trading Strategy"
   - TradingView: "Standard Deviation Channels"

2. **Isolation Forest算法**
   - Liu et al. (2008): "Isolation Forest"
   - scikit-learn文档: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

3. **量化交易最佳实践**
   - Bloomberg Quantitative Research
   - Two Sigma Technical Blog

### 9.2 术语表

| 术语 | 说明 |
|-----|------|
| Z-Score | 标准分数，衡量数据点偏离均值的标准差数量 |
| Isolation Forest | 基于随机森林的异常检测算法 |
| Contamination | 预期异常比例，Isolation Forest参数 |
| 多级过滤 | 使用多个检测层逐步筛选异常的方法 |
| 去重 | 避免重复报告同一异常的策略 |

### 9.3 版本历史

| 版本 | 日期 | 说明 |
|-----|------|------|
| 1.0 | 2026-04-03 | 初始版本 |

---

**文档结束**