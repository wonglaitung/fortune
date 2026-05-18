# 经验教训

> **版本**：v8.2 (2026-05-18) | **状态**：当前有效

---

## 一、特征工程

### 1. 绝对价格特征标准化 ⭐⭐⭐

**问题**：绝对价格特征跨股票量级差异大，可能导致模型学到无意义模式

**现象**：
- 腾讯 Channel_High_20d ≈ 400元 vs 小盘股 ≈ 5元
- CatBoost 偏向选择高方差特征，导致绝对价格特征进入 Top 20
- 模型可能学到"高价股=好"这种无意义模式

**解决方案**：
```python
# 标准化：除以前一日收盘价
prev_close = df['Close'].shift(1)  # 避免数据泄漏
df['Channel_High_Ratio_20d'] = df['Channel_High_20d'] / prev_close
df['Support_Ratio_120d'] = df['Support_120d'] / prev_close
df['MA_Ratio_20d'] = df['MA20'] / prev_close
df['ATR_Pct'] = df['ATR'] / prev_close
```

**标准化特征意义**：
| 特征 | 值范围 | 解读 |
|------|--------|------|
| MA_Ratio_20d = 1.02 | >1 | 当前价格比20日均线低2%（偏空） |
| Support_Ratio_120d = 0.90 | <1 | 支撑位在当前价格下方10% |
| ATR_Pct = 0.03 | - | 日波动率3% |

**需标准化的特征类型**：
| 类型 | 绝对特征 | 标准化替代 |
|------|----------|------------|
| 价格通道 | Channel_High/Low_20d | Channel_High/Low_Ratio_20d |
| 支撑阻力 | Support/Resistance_120d | Support/Resistance_Ratio_120d |
| 均线 | MA5/20/60/120/250 | MA_Ratio_5/20/60/120/250d |
| 布林带 | BB_upper/lower/middle | BB_Upper/Lower/Middle_Ratio |
| 波动率 | ATR | ATR_Pct |
| 成交量 | Volume_MA7/120/250 | Volume_Ratio_7d/20d/120d |
| OBV | OBV 绝对值 | OBV_Trend, OBV_Change_5d |

**不需要标准化的特征**：
- 技术指标极值（RSI_High_5d_History 等）：本身是 0-100 范围
- 网络特征、收益率特征：本身就是比率

**教训**：
- 树模型虽然对特征量级不敏感，但跨股票混合训练时需要标准化
- 标准化特征更直观易解读

---

### 2. 特征架构单一真相源 ⭐⭐⭐

**问题**：特征处理逻辑在多个文件中重复，维护困难且易不一致

**现象**：
- `feature_selection.py` 和 `ml_trading_model.py` 各自维护排除列表
- `ABSOLUTE_PRICE_FEATURES` 需要手动同步
- 新增特征时容易遗漏

**解决方案**：
```python
# ml_trading_model.py 中定义模块级常量
ABSOLUTE_PRICE_FEATURES = [...]

# 新增方法，封装所有特征处理逻辑
def prepare_features_for_selection(self, codes, horizon=20, sample_size=10):
    """为特征选择准备数据，确保与训练一致"""
    # 预加载社区 ID
    # 调用 prepare_data
    # 删除 NaN
    # 获取特征列（已排除绝对值）
    # 编码分类特征
    return X, y, feature_columns

# feature_selection.py 直接导入和使用
from ml_trading_model import CatBoostModel, ABSOLUTE_PRICE_FEATURES
X, y, feature_columns = model.prepare_features_for_selection(...)
```

**教训**：
- 特征处理逻辑只维护一处，其他模块通过导入或方法调用复用
- 模块级常量作为单一真相源，自动同步到所有使用方

---

### 3. 网络社区特征一致性 ⭐⭐⭐

**问题**：训练时动态提取社区 ID，预测时使用保存的社区 ID，导致特征不一致

**现象**：
```
WARNING | 动态提取社区 ID（可能导致训练/预测不一致）: [np.int64(0)]
```

**根本原因**：
- 缓存命中时跳过交叉特征计算
- 缓存中的交叉特征是旧的网络特征文件生成的
- 网络特征文件更新后，缓存未同步更新

**解决方案**：
```python
# 1. 预加载社区 ID
preloaded_community_ids = extract_community_ids_from_network_file()

# 2. 无论缓存是否命中，都重新计算交叉特征
stock_df = create_market_network_interaction_features(
    stock_df, community_ids=preloaded_community_ids)

# 3. 更新缓存
save_feature_cache(stock_df)
```

**教训**：
- 缓存命中时也需要重新计算依赖外部数据的特征
- 网络特征文件更新后，缓存中的交叉特征可能过时

**代码**：`ml_services/ml_trading_model.py:4517-4561`

---

### 2. 默认值设计原则 ⭐⭐

**问题**：默认值设计不当会导致模型无法区分"缺失"和"有效值"

**设计原则**：

| 原则 | 说明 | 示例 |
|------|------|------|
| 分离原则 | 默认值应与有效值范围完全分离 | `net_community_centrality_rank`: -1（默认）vs [0,1]（有效）|
| 语义原则 | 默认值应有明确语义 | `net_constraint=1.0` 表示"高约束=无机会" |
| 中性原则 | 中性默认值用于无法判断的情况 | `sector_rising_ratio=0.5` 表示"50%上涨" |

**常见错误**：

| 错误 | 问题 | 修正 |
|------|------|------|
| 默认值=中位数 | 无法区分"未知"和"中等" | 0.5 → -1 |
| 默认值=边界值 | 无法区分"未知"和"第一名" | 0 → -1 |
| 不合理的值 | 现实中不存在 | PE=0 → NaN |

---

### 3. 训练时保留特征 NaN ⭐⭐

**问题**：`df.dropna()` 删除所有含 NaN 的行，导致数据丢失

**解决方案**：
```python
# 只删除标签和关键列的 NaN
df = df.dropna(subset=['Label'])
critical_cols = ['Return_1d', 'Return_5d', 'Return_20d', 'Close', 'Volume']
df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
# 基本面特征的 NaN 保留，让模型自动处理
```

**技术原理**：LightGBM/XGBoost/CatBoost 原生支持 NaN

---

### 4. 市场级特征需与股票特征交叉 ⭐⭐

**问题**：市场级特征（如 HSI_Return、VIX）对所有股票同值，无法区分个股

**解决方案**：
- 使用 `_build_market_level_features()` 动态构建，自动同步特征模块
- 与网络社区特征交叉：`HSI_Return_1d * net_community_id`

**教训**：新增特征模块时，必须在 `get_feature_names()` 中定义，确保自动同步

---

### 5. 特征缓存版本控制 ⭐

**问题**：新增特征后，旧缓存缺少新特征列

**解决方案**：
- 缓存验证时检查必需特征列是否存在
- 缺少新特征时标记缓存无效，重新计算

---

## 二、验证方法

### 6. 特征缓存数据泄漏 ⭐⭐⭐⭐

**问题**：特征缓存键未区分 backtest 和 production 模式，导致 Walk-forward 验证准确率异常高（68-71%）

**现象**：
- Walk-forward 验证准确率超过 65% 数据泄漏阈值
- Fold 1-4 准确率 68-71%，明显高于正常范围（50-55%）

**根本原因**：
- `_get_feature_cache_key()` 未包含 `use_shift` 参数
- backtest 模式（`use_shift=True`）和 production 模式（`use_shift=False`）使用相同缓存键
- production 模式缓存包含当日数据，被 backtest 模式误用

**解决方案**：
```python
# 缓存键添加 use_shift 参数
def _get_feature_cache_key(stock_code, last_date, use_shift=True):
    shift_suffix = "shift" if use_shift else "noshift"
    return f"{stock_code}_{last_date}_{shift_suffix}"

# 保存缓存时记录 use_shift
def _save_feature_cache(cache_file_path, feature_data, use_shift=True):
    pickle.dump({
        'data': feature_data,
        'timestamp': datetime.now().isoformat(),
        'use_shift': use_shift
    }, f)

# 加载缓存时验证 use_shift
def _load_feature_cache(cache_file_path, use_shift=None):
    if use_shift is not None and 'use_shift' in cache:
        if cached_use_shift != use_shift:
            logger.warning(f"缓存 use_shift={cached_use_shift} 与期望值 {use_shift} 不匹配")
            return None
    return cache['data']
```

**验证结果**：
- 修复后准确率从 68-71% 降至 54.44%，符合正常范围
- 所有指标恢复正常：夏普比率 6.05，最大回撤 -0.95%

**教训**：
- 特征缓存必须区分不同预测模式
- 准确率 >65%（个股）或 >80%（恒指）是数据泄漏信号
- 发现异常高准确率时，立即排查特征缓存和数据流

---

### 7. Walk-forward 收益率计算 ⭐⭐⭐

**问题**：数据过滤后计算的收益率不完整

**现象**：
- 修复前：所有负收益率都在 [-1%, 0%] 区间
- 修复后：收益率分布正常，范围 [-73%, 125%]

**解决方案**：
```python
# 优先使用过滤前已计算好的收益率列
if 'Future_Return' in df.columns and df['Future_Return'].notna().sum() > 0:
    df['actual_return'] = df['Future_Return']
else:
    df['actual_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
```

**教训**：数据过滤后计算的收益率可能不完整

---

### 7. IC 计算必须与训练一致 ⭐⭐

**问题**：IC 计算使用二元标签而非实际收益率

**解决方案**：
```python
# 正确：与训练一致的累积收益率
df['actual_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
ic = df['probability'].corr(df['actual_return'])
```

**教训**：IC 应计算预测概率与实际收益率的相关性，而非二元标签

---

### 8. 夏普比率需结合波动率解读 ⭐⭐

**问题**：高夏普比率可能来自低波动率，而非高收益

**高夏普比率原因**：

| Fold | 正收益批次% | 夏普比率 | 原因 |
|------|------------|---------|------|
| 1 | 100% | 6.51 | 批次收益全正，标准差小 |
| 12 | 100% | 10.27 | 批次收益全正，标准差小 |

**教训**：
- 高夏普比率可能来自低波动率，需检查批次收益分布
- 关注夏普比率标准差，稳定性比平均值更重要

---

### 9. 数据泄漏阈值因模型而异 ⭐⭐

**阈值标准**：

| 模型类型 | 正常范围 | 数据泄漏信号 |
|---------|---------|-------------|
| 个股 | 50-60% | >65% |
| 恒指 | 60-82% | >80% |

**教训**：恒指是"均值"，噪声被对冲，预测更容易

---

## 三、模型训练

### 10. CatBoost 分类特征 NaN 处理 ⭐⭐

**问题**：`predict_proba` 函数缺少分类特征 NaN 处理

**解决方案**：
```python
for col in self.categorical_encoders.keys():
    test_df[col] = test_df[col].fillna('unknown').astype(str)
    encoder = self.categorical_encoders[col]
    test_df[col] = test_df[col].apply(
        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
    )
```

**教训**：训练和预测的预处理逻辑必须一致

---

### 10.1 CatBoost cat_features 参数与数据类型一致性 ⭐⭐

**问题**：CatBoost 训练时报错 `'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features`

**根本原因**：
- 分类特征用 LabelEncoder 编码为整数（如 `0, 1, 2...`）
- 调用 `df[feature_columns].values` 时，numpy 将整个数组转为统一类型
- 因为其他特征是 float，整个数组变成 float 类型
- CatBoost 的 `cat_features` 参数不接受 float 数组

**解决方案**：
```python
# 方案1：不使用 cat_features（推荐，改动最小）
# 分类特征作为普通数值特征处理
catboost_params = {
    'loss_function': 'Logloss',
    # 不设置 cat_features
}
train_pool = Pool(data=X, label=y)  # 不传递 cat_features

# 方案2：保持 DataFrame 格式（更优但改动大）
# 不调用 .values，保持 DataFrame 格式
train_pool = Pool(data=df[feature_columns], label=y, cat_features=[0, 1])
```

**验证**：
```python
# float 数组 + cat_features → 报错
# int 数组 + cat_features → 正常
# float 数组不使用 cat_features → 正常
```

**教训**：
- 训练和预测必须使用相同的 `cat_features` 设置
- 分类特征编码为数值后，CatBoost 仍能学到其模式
- numpy 数组类型转换可能导致意外行为

---

### 11. 特征冗余清理需谨慎 ⭐

**问题**：清理冗余特征后夏普比率下降 15-20%

**教训**：
- r=1.0 的"冗余"特征可能有预测价值
- 清理特征后需重新调参，不能直接使用旧参数

---

### 12. 训练窗口不是越大越好 ⭐

**验证**：12个月 60.11% > 18个月 56.57% > 24个月 54.85%

**教训**：港股近年结构性变化，近期数据比更多历史数据更有预测价值

---

## 四、交易策略

### 13. 高置信度预测错误损失不一定小 ⭐⭐⭐

**验证数据**（置信度 >= 0.65）：

| 指标 | 值 |
|------|-----|
| 平均损失 | **-6.91%** |
| 最大损失 | **-72.96%** |
| 损失 <= -5% | **49.4%** |

**教训**：高置信度不代表低风险，必须配合止损策略

---

### 14. 恒指与个股预测差距大 ⭐⭐⭐

**验证数据**：

| 指标 | 恒指 | 个股 |
|------|------|------|
| 假突破(101)准确率 | **95.00%** | **54.58%** |
| 5d预测概率与20d实际方向相关性 | +0.35 | +0.03 |

**教训**：个股预测准确率接近随机，不可将恒指策略直接套用于个股

---

### 15. 异常检测策略有效性 ⭐⭐

| 策略 | 5日收益 | 胜率 | 操作 |
|------|---------|------|------|
| 价格异常 + 当日下跌 | **+4.12%** | **72%** | 🟢 抄底 |
| IF high | -3.04% | 43% | 🔴 减仓 |

**注意**：股票策略不适用于加密货币

---

### 16. 问题 Fold 盈亏比倒挂根因 ⭐⭐⭐

**验证数据**（2026-05-18，12 folds）：

| Fold | 盈亏比 | 极端损失次数 | 极端损失平均 | 高置信准确率 |
|------|--------|-------------|-------------|-------------|
| 3 | 0.53 | 501 | -18.08% | 26.6% |
| 5 | 1.33 | 44 | -31.38% | 80.3% |
| 10 | 0.90 | 193 | -15.12% | 59.5% |
| 11 | 0.77 | 124 | -13.82% | 44.7% |

**特征重要性差异**：

| 特征类别 | 问题Fold | 正常Fold | 差异 |
|---------|---------|---------|------|
| Network | 38.03 | 48.12 | -21% |
| Market | 45.93 | 55.02 | -16.5% |
| Volatility | 14.58 | 13.87 | +5.2% |

**根因分析**：
- 模型已正确识别震荡期（自动降低 Network/Regime 重要性）
- 但仍给出高置信度预测，实际准确率低
- 问题不在特征设计，而在风险控制

**解决方案**：
- 添加 `Regime_Duration` 检测，< 5 天时降低仓位
- 设置止损策略，单次亏损 > 10% 强制止损
- 震荡期降低高置信度预测的可信度

**教训**：
- 特征重要性变化是模型的**自适应行为**，不是问题根源
- 盈亏比倒挂需从交易策略层面解决，而非特征层面

---

### 17. Regime_Duration 状态稳定性检测 ⭐⭐

**状态稳定性判断**：

| Regime_Duration | 稳定性 | 建议 |
|-----------------|--------|------|
| < 5 天 | ⚠️ 不稳定 | 降低仓位 |
| 5-15 天 | 🟡 中等 | 正常交易 |
| > 15 天 | ✅ 稳定 | 趋势明确 |

**教训**：市场状态持续时间短时，模型预测可靠性下降

---

## 五、开发规范

### CatBoost 参数调优原则

| 参数 | 特征减少时调整方向 | 原因 |
|------|-------------------|------|
| n_estimators | 增加 ↑ | 特征减少需要更多树 |
| depth | 增加 ↑ | 允许更复杂的决策边界 |
| l2_leaf_reg | 降低 ↓ | 过拟合风险降低 |
| learning_rate | 降低 ↓ | 提升稳定性 |

### 数据泄漏防护

```python
# ❌ 错误：使用当日数据
df['Feature'] = df['Close'].pct_change()

# ✅ 正确：使用昨日数据
df['Feature'] = df['Close'].pct_change().shift(1)
```

**高风险特征**：所有 `.rolling()` 计算的特征必须 `.shift(1)`

### 阈值配置

| 用途 | 阈值 | 说明 |
|------|------|------|
| 方向预测 | 0.5 | 概率 > 0.5 预测上涨 |
| 置信度分级 | 0.65/0.55 | 信号强弱，不影响方向 |

---

## 五、消息服务模块

### 1. 邮件发送逻辑统一 ⭐⭐⭐

**问题**：10+ 个文件各自实现邮件发送，代码重复、难以维护

**现象**：
- 每个文件独立实现 SMTP 连接、认证、发送
- 配置修改需要改多处
- 新增通知渠道（如微信）需要改多个文件

**解决方案**：
```python
# 创建统一消息服务模块
message_services/
├── email_sender.py      # 统一邮件发送器
├── wechat_work_bot.py   # 企微机器人
├── wxpusher_bot.py      # WxPusher 推送
├── message_formatter.py # 消息格式化
└── notifier.py          # 统一通知接口

# 使用
from message_services import send_email, notify_all
send_email("主题", "内容", html_content="<html>...</html>")
notify_all("标题", "内容")  # 发送到所有已配置渠道
```

**教训**：
- 重复代码超过 3 处就应抽取为公共模块
- 通知渠道统一管理，新增渠道只需修改一处
- 保留备用实现，确保向后兼容

---

### 2. 企业微信机器人 vs WxPusher ⭐⭐

**对比**：

| 特性 | 企业微信机器人 | WxPusher |
|------|---------------|----------|
| 接收方式 | 企业微信群 | 个人微信服务号 |
| 免费额度 | 无限制 | 200 条/天 |
| 配置复杂度 | 需要企业微信 | 扫码关注即可 |
| 适用场景 | 团队通知 | 个人推送 |

**教训**：团队使用选企微机器人，个人使用选 WxPusher

---

## 六、网络可视化

### 1. 网络图边过滤 ⭐⭐

**问题**：56个节点、数百条边导致图太密集，完全看不清

**解决方案**：
```python
# 两者结合
p_threshold = 0.01  # 提高阈值（原 0.05）
top_k_edges = 50    # 限制边数
```

**教训**：
- 大网络必须过滤，否则信息过载
- 过滤后仍需保留最显著的关系

---

### 2. 分层布局优于力导向布局 ⭐⭐

**问题**：力导向布局节点位置随机，难以解读

**解决方案**：
- 按出度分层：领导者在上，跟随者在下
- 信息流从上到下，语义清晰

**实现**：
```python
def create_layered_layout(G):
    """出度高的在上层"""
    out_degrees = dict(G.out_degree())
    pos = {}
    for node in G.nodes():
        y = out_degrees.get(node, 0) / max(out_degrees.values())
        pos[node] = [x, y]
    return pos
```

---

### 3. 社区子图需包含跨社区边 ⭐⭐

**问题**：社区子图只显示内部边，银行股没有边显示

**原因**：
- 银行股之间没有显著的Granger因果关系
- 银行股的领先滞后边都是跨社区的（如 中国平安证券 → 工商银行）

**解决方案**：
```python
# 社区子图包含所有指向该社区内节点的边
for u, v in digraph.edges():
    if v in members:  # 目标节点在社区内
        subgraph.add_edge(u, v, **digraph.edges[u, v])
```

**教训**：有向网络子图应包含跨社区入边，否则重要关系会丢失

---

### 4. 边颜色应与节点关联 ⭐

**问题**：统一蓝色边无法区分领先股票来自哪个板块

**解决方案**：
- 边颜色 = 源节点（领先股票）的板块颜色
- 显著性越高颜色越深

**效果**：一眼看出领先关系来自哪个板块

---

### 16. 非对称损失函数单独使用效果有限 ⭐⭐

**问题**：问题 Fold (3, 10, 11) 盈亏比 < 1，False Positive 占主导

**尝试方案**：对 FP 错误施加惩罚（`class_weights={0: fp_penalty, 1: 1.0}`）

**完整对比测试**（fp_penalty=2.5/3.0/3.5，12 folds，57只股票）：

| 指标 | baseline | fp=2.5 | fp=3.0 | fp=3.5 |
|------|----------|--------|--------|--------|
| 平均准确率 | 55.06% | 52.54% | 51.43% | 51.18% |
| 平均夏普比率 | 5.10 | 4.58 | 4.42 | 4.36 |
| 总预测涨次数 | 7262 | 6043 | 5828 | 5102 |
| 总 FP 次数 | 2646 | 2222 | 2162 | 1842 |
| 总 FP 率 | 36.40% | 36.77% | 37.10% | **36.10%** |
| 盈亏比 | 1.53 | 1.53 | 1.53 | 1.53 |

**变化分析**（相对于 baseline）：

| 参数 | 预测涨变化 | FP变化 | FP率变化 |
|------|-----------|--------|----------|
| fp=2.5 | -16.8% | -16.0% | +0.37% |
| fp=3.0 | -19.7% | -18.3% | +0.70% |
| **fp=3.5** | **-29.7%** | **-30.4%** | **-0.30%** |

**关键发现**：
- fp_penalty=3.5 是唯一能降低 FP 率的设置
- 代价：准确率下降 3.88 个百分点，夏普比率下降 14.5%
- FP 减少主要来自模型更保守（预测涨次数减少），而非 FP 率改善
- 盈亏比不变，非对称损失不影响盈亏比

**正确用法**：
1. 配合阈值调整：提高 `confidence_threshold` 到 0.65-0.70
2. 市场状态自适应：熊市用更高惩罚，牛市用标准参数
3. 叠加规则过滤：VIX > 30 时限制看涨信号

**教训**：单一技术手段难以解决复杂问题，需多维度配合

---

### 17. 市场环境感知至关重要 ⭐⭐⭐

**问题**：问题Fold (3, 11) 盈亏比 < 1，以为是"高风险股票暴雷"

**根本原因分析**：

| Fold | 上涨比例 | 平均收益 | 盈亏比 | 诊断 |
|------|---------|---------|--------|------|
| 3 | **18.6%** | -8.44% | 0.66 | ❌ 极端下跌市场 |
| 10 | 48.9% | -0.52% | **1.11** | ✅ 正常 |
| 11 | **26.0%** | -3.09% | 0.93 | ❌ 下跌市场 |

**关键发现**：
- 高风险股票（risk >= 85）只贡献 ~20% 的 FP 亏损
- 过滤高风险股票只能解决小部分问题
- 核心问题是"极端β市场"：市场普跌时模型仍过度乐观

**精细化解决方案**：

```python
if market_up_ratio < 20:      # 极端下跌（如 Fold 3）
    suspend_trading = True     # 暂停交易
elif market_up_ratio < 30:    # 下跌市场（如 Fold 11）
    threshold = 0.70           # 大幅提高阈值
elif market_up_ratio < 40:    # 弱震荡
    threshold = 0.65
else:                          # 正常市场
    threshold = 0.50
```

**验证效果**：

| Fold | 原始盈亏比 | 阈值0.70后盈亏比 | 净收益变化 |
|------|-----------|-----------------|-----------|
| 3 | 0.66 | 0.77 | 仍为负，应暂停 |
| 11 | 0.93 | **1.09** | -823% → **+27%** |

**教训**：
- 问题本质是"极端β市场未感知"，而非"α预测失效"
- 市场环境感知比股票过滤更有效
- 不同下跌程度需要不同应对策略

---

### 18. 市场情绪过滤器设计与实施 ⭐⭐⭐

**问题**：模型在极端市场环境下仍过度乐观，导致大量 FP 亏损

**核心发现**：
- 市场上涨比例有强自相关性（lag=1 自相关系数 0.929）
- 滞后1天数据能有效识别极端市场环境（精确率80%，召回率80%）
- 问题本质是"市场普跌时模型仍过度乐观"，而非"选错股"

**设计方案**：

```python
# 使用滞后1天数据，避免前瞻性偏差
market_up_ratio_lag1 = daily_up_ratio.shift(1)

# 阈值分层
if market_up_ratio_lag1 < 0.20:      # 极端熊市
    threshold = 1.0                   # 暂停交易
elif market_up_ratio_lag1 < 0.30:    # 熊市
    threshold = 0.70                  # 高置信
elif market_up_ratio_lag1 < 0.40:    # 弱震荡
    threshold = 0.65                  # 谨慎
else:                                 # 正常市场
    threshold = 0.50                  # 标准
```

**验证效果**（12 folds，57只股票）：

| 指标 | 过滤前 | 过滤后 | 变化 |
|------|--------|--------|------|
| 准确率 | 62.0% | 70.7% | **+8.7%** |
| 总收益 | 242.13 | 305.57 | **+63.44** |
| FP | 2217 | 1424 | **-793** |

**关键 Fold 效果**：

| Fold | 市场上涨 | 原始收益 | 过滤后收益 | 收益变化 |
|------|---------|---------|-----------|---------|
| 3 | 18.6% | -51.54 | +1.87 | **+53.41** |
| 11 | 26.0% | -3.91 | +1.82 | **+5.73** |

**实施要点**：
1. **无前瞻性偏差**：严格使用滞后1天数据
2. **预计算优化**：在 Walk-Forward 开始前预计算所有交易日阈值
3. **O(1) 查询**：预测时直接查表，不影响性能
4. **可解释性**：每个信号附带 market_layer / up_ratio

**代码**：`ml_services/market_regime.py`、`ml_services/walk_forward_validation.py`

**教训**：
- 滞后数据在强自相关场景下有效（自相关系数 > 0.9）
- 市场环境感知比个股过滤更有效
- 极端市场日应完全暂停交易，而非提高阈值

---

### 19. 市场情绪过滤器生产集成 ⭐⭐⭐

**问题**：验证方案与生产环境数据源不一致，可能导致效果偏差

**现象**：
- Walk-forward 验证使用所有股票收益率计算上涨比例
- 生产环境最初使用 HSI 单一指数数据
- HSI 数据每天只有一个值，上涨比例只能是 0% 或 100%

**解决方案**：

```python
# 生产环境：获取所有股票收益率数据
all_returns = []
for stock_code in stock_codes:
    stock_df = get_hk_stock_data_tencent(stock_code, period_days=90)
    stock_df['Return_1d'] = stock_df['Close'].pct_change()
    all_returns.append(stock_df[['Date', 'Return_1d']])

returns_df = pd.concat(all_returns, ignore_index=True)
market_filter.prepare_market_schedule(returns_df, date_col='Date', ret_col='Return_1d')
```

**邮件显示增强**：

| 显示位置 | 内容 |
|---------|------|
| 邮件开头 | 市场层级、滞后1天上涨比例、动态阈值 |
| 表格新增列 | "市场调整"：显示每只股票的市场调整状态 |
| JSON 新增字段 | `probability_display`、`market_layer`、`dynamic_threshold` |
| 极端熊市警告 | 邮件末尾显示警告信息 |

**教训**：
- 验证方案与生产环境必须使用相同数据源
- 邮件显示应完整呈现市场情绪调整逻辑
- 传给大模型的数据应包含市场调整标注

**代码**：`comprehensive_analysis.py:777-820`（市场情绪初始化）、`comprehensive_analysis.py:955-1017`（动态阈值判断）

---

### 20. 市场情绪过滤器聚合方式 ⭐⭐

**问题**：市场上涨比例计算方式错误，导致过滤效果与预期不符

**现象**：
- 上午验证 Fold 3 收益 +1.87%，下午验证 -5.66%
- 市场上涨比例显示异常

**根本原因**：
- 正确做法：按日期聚合所有股票收益率，计算当天上涨股票占比
- 错误做法：直接使用每只股票的 Return_1d，未按日期聚合
- 结果：每个股票单独计算"上涨比例"，而非整体市场上涨比例

**解决方案**：
```python
# ❌ 错误：直接使用 Return_1d
# 每个股票的 Return_1d 只能是正或负，"上涨比例"无意义

# ✅ 正确：按日期聚合后计算
daily_up_ratio = returns_df.groupby('Date')['Return_1d'].apply(
    lambda x: (x > 0).mean()  # 当天所有股票中上涨的比例
).reset_index()
daily_up_ratio.columns = ['Date', 'Return_1d']
```

**教训**：
- 市场级指标必须先聚合再计算，不能直接使用个股数据
- 验证结果差异大时，首先检查数据聚合逻辑
- 市场上涨比例 = 当天上涨股票数 / 当天总股票数

**代码**：`ml_services/walk_forward_validation.py:_apply_market_filter`

---

### 21. Walk-forward 验证结果展示指标选择 ⭐⭐

**问题**：多股票混合信号的验证结果展示指标选择不当

**现象**：
- 使用 `np.cumprod(1 + returns)` 计算累积收益曲线的最大回撤
- 不同股票的信号混合在一起，累积收益曲线没有实际意义
- 最大回撤显示 -99.97%，明显不合理

**正确做法**：

| 指标 | 适用场景 | 计算方式 |
|------|---------|---------|
| 单笔最大亏损 | 多股票混合信号 | `returns.min()` |
| 亏损>10%比例 | 风险分布评估 | `(returns < -0.10).mean()` |
| 最大回撤 | 单一策略/单股票 | 累积收益曲线的峰值到谷值 |

**教训**：
- 多股票混合信号不应使用累积收益曲线计算最大回撤
- 单笔最大亏损更适合评估多股票策略的风险
- 胜率和准确率在看涨信号场景下相同，只保留一个即可

---

## 六、双模式预测系统

### 22. 特征时点控制参数设计 ⭐⭐⭐

**问题**：项目有两个特征生成场景，需要区分特征时点

**场景差异**：

| 场景 | 文件 | 特征时点 | 目的 |
|------|------|---------|------|
| 收市后预测 | `comprehensive_analysis.py` | 当日数据 | 最优信息利用 |
| Walk-forward 验证 | `walk_forward_validation.py` | T-1 数据 | 防止数据泄漏 |

**解决方案**：

```python
# 添加 use_shift 参数控制特征时点
def calculate_technical_features(self, df, use_shift=True):
    """
    Args:
        use_shift: 特征时点控制
            - True: 使用 T-1 数据（Walk-forward 验证）
            - False: 使用当日数据（收市后预测）
    """
    shift_val = 1 if use_shift else 0

    # 所有 .shift(1) 改为 .shift(shift_val)
    df['RSI'] = calculate_rsi().shift(shift_val)
    df['MACD'] = calculate_macd().shift(shift_val)
    # ...
```

**默认值设计原则**：

| 方法 | 默认值 | 原因 |
|------|--------|------|
| `prepare_data(mode='backtest')` | backtest | 训练/验证必须防止泄漏 |
| `predict(mode='production')` | production | 收市后预测使用当日数据 |

**涉及文件**（共 7 个）：

| 文件 | 修改内容 |
|------|---------|
| `ml_services/ml_trading_model.py` | FeatureEngineer 类添加 `use_shift` 参数（约 80 处） |
| `ml_services/hsi_ml_model.py` | 特征计算添加 `use_shift` 参数 |
| `data_services/volatility_model.py` | GARCH 特征添加 `use_shift` 参数 |
| `data_services/regime_detector.py` | HMM 特征添加 `use_shift` 参数 |
| `data_services/multiscale_features.py` | 多尺度特征添加 `use_shift` 参数 |
| `data_services/info_decay_analyzer.py` | 信息衰减特征添加 `use_shift` 参数 |
| `data_services/technical_analysis.py` | 布林带计算添加 `use_shift` 参数 |

**教训**：
- 默认值设计应优先考虑"安全"场景（防止泄漏）
- 参数命名应清晰表达意图（`use_shift` 比 `shift` 更明确）
- 所有数据服务模块必须支持双模式，否则会出现不一致

---

### 23. 市场情绪过滤器时点配置 ⭐⭐

**问题**：市场情绪过滤器在收市后预测和 Walk-forward 验证中使用不同时点

**配置差异**：

| 场景 | `lookback_days` | 说明 |
|------|----------------|------|
| 收市后预测 | 0 | 使用当日上涨比例（收市后已知） |
| Walk-forward 验证 | 1 | 使用滞后1天数据（避免前瞻性偏差） |

**关键发现**：
- 市场上涨比例有强自相关性（lag=1 自相关系数 0.929）
- 滞后1天数据能有效识别极端市场环境
- 收市后预测时，当日上涨比例已确定，可以使用

**代码位置**：
- `comprehensive_analysis.py`: `MarketSentimentFilter(lookback_days=0)`
- `ml_services/walk_forward_validation.py`: `MarketSentimentFilter(lookback_days=1)`

**教训**：
- 时点配置必须与场景匹配
- 收市后预测可以使用当日数据，因为市场已收盘
- Walk-forward 验证必须使用滞后数据，模拟真实预测环境

---

### 24. 异常特征预测价值有限 ⭐⭐

**问题**：异常检测策略（抄底信号 72% 胜率）在 20 天模型中特征重要性较低

**特征重要性分析**（19 个异常特征）：

| 特征 | 排名 | 状态 |
|------|------|------|
| `anomaly_count_30d` | **112** | ✅ 进入 Top 500 |
| `anomaly_count_7d` | **432** | ✅ 进入 Top 500 |
| `Anomaly_Buy_Signal`（抄底信号） | - | ❌ 未进入 Top 500 |
| 其他 15 个异常特征 | - | ❌ 未进入 Top 500 |

**原因分析**：

1. **预测周期不匹配**：
   - 异常策略（抄底信号）验证时使用 5 天收益，胜率 72%
   - 模型预测 20 天收益，异常信号对长周期预测贡献有限

2. **异常事件稀少**：
   - 异常事件（Z-Score > 3.0）发生频率低
   - 模型难以从稀少事件中学习到稳定模式

3. **累积次数更有价值**：
   - `anomaly_count_30d` 排名 112，反映股票异常频率
   - 异常频率高的股票可能有不同的风险特性

**结论**：
- 异常检测特征对 20 天预测贡献有限
- 异常策略更适合短线交易（5 天），而非中长线预测
- 累积异常次数（`anomaly_count_*`）比单次异常信号更有预测价值

---

### 25. 异常检测特征向量化优化 ⭐⭐⭐

**问题**：`create_anomaly_features` 使用逐行循环，导致预测和回测时间成倍增加

**现象**：
- Walk-forward 验证时间从 ~10 分钟增加到 ~30+ 分钟
- 每只股票特征计算耗时 2-5 秒

**根本原因**：

```python
# 原实现：O(n) 逐行循环
for i in range(len(df)):
    # Z-Score 检测（价格）
    price_history = df['Close'].iloc[:i+1]
    price_anomaly = zscore_detector.detect_anomaly(...)

    # Z-Score 检测（成交量）
    volume_history = df['Volume'].iloc[:i+1]
    volume_anomaly = zscore_detector.detect_anomaly(...)

    # Isolation Forest 检测
    if_anomalies = if_detector.detect_anomalies_by_date(...)
```

**性能影响计算**：

| 参数 | 数值 |
|------|------|
| Walk-forward Folds | 12 |
| 股票数量 | 57 |
| 每只股票数据行数 | ~500-1000 |
| 每行操作 | Z-Score × 2 + Isolation Forest |
| **总循环次数** | 12 × 57 × ~750 = **~513,000 次** |

**解决方案**：向量化计算

```python
# 优化后：向量化 Z-Score 计算
price_rolling_mean = df['Close'].rolling(window=30).mean()
price_rolling_std = df['Close'].rolling(window=30).std()
price_zscore = (df['Close'] - price_rolling_mean) / price_rolling_std

# 异常标志（|Z-Score| >= threshold）
anomaly_price_flag = (price_zscore.abs() >= 3.0).astype(int)

# 严重程度：向量化分类
price_severity = pd.cut(
    price_zscore.abs(),
    bins=[-np.inf, 3.0, 4.0, np.inf],
    labels=[0, 1, 2]
).astype(int)

# Isolation Forest：一次性获取所有样本的异常分数
all_scores = if_detector.model.decision_function(features)
```

**性能对比**：

| 指标 | 优化前（逐行） | 优化后（向量化） | 提升 |
|------|--------------|----------------|------|
| 500 行数据耗时 | ~2.5 秒 | **0.128 秒** | **20x** |
| 每行耗时 | ~5 毫秒 | **0.255 毫秒** | **20x** |
| Walk-forward 总时间 | ~30 分钟 | **~1 分钟** | **30x** |

**教训**：
- pandas/numpy 向量化操作比 Python 循环快 10-100 倍
- 特征计算应优先使用 `rolling()`、`shift()` 等向量化方法
- Isolation Forest 的 `decision_function()` 支持批量预测，应一次性调用

**代码**：`ml_services/ml_trading_model.py:FeatureEngineer.create_anomaly_features`

---

## 七、快速参考

### 模型可信度

| 模型 | 可信度 | 推荐 |
|------|--------|------|
| CatBoost 20天 | ⭐⭐⭐⭐⭐ | **推荐** |
| CatBoost 5天 | ⭐⭐⭐ | 谨慎使用 |
| CatBoost 1天 | ⭐ | 不推荐 |
| LSTM/Transformer | ⭐ | 不推荐（F1≈0） |

### 验证方法

| 方法 | 可信度 | 说明 |
|------|--------|------|
| Walk-forward | ✅ 唯一可信 | 每个 fold 重新训练 |
| 简单评估 | ❌ 不可信 | 结果虚高 |

### 样本/特征比例

| 比例 | 状态 | 建议 |
|------|------|------|
| < 10:1 | 过拟合风险 | 增加样本或减少特征 |
| 15:1 | 可接受 | 当前状态 |
| 50:1+ | 理想 | 推荐 |

---

## 七、更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-05-16 | v7.9 | 新增：异常检测特征向量化优化经验（性能提升20-40倍） |
| 2026-05-16 | v7.8 | 新增：异常特征预测价值有限经验（特征重要性分析、预测周期不匹配） |
| 2026-05-16 | v7.7 | 新增：双模式预测系统章节（特征时点控制参数设计、市场情绪过滤器时点配置） |
| 2026-05-12 | v7.6 | 新增：市场情绪过滤器聚合方式经验（按日期聚合计算上涨比例） |
| 2026-05-12 | v7.5 | 新增：CatBoost cat_features 参数与数据类型一致性经验 |
| 2026-05-12 | v7.4 | 新增：Walk-forward 验证结果展示指标选择经验（单笔最大亏损 vs 最大回撤） |
| 2026-05-12 | v7.3 | 新增：市场情绪过滤器生产集成经验（数据源一致性、邮件显示增强、大模型标注） |
| 2026-05-12 | v7.2 | 新增：市场情绪过滤器设计与实施经验（滞后数据有效性、阈值分层、验证效果） |
| 2026-05-11 | v7.2 | 新增：市场环境感知经验（问题Fold根本原因分析） |
| 2026-05-11 | v7.1 | 完善：非对称损失函数完整对比测试（fp_penalty=2.5/3.0/3.5） |
| 2026-05-09 | v7.0 | 新增：网络可视化章节（边过滤、分层布局、社区子图、边颜色） |
| 2026-05-09 | v6.2 | 添加：特征架构单一真相源经验；完善成交量绝对值排除列表 |
| 2026-05-09 | v6.1 | 添加：绝对价格特征标准化经验 |
| 2026-05-08 | v6.0 | 重构：精简至核心内容，分类整理 |
| 2026-05-08 | v5.11 | 添加：网络社区特征缓存一致性修复 |
| 2026-05-08 | v5.10 | 添加：夏普比率解读经验 |
| 2026-05-07 | v5.7 | 添加：Walk-forward 收益率计算 Bug |
| 2026-05-05 | v5.5 | 添加：IC计算一致性、市场级特征交叉 |
| 2026-04-28 | v5.2 | 添加：特征冗余清理经验、CatBoost参数调优 |
| 2026-04-27 | v5.0 | 添加：恒指vs个股因果链差异 |
| 2026-04-23 | v4.0 | 重构：精简至核心内容 |
