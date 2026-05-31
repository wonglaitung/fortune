# 经验教训

> **版本**：v10.3 (2026-05-31) | **状态**：当前有效

---

## 一、项目架构

### 1. 数据源选择：yfinance vs 腾讯财经接口 ⭐⭐⭐

**问题**：yfinance 盘中数据不准确，异常检测邮件显示错误价格

**现象**：美团邮件显示 85.00，实际收盘价 82.10

**解决方案**：
```python
# 获取历史K线 + 实时报价更新
df = get_hk_stock_data_tencent(stock_code, period_days=90)
realtime_info = get_hk_stock_info_tencent(stock_code)
df.loc[df.index[-1], 'Close'] = realtime_info['current_price']
```

**教训**：yfinance 盘中数据可能不准确，腾讯接口实时报价更可靠

---

### 2. 目录定位清晰化 ⭐⭐⭐

**问题**：`output/` 混合人类可读报告和机器可读数据

**解决方案**：
```
data/  - 数据存储（机器可读、程序运行时读写）
output/ - 输出报告（人类可读、知识库材料）
```

**教训**：数据和报告分离，便于管理和版本控制

---

### 3. 目录重构需检查所有路径变更 ⭐⭐

**问题**：路径变更后遗漏创建新目录的代码

**解决方案**：
```python
# 写入文件前创建目录（就近原则）
os.makedirs(os.path.join(data_dir, 'hsi_prediction_reports'), exist_ok=True)
```

**教训**：目录重构时必须检查所有路径变更，确保新目录有创建代码

---

### 4. 邮件发送逻辑统一 ⭐⭐⭐

**问题**：10+ 个文件各自实现邮件发送，代码重复

**解决方案**：
```python
# 创建统一消息服务模块
message_services/
├── email_sender.py      # 统一邮件发送器
├── notifier.py          # 统一通知接口

# 使用
from message_services import send_email
send_email("主题", "内容", html_content="<html>...</html>")
```

**教训**：重复代码超过 3 处就应抽取为公共模块

---

## 二、特征工程

### 1. 绝对价格特征标准化 ⭐⭐⭐

**问题**：绝对价格特征跨股票量级差异大，模型可能学到无意义模式

**解决方案**：
```python
# 标准化：除以前一日收盘价
df['Channel_High_Ratio_20d'] = df['Channel_High_20d'] / df['Close'].shift(1)
df['MA_Ratio_20d'] = df['MA20'] / df['Close'].shift(1)
```

**需标准化的特征**：Channel、Support、MA、BB、ATR、Volume

**教训**：跨股票混合训练时，绝对价格特征必须标准化

---

### 2. 特征架构单一真相源 ⭐⭐⭐

**问题**：特征处理逻辑在多个文件中重复，维护困难

**解决方案**：
```python
# ml_trading_model.py 中定义模块级常量和方法
ABSOLUTE_PRICE_FEATURES = [...]
def prepare_features_for_selection(self, codes, horizon=20):
    """为特征选择准备数据，确保与训练一致"""

# feature_selection.py 直接导入使用
from ml_trading_model import CatBoostModel, ABSOLUTE_PRICE_FEATURES
```

**教训**：特征处理逻辑只维护一处，其他模块通过导入复用

---

### 3. 网络社区特征一致性 ⭐⭐⭐

**问题**：训练时动态提取社区 ID，预测时使用保存的社区 ID，导致特征不一致

**解决方案**：
```python
# 预加载社区 ID，无论缓存是否命中都重新计算交叉特征
preloaded_community_ids = extract_community_ids_from_network_file()
stock_df = create_market_network_interaction_features(stock_df, community_ids=preloaded_community_ids)
```

**教训**：缓存命中时也需要重新计算依赖外部数据的特征

---

### 4. 默认值设计原则 ⭐⭐

| 原则 | 说明 | 示例 |
|------|------|------|
| 分离原则 | 默认值与有效值范围分离 | `net_community_centrality_rank`: -1（默认）vs [0,1]（有效）|
| 语义原则 | 默认值有明确语义 | `net_constraint=1.0` 表示"高约束=无机会" |

---

### 5. 特征模块参数传递 ⭐⭐

**问题**：新增特征模块时，内部使用了变量但函数签名未定义参数

**现象**：
```
WARNING | LSTM-GARCH 混合特征计算失败，使用默认值: name 'code' is not defined
```

**根本原因**：
```python
# 函数内部使用了 code 变量
df = hybrid_model.calculate_features(df, symbol=code, ...)

# 但函数签名没有 code 参数
def calculate_technical_features(self, df, use_shift=True):  # 缺少 code
```

**解决方案**：
```python
# 修改函数签名添加参数
def calculate_technical_features(self, df, use_shift=True, code=None):

# 所有调用处传递参数（共 6 处）
stock_df = self.feature_engineer.calculate_technical_features(stock_df, code=code)
```

**教训**：新增特征模块时，检查函数签名是否包含内部使用的所有参数，并更新所有调用处

---

### 5. 训练时保留特征 NaN ⭐⭐

**问题**：`df.dropna()` 删除所有含 NaN 的行，导致数据丢失

**解决方案**：
```python
# 只删除标签和关键列的 NaN
df = df.dropna(subset=['Label'])
# 基本面特征的 NaN 保留，让模型自动处理
```

**教训**：LightGBM/XGBoost/CatBoost 原生支持 NaN

---

### 6. 市场级特征需与股票特征交叉 ⭐⭐

**问题**：市场级特征（HSI_Return、VIX）对所有股票同值，无法区分个股

**解决方案**：与网络社区特征交叉：`HSI_Return_1d * net_community_id`

**教训**：新增特征模块时，必须在 `get_feature_names()` 中定义

---

## 三、验证方法

### 1. 特征缓存数据泄漏 ⭐⭐⭐⭐

**问题**：缓存键未区分 backtest 和 production 模式，Walk-forward 准确率异常高（68-71%）

**解决方案**：
```python
# 缓存键添加 use_shift 参数
def _get_feature_cache_key(stock_code, last_date, use_shift=True):
    shift_suffix = "shift" if use_shift else "noshift"
    return f"{stock_code}_{last_date}_{shift_suffix}"
```

**验证结果**：修复后准确率从 68-71% 降至 54.44%

**教训**：准确率 >65%（个股）或 >80%（恒指）是数据泄漏信号

---

### 2. Walk-forward 收益率计算 ⭐⭐⭐

**问题**：数据过滤后计算的收益率不完整

**解决方案**：优先使用过滤前已计算好的收益率列

**教训**：数据过滤后计算的收益率可能不完整

---

### 3. IC 计算必须与训练一致 ⭐⭐

**问题**：IC 计算使用二元标签而非实际收益率

**解决方案**：
```python
# 正确：与训练一致的累积收益率
ic = df['probability'].corr(df['actual_return'])
```

**教训**：IC 应计算预测概率与实际收益率的相关性

---

### 4. 数据泄漏阈值因模型而异 ⭐⭐

| 模型类型 | 正常范围 | 数据泄漏信号 |
|---------|---------|-------------|
| 个股 | 50-60% | >65% |
| 恒指 | 60-82% | >80% |

**教训**：恒指是"均值"，噪声被对冲，预测更容易

---

## 四、模型训练

### 1. CatBoost 分类特征 NaN 处理 ⭐⭐

**问题**：`predict_proba` 函数缺少分类特征 NaN 处理

**解决方案**：
```python
for col in self.categorical_encoders.keys():
    test_df[col] = test_df[col].fillna('unknown').astype(str)
    test_df[col] = test_df[col].apply(
        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
    )
```

**教训**：训练和预测的预处理逻辑必须一致

---

### 2. CatBoost cat_features 参数 ⭐⭐

**问题**：分类特征编码后 numpy 自动转为 float，CatBoost 不接受

**解决方案**：不使用 `cat_features` 参数，分类特征作为普通数值特征处理

---

### 3. 训练窗口不是越大越好 ⭐

**验证**：12个月 60.11% > 18个月 56.57% > 24个月 54.85%

**教训**：港股近年结构性变化，近期数据更有预测价值

---

## 五、交易策略

### 1. 高置信度预测错误损失不一定小 ⭐⭐⭐

**验证数据**（置信度 >= 0.65）：

| 指标 | 值 |
|------|-----|
| 平均损失 | **-6.91%** |
| 最大损失 | **-72.96%** |

**教训**：高置信度不代表低风险，必须配合止损策略

---

### 2. 恒指与个股预测差距大 ⭐⭐⭐

| 指标 | 恒指 | 个股 |
|------|------|------|
| 假突破(101)准确率 | **87.32%** | **54.58%** |

**教训**：个股预测准确率接近随机，不可将恒指策略直接套用于个股

---

### 3. 异常检测策略有效性 ⭐⭐

| 策略 | 5日收益 | 胜率 | 操作 |
|------|---------|------|------|
| 价格异常 + 当日下跌 | **+4.12%** | **72%** | 🟢 抄底 |

**注意**：股票策略不适用于加密货币

---

### 4. 市场情绪过滤器 ⭐⭐⭐

**核心发现**：市场上涨比例滞后1天数据能有效识别极端市场环境

**阈值分层**：

| 层级 | 上涨比例 | 阈值 | 操作 |
|------|---------|------|------|
| extreme_bear | <20% | 1.0 | 暂停交易 |
| bear | 20-30% | 0.70 | 高置信 |
| weak | 30-40% | 0.65 | 谨慎 |
| normal | >40% | 0.50 | 标准 |

**验证效果**：准确率 62.0% → **70.7%**（+8.7%）

**教训**：市场环境感知比个股过滤更有效

---

### 5. Regime_Duration 状态稳定性 ⭐⭐

| Regime_Duration | 稳定性 | 建议 |
|-----------------|--------|------|
| < 5 天 | ⚠️ 不稳定 | 降低仓位 |
| 5-15 天 | 🟡 中等 | 正常交易 |
| > 15 天 | ✅ 稳定 | 趋势明确 |

---

## 六、双模式预测系统

### 1. 特征时点控制参数 ⭐⭐⭐

| 场景 | `use_shift` | `mode` | 特征时点 |
|------|-------------|--------|---------|
| 收市后预测 | False | production | 当日数据 |
| Walk-forward 验证 | True | backtest | T-1 数据 |

**教训**：默认值设计应优先考虑"安全"场景（防止泄漏）

---

### 2. 市场情绪过滤器时点配置 ⭐⭐

| 场景 | `lookback_days` | 说明 |
|------|----------------|------|
| 收市后预测 | 0 | 使用当日上涨比例（收市后已知） |
| Walk-forward 验证 | 1 | 使用滞后1天数据（避免前瞻性偏差） |

---

## 七、开发规范

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

## 八、快速参考

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

---

## 九、更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-05-31 | v10.3 | 新增：特征模块参数传递经验（函数签名需包含内部使用的参数） |
| 2026-05-24 | v10.2 | 新增：市场情绪过滤器验证确认（extreme_bear 完全过滤生效） |
| 2026-05-21 | v10.0 | 重构：精简至核心内容，分类整理 |
| 2026-05-21 | v9.1 | 新增：数据源选择经验（yfinance vs 腾讯接口） |
| 2026-05-20 | v9.0 | 新增：目录重构需检查所有路径变更 |
| 2026-05-18 | v8.3 | 新增：HMM 状态转换概率解读、异常检测方向标识 |
| 2026-05-16 | v7.7 | 新增：双模式预测系统章节 |
| 2026-05-12 | v7.3 | 新增：市场情绪过滤器设计与实施 |
| 2026-05-09 | v7.0 | 新增：网络可视化章节 |
| 2026-05-08 | v6.0 | 重构：精简至核心内容 |