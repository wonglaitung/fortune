# 特征与相对标签兼容性修复计划

## 背景

个股模型使用**相对标签**（Label = Future_Return > Daily_Median_Return），目标是选股（"谁比谁强"）。但当前特征管线存在根本性不匹配，导致模型 IC=-0.0214（选股能力几乎为零）。本计划整合审计发现和批量预测架构改造，系统性修复此问题。

---

## 审计发现（5个问题）

| # | 问题 | 严重程度 | 表现 |
|---|------|---------|------|
| 1 | **市场级特征混入** | 致命 | `HSI_Return_60d` 排名 #1，模型变成宏观择时器 |
| 2 | **截面化覆盖率仅 6%** | 严重 | 779 特征中仅 25 个做了截面化 |
| 3 | **单股预测截面特征失效** | 严重 | `_CS_Pct` 全 0.5，`_CS_ZScore` 全 0.0，用原始值回退 |
| 4 | **残差化后原始特征共存** | 中等 | 模型可走捷径用含宏观成分的原始版本 |
| 5 | **交叉特征含宏观成分** | 中等 | ~40% 交叉特征区分力被稀释 |

---

## 实施步骤

### P0-1：定义并排除市场级特征

**文件**：`ml_services/ml_trading_model.py`

1. 在 CatBoostModel 类中新增 `MARKET_LEVEL_FEATURES` 常量（~4130 行附近）：

```python
MARKET_LEVEL_FEATURES = [
    # 恒指收益（同日所有股票值相同）
    'HSI_Return_1d', 'HSI_Return_3d', 'HSI_Return_5d',
    'HSI_Return_10d', 'HSI_Return_20d', 'HSI_Return_60d',
    # HSI 市场状态
    'HSI_Market_Regime', 'HSI_Regime_Prob_0', 'HSI_Regime_Prob_1',
    'HSI_Regime_Prob_2', 'HSI_Regime_Duration', 'HSI_Regime_Transition_Prob',
    # 美股
    'SP500_Return', 'SP500_Return_5d', 'SP500_Return_20d',
    'NASDAQ_Return', 'NASDAQ_Return_5d', 'NASDAQ_Return_20d',
    # VIX
    'VIX_Level', 'VIX_Change', 'VIX_Ratio_MA20',
    # 美债
    'US_10Y_Yield', 'US_10Y_Yield_Change',
    'US10Y_Yield', 'US10Y_Yield_Change_5d',
    # 市场状态 one-hot
    'Market_Regime_Ranging', 'Market_Regime_Normal', 'Market_Regime_Trending',
]
```

2. 修改 `get_feature_columns()`（~4850 行），排除市场级特征：

```python
market_exclude = set(self.MARKET_LEVEL_FEATURES) if hasattr(self, 'MARKET_LEVEL_FEATURES') else set()
feature_columns = [col for col in df.columns if col not in exclude_columns and col not in market_exclude]
```

**注意**：交叉特征（如 `10d_Trend_HSI_Return_60d`）保留。`10d_Trend` 部分提供截面区分度，且分类乘数使不同股票值不同。仅排除"裸"市场级特征。

**注意**：市场级特征保留在 DataFrame 中（残差化需要用），只是不进入模型特征列表。

### P0-2：扩展截面化特征覆盖

**文件**：`ml_services/ml_trading_model.py`（~4130、4147 行）

将 `CROSS_SECTIONAL_PERCENTILE_FEATURES` 从 12 个扩展到 ~55 个，覆盖所有主要股票特异性特征族：

```python
CROSS_SECTIONAL_PERCENTILE_FEATURES = [
    # 波动率（5→8）
    'Volatility_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_60d',
    'Volatility_120d', 'GARCH_Conditional_Vol', 'GARCH_Vol_Ratio', 'Intraday_Range',
    # ATR（2→4）
    'ATR', 'ATR_Ratio', 'ATR_Risk_Score', 'ATR_Change_5d',
    # 成交量（2→8）
    'Volume_Ratio_5d', 'Volume_Ratio_20d', 'Volume_Volatility',
    'OBV', 'CMF', 'Volume_Confirmation_Adaptive',
    'Turnover_Change_5d', 'Turnover_Rate_Change_5d',
    # 动量（1→6）
    'Momentum_20d', 'Momentum_Accel_5d', 'Momentum_Accel_10d',
    'MACD_histogram', 'Price_Pct_20d', 'Close_Position',
    # RSI（1→4）
    'RSI', 'RSI_Deviation', 'RSI_ROC', 'RSI_Deviation_MA20',
    # 相对强度（0→5）—— 替代市场级特征的关键截面信号
    'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
    'Relative_Return',
    # 布林带/位置（0→5）
    'BB_Position', 'BB_Width', 'BB_Width_Normalized',
    'MA5_Deviation_Std', 'MA20_Deviation_Std',
    # 风险（0→4）
    'Max_Drawdown_20d', 'Max_Drawdown_60d',
    'Vol_Z_Score', 'Kurtosis_20d',
    # 资金流向（0→4）
    'Smart_Money_Score', 'Accumulation_Score',
    'Net_Flow_5d', 'Net_Flow_20d',
    # 基本面（0→4）
    'PE', 'PB', 'ROE', 'Market_Cap',
]
```

同步扩展 `CROSS_SECTIONAL_ZSCORE_FEATURES`，加入相同特征 + `Volume`, `Turnover`, `Turnover_Mean_20` 等。

**注意**：实际存在的特征名需在开发时验证（参考 `output/cumulative_importance_features_latest.txt`），不存在的自动跳过。

### P0-3：批量预测架构（方案 A）

**文件**：`ml_services/ml_trading_model.py`

#### 3a. 新增 `_extract_raw_features_single()` 方法（~5490 行）

从 `predict()` 中提取特征计算部分（5253-5346 行的逻辑），返回带 `Code` 列的 `stock_df`，不计算截面特征也不预测：

```python
def _extract_raw_features_single(self, code, predict_date=None, horizon=None, use_feature_cache=True):
    """提取单只股票的原始特征（不含截面特征），供批量预测使用"""
    # 复用 predict() 中 5253-5346 行的特征计算逻辑
    # 返回 stock_df（含 Code 列），不提取 latest_data，不预测
    # 包含残差化处理
```

#### 3b. 新增 `CatBoostModel.predict_batch()` 方法

```python
def predict_batch(self, codes, predict_date=None, horizon=None, use_feature_cache=True):
    """批量预测：先提取所有股票特征，再统一计算截面特征，最后逐只预测

    核心改进：截面特征（_CS_Pct, _CS_ZScore）在所有股票数据上联合计算，
    确保训练/预测一致，而非单只股票时退化为 0.5/0.0。
    """
    # 阶段1：逐只提取原始特征
    all_features = {}
    for code in codes:
        stock_df = self._extract_raw_features_single(code, predict_date, horizon, use_feature_cache)
        if stock_df is not None:
            all_features[code] = stock_df

    if not all_features:
        return []

    # 阶段2：合并所有股票，计算截面特征
    combined = pd.concat(all_features.values())

    if self.use_cross_sectional_percentile:
        combined = self._calculate_cross_sectional_percentile_features(combined)
    if self.use_cross_sectional_zscore:
        combined = self._calculate_cross_sectional_zscore_features(combined)

    # 阶段3：逐只预测（使用正确的截面特征）
    results = []
    for code, _ in all_features.items():
        stock_data = combined[combined['Code'] == code]
        latest = stock_data.iloc[-1:]
        # 处理分类特征、填充缺失值、预测
        result = self._predict_from_features(code, latest, horizon)
        if result:
            results.append(result)

    return results
```

#### 3c. 新增 `_predict_from_features()` 辅助方法

从 `predict()` 中提取预测逻辑（5398-5490 行）：获取 latest_data → 处理分类特征 → CatBoost predict_proba → 格式化输出。供 `predict()` 和 `predict_batch()` 共用。

#### 3d. 更新 `EnsembleModel.predict_batch()`（6315-6330 行）

```python
def predict_batch(self, codes, predict_date=None):
    # CatBoost 使用批量预测（含正确截面特征）
    catboost_results = self.catboost_model.predict_batch(codes, predict_date)
    # LightGBM/GBDT 不使用截面特征，保留原有逐只预测
    # ... 融合逻辑
```

### P0-4：单股预测的截面特征回退机制

**文件**：`ml_services/ml_trading_model.py`

#### 4a. 训练时保存截面统计量

在 `prepare_data()` 末尾（~4820 行），截面特征计算完成后：

```python
# 保存截面特征的训练集统计量，供单只股票预测时回退使用
self.cs_feature_stats = {}
for col in df.columns:
    if col.endswith('_CS_Pct') or col.endswith('_CS_ZScore'):
        self.cs_feature_stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
        }
```

#### 4b. 在 `save_model()` / `load_model()` 中持久化

`save_model()`（~5491 行）增加 `'cs_feature_stats': self.cs_feature_stats`
`load_model()`（~5513 行）增加 `self.cs_feature_stats = model_data.get('cs_feature_stats', {})`

#### 4c. 修改 `predict()` 中的截面特征回退（5416-5431 行）

```python
# 当前：用原始特征值替代（导致训练-预测分布不一致）
# 改为：用训练集统计量的均值替代（更合理的中性值）
for suffix in ['_CS_Pct', '_CS_ZScore']:
    cs_features = [col for col in self.feature_columns if col.endswith(suffix)]
    for cs_feat in cs_features:
        if cs_feat not in latest_data.columns:
            if hasattr(self, 'cs_feature_stats') and cs_feat in self.cs_feature_stats:
                latest_data[cs_feat] = self.cs_feature_stats[cs_feat]['mean']
            else:
                # 最终回退：_CS_Pct 用 0.5（中位数），_CS_ZScore 用 0.0（均值）
                latest_data[cs_feat] = 0.5 if suffix == '_CS_Pct' else 0.0
```

#### 4d. `predict()` 增加降级警告

```python
if self.use_cross_sectional_percentile or self.use_cross_sectional_zscore:
    logger.warning("predict() 单股预测时截面特征使用训练集均值回退，精度降低。"
                   "建议使用 predict_batch() 获取正确的截面特征。")
```

### P1-5：残差化策略改进

**文件**：`ml_services/ml_trading_model.py`、`data_services/feature_residualizer.py`

#### 5a. `keep_original=False`（~4798 行、~5388 行）

训练和预测时将 `keep_original=True` 改为 `keep_original=False`。残差版本直接替换原始特征名，不再产生 `_Residual` 后缀列。

```python
# prepare_data() 中：
df = residualizer.residualize(df, inplace=True, keep_original=False)

# predict() / predict_batch() 中：
stock_df = self.residualizer.residualize(stock_df, inplace=True, keep_original=False)
```

**风险**：这是破坏性变更，旧模型 pkl 不兼容。必须配合完整重训练。

#### 5b. 扩展 `MICRO_FEATURES` 列表

**文件**：`data_services/feature_residualizer.py`（~75-132 行）

将 30 个微观特征扩展到覆盖所有主要股票特异性特征。新增：
- 更多动量特征：`Momentum_5d`, `Momentum_10d`, `Momentum_60d`
- 更多波动率特征：`Volatility_5d`, `Volatility_10d`
- 更多成交量特征：`Volume_Volatility`, `Turnover_Change_5d`
- 布林带特征：`BB_Position`, `BB_Width`
- 资金流向特征：`Smart_Money_Score`, `Accumulation_Score`

### P2-6：调用方迁移

#### 6a. `comprehensive_analysis.py`

找到 `model.predict(stock_code)` 调用，改为先收集所有 stock_codes，再调用 `model.predict_batch(codes)` 批量预测，从结果字典中查找各股票预测。

#### 6b. CLI 入口（`ml_trading_model.py` ~6718 行）

将逐只 `model.predict(code)` 改为 `model.predict_batch(WATCHLIST)`。

---

## 实施顺序

```
阶段 1（P0，可独立交付）：
  Step 1 → 定义 MARKET_LEVEL_FEATURES 常量
  Step 2 → 排除市场级特征 from feature_columns
  Step 3 → 扩展截面化特征列表
  Step 4 → _extract_raw_features_single()
  Step 5 → CatBoostModel.predict_batch()
  Step 6 → 截面统计量回退机制
  Step 7 → 完整重训练 + Walk-forward 验证

阶段 2（P1，在阶段 1 验证后）：
  Step 8  → keep_original=False + 扩展 MICRO_FEATURES
  Step 9  → 再次重训练 + Walk-forward 验证
  Step 10 → 调用方迁移（comprehensive_analysis.py, CLI）

阶段 3（验证收尾）：
  Step 11 → IC/Rank IC 对比验证
  Step 12 → 更新 CLAUDE.md、progress.txt
```

---

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `ml_services/ml_trading_model.py` | P0 全部 + P1 全部（主战场） |
| `data_services/feature_residualizer.py` | 扩展 MICRO_FEATURES + keep_original 默认值 |
| `comprehensive_analysis.py` | 调用方迁移到 predict_batch |
| `ml_services/walk_forward_validation.py` | 验证兼容性，可能微调 |
| `config.py` | 确认 TRAINING_STOCKS 列表（截面计算基数） |

---

## 验证策略

### 语法和单元验证
```bash
python3 -m py_compile ml_services/ml_trading_model.py
python3 -m py_compile data_services/feature_residualizer.py
python3 -m pytest tests/ -v
```

### 功能验证
1. **截面特征正确性**：`predict_batch()` 对 3 只股票返回不同 `_CS_Pct` 值（非全 0.5）
2. **市场级特征排除**：训练后 `feature_columns` 中不包含 `HSI_Return_*` 等
3. **单股回退**：`predict()` 缺失截面特征时填充训练均值 + 打印警告

### 模型验证（`/model_validation` skill）
4. **Walk-forward 验证**：`python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20`
5. **IC 对比**：目标 IC 从 -0.02 提升至正数，夏普比率不低于当前 0.9677
6. **准确率范围**：59-62%（超过 65% 需排查数据泄漏）

### 回归保护
7. **特征数量日志**：训练时记录特征总数，确保在 700-1000 范围内
8. **predict_batch 一致性**：批量预测 vs 逐只预测结果，截面特征部分不同但原始特征相同

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 移除市场级特征后短期性能下降 | 保留 `use_market_features` 开关（默认 False）用于 A/B 对比 |
| `keep_original=False` 使旧模型不兼容 | 模型版本号区分，强制重训练 |
| 截面特征在 28 只 WATCHLIST 上较粗 | 用 45 只 TRAINING_STOCKS 做截面计算，提升区分度 |
| `predict_batch()` 比 `predict()` 慢 | 批量预测适用于每日批量任务；单股预测保留降级路径 |
