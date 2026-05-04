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

## P3 阶段：Rank IC 提升（排序模型 + 特征修剪 + 指标对齐）

**背景**：P0+P1 实施后 Rank IC 从 -0.0214 改善至 -0.0013（改善 93.9%），但仍接近零。根本原因是**模型目标与评估指标不对齐**：

- 模型使用 `CatBoostClassifier` + `Logloss` 优化二分类准确率（"跑赢/跑输中位数"）
- 评估指标 Rank IC 衡量预测排序与实际收益排序的 Spearman 相关性
- 分类模型丢弃收益幅度信息（top 1% 收益和 barely-above-median 收益标签相同）
- 801 个特征中 280 个重要性为零，浪费模型容量、稀释梯度信号

**目标**：Rank IC 从 -0.0013 提升至 >0.02

### P3-7：eval_metric 对齐

**文件**：`ml_services/ml_trading_model.py:5808`

将 `CatBoostModel` 的 `eval_metric` 从 `Accuracy` 改为 `AUC`：

```python
catboost_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',  # Changed from 'Accuracy' — AUC 直接衡量排序能力
    ...
}
```

**理由**：AUC 衡量正样本排在负样本前面的概率，与排序目标对齐，改善早停决策。

### P3-8：特征修剪

**文件**：`ml_services/ml_trading_model.py`、`ml_services/walk_forward_validation.py`

#### 8a. 添加 `feature_importance_threshold` 参数

在 `CatBoostModel.__init__`（~4933行）添加参数：

```python
def __init__(self, class_weight='balanced', ...,
             feature_importance_threshold=0.0):  # 0.0 = 不修剪
    self.feature_importance_threshold = feature_importance_threshold
```

#### 8b. 训练后修剪低重要性特征

在 `train()` 方法特征重要性计算后（~5970行），修剪并重新训练：

```python
# 特征修剪：移除重要性低于阈值的特征
if self.feature_importance_threshold > 0 and len(feat_imp) > 0:
    kept_features = feat_imp[feat_imp['Importance'] >= self.feature_importance_threshold]['Feature'].tolist()
    pruned = len(self.feature_columns) - len(kept_features)
    if pruned > 0 and len(kept_features) >= 50:  # 安全：至少保留 50 个特征
        logger.info(f"特征修剪：{len(self.feature_columns)} → {len(kept_features)}（移除 {pruned} 个，阈值={self.feature_importance_threshold}）")
        self.feature_columns = [f for f in self.feature_columns if f in kept_features]
        # 用修剪后的特征重新训练
        X_pruned = df[self.feature_columns].values
        # ... 重新创建 Pool 和 fit（复用 train() 中相同逻辑）
```

**推荐阈值**：

| 阈值 | 移除特征数 | 保留特征数 | 风险 |
|------|-----------|-----------|------|
| 0.0 | 0 | 801 | 无修剪（基线） |
| 0.01 | ~327 | ~474 | 低（仅移除近零特征） |
| 0.05 | ~480 | ~321 | 中（可能过度修剪） |

#### 8c. 持久化和传递

- `save_model()` / `load_model()` 添加 `feature_importance_threshold` 字段
- `WalkForwardValidator.__init__` 添加参数，`_validate_fold()` 传递给模型构造

### P3-9：CatBoostRanker 排序模型（核心改动）

**文件**：`ml_services/ml_trading_model.py`、`ml_services/walk_forward_validation.py`

创建 `CatBoostRankerModel` 类，直接优化排序，从根本上解决模型-评估不对齐问题。

#### 9a. CatBoostRanker vs CatBoostClassifier 对比

| 项目 | CatBoostClassifier（当前） | CatBoostRanker（新增） |
|------|--------------------------|----------------------|
| 模型类 | `CatBoostClassifier` | `CatBoostRanker` |
| 损失函数 | `Logloss`（二分类） | `YetiRank`（排序） |
| 标签 | 二元 (0/1) | 连续 (`Future_Return`) |
| eval_metric | `AUC` | `NDCG` |
| group_id | 无 | 日期（每天所有股票为一组） |
| 输出 | 概率 [0,1] | 排序分数（实数） |
| class_weight | `Balanced` | 不适用（排序无需类别权重） |

#### 9b. 已验证的技术可行性

- ✅ `CatBoostRanker` 可用，支持 `YetiRank` 和 `YetiRankPairwise`
- ✅ `group_id` 参数在 `Pool` 中可用
- ✅ `monotone_constraints`、`has_time=True`、`early_stopping_rounds` + `eval_metric='NDCG'` 均可用
- ✅ `predict()` 返回实数排序分数，sigmoid 变换后兼容现有接口
- ✅ `get_feature_importance(train_pool, prettified=True)` 需传入训练池
- ❌ `PairLogProb` 和 `QueryCrossEntropy` 不支持 CPU 学习

#### 9c. 类结构

在 `CatBoostModel` 之后（~6700行）插入新类：

```python
class CatBoostRankerModel(BaseTradingModel):
    """CatBoost 排序模型 - 直接优化股票排序，最大化 Rank IC

    与 CatBoostClassifier 的关键区别：
    1. 使用 CatBoostRanker（排序模型）而非 CatBoostClassifier（分类模型）
    2. 标签为连续 Future_Return（而非二元 0/1），保留收益幅度信息
    3. 使用 group_id=date 分组，每个日期形成一个排序组
    4. 损失函数为 YetiRank（直接优化排序），而非 Logloss（优化分类）
    5. 输出为排序分数（而非概率），分数越高表示预期收益越高
    """

    # 复用 CatBoostModel 的特征列表
    MONOTONE_CONSTRAINT_MAP = CatBoostModel.MONOTONE_CONSTRAINT_MAP
    MARKET_LEVEL_FEATURES = CatBoostModel.MARKET_LEVEL_FEATURES
    CROSS_SECTIONAL_PERCENTILE_FEATURES = CatBoostModel.CROSS_SECTIONAL_PERCENTILE_FEATURES
    CROSS_SECTIONAL_ZSCORE_FEATURES = CatBoostModel.CROSS_SECTIONAL_ZSCORE_FEATURES
    ROLLING_PERCENTILE_FEATURES = CatBoostModel.ROLLING_PERCENTILE_FEATURES

    def __init__(self, loss_function='YetiRank',
                 use_monotone_constraints=True,
                 time_decay_lambda=0.5,
                 use_rolling_percentile=False,
                 use_cross_sectional_percentile=True,
                 use_cross_sectional_zscore=True,
                 feature_importance_threshold=0.0):
        super().__init__()
        self.ranker_model = None
        self.model_type = 'catboost_ranker'
        self.loss_function = loss_function
        self.use_monotone_constraints = use_monotone_constraints
        self.time_decay_lambda = time_decay_lambda
        # ... 其余参数同 CatBoostModel
```

#### 9d. `train()` 方法核心逻辑

```python
def train(self, codes, start_date=None, end_date=None, horizon=1, ...):
    # 1. 复用 CatBoostModel.prepare_data()（Future_Return 已存在于输出中）
    df = self.prepare_data(codes, start_date, end_date, horizon, for_backtest=False)

    # 2. 使用 Future_Return 作为标签（连续值），而非 Label（二元）
    y = df['Future_Return'].values

    # 3. 构建 group_id（每天所有股票为一个排序组）
    df = df.sort_index()  # 确保按日期排序
    unique_dates = sorted(df.index.normalize().unique())
    date_to_gid = {d: i for i, d in enumerate(unique_dates)}
    group_ids = df.index.normalize().map(date_to_gid).values.astype(int)

    # 4. CatBoostRanker 参数
    from catboost import CatBoostRanker, Pool
    ranker_params = {
        'loss_function': self.loss_function,  # 'YetiRank'
        'eval_metric': 'NDCG',
        'depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 600,
        'l2_leaf_reg': 2,
        'subsample': 0.75,
        'colsample_bylevel': 0.75,
        'random_seed': 2020,
        'early_stopping_rounds': 80,
        'has_time': True,  # 告诉 CatBoost 数据是时间有序的
        'thread_count': -1,
        'allow_writing_files': False,
    }

    # 5. 单调约束（与 CatBoostModel 一致）
    monotone_constraints = self._build_monotone_constraints(self.feature_columns)
    if monotone_constraints is not None:
        ranker_params['monotone_constraints'] = monotone_constraints

    self.ranker_model = CatBoostRanker(**ranker_params)

    # 6. 时间序列 CV（注意：CV 可能切分同日样本，仅影响诊断指标）
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # 提取子集的 group_ids
        group_ids_train = group_ids[train_idx]
        group_ids_val = group_ids[val_idx]

        train_pool = Pool(
            data=X[train_idx], label=y[train_idx],
            group_id=group_ids_train,
            weight=fold_weights,  # 时间衰减权重
            cat_features=categorical_features if categorical_features else None
        )
        val_pool = Pool(
            data=X[val_idx], label=y[val_idx],
            group_id=group_ids_val,
            cat_features=categorical_features if categorical_features else None
        )
        self.ranker_model.fit(train_pool, eval_set=val_pool, verbose=False)

    # 7. 全量数据重训练
    full_pool = Pool(data=X, label=y, group_id=group_ids,
                     weight=self.sample_weights,
                     cat_features=categorical_features if categorical_features else None)
    self.ranker_model.fit(full_pool, verbose=100)

    # 8. 特征重要性（需传入训练池）
    feat_importance = self.ranker_model.get_feature_importance(full_pool, prettified=True)
```

#### 9e. `predict_proba()` — 兼容 WalkForwardValidator

WalkForwardValidator 使用 `prediction_proba[:, 1]` 获取预测值。Ranker 输出实数分数，需 sigmoid 变换：

```python
def predict_proba(self, X):
    """预测排序分数（兼容 predict_proba 接口）"""
    from scipy.special import expit  # sigmoid

    test_pool = Pool(data=test_df,
                     cat_features=categorical_features if categorical_features else None)
    scores = self.ranker_model.predict(test_pool)

    # sigmoid 变换到 [0,1]，保持排序不变
    proba_col1 = expit(scores)
    return np.column_stack([1 - proba_col1, proba_col1])
```

**关键**：sigmoid 是单调变换，排序完全保留。`spearmanr(expit(scores), returns) == spearmanr(scores, returns)`，Rank IC 不受影响。

#### 9f. `_predict_from_features()` / `predict()` / `predict_batch()`

复用 CatBoostModel 的 3 阶段批量预测架构（`_extract_raw_features_single` → 合并计算截面特征 → `_predict_from_features`）。

输出格式保持兼容：

```python
return {
    'code': code,
    'prediction': 1 if score > 0 else 0,  # 正分数 = 预测跑赢
    'probability': float(expit(score)),     # sigmoid 变换，兼容阈值逻辑
    'rank_score': float(score),             # 原始排序分数
    'current_price': ...,
    'date': ...,
}
```

#### 9g. WalkForwardValidator 集成

**文件**：`ml_services/walk_forward_validation.py`

1. 添加模型类映射（line ~102）：

```python
from ml_services.ml_trading_model import CatBoostModel, CatBoostRankerModel, ...

self.model_classes = {
    'catboost': CatBoostModel,
    'catboost_ranker': CatBoostRankerModel,  # 新增
    'lightgbm': LightGBMModel,
    'gbdt': GBDTModel,
}
```

2. 在 `_validate_fold()` 添加 ranker 分支（line ~284）：

```python
elif self.model_type == 'catboost_ranker':
    model = self.model_class(
        loss_function='YetiRank',
        use_monotone_constraints=self.use_monotone_constraints,
        time_decay_lambda=self.time_decay_lambda,
        use_cross_sectional_percentile=self.use_cross_sectional_percentile,
        use_cross_sectional_zscore=self.use_cross_sectional_zscore,
        feature_importance_threshold=self.feature_importance_threshold,
    )
```

3. 对 ranker 跳过 `Label` 多样性检查（ranker 使用 `Future_Return`）：

```python
# 在 line 310-315 的 Label 多样性检查前添加：
if self.model_type == 'catboost_ranker':
    # Ranker 使用 Future_Return 作为标签，不需要 Label 列
    pass
elif 'Label' in train_data.columns:
    # 现有检查逻辑
```

---

## P4 阶段：损失函数优化（YetiRankPairwise）

**背景**：P3 阶段 CatBoostRanker 验证结果：
- IC 从 -0.0216 改善至 +0.0122（✅ 从负变正）
- Rank IC 未达标（-0.0060，目标 >0.02）
- 索提诺比率显著提升（3.54 → 5.61，+58%）

**目标**：通过损失函数优化，将 Rank IC 提升至 >0.02

### P4-10：YetiRankPairwise 损失函数（已完成 ✅）

**文件**：`ml_services/walk_forward_validation.py:300`

```python
# 已修改为
loss_function='YetiRankPairwise',
```

**原理**：
- YetiRank：基于列表式排序，优化整体排序质量
- YetiRankPairwise：成对比较式排序，直接优化"股票 A vs 股票 B"
- 截面选股本质是成对比较，YetiRankPairwise 更适合

**配套配置**：
- `eval_metric='NDCG'`（ml_trading_model.py:7336）— 排序模型用排序指标监控

### P4-11：温度参数优化（❌ 不推荐）

**分析结论**：在 Rank IC 未稳住（>0.02）前调整温度参数会放大噪音，不建议使用。

| 策略 | 评估 | 原因 |
|------|------|------|
| 固定温度 0.5 | ❌ 不推荐 | 放大噪音，等 Rank IC > 0.02 后再考虑 |
| MinMax 归一化 | ❌ 不推荐 | 输出端归一化不改变排名，无意义 |

**正确方向**：如 Rank IC 仍未达标，应在输入特征端做 Rank 归一化，而非输出端

---

## 实施顺序

```
阶段 1（P0，已完成 ✅）：
  Step 1 → 定义 MARKET_LEVEL_FEATURES 常量
  Step 2 → 排除市场级特征 from feature_columns
  Step 3 → 扩展截面化特征列表
  Step 4 → _extract_raw_features_single()
  Step 5 → CatBoostModel.predict_batch()
  Step 6 → 截面统计量回退机制
  Step 7 → 完整重训练 + Walk-forward 验证

阶段 2（P1，已完成 ✅）：
  Step 8  → keep_original=False + 扩展 MICRO_FEATURES
  Step 9  → 再次重训练 + Walk-forward 验证

阶段 3（P2，部分完成 🚧）：
  Step 10 → 调用方迁移（comprehensive_analysis.py 已支持 predict_batch，CLI 待迁移）

阶段 4（P3，已完成 ✅）：
  Step 11 → eval_metric 对齐（Accuracy → AUC）
  Step 12 → 特征修剪参数（feature_importance_threshold）
  Step 13 → CatBoostRankerModel 类实现
  Step 14 → WalkForwardValidator 集成
  Step 15 → Walk-forward 验证对比（catboost vs catboost_ranker）

阶段 5（P4，已完成 ✅）：
  Step 16 → YetiRankPairwise 损失函数替换
  Step 17 → Walk-forward 验证（Rank IC 目标 >0.02）— ✅ Rank IC 从负变正（+0.0038）
  Step 18 → MinMax 归一化验证 — ❌ 失败，已回退

阶段 6（P5，已完成 ❌）：
  Step 19 → 软标签（use_soft_label=True）实现
  Step 20 → Walk-forward 验证 — ❌ 失败，IC/Rank IC 双双变负
  Step 21 → 回退到 P4 最优配置（use_soft_label=False）

阶段 7（P6，已完成 ✅）：
  Step 22 → 特征重要性审计（Classifier vs Ranker 对比）
  Step 23 → 诊断模型是否在"偷懒"（通过交叉特征获取市场信号）
  Step 24 → 制定下一步优化方向（强化截面特征、剔除交叉特征中的市场成分）

阶段 8（P7，已完成 ⚠️）：
  Step 25 → 截面特征扩展（55 → ~100）
  Step 26 → 剔除宏观交叉特征（MACRO_CROSS_FEATURES）
  Step 27 → Walk-forward 验证 — ⚠️ Rank IC 变负，效果不如预期
  Step 28 → 建议回退到 P4 最优配置

阶段 9（P8，已完成 ✅）：
  Step 29 → 训练与预测一致性诊断（网络特征重要性极低）
  Step 30 → 启用动量网络特征（predict_batch 添加网络特征计算）
  Step 31 → 移除占位符特征（net_sector_community_match, net_mst_neighbor_sectors）
  Step 32 → 新增"训练与预测一致性"规则（docs/programmer_skill.md）

阶段 10（验证收尾，待执行）：
  Step 33 → 重新训练模型（含修复后的网络特征）
  Step 34 → Walk-forward 验证（检查网络特征重要性排名）
  Step 35 → 最终 IC/Rank IC 验证
  Step 36 → 文档更新
```

---

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `ml_services/ml_trading_model.py` | P0 全部 + P1 全部 + P3 全部 + P8 全部（主战场） |
| `data_services/feature_residualizer.py` | 扩展 MICRO_FEATURES + keep_original 默认值 |
| `comprehensive_analysis.py` | 调用方迁移到 predict_batch |
| `ml_services/walk_forward_validation.py` | 验证兼容性 + P3 Ranker 支持 + 特征修剪参数 |
| `config.py` | 确认 TRAINING_STOCKS 列表（截面计算基数） |

---

## 验证策略

### 语法和单元验证
```bash
python3 -m py_compile ml_services/ml_trading_model.py
python3 -m py_compile data_services/feature_residualizer.py
python3 -m py_compile ml_services/walk_forward_validation.py
python3 -m pytest tests/ -v
```

### 功能验证
1. **截面特征正确性**：`predict_batch()` 对 3 只股票返回不同 `_CS_Pct` 值（非全 0.5）
2. **市场级特征排除**：训练后 `feature_columns` 中不包含 `HSI_Return_*` 等
3. **单股回退**：`predict()` 缺失截面特征时填充训练均值 + 打印警告

### 模型验证（`/model_validation` skill）
4. **Walk-forward 验证（CatBoost 分类器）**：
   ```bash
   python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20
   ```
5. **Walk-forward 验证（CatBoost Ranker）**：
   ```bash
   python3 ml_services/walk_forward_validation.py --model-type catboost_ranker --horizon 20
   ```
6. **IC 对比**：目标 Rank IC 从 -0.0013 提升至 >0.02，夏普比率不低于当前 0.8291
7. **准确率范围**：59-62%（超过 65% 需排查数据泄漏）

### P3 专项验证
8. **特征修剪效果**：
   ```bash
   # 基线（无修剪）
   python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

   # 修剪阈值 0.01
   python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20 --feature-importance-threshold 0.01
   ```
9. **Ranker vs Classifier 对比**：
   | 模型 | 预期 Rank IC | 预期夏普 |
   |------|-------------|---------|
   | CatBoostClassifier（基线） | ~0.0 | ~0.83 |
   | + eval_metric=AUC | ~0.01 | ~0.85 |
   | + 特征修剪 | ~0.015 | ~0.90 |
   | **CatBoostRanker** | **>0.02** | **>1.0** |

### P4 专项验证
10. **YetiRankPairwise 验证**：
    ```bash
    python3 ml_services/walk_forward_validation.py --model-type catboost_ranker --horizon 20
    ```

11. **损失函数对比**：
    | 指标 | YetiRank（基线） | YetiRankPairwise | 目标 | 状态 |
    |------|-----------------|------------------|------|------|
    | IC | +0.0122 | **+0.0066** | >0.01 | ✅ 正值 |
    | Rank IC | -0.0060 | **+0.0038** | >0.02 | ⚠️ 未达标 |
    | 准确率 | 49.69% | 49.80% | - | ➡️ 持平 |
    | 夏普比率 | 0.7945 | **0.8103** | >0.8 | ✅ 提升 |
    | 索提诺比率 | 5.61 | 5.06 | >5.0 | ✅ 达标 |
    | 最大回撤 | -0.22% | **-0.23%** | <-20% | ✅ 极佳 |

12. **MinMax 归一化验证（失败）**：
    | 指标 | YetiRankPairwise | YetiRankPairwise + MinMax | 变化 |
    |------|------------------|---------------------------|------|
    | IC | +0.0066 | +0.0039 | ❌ -0.0027 |
    | Rank IC | +0.0038 | +0.0014 | ❌ -0.0024 |
    | 夏普比率 | 0.8103 | 0.8331 | ➡️ +0.0228 |
    | 索提诺比率 | 5.06 | 5.43 | ➡️ +0.37 |

    **结论**：MinMax 归一化导致 IC/Rank IC 双降，不适合 Ranker 分数转概率。自动温度 sigmoid 是最佳策略。

13. **关键诊断**：IC 正但 Rank IC 低 → 模型抓住了极端收益的"妖股"，但整体排序混乱。YetiRankPairwise 改善了此问题（Rank IC 从负变正），但仍未达到 >0.02 目标。

14. **Ranker 特有调参方向**（如需调参）：

    | 参数 | 建议方向 | 理由 |
    |------|----------|------|
    | `random_strength` | 1 → 5 | 防止模型过度拟合局部样本顺序噪声 |
    | `l2_leaf_reg` | 2 → 10 | Ranker 更容易过拟合极端收益样本 |
    | `bagging_temperature` | 0.2 ~ 1.0 | 控制样本抽样随机性，关注确定性样本 |

15. **进阶决策树**：

    ```
    Rank IC 结果
        │
        ├─ 情况 A：达标但波动大（ICIR 低）
        │   └─ 减小 learning_rate，增加 l2_leaf_reg
        │       （牺牲精度换稳定性）
        │
        └─ 情况 B：不达标且特征集中
            └─ 减小 colsample_bylevel（0.75 → 0.5）
                （强迫模型关注更多特征）
    ```

16. **核心原则**：如果 Rank IC ≥ 0.015，且 ICIR 高、Sortino > 5.0，不要过度追求超参优化。港股非线性极强，过度调参往往是过拟合的开始。

17. **P4 验证结论**：
    - ✅ YetiRankPairwise 是当前最优配置（Rank IC +0.0038）
    - ❌ MinMax 归一化失败，已回退到自动温度 sigmoid
    - ⚠️ Rank IC 目标（>0.02）未达成，但相比 Classifier 已有显著改善
    - 建议：接受当前配置，后续可尝试特征工程改进或模型融合

---

## P5 阶段：软标签实验（Soft Label）

**背景**：P4 验证后 Rank IC = +0.0038，仍未达到 >0.02 目标。分析发现原始收益率作为标签时，极端收益的"妖股"会主导训练，导致模型过度关注少数样本而非整体排序。

**目标**：通过软标签（截面排名百分位）替代原始收益率，让 YetiRankPairwise 学到"好多少"而非仅仅"谁更好"。

### P5-12：软标签实现

**文件**：`ml_services/ml_trading_model.py:7198-7308`

```python
# 新增参数
def __init__(self, ..., use_soft_label=True):

# 标签计算逻辑
if self.use_soft_label:
    # 计算每日截面排名百分位（0 到 1 之间）
    df['Return_Rank_Pct'] = df.groupby(df.index.normalize())['Future_Return'].transform(
        lambda x: x.rank(pct=True)
    )
    y = df['Return_Rank_Pct'].values
else:
    # 使用原始收益率作为标签
    y = df['Future_Return'].values
```

### P5-13：软标签 vs 原始收益率对比

| 标签类型 | 值范围 | 含义 | 优势 |
|----------|--------|------|------|
| 原始收益率 | 实数（可正可负） | "赚多少" | 保留收益幅度信息 |
| 软标签（排名百分位） | 0 ~ 1 | "排第几" | 消除极端值影响，与截面选股目标一致 |

### P5-14：验证结果（❌ 失败）

```bash
python3 ml_services/walk_forward_validation.py --model-type catboost_ranker --horizon 20 --n-jobs -1
```

| 指标 | P4（原始收益率） | P5（软标签） | 变化 | 目标 |
|------|-----------------|--------------|------|------|
| IC | +0.0066 | **-0.0022** | ❌ -0.0088 | >0.01 |
| Rank IC | +0.0038 | **-0.0063** | ❌ -0.0101 | >0.02 |
| 夏普比率 | 0.8103 | 0.7146 | ❌ -0.0957 | >0.8 |
| 索提诺比率 | 5.06 | 3.14 | ❌ -1.92 | >5.0 |
| 准确率 | 49.80% | 48.57% | -1.23% | - |

**失败原因分析**：
- 软标签将收益率转换为排名百分位（0~1），丢失了收益幅度信息
- YetiRankPairwise 成对比较时，无法区分"大幅跑赢"和"小幅跑赢"
- 原始收益率保留了幅度信息，模型可以学习到"好多少"而非仅仅"谁更好"
- 截面排名百分位与 YetiRankPairwise 的成对比较机制不兼容

**结论**：软标签不适合 YetiRankPairwise 损失函数，已回退到 P4 最优配置（原始收益率 + YetiRankPairwise）

### 回归保护
10. **特征数量日志**：训练时记录特征总数，确保在 400-800 范围内
11. **predict_batch 一致性**：批量预测 vs 逐只预测结果，截面特征部分不同但原始特征相同
12. **Ranker 输出兼容性**：`predict_proba()[:, 1]` 输出范围在 [0, 1]，排序与原始分数一致

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 移除市场级特征后短期性能下降 | 保留 `use_market_features` 开关（默认 False）用于 A/B 对比 |
| `keep_original=False` 使旧模型不兼容 | 模型版本号区分，强制重训练 |
| 截面特征在 28 只 WATCHLIST 上较粗 | 用 45 只 TRAINING_STOCKS 做截面计算，提升区分度 |
| `predict_batch()` 比 `predict()` 慢 | 批量预测适用于每日批量任务；单股预测保留降级路径 |
| **P3 新增风险** | |
| group_id 排序问题导致 Ranker 输出错误 | `df.sort_index()` 确保按日期排序后再分配 group_id |
| CV 切分同日样本影响诊断指标 | 仅影响 CV 诊断，最终模型全量重训练；可接受 |
| Ranker 分数不等于概率，阈值逻辑失效 | sigmoid 变换保持排序不变，兼容现有阈值逻辑；或改用 rank_score 直接排序 |
| 特征修剪过拟合到历史重要性 | 从阈值 0.01 开始（仅移除零/近零特征），Walk-forward 验证确认 |
| CatBoostRanker 与 EnsembleModel 不兼容 | P3 阶段 Ranker 作为独立模型评估，不加入 Ensemble（后续可扩展） |
| `get_feature_importance()` 需传入训练池 | 在 `train()` 中保存 `train_pool` 或在需要时重建 |
| **P4 新增风险** | |
| YetiRankPairwise 训练时间更长 | 先用少量 fold 测试，确认效果后再完整验证 |
| Rank IC 仍未达标 | 备选方案：输入特征端 Rank 归一化、特征工程改进、模型融合 |
| MinMax 归一化信息损失 | ❌ 已放弃：验证证明 MinMax 导致 IC/Rank IC 双降 |
| **P5 新增风险** | |
| 软标签丢失收益幅度信息 | ❌ 已验证失败：软标签导致 IC/Rank IC 双双变负 |
| 软标签与 YetiRankPairwise 不兼容 | ❌ 已验证失败：成对比较需要幅度信息 |

---

## P6 阶段：特征重要性审计（模型是否在"偷懒"）

**背景**：P5 软标签实验失败后，需要深入分析模型是否仍在通过"代理特征"猜测大盘环境，而非学习个股间的超额收益（Alpha）。

**核心问题**：模型是否在"偷懒"？

### P6-15：特征重要性对比分析

**对比对象**：CatBoostClassifier vs CatBoostRanker（YetiRankPairwise）

| 类型 | Classifier | Ranker | 变化 | 评估 |
|------|-----------|--------|------|------|
| **市场/宏观** | 39.7% | 29.4% | **-10.4%** | ✅ 显著改善 |
| **截面特征** | 0.0% | 29.2% | **+29.2%** | ✅ 新增有效 |
| **个股特异性** | 60.3% | 41.5% | -18.8% | ⚠️ 下降 |

**关键发现**：

1. **Ranker 成功降低市场特征依赖**（-10.4%）
   - Classifier Top 20 中市场特征占 11 席（55%）
   - Ranker Top 20 中市场特征仅占 3 席（15%）

2. **截面特征成为 Ranker 的核心信号**（+29.2%）
   - Top 1 特征：`CMF_CS_Pct`（资金流向截面排名）
   - Top 7 特征：`CMF_CS_ZScore`
   - 截面特征直接衡量"这只股票今天在所有股票中排第几"，与选股目标一致

3. **但模型仍在"偷懒"的迹象**：
   - `60d_Trend_HSI_Return_60d` 排名 #2（市场特征仍是 Top 2）
   - `10d_Trend_HSI_Regime_Prob_1` 排名 #5
   - `Outperforms_HSI_*` 类特征占 6.27%（46 个特征）
   - Trend × 市场交叉特征占 21.24%（66 个特征）

### P6-16：Classifier vs Ranker Top 20 对比

**Classifier 独有的 Top 20 特征（纯市场特征）**：
| 特征 | Classifier 排名 | Ranker 排名 |
|------|----------------|------------|
| `US_10Y_Yield` | #1 | N/A（已排除） |
| `HSI_Regime_Duration` | #2 | N/A |
| `HSI_Regime_Prob_0` | #3 | N/A |
| `HSI_Return_60d` | #4 | N/A |
| `VIX_Level` | #5 | N/A |
| `HSI_Return_10d/20d` | #13/#20 | N/A |

**Ranker 独有的 Top 20 特征（截面 + 个股）**：
| 特征 | Ranker 排名 | Classifier 排名 | 类型 |
|------|------------|----------------|------|
| `CMF_CS_Pct` | #1 | N/A | 截面 |
| `Skewness_10d` | #3 | #285 | 个股 |
| `Anomaly_Severity_Score` | #8 | #469 | 个股 |
| `Close_Position` | #12 | #227 | 个股 |
| `Kurtosis_10d` | #10 | #126 | 个股 |

**结论**：Ranker 成功将"纯市场特征"挤出 Top 20，但仍依赖"交叉特征"间接获取市场信号。

### P6-17：下一步优化方向

**问题诊断**：模型通过交叉特征（如 `60d_Trend_HSI_Return_60d`）间接获取市场信号，而非学习个股 Alpha。

**优化方向**：

| 方向 | 具体措施 | 预期效果 |
|------|---------|---------|
| **A. 强化截面特征** | 对更多个股特异性特征做截面化（目前仅 85 个） | 提升截面特征占比至 >40% |
| **B. 剔除交叉特征中的市场成分** | 移除 `Trend × HSI_*` 类交叉特征 | 强迫模型关注个股信号 |
| **C. 增加纯 Alpha 特征** | 新增 `Relative_Return_5d`、`Sector_Relative_Return` 等 | 提供更多截面信号 |
| **D. 特征权重调整** | 训练时对截面特征赋予更高权重 | 强制模型优先学习 Alpha |

**推荐优先级**：A > B > C > D

### P6-18：具体实施建议

#### A. 扩展截面特征覆盖（推荐）

当前截面特征仅覆盖 85 个，建议扩展至 150+：

```python
# 新增截面化特征候选
NEW_CS_FEATURES = [
    # 动量类（当前仅 3 个，扩展至 10）
    'Momentum_5d', 'Momentum_10d', 'Momentum_60d',
    'Momentum_Accel_5d', 'Momentum_Accel_10d', 'Momentum_Accel_120d',
    'MACD_histogram', 'MACD_Hist_ROC', 'Price_Pct_5d', 'Price_Pct_10d',

    # 风险类（当前仅 4 个，扩展至 8）
    'Max_Drawdown_60d', 'Max_Drawdown_120d',
    'Vol_Z_Score', 'Kurtosis_20d', 'Kurtosis_60d',
    'Skewness_5d', 'Skewness_10d', 'Skewness_20d',

    # 相对强度类（关键 Alpha 信号）
    'RS_Ratio_5d', 'RS_Ratio_20d', 'RS_Diff_5d', 'RS_Diff_20d',
    'Relative_Return_5d', 'Relative_Return_10d', 'Relative_Return_20d',

    # 资金流向类（当前仅 4 个，扩展至 8）
    'Smart_Money_Score', 'Accumulation_Score',
    'Net_Flow_5d', 'Net_Flow_20d',
    'CMF', 'OBV', 'Volume_Confirmation_Adaptive',

    # 基本面类
    'PE', 'PB', 'ROE', 'Market_Cap', 'Dividend_Yield',
]
```

#### B. 剔除交叉特征中的市场成分

```python
# 在 MARKET_LEVEL_FEATURES 中新增
MARKET_LEVEL_FEATURES.extend([
    # 交叉特征中的市场成分
    '60d_Trend_HSI_Return_60d',
    '10d_Trend_HSI_Return_20d',
    '10d_Trend_HSI_Regime_Prob_1',
    '20d_Trend_NASDAQ_Return_5d',
    '20d_Trend_SP500_Return_5d',
    # ... 其他 Trend × 市场交叉特征
])
```

**风险**：过度剔除可能导致模型容量不足，需逐步验证。

---

## P7 阶段：特征优化验证（截面特征扩展 + 宏观交叉特征剔除）

**背景**：P6 特征重要性审计发现模型仍通过交叉特征（如 `60d_Trend_HSI_Return_60d`）间接获取市场信号，而非学习个股 Alpha。

**优化措施**：
1. **A. 截面特征扩展**：`CROSS_SECTIONAL_PERCENTILE_FEATURES` 从 55 扩展至 ~100
2. **B. 剔除宏观交叉特征**：`MACRO_CROSS_FEATURES`（27 个特征）在 `get_feature_columns()` 中排除
3. **C. 增加纯 Alpha 特征**：新增 `Anomaly_*`、`Trend_*`、K线形态等截面化特征

### P7-19：验证结果（⚠️ 效果不如预期）

```bash
python3 ml_services/walk_forward_validation.py --model-type catboost_ranker --horizon 20 --n-jobs -1
```

| 指标 | P4（基线） | P5（软标签） | P7（特征优化） | 目标 | 评估 |
|------|-----------|--------------|----------------|------|------|
| **IC** | +0.0066 | -0.0022 | **+0.0033** | >0.01 | ⚠️ 下降 |
| **Rank IC** | +0.0038 | -0.0063 | **-0.0025** | >0.02 | ❌ 变负 |
| **准确率** | 49.80% | 48.57% | **49.29%** | - | ➡️ 持平 |
| **夏普比率** | 0.8103 | 0.7146 | **0.7786** | >0.8 | ⚠️ 略低 |
| **索提诺比率** | 5.06 | 3.14 | **5.2051** | >5.0 | ✅ 最优 |
| **最大回撤** | -0.23% | -0.21% | **-0.24%** | <-20% | ✅ 极佳 |

### P7-20：失败原因分析

**Rank IC 变负的原因**：

1. **截面特征扩展引入噪声**
   - 从 55 扩展至 ~100，新增特征可能包含低质量信号
   - 部分特征（如 `Skewness_*`、`Kurtosis_*`）在截面排名中区分度有限

2. **宏观交叉特征剔除过度**
   - `MACRO_CROSS_FEATURES`（27 个特征）被剔除
   - 这些特征虽含宏观成分，但也提供了个股与市场的相对强度信息
   - 剔除后模型失去了部分预测能力

3. **特征冗余导致信息稀释**
   - 截面特征过多（~100 个），部分特征间高度相关
   - 模型难以区分有效信号和噪声

### P7-21：关键教训

| 教训 | 说明 |
|------|------|
| **截面特征非越多越好** | 质量比数量重要，低质量特征会稀释有效信号 |
| **宏观交叉特征有双重作用** | 既含宏观成分（干扰），也含相对强度信息（有用） |
| **剔除特征需逐步验证** | 过度剔除可能导致模型容量不足 |
| **索提诺比率最优** | P7 的风险调整收益最佳，但排序能力下降 |

### P7-22：下一步建议

| 方向 | 说明 | 优先级 |
|------|------|--------|
| **回退到 P4 配置** | P4（YetiRankPairwise + 原始收益率）仍是当前最优 | ⭐⭐⭐ |
| **精简截面特征** | 从 ~100 回退到 55，仅保留高质量特征 | ⭐⭐⭐ |
| **部分恢复宏观交叉特征** | 仅剔除纯市场特征，保留 `Trend × HSI_Return` 类 | ⭐⭐ |
| **模型融合** | P4（排序）+ Classifier（分类）融合 | ⭐ |

---

## P8 阶段：训练与预测一致性修复

**背景**：P7 验证后发现网络特征重要性极低（排名 737/818、786/818、802/818），经排查发现训练和预测时特征计算不一致。

### P8-23：问题诊断

**网络特征训练-预测不一致**：

| 问题 | 训练时 | 预测时（修复前） | 后果 |
|------|--------|-----------------|------|
| 网络特征（批量预测） | 实时计算 | 默认值 0/-1 | 特征重要性极低 |
| 网络特征（单股预测） | 实时计算 | 默认值 0/-1 | 无法修复（设计限制） |

**占位符特征未实际计算**：

| 特征 | 训练时 | 预测时 | 说明 |
|------|--------|--------|------|
| `net_sector_community_match` | 设为 0 | 设为 0 | 占位符，从未实现 |
| `net_mst_neighbor_sectors` | 设为 0 | 设为 0 | 占位符，从未实现 |

### P8-24：修复措施

**1. 启用动量网络特征（predict_batch）**

修改 `CatBoostModel.predict_batch()` 和 `CatBoostRankerModel.predict_batch()`，添加网络特征实时计算：

```python
# 批量预测时正确计算网络特征
from data_services.network_features import get_network_calculator
network_calc = get_network_calculator()

# 计算网络洞察（中心性、社区）
insights = network_calc.calculate_network_insights(unique_codes, force_refresh=False)

# 计算节点偏离度（动量网络特征）
deviations = network_calc.calculate_node_deviation(unique_codes, score_type='momentum', window=20)

# 填充网络特征
for code in unique_codes:
    mask = combined['Code'] == code
    combined.loc[mask, 'net_composite_centrality'] = insights[code].get('composite_centrality', 0)
    combined.loc[mask, 'net_community_id'] = insights[code].get('community', -1)
    combined.loc[mask, 'net_node_deviation'] = deviations[code].get('node_deviation', 0)
```

**2. 移除占位符特征**

移除 `net_sector_community_match` 和 `net_mst_neighbor_sectors`，保留 3 个有效网络特征：

| 保留特征 | 说明 |
|---------|------|
| `net_composite_centrality` | 综合中心性 |
| `net_community_id` | 社区ID（分类特征） |
| `net_node_deviation` | 节点偏离度（动量网络特征） |

**3. 新增"训练与预测一致性"规则**

在 `docs/programmer_skill.md` 新增关键警告：

- 常见不一致问题类型
- 检查清单
- 批量预测优先原则

### P8-25：修改位置

| 文件 | 修改内容 |
|------|---------|
| `ml_services/ml_trading_model.py` | `prepare_data()` 网络特征计算（移除占位符） |
| `ml_services/ml_trading_model.py` | `predict()` 单股预测网络特征默认值（移除占位符） |
| `ml_services/ml_trading_model.py` | `_extract_raw_features_single()` 特征提取（移除占位符） |
| `ml_services/ml_trading_model.py` | `predict_batch()` 批量预测网络特征计算（CatBoostModel 和 CatBoostRankerModel） |
| `docs/programmer_skill.md` | 新增"训练与预测必须一致"关键警告 |

### P8-26：提交记录

```
cd05433 feat: 启用动量网络特征 - predict_batch 添加网络特征计算
0ab3c99 refactor: 移除占位符网络特征 + 文档新增"训练与预测一致性"规则
```

### P8-27：待验证

| 验证项 | 说明 |
|--------|------|
| 网络特征重要性提升 | 重新训练后检查 `net_node_deviation` 排名 |
| 训练-预测一致性 | 确保 `predict_batch()` 网络特征与训练时一致 |
| Walk-forward 验证 | 验证 Rank IC 是否改善 |

---

**最后更新**：2026-05-04（P8 训练与预测一致性修复：启用动量网络特征、移除占位符、新增一致性规则）
