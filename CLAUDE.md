# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：开发流程、系统设计决策请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)
> **📅 进度跟踪**：[progress.txt](progress.txt) - 项目当前进展

---

## ⚡ 常用命令

### 测试与验证

```bash
# 语法检查（每次修改后必须执行）
python3 -m py_compile <文件路径>

# 运行所有测试
python3 -m pytest tests/ -v

# 运行单个测试
python3 -m pytest tests/test_anomaly_integrator.py -v
```

### 核心功能命令

#### 港股系统

| 任务 | 命令 | 运行时机 |
|------|------|---------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` | 收市后 |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` | ⚠️ 收市后（16:00 HKT） |
| **股票分析邮件** | `python3 scripts/send_stock_analysis_email.py --stocks 2318.HK --email` | 收市后 |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` | 收市后推荐 |
| **个股Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` | - |
| **恒指Walk-forward验证** | `python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20` | - |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection` | - |
| **模型预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost --use-feature-selection` | - |
| **特征选择** | `python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20` | - |
| **超参数调优** | `python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30` | - |
| **股票网络分析** | `python3 ml_services/stock_network_analysis.py --skip-pmfg` | - |
| **性能监控** | `python3 ml_services/performance_monitor.py --mode all --no-email` | - |

#### A股系统

| 任务 | 命令 | 运行时机 |
|------|------|---------|
| **A股综合分析（完整流程）** | `./scripts/run_a_stock_analysis.sh` | ⚠️ 收市后（15:15 CST） |
| **A股模型训练** | `python3 a_stock_ml_model.py --mode train --horizon 20` | - |
| **A股模型预测** | `python3 a_stock_ml_model.py --mode predict --horizon 20 --core-only` | - |
| **A股大模型建议** | `python3 a_stock_email.py --force --no-email` | - |
| **A股综合分析** | `python3 a_stock_comprehensive_analysis.py --llm-file data/a_stock_llm_*.txt --use-cached-predictions` | - |
| **A股Walk-forward验证** | `python3 a_stock_walk_forward.py --horizon 20` | - |

### 缓存管理

```bash
# 清除港股特征缓存（新增特征后必须执行）
rm -rf data/feature_cache/*.pkl

# 清除港股原始数据缓存
rm -rf data/stock_cache/*.pkl

# 清除A股特征缓存
rm -rf data/a_stock_feature_cache/*.pkl

# 清除A股原始数据缓存
rm -rf data/a_stock_cache/*.pkl
```

---

## ⚠️ 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward准确率 >65%（个股）或 >80%（恒指）通常是数据泄漏信号 |
| **IC 计算** | IC 必须用实际收益率，不能用二元标签；收益率计算必须与训练一致 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **CatBoost 1天模型** | 噪音大，仅供参考 |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **恒指 vs 个股** | 恒指准确率显著高于个股（81% vs 54%），个股预测需谨慎 |
| **高置信度风险** | 高置信度预测错误时损失可达 -73%，必须设置止损 |
| **网络社区特征一致性** | 训练时保存 `model.community_ids`，预测时使用相同社区 ID 列表 |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测预处理必须一致 |
| **默认值设计** | 默认值必须与有效值范围分离，使用 -1 表示"未知"，基本面特征用 NaN |
| **训练时 NaN** | 不要用 `df.dropna()` 删除所有 NaN，只删除标签和关键列 |
| **绝对值特征** | 跨股票训练时，绝对价格/成交量特征必须标准化或排除 |
| **市场情绪数据源** | 必须使用所有股票收益率计算上涨比例，与 walk-forward 验证一致 |
| **双模式预测** | 收市后预测使用 `mode='production'`（当日数据），Walk-forward 使用 `mode='backtest'`（T-1 数据） |
| **yfinance 盘中数据** | yfinance 日线数据在盘中可能不准确，推荐使用腾讯财经接口获取实时报价 |
| **A股涨跌停差异** | 主板10%涨跌停，创业板20%涨跌停，混合训练时需标签标准化 |
| **A股股票代码前导零** | 保存CSV时必须用字符串格式 `zfill(6)`，否则前导零丢失（002655→2655） |
| **A股样本权重** | 核心股权重3.0倍，扩展股1.0倍，训练时需传入 `sample_weight` |
| **A股数据泄漏阈值** | 个股准确率正常范围50-60%，>65%为数据泄漏信号 |

---

## 📐 数据流架构

```
外部数据源 → data_services/ → 分析层 → ml_services/ → 输出
    ↓              ↓              ↓            ↓          ↓
腾讯财经      技术指标计算    异常检测     CatBoost    邮件报告
AKShare      基本面数据      综合分析     Walk-forward  JSON文件
             南向资金        主力追踪     性能监控     微信通知
```

**关键依赖关系**：
- `comprehensive_analysis.py` 整合：大模型建议 + CatBoost预测 + 异常检测 + 板块分析
- `hsi_prediction.py` 调用 `ml_services/hsi_ml_model.py` 进行CatBoost预测
- `detect_stock_anomalies.py` 使用 `anomaly_detector/` 模块的双层检测（Z-Score + Isolation Forest）
- `config.py` 定义股票板块映射 `STOCK_SECTOR_MAPPING` 和自选股列表 `WATCHLIST`（31只）
- `message_services/` 统一管理邮件和微信通知

**特征模块**（动态构建，自动同步）：
- `data_services/calendar_features.py` - 日历效应（22个特征）
- `data_services/volatility_model.py` - GARCH 波动率（4个特征）
- `data_services/regime_detector.py` - HMM 市场状态检测（10个特征）
- `ml_services/stock_network_analysis.py` - 股票网络分析（社区ID、中心性等）
- `ml_services/hybrid_volatility_model.py` - LSTM-GARCH 混合波动率（3个特征）

**消息服务模块**：
- `message_services/email_sender.py` - 统一邮件发送
- `message_services/wechat_work_bot.py` - 企业微信机器人
- `message_services/wxpusher_bot.py` - WxPusher 推送
- `message_services/notifier.py` - 统一通知接口

**A股核心模块**：
- `a_stock_config.py` - A股配置（股票池53只、板块映射、样本权重）
- `a_stock_ml_model.py` - A股模型训练与预测（1077特征）
- `a_stock_comprehensive_analysis.py` - A股综合分析（买卖建议+异常检测+板块分析）
- `a_stock_email.py` - A股大模型建议生成（通义千问）
- `a_stock_recommendation_generator.py` - 综合买卖建议生成器
- `data_services/a_stock_data.py` - A股数据获取（AKShare+腾讯财经）
- `data_services/a_stock_market_features.py` - A股市场特征（涨跌停、北向资金、跨市场联动）

**数据存储**（`data/` - 机器可读）：
- `data/hsi_models/` - 恒指CatBoost模型（.cbm）和特征配置（.json）
- `data/stock_cache/` - 原始数据缓存（7天有效期）
- `data/feature_cache/` - 特征缓存（7天有效期，170x加速）
- `data/feature_selection/` - 特征选择结果（CSV/TXT）
- `data/hsi_prediction_reports/` - 恒指预测报告（JSON）
- `data/network_features/` - 网络特征（JSON）
- `data/walk_forward_results/` - Walk-forward 验证结果（CSV/JSON）
- `data/hyperparams/` - 超参数记录（JSON）
- `data/analysis_results/` - 分析结果（CSV/JSON）

**A股数据存储**：
- `data/a_stock_models/` - A股CatBoost模型（1d/5d/20d）和特征重要性
- `data/a_stock_cache/` - A股原始数据缓存
- `data/a_stock_feature_cache/` - A股特征缓存
- `data/a_stock_network_features/` - A股网络特征（JSON）
- `data/a_stock_llm_recommendations_*.txt` - 通义千问大模型建议
- `data/a_stock_comprehensive_recommendations_*.txt` - 综合买卖建议

**输出报告**（`output/` - 人类可读）：
- `output/*.md` - Markdown 分析报告
- `output/*.txt` - 文本分析报告
- `output/*.png` - 可视化图表
- `output/*_catboost_20d/` - Walk-forward 验证结果目录
- `output/comprehensive_reports/` - 综合分析报告（知识库材料）

---

## 🏗️ 特征架构（单一真相源）

**核心原则**：特征处理逻辑只在 `ml_trading_model.py` 中维护，其他模块通过导入或方法调用复用。

```
ml_trading_model.py
├── 模块级常量
│   ├── ABSOLUTE_PRICE_FEATURES（40个绝对值特征）
│   ├── NETWORK_FEATURE_MONOTONICITY（7个网络特征单调性）
│   └── MARKET_FEATURE_MONOTONICITY（34个市场特征单调性）
│
├── BaseTradingModel 类
│   ├── get_feature_columns()     # 排除绝对值特征，返回有效特征列表
│   ├── prepare_features_for_selection()  # 特征选择专用方法
│   └── prepare_data()            # 完整特征准备
│
└── FeatureEngineer 类
    ├── 计算技术指标
    ├── 生成交叉特征
    ├── create_monotonic_interaction()  # 智能交叉（保持单调性）
    └── 处理 NaN 和默认值

feature_selection.py
└── model.prepare_features_for_selection()  # 直接调用，无需维护重复逻辑
```

### 绝对价格特征排除列表（40个）

所有绝对值特征都有标准化替代：

| 类别 | 绝对值特征 | 标准化替代 |
|------|-----------|-----------|
| 价格通道 | Channel_High/Low_20d | Channel_High/Low_Ratio_20d |
| 支撑阻力 | Support/Resistance_120d | Support/Resistance_Ratio_120d |
| 均线 | MA5~MA250 | MA_Ratio 系列 |
| 布林带 | BB_upper/lower/middle | BB_Ratio 系列 |
| ATR | ATR, ATR_MA 等 | ATR_Pct, ATR_Ratio |
| 成交额 | Turnover, Turnover_Mean/Std_20 | Turnover_Z_Score |
| 成交量 | Volume_MA7/120/250, Volume_Mean/Std_30d | Volume_Ratio 系列 |
| OBV | OBV, OBV_MA5 | OBV_Trend, OBV_Change_5d |
| VWAP | VWAP | VWAP_Ratio |
| 技术指标 | MACD, MACD_signal, TP | 比率版本 |

### 特征单调性与智能交叉

交叉特征时必须保持逻辑单调性：

| 交叉类型 | 交叉方式 | 示例 |
|---------|---------|------|
| 正向 × 正向 | 乘法 | 中心性 × 收益率 |
| 负向 × 负向 | 风险放大 | 约束度 × VIX → `-|X| × |Y|` |
| 正向 × 负向 | 风险调整 | 中心性 × VIX → `X / (|Y| + ε)` |
| 涉及中性 | 乘法 | × 日历效应 |

**关键**：市场级特征（HSI_Return、VIX 等）对所有股票同值，必须与网络社区特征交叉才能区分个股。

### 新增特征时

只需修改 `ml_trading_model.py`：

```python
# 1. 在特征计算处添加标准化特征
df['New_Ratio'] = df['New_Value'] / df['Close'].shift(1)

# 2. 如果是绝对值，添加到排除列表
ABSOLUTE_PRICE_FEATURES = [..., 'New_Value']

# 3. 如果是市场级特征，添加到 _build_market_level_features()
# 4. 定义单调性（如需要交叉）
```

**feature_selection.py 自动同步，无需修改。**

---

## 🤖 机器学习模型

### 模型可信度（Walk-forward 验证）

**恒指增强模型**（2026-05-18 验证，33特征）：

| 周期 | 准确率 | 推荐度 |
|------|--------|--------|
| **20天** | **81.22%** | ⭐⭐⭐⭐⭐ 推荐 |
| 5天 | 65.86% | ⭐⭐⭐ 趋势确认 |
| 1天 | 51.49% | ⚠️ 噪音大 |

**个股完整模型**（2026-05-24 验证，12 folds，57只股票，Top 500特征，市场情绪过滤器启用）：

| 指标 | 数值 | 评估 |
|------|------|------|
| 综合评分 | **90/100** | ⭐⭐⭐⭐⭐ 优秀 |
| 平均准确率 | **58.77%** | ✅ 正常范围（<65% 无数据泄漏） |
| 平均夏普比率 | **6.45** | ✅ 优秀 |
| 平均最大回撤 | **-1.03%** | ✅ 优秀 |
| 平均胜率 | 63.30% | 良好 |
| 平均收益率 | +4.88% | ✅ 正收益 |
| 平均 IC | 0.203 | ✅ 有效 |
| 稳定性评级 | 低（需改进） | ⚠️ |

**新增利率特征**（2026-05-23）：
- 多期限美债收益率：US_2Y_Yield, US_10Y_Yield, US_30Y_Yield
- 中国国债收益率：CN_10Y_Yield
- 期限利差：US_2Y_10Y_Spread（收益率曲线斜率）, US_10Y_30Y_Spread
- 中美利差：CN_US_10Y_Spread（资金流向驱动）, CN_US_Spread_Change_5d, CN_US_Spread_Z_Score
- 通过网络交叉特征区分个股（如 `net_centrality_CN_US_10Y_Spread`）

**特征选择**：使用 Top 500 特征，特征减少 55.8%，性能优于全量特征

**Walk-forward 输出文件**（保存到 `output/YYYYMMDD_HHMMSS_catboost_20d/`）：
- `fold_metrics_detail.json` - 每个 Fold 的指标 + **Top 100 特征重要性**
- `prediction_analysis.csv` - 所有预测详情（用于 Fold 盈亏比分析）
- `validation_summary.json` - 总体验证结果

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨 |
| 特征数量 | ~1450 → 500 | 推荐使用 Top 500 特征选择 |
| 随机种子 | 42（固定） | 确保可重现性 |

**20天模型参数**（超参数优化后）：

| 参数 | 值 |
|------|-----|
| n_estimators | 400 |
| depth | 8 |
| learning_rate | 0.06 |
| l2_leaf_reg | 2 |
| subsample | 0.75 |
| colsample_bylevel | 0.8 |

### 新特征上线验证清单

详见 [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)，8个验证步骤：

1. 泄漏检查 - 所有特征使用 `shift(1)`
2. **绝对值特征标准化** - 跨股票训练必须标准化
3. **市场级特征交叉** - 对所有股票同值的特征必须交叉
4. **特征单调性** - 交叉特征保持逻辑单调性
5. Walk-forward 验证 - 准确率达标
6. SHAP 排名 - 进入 top 30
7. Pearson 相关性 - 与现有特征 < 0.8
8. 随机种子稳定性 - 波动 < 2%

### 市场情绪过滤器

**核心原理**：市场上涨比例有强自相关性（lag=1 自相关系数 0.929），滞后1天数据能有效识别极端市场环境。

**阈值分层**：

| 层级 | 上涨比例 | 动态阈值 | 操作 |
|------|---------|---------|------|
| extreme_bear | <20% | 1.0 | 暂停交易 |
| bear | 20-30% | 0.70 | 高置信 |
| weak | 30-40% | 0.65 | 谨慎 |
| normal | >40% | 0.50 | 标准 |

**验证效果**：准确率 62.0% → 70.7%（+8.7%），总收益 +63.44

**代码**：`ml_services/market_regime.py` - MarketSentimentFilter 类

**使用要点**：
- 数据源：使用所有股票收益率计算上涨比例，与 walk-forward 验证一致
- 无前瞻性偏差：严格使用滞后1天数据（`lookback_days=1`）
- 生产集成：`comprehensive_analysis.py` 中已集成

### 市场状态稳定性检测

**核心原理**：HMM 市场状态持续时间（Regime_Duration）反映状态稳定性，短持续时间意味着频繁转换，预测可靠性下降。

**状态稳定性判断**：

| Regime_Duration | 稳定性 | 建议 |
|-----------------|--------|------|
| < 5 天 | ⚠️ 不稳定 | 降低仓位 |
| 5-15 天 | 🟡 中等 | 正常交易 |
| > 15 天 | ✅ 稳定 | 趋势明确 |

**代码**：`data_services/regime_detector.py` - RegimeDetector 类

**生产集成**：
- `comprehensive_analysis.py` 的 `get_current_market_state()` 函数
- 邮件报告中展示市场状态持续时间

---

## ⚙️ 环境配置

### 必填环境变量

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP 服务器地址 |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱应用密码 |
| `RECIPIENT_EMAIL` | 收件人邮箱列表 |
| `QWEN_API_KEY` | 通义千问 API 密钥 |
| `WECHAT_WORK_WEBHOOK` | 企业微信机器人 Webhook（可选） |
| `WXPUSHER_TOKEN` | WxPusher Token（可选） |
| `WXPUSHER_UIDS` | WxPusher 用户 UID（可选） |

### 主要依赖

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm` `hmmlearn` `arch` `networkx`

---

## 🔧 开发规范

### 代码修改原则

1. **修改完即测试**：每次修改后立即执行 `python3 -m py_compile <文件>`
2. **避免硬编码路径**：使用 `os.path.dirname(os.path.abspath(__file__))` 获取脚本目录
3. **HTTP API 超时处理**：调用 API 时必须设置超时时间
4. **语言规范**：对话和注释使用简体中文，变量名/函数名使用英文

### 数据泄漏防护

高风险特征必须使用 `.shift(1)` 避免使用当日数据：
- 所有 `.rolling()` 计算的特征
- `future_return` 必须使用 `.shift(-N)` 计算未来收益

```python
# ❌ 错误：使用当日数据
future_return = returns.rolling(5).sum()

# ✅ 正确：使用未来数据
future_return = returns.rolling(5).sum().shift(-5)
```

### CatBoost 分类特征处理

训练和预测时必须一致处理分类特征 NaN：
```python
# 训练时
df[col] = df[col].fillna('unknown').astype(str)
encoder = LabelEncoder()
df[col] = encoder.fit_transform(df[col])

# 预测时
test_df[col] = test_df[col].fillna('unknown').astype(str)
test_df[col] = test_df[col].apply(
    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
)
```

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`

---

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

**模型更新后**：运行 Walk-forward 验证确认性能，使用 `/model_validation` 技能执行标准验证流程

**特征修改后**：清除缓存 `rm -rf data/feature_cache/*.pkl`

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **特征工程**：[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) - 完整指南（含案例分析）
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **验证方法**：[docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md)
- **A股设计**：[docs/A_STOCK_DESIGN.md](docs/A_STOCK_DESIGN.md) - A股系统完整设计文档
