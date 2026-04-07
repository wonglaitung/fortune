# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法、异常检测等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：所有关键警告和最佳实践请参阅 [lessons.md](lessons.md)

---

## ⚡ 快速参考

### 常用命令速查

| 任务 | 命令 |
|------|------|
| **训练模型** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost` |
| **生成预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` |
| **批量回测** | `python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep --date 2026-04-03` (每日) 或 `--time-interval hour` (每小时) |
| **加密货币异常检测** | `python3 crypto_email.py --mode deep --date 2026-04-03` |
| **Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **板块Walk-forward验证** | `python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20` |
| **恒生指数监控** | `python3 hsi_email.py` |
| **主力资金追踪** | `python3 hk_smart_money_tracker.py` |
| **预测性能监控** | `python3 ml_services/performance_monitor.py --mode all --horizon 20` |
| **信号与异常关联性验证** | `python3 ml_services/validate_signal_anomaly_correlation.py --symbol ETH-USD --mode all` |
| **港股异常因果分析** | `python3 analyze_hk_stock_anomaly_causality.py --start-date 2024-04-01 --end-date 2026-04-01` |
| **趋势延续性验证** | `python3 analyze_trend_continuation.py --start-date 2024-04-01 --end-date 2026-04-01` |
| **异常后表现分析** | `python3 analyze_anomaly_post_performance.py` |

### 主要入口脚本

| 脚本 | 用途 |
|------|------|
| `comprehensive_analysis.py` | 综合分析（每日交易信号生成） |
| `ml_services/ml_trading_model.py` | ML模型训练/预测 |
| `hsi_email.py` | 恒生指数监控 |
| `hk_smart_money_tracker.py` | 主力资金追踪 |
| `ml_services/batch_backtest.py` | 批量回测 |
| `ml_services/performance_monitor.py` | 预测性能监控 |
| `detect_stock_anomalies.py` | 港股异常检测（支持每日/每小时，交易时段检测） |
| `crypto_email.py` | 加密货币异常检测 |
| `ml_services/walk_forward_validation.py` | Walk-forward验证 |
| `ml_services/validate_signal_anomaly_correlation.py` | 交易信号与异常关联性验证 |
| `ml_services/walk_forward_by_sector.py` | 板块Walk-forward验证 |
| `ml_services/walk_forward_anomaly_strategy_validation.py` | 异常策略Walk-forward验证 |
| `analyze_hk_stock_anomaly_causality.py` | 港股异常因果关系分析（两年数据） |
| `analyze_trend_continuation.py` | 趋势延续性验证工具 |
| `analyze_anomaly_post_performance.py` | 异常后股价表现分析 |

### 核心警告 ⚠️

| 警告类型 | 关键信息 |
|---------|---------|
| **数据泄漏** | 准确率>65%通常是数据泄漏信号 |
| **CatBoost 1天模型** | 严重过拟合，不推荐使用 |
| **融合模型** | 信号稀释，表现远不如CatBoost单模型 |
| **深度学习** | LSTM/Transformer表现远不如CatBoost，不推荐 |
| **Walk-forward验证** | 唯一可信的模型验证方法 |
| **交易时段误报** | 交易时段数据不完整，应在收盘后检测 |
| **异常是"放大器"** | 异常后波动率增加+79%，胜率接近50%，需降低仓位 |
| **趋势延续假设错误** | "升的继续升，跌的继续跌"假设**错误**，实际是均值回归信号（两年数据验证） |
| **最强抄底信号** | 价格异常+当日下跌，5天胜率71.7%，10天胜率72.8% |

> **详细说明**：参阅 [lessons.md](lessons.md)

---

## 📂 项目架构

```
金融信息监控与智能交易系统
├── 数据获取层
│   ├── 加密货币异常检测 (crypto_email.py)
│   ├── 港股异常检测 (detect_stock_anomalies.py)
│   ├── 恒生指数预测 (hsi_prediction.py)
│   └── 黄金监控 (gold_analyzer.py)
├── 异常检测模块 (anomaly_detector/)
│   ├── zscore_detector.py           # Z-Score检测器（支持分钟/小时/天/周）
│   ├── isolation_forest_detector.py # Isolation Forest检测器（支持通用时间区间）
│   ├── feature_extractor.py         # 特征提取器
│   ├── anomaly_integrator.py        # 异常整合器
│   └── cache.py                    # 异常缓存
├── 数据服务层 (data_services/)
│   ├── 技术分析工具 (technical_analysis.py)
│   ├── 基本面数据 (fundamental_data.py)
│   └── 板块分析 (hk_sector_analysis.py)
├── 分析层
│   ├── 综合分析 (comprehensive_analysis.py)
│   ├── 恒生指数监控 (hsi_email.py)
│   ├── 主力资金追踪 (hk_smart_money_tracker.py)
│   └── ML 模块 (ml_services/)
│       ├── 机器学习模型 (ml_trading_model.py)
│       ├── 批量回测 (batch_backtest.py)
│       ├── Walk-forward验证 (walk_forward_validation.py)
│       ├── 板块Walk-forward验证 (walk_forward_by_sector.py)
│       ├── 异常策略验证 (walk_forward_anomaly_strategy_validation.py)
│       ├── 信号异常关联性验证 (validate_signal_anomaly_correlation.py)
│       └── 预测性能监控 (performance_monitor.py)
├── 交易层
│   └── 模拟交易 (simulation_trader.py)
├── 服务层 (llm_services/)
│   ├── 大模型接口 (qwen_engine.py)
│   └── 情感分析 (sentiment_analyzer.py)
├── 工具脚本 (scripts/)
├── 测试脚本 (tests/)
└── 文档 (docs/)
    ├── FEATURE_ENGINEERING.md        # 特征工程完整指南
    ├── VALIDATION_GUIDE.md           # 验证方法完整指南
    ├── ANOMALY_DETECTION_GUIDE.md    # 异常检测完整指南
    ├── BACKTEST_GUIDE.md             # 回测系统使用指南
    ├── CATBOOST_USAGE.md             # CatBoost 使用指南
    └── WALK_FORWARD_GUIDE.md         # Walk-forward验证指南
```

---

## 🎯 核心功能模块

### 港股异常检测
- **双异常检测**：价格 + 成交量异常（基于Z-Score）
- **深度分析模式**：Z-Score + Isolation Forest（多维特征检测）
- **时间间隔支持**：支持每日（`--time-interval day`，默认）和每小时（`--time-interval hour`）两种检测模式
- **日期参数支持**：支持检测指定日期的异常（`--date` 参数）
- **异常原因分析**：分析价格、成交量、多维特征异常的具体原因
- **异常日期数据**：使用异常发生日期的技术指标（RSI、布林带、MACD、涨跌幅）
- **邮件表格增强**：显示异常日期、异常原因、技术指标等详细信息
- **交易时段误报防护**：每天凌晨2点检测，避免交易时段数据不完整导致误报
- **ML特征集成**：26个异常检测特征已集成到CatBoost模型（使用shift(1)防止数据泄漏）
- **详细指南**：[docs/ANOMALY_DETECTION_GUIDE.md](docs/ANOMALY_DETECTION_GUIDE.md)

### 加密货币异常检测
- **双层检测**：Z-Score（每小时）+ Isolation Forest（凌晨2点）
- **日期参数支持**：支持检测指定日期的异常（`--date` 参数）
- **异常原因分析**：分析价格、成交量、多维特征异常的具体原因
- **异常日期数据**：使用异常发生日期的技术指标（RSI、布林带、MACD、涨跌幅）
- **邮件表格增强**：显示异常日期、异常原因、技术指标等详细信息
- **只展示指定日期异常**：从检测7天内异常改为只展示当天或指定日期异常
- **小时级监控优化**：使用1个月小时级数据（720个数据点），精度提升24倍

### 交易信号与异常关联性验证
- **验证脚本**：`ml_services/validate_signal_anomaly_correlation.py`
- **验证方法**：
  - 相关性分析（Pearson + Spearman）
  - Granger因果检验（异常是否预示信号变化）
  - 时间序列交叉相关分析（异常前置/滞后效应）
  - 事件研究法（异常前后信号表现）
  - 回测对比（有/无异常信号收益率对比）
- **关键发现**：
  - 异常与交易信号显著相关（相关系数0.1843）
  - 异常是信号变化的Granger原因（4个滞后期显著）
  - 异常后信号数量增加+65%
  - 异常期间收益率提升+245%，但波动率增加+79%
- **动态阈值策略**：
  - 高异常：阈值 +0.08，仓位 40%
  - 中异常：阈值 +0.04，仓位 60%
  - 低异常：阈值 +0.02，仓位 80%

### 港股异常因果关系分析（两年数据验证）
- **分析脚本**：`analyze_hk_stock_anomaly_causality.py`
- **分析周期**：2024-04-01 至 2026-04-01（938个异常）
- **验证方法**：
  - 趋势延续性分析（验证"升的继续升"假设）
  - Granger因果检验
  - 交叉相关分析
  - 事件研究法
- **关键发现**：
  - **趋势延续假设错误**：延续率49-51%（接近随机），相关系数-0.10至-0.14（均值回归）
  - **最强抄底信号**：价格异常+当日下跌，5天胜率71.7%，10天胜率72.8%
  - **波动率下降**：异常后波动率下降28.5%（市场趋于平静）
- **策略建议**：
  - 价格异常+当日下跌 → 考虑抄底（胜率71.7%）
  - 价格异常+当日上涨 → 观望（胜率53.7%）
  - 成交量异常 → 谨慎使用（预测能力较弱）

### 异常检测特征作为ML特征（2026-04-08）
- **实现脚本**：`ml_services/ml_trading_model.py`（`create_smart_money_features`方法）
- **特征数量**：26个异常检测特征
- **数据泄漏防护**：所有特征使用`.shift(1)`，只使用历史数据
- **主要特征**：
  - 价格异常：`Price_Anomaly_ZScore`、`Price_Anomaly_Flag`、`Anomaly_Buy_Signal`
  - 成交量异常：`Volume_Anomaly_ZScore`、`Volume_Anomaly_Flag`
  - 波动率异常：`Volatility_Anomaly_ZScore`、`Volatility_Anomaly_Flag`
  - 抄底信号：`Anomaly_Buy_Signal`（价格异常+下跌，胜率71.7%）
- **验证结果**：
  - 准确率：61.44%（±1.75%），无数据泄漏
  - `Volume_Anomaly_ZScore`重要性：0.165
  - `Volatility_Anomaly_Flag`重要性：0.062
  - 银行股板块胜率提升：48.66% → 50.26%（+1.6%）
- **详细说明**：参阅 [lessons.md](lessons.md) "异常检测特征作为ML特征经验"

### Walk-forward 异常策略验证
- **验证脚本**：`ml_services/walk_forward_anomaly_strategy_validation.py`
- **验证方法**：业界标准 Walk-forward 验证（每个fold重新训练）
- **验证结果**：
  - 无异常信号：平均收益率 -0.31%，胜率 47.94%，标准差 3.77%
  - 有异常信号：平均收益率 +0.45%，胜率 52.38%，标准差 6.76%（+79%）
- **关键发现**：
  - 异常期间波动率显著增加（+79%），需降低仓位
  - 当前策略（40%仓位）过于保守，应提高至55-60%
  - 阈值调整+0.08过于严格，应降至+0.02-0.03
  - 必须引入动态止损机制（基于ATR）

### 港股主力资金追踪
- 建仓/出货信号分析
- 筹码分布分析集成
- 1-6层分析框架

### 恒生指数及自选股分析
- 实时技术指标监控
- 短期/中期大模型分析
- 板块分析和龙头股识别

### 综合分析系统
- 整合大模型建议和CatBoost预测
- 每日自动执行，生成买卖建议
- 集成异常检测功能（可选用深度分析模式）
- 支持深度分析模式（`--deep-analysis` 参数）

---

## 🤖 机器学习模型

### 模型可信度

| 模型 | 可信度 | 说明 |
|------|--------|------|
| **CatBoost 20天** | 高可信度 | **推荐使用** |
| CatBoost 5天 | 中等可信度 |
| CatBoost 1天 | 低可信度，**不推荐** |
| LSTM/Transformer | 低可信度，**不推荐** |
| 融合模型 | 低可信度，**不推荐** |

### CatBoost 配置（推荐）
- **准确率**：61.44%（±1.75%）（含异常检测特征）
- **置信度阈值**：0.65
- **特征数量**：918个全量特征（含26个异常检测特征）
- **随机种子**：42（固定）
- **事件驱动特征**：9个（分红3个、财报日期3个、财报超预期3个）
- **异常检测特征**：26个（价格异常、成交量异常、波动率异常等）

### Walk-forward 验证结果（2026-04-08）

#### 全局验证（12 Fold，28只股票）
| 指标 | 数值 | 说明 |
|------|------|------|
| **夏普比率** | 0.1455 | ✅ 正值，风险调整后收益为正 |
| **平均收益率** | 3.94% | 20天持有期 |
| **平均准确率** | 59.43% | 无数据泄漏信号 |
| **稳定性评级** | 高（优秀） | 收益率标准差 1.87% |

#### 季度表现规律
| 季度 | 平均夏普 | 平均胜率 | 评价 |
|------|---------|---------|------|
| **Q3（7-9月）** | 0.1856 | 31.49% | **最佳** |
| Q1（1-3月） | 0.1624 | 22.98% | 准确率高 |
| Q2（4-6月） | 0.1110 | 28.93% | 中等 |
| Q4（10-12月） | 0.1230 | 19.45% | 偏弱 |

### 板块模型（Walk-forward验证，阈值0.55）

| 板块 | 夏普比率 | 胜率 | 推荐度 |
|------|---------|------|--------|
| 消费股 | 0.7445 | 54.80% | 强烈推荐 |
| **银行股** | **-0.0296** | **50.26%** | 推荐（含异常特征） |
| 半导体股 | 0.1260 | 49.87% | 推荐 |
| 保险股 | -0.3160 | 49.02% | 谨慎使用 |
| 房地产股 | -0.1352 | 47.75% | 谨慎使用 |

> **详细说明**：参阅 [lessons.md](lessons.md)

---

## ⚙️ 配置与运行

### 环境变量（必填）

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP服务器地址（如 smtp.163.com） |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱应用密码 |
| `RECIPIENT_EMAIL` | 收件人邮箱列表（逗号分隔） |
| `QWEN_API_KEY` | 通义千问 API 密钥 |

### 文件路径约定

- **使用相对路径**：基于脚本目录构建
- **禁止硬编码绝对路径**：如 `/data/fortune/...`
- **输出目录**：`output/`
- **数据目录**：`data/`

---

## 📊 数据文件结构

```
data/
├── model_accuracy.json              # 模型准确率信息
├── prediction_history.json          # 预测历史记录
├── anomaly_cache.json              # 异常缓存
├── llm_recommendations_*.txt       # 大模型建议
└── simulation_transactions.csv      # 交易历史记录

output/
├── batch_backtest_*.json           # 批量回测数据
├── walk_forward_catboost_20d_*.json # Walk-forward全局验证数据
├── walk_forward_fold_analysis_*.md # Fold详细分析报告
├── walk_forward_sector_*.md        # 板块Walk-forward验证报告
├── walk_forward_anomaly_*.md       # 异常策略Walk-forward验证报告
├── signal_anomaly_correlation_*.md # 交易信号与异常关联性验证报告
└── performance_report_*.md         # 性能月度报告
```

---

## 🚀 自动化调度

### GitHub Actions 工作流

| 文件 | 功能 | 执行时间 |
|------|------|----------|
| `stock-anomaly-detection.yml` | 港股异常检测（每日） | 每天凌晨2点（香港时间） |
| `hourly-stock-monitor.yml` | 港股异常检测（每小时） | 交易日每小时（09:30-12:00, 13:00-16:00） |
| `hourly-crypto-monitor.yml` | 加密货币异常检测 | 每小时（深度模式） |
| `hourly-gold-monitor.yml` | 黄金监控 | 每小时 |
| `comprehensive-analysis.yml` | 综合分析邮件 | 周一到周五 UTC 08:00（香港时间16:00） |
| `hsi-prediction.yml` | 恒生指数涨跌预测 | 周一到周五 UTC 22:00（香港时间早上6:00） |
| `bull-bear-analysis.yml` | 牛熊市分析 | 每周日 UTC 17:00（香港时间凌晨1:00） |
| `sector-analysis.yml` | 板块表现分析 | 每月1号 UTC 19:00（香港时间凌晨3:00） |
| `ranking-analysis.yml` | 股票表现排名分析 | 每月1号 UTC 19:00（香港时间凌晨3:00） |
| `performance-monitor.yml` | 预测性能月度报告 | 每月1号 UTC 20:00（香港时间凌晨4:00） |
| `batch-stock-news-fetcher.yml` | 批量股票新闻获取 | 每天 UTC 22:00 |
| `daily-ipo-monitor.yml` | IPO 信息监控 | 每天 UTC 02:00 |
| `daily-ai-trading-analysis.yml` | AI 交易分析日报 | 周一到周五 UTC 08:30 |
| `weekly-comprehensive-analysis.yml` | 周综合交易分析 | 每周日 UTC 01:00 |

**注意**：工作流调度时间基于 UTC 时间，已转换为香港时间便于理解。

---

## 🔗 快速链接

### 开发参考
- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践（含交易时段误报问题）
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展（含异常检测改进详情）
- **编码规范**：[.iflow/commands/programmer_skill.md](.iflow/commands/programmer_skill.md) - 开发规范

### 详细文档

**特征工程与验证**：
- [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) - 特征工程完整指南（892个特征详情）
- [docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md) - 验证方法完整指南（Walk-forward、回测等）
- [docs/ANOMALY_DETECTION_GUIDE.md](docs/ANOMALY_DETECTION_GUIDE.md) - 异常检测完整指南
- [docs/WALK_FORWARD_GUIDE.md](docs/WALK_FORWARD_GUIDE.md) - Walk-forward验证指南

**模型与回测**：
- [docs/BACKTEST_GUIDE.md](docs/BACKTEST_GUIDE.md) - 回测系统使用指南
- [docs/CATBOOST_USAGE.md](docs/CATBOOST_USAGE.md) - CatBoost 使用指南

**分析与策略**：
- [lessons.md](lessons.md) - 经验教训和最佳实践（含交易时段误报问题）
- [progress.txt](progress.txt) - 项目进度跟踪（含异常检测改进详情）

---

## 📝 Session Workflow

**会话开始时必须执行**：
1. 读取 `progress.txt` 文件，了解项目当前进展
2. 审查 `lessons.md` 文件，检查是否有错误需要纠正

**功能更新后**：
1. 更新 `progress.txt`，记录新的进展
2. 如有新的学习心得或经验教训，更新 `lessons.md`

---

**最后更新**：2026-04-08（Walk-forward验证完成，异常检测特征集成）
