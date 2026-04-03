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
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **加密货币异常检测** | `python3 crypto_email.py --mode quick` (快速) / `--mode deep` (深度) |
| **Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **板块Walk-forward验证** | `python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20` |
| **恒生指数监控** | `python3 hsi_email.py` |
| **主力资金追踪** | `python3 hk_smart_money_tracker.py` |
| **预测性能监控** | `python3 ml_services/performance_monitor.py --mode all --horizon 20` |

### 主要入口脚本

| 脚本 | 用途 |
|------|------|
| `comprehensive_analysis.py` | 综合分析（每日交易信号生成） |
| `ml_services/ml_trading_model.py` | ML模型训练/预测 |
| `hsi_email.py` | 恒生指数监控 |
| `hk_smart_money_tracker.py` | 主力资金追踪 |
| `ml_services/batch_backtest.py` | 批量回测 |
| `ml_services/performance_monitor.py` | 预测性能监控 |
| `detect_stock_anomalies.py` | 港股异常检测 |
| `crypto_email.py` | 加密货币异常检测 |
| `ml_services/walk_forward_validation.py` | Walk-forward验证 |
| `ml_services/validate_signal_anomaly_correlation.py` | 交易信号与异常关联性验证 |
| `ml_services/walk_forward_by_sector.py` | 板块Walk-forward验证 |
| `ml_services/walk_forward_anomaly_strategy_validation.py` | 异常策略Walk-forward验证 |

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
│   ├── zscore_detector.py
│   ├── isolation_forest_detector.py
│   ├── feature_extractor.py
│   ├── anomaly_integrator.py
│   └── cache.py
├── 数据服务层 (data_services/)
│   ├── 技术分析工具 (technical_analysis.py)
│   ├── 基本面数据 (fundamental_data.py)
│   └── 板块分析 (hk_sector_analysis.py)
├── 分析层
│   ├── 综合分析 (comprehensive_analysis.py)
│   ├── 恒生指数监控 (hsi_email.py)
│   ├── 主力资金追踪 (hk_smart_money_tracker.py)
│   └── ML 模块 (ml_services/)
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
- **日期参数支持**：支持检测指定日期的异常（`--date` 参数）
- **异常原因分析**：分析价格、成交量、多维特征异常的具体原因
- **异常日期数据**：使用异常发生日期的技术指标（RSI、布林带、MACD、涨跌幅）
- **邮件表格增强**：显示异常日期、异常原因、技术指标等详细信息
- **详细指南**：[docs/ANOMALY_DETECTION_GUIDE.md](docs/ANOMALY_DETECTION_GUIDE.md)

### 加密货币异常检测
- **双层检测**：Z-Score（每小时）+ Isolation Forest（凌晨2点）
- **日期参数支持**：支持检测指定日期的异常（`--date` 参数）
- **异常原因分析**：分析价格、成交量、多维特征异常的具体原因
- **异常日期数据**：使用异常发生日期的技术指标（RSI、布林带、MACD、涨跌幅）
- **邮件表格增强**：显示异常日期、异常原因、技术指标等详细信息
- **只展示指定日期异常**：从检测7天内异常改为只展示当天或指定日期异常

### 港股主力资金追踪
- 建仓/出货信号分析
- 筹码分布分析集成

### 恒生指数及自选股分析
- 实时技术指标监控
- 短期/中期大模型分析
- 板块分析和龙头股识别

### 综合分析系统
- 整合大模型建议和CatBoost预测
- 每日自动执行，生成买卖建议
- 集成异常检测功能（可选用深度分析模式）

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
- **准确率**：60.99%（±2.00%）
- **置信度阈值**：0.60
- **特征数量**：892个全量特征
- **随机种子**：42（固定）
- **事件驱动特征**：9个（分红3个、财报日期3个、财报超预期3个）

### 板块模型（Walk-forward验证，阈值0.6）

| 板块 | 夏普比率 | 胜率 | 推荐度 |
|------|---------|------|--------|
| 消费股 | 0.7445 | 54.80% | 强烈推荐 |
| 银行股 | 0.1546 | 50.44% | 推荐 |
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
├── model_accuracy.json          # 模型准确率信息
├── prediction_history.json        # 预测历史记录
├── anomaly_cache.json            # 异常缓存
├── llm_recommendations_*.txt    # 大模型建议
└── simulation_transactions.csv    # 交易历史记录

output/
├── batch_backtest_*.json         # 批量回测数据
├── walk_forward_sector_*.md        # Walk-forward验证报告
├── walk_forward_anomaly_*.md      # 异常策略Walk-forward验证报告
├── signal_anomaly_correlation_*.md # 交易信号与异常关联性验证报告
└── performance_report_*.md       # 性能月度报告
```

---

## 🚀 自动化调度

### GitHub Actions 工作流

| 文件 | 功能 | 执行时间 |
|------|------|----------|
| `stock-anomaly-detection.yml` | 港股异常检测 | 每天凌晨2点（香港时间） |
| `hourly-crypto-monitor.yml` | 加密货币异常检测 | 每小时（快速）+ 凌晨2点（深度） |
| `comprehensive-analysis.yml` | 综合分析邮件 | 周一到周五 UTC 8:00（香港时间16:00） |
| `hsi-prediction.yml` | 恒生指数涨跌预测 | 周一到周五 UTC 22:00（香港时间早上6:00） |
| `bull-bear-analysis.yml` | 牛熊市分析 | 每周一 UTC 17:00（香港时间凌晨1:00） |
| `sector-analysis.yml` | 板块表现分析 | 每月1号 UTC 19:00（香港时间凌晨3:00） |
| `ranking-analysis.yml` | 股票表现排名分析 | 每月1号 UTC 19:00（香港时间凌晨3:00） |
| `performance-monitor.yml` | 预测性能月度报告 | 每月1号 UTC 20:00（香港时间凌晨4:00） |

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
- [lessons.md](lessons.md) - 经验教训和最佳实践（含交易时段误报问题）⭐ 新增
- [progress.txt](progress.txt) - 项目进度跟踪（含异常检测改进详情）⭐ 新增

---

## 📝 Session Workflow

**会话开始时必须执行**：
1. 读取 `progress.txt` 文件，了解项目当前进展
2. 审查 `lessons.md` 文件，检查是否有错误需要纠正

**功能更新后**：
1. 更新 `progress.txt`，记录新的进展
2. 如有新的学习心得或经验教训，更新 `lessons.md`

---

**最后更新**：2026-04-04