# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法、异常检测等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：所有关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：规范化开发流程、系统设计决策、测试验证要求请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)

---

## ⚡ 快速参考

### 常用命令速查

| 任务 | 命令 |
|------|------|
| **训练模型** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost` |
| **生成预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` |
| **批量回测** | `python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6` |
| **港股异常检测（每日）** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep --date 2026-04-08` |
| **港股异常检测（每小时）** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep --time-interval hour` |
| **加密货币异常检测** | `python3 crypto_email.py --mode deep --date 2026-04-08` |
| **Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **恒生指数监控** | `python3 hsi_email.py` |
| **主力资金追踪** | `python3 hk_smart_money_tracker.py` |
| **预测性能监控** | `python3 ml_services/performance_monitor.py --mode all --horizon 20` |

### 主要入口脚本

| 脚本 | 用途 |
|------|------|
| `comprehensive_analysis.py` | 综合分析（每日交易信号生成） |
| `detect_stock_anomalies.py` | 港股异常检测（统一入口） |
| `crypto_email.py` | 加密货币异常检测 |
| `ml_services/ml_trading_model.py` | ML模型训练/预测 |
| `hsi_email.py` | 恒生指数监控 |
| `hk_smart_money_tracker.py` | 主力资金追踪 |

### 核心警告 ⚠️

| 警告类型 | 关键信息 |
|---------|---------|
| **数据泄漏** | 准确率>65%通常是数据泄漏信号 |
| **CatBoost 1天模型** | 严重过拟合，不推荐使用 |
| **融合模型** | 信号稀释，表现远不如CatBoost单模型 |
| **深度学习** | LSTM/Transformer表现远不如CatBoost，不推荐 |
| **Walk-forward验证** | 唯一可信的模型验证方法 |
| **交易时段误报** | 交易时段数据不完整，应在收盘后检测 |
| **异常是"放大器"** | 异常后波动率增加+79%，需降低仓位 |
| **最强抄底信号** | 价格异常+当日下跌，5天胜率72% |

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
│   ├── zscore_detector.py           # Z-Score检测器
│   ├── isolation_forest_detector.py # Isolation Forest检测器
│   ├── feature_extractor.py         # 特征提取器
│   ├── anomaly_integrator.py        # 异常整合器
│   └── cache.py                     # 异常缓存
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
```

---

## 🎯 核心功能模块

### 港股异常检测
- **统一入口函数**：`run_stock_anomaly_detection()` 封装完整检测逻辑
- **双层检测**：Z-Score + Isolation Forest（多维特征检测）
- **市场整体异常检测**：恒指涨跌 ≥ 2% 时显示提示，区分市场驱动与个股异常
- **时间间隔支持**：每日（`--time-interval day`）和每小时（`--time-interval hour`）
- **日期参数支持**：支持检测指定日期的异常（`--date` 参数）
- **异常严重级别**：high（🔴）、medium（⚠️）、low（ℹ️）
- **ML特征集成**：26个异常检测特征已集成到CatBoost模型

### 异常策略（两年数据验证）

| 异常类型 | 含义 | 5日收益 | 胜率 | 策略 |
|---------|------|---------|------|------|
| **IF high** | 多维异常（系统风险） | -3.04% | 43% | 🔴 减仓 |
| **价格异常+当日下跌** | 超跌反弹 | +4.12% | 72% | 🟢 抄底 |
| 价格异常+当日上涨 | 追涨风险 | +1.96% | 54% | ⚠️ 观望 |

**核心逻辑**：IF high 预警风险，Z-Score 价格异常+当日下跌 = 抄底机会

### 加密货币异常检测
- **双层检测**：Z-Score + Isolation Forest
- **小时级监控**：使用1个月小时级数据（720个数据点）

---

## 🤖 机器学习模型

### 模型可信度

| 模型 | 可信度 | 说明 |
|------|--------|------|
| **CatBoost 20天** | 高可信度 | **推荐使用** |
| CatBoost 5天 | 中等可信度 |
| CatBoost 1天 | 低可信度，**不推荐** |
| LSTM/Transformer | 低可信度，**不推荐** |

### CatBoost 配置（推荐）
- **准确率**：61.44%（±1.75%）
- **置信度阈值**：0.65
- **特征数量**：918个全量特征
- **随机种子**：42（固定）

### Walk-forward 验证结果

| 指标 | 数值 |
|------|------|
| **夏普比率** | 0.1455 |
| **平均收益率** | 3.94% |
| **平均准确率** | 59.43% |

### 板块模型（Walk-forward验证）

| 板块 | 夏普比率 | 胜率 | 推荐度 |
|------|---------|------|--------|
| 消费股 | 0.7445 | 54.80% | 强烈推荐 |
| 银行股 | 0.1546 | 50.44% | 推荐 |
| 半导体股 | 0.1260 | 49.87% | 推荐 |
| 保险股 | -0.3160 | 49.02% | 谨慎使用 |
| 房地产股 | -0.1352 | 47.75% | 谨慎使用 |

---

## ⚙️ 配置与运行

### 环境变量（必填）

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP服务器地址 |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱应用密码 |
| `RECIPIENT_EMAIL` | 收件人邮箱列表 |
| `QWEN_API_KEY` | 通义千问 API 密钥 |

---

## 🚀 自动化调度

### GitHub Actions 工作流

| 文件 | 功能 | 执行时间 |
|------|------|----------|
| `stock-anomaly-detection.yml` | 港股异常检测（每日） | 每天凌晨2点 |
| `hourly-stock-monitor.yml` | 港股异常检测（交易时段） | 10:00-15:00 每小时 |
| `hourly-crypto-monitor.yml` | 加密货币异常检测 | 每小时 |
| `comprehensive-analysis.yml` | 综合分析邮件 | 周一到周五 16:00 |
| `hsi-prediction.yml` | 恒生指数预测 | 周一到周五 06:00 |
| `performance-monitor.yml` | 性能月度报告 | 每月1号 |

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md)
- **进度跟踪**：[progress.txt](progress.txt)
- **详细文档**：[docs/](docs/)

---

## 📝 Session Workflow

**会话开始时必须执行**：
1. 读取 `progress.txt` 文件，了解项目当前进展
2. 审查 `lessons.md` 文件，检查是否有错误需要纠正

**功能更新后**：
1. 更新 `progress.txt`，记录新的进展
2. 如有新的学习心得，更新 `lessons.md`

---

**最后更新**：2026-04-09（重构统一入口函数、精简异常策略说明）
