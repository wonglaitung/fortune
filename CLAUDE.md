# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法、异常检测等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：所有关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：规范化开发流程、系统设计决策、测试验证要求请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)

---

## ⚡ 快速参考

### 测试命令

| 测试类型 | 命令 |
|---------|------|
| **语法检查** | `python3 -m py_compile <文件路径>` |
| **运行所有测试** | `python3 -m pytest tests/ -v` |
| **运行单个测试** | `python3 -m pytest tests/test_zscore_detector.py -v` |
| **测试异常检测模块** | `python3 -m pytest tests/test_*.py -v` |

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
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **恒生指数预测验证** | `python3 hsi_prediction.py --verify` |
| **恒指模型滚动训练** | `python3 ml_services/hsi_ml_model.py --mode rolling --window 18` |
| **恒指Walk-forward验证** | `python3 ml_services/hsi_walk_forward.py --horizon 20` |
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
| `hsi_prediction.py` | 恒生指数预测（双模型对比） |
| `hk_smart_money_tracker.py` | 主力资金追踪 |
| `ml_services/hsi_ml_model.py` | 恒指CatBoost模型训练/预测 |
| `ml_services/hsi_walk_forward.py` | 恒指Walk-forward验证 |


### 语言规范
- 所有的对话沟通、代码解释和文档注释必须使用 **简体中文**。
- 如果输出包含技术术语，建议在中文后用括号标注英文（例如：异步处理 (Asynchronous)）。

### 代码风格
- 遵循 PEP8 规范。
- 变量名和函数名必须使用英文，但注释必须是中文。


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
| **三周期一致策略** | 1天/5天/20天预测一致时，至少一周期正确率89% |

> **详细说明**：参阅 [lessons.md](lessons.md)

### 指标计算关键修正（固定持有期策略）

⚠️ 对于固定持有期（如20天）策略，传统指标计算方法会出错：

| 指标 | 错误计算 | 正确计算 |
|------|---------|---------|
| **年化收益** | `avg_return * (252/horizon)` | `avg_return * 12` |
| **年化标准差** | `return_std * sqrt(252/horizon)` | `batch_std * sqrt(12)` |
| **最大回撤** | 时间序列回撤 | 批次回撤 |

**原因**：固定持有期策略的多笔交易是**同时持有**的，不能假设一年可以做 `252/horizon` 次独立交易。

### 数据合理性问题

⚠️ **平均月收益 9.42% 偏高**，数据合理性待验证：

1. **avg_return 定义**：未来第20天的单日收益平均值（不是累积收益）
2. **年化收益 94%** 远超业界标准 15-30%
3. **可能原因**：
   - 高置信度阈值(0.65)筛选后的信号效果较好
   - 实际收益需要考虑交易成本、滑点等
   - 建议实盘时降低预期收益

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
│   ├── 板块分析 (hk_sector_analysis.py)
│   └── 南向资金数据 (southbound_data.py)
├── 分析层
│   ├── 综合分析 (comprehensive_analysis.py)
│   ├── 恒生指数监控 (hsi_email.py)
│   ├── 恒指预测双模型 (hsi_prediction.py + hsi_ml_model.py)
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

### 恒生指数预测系统（2026-04-16 新增）

**双模型架构**：
- **评分模型**：基于特征重要性加权的规则模型
- **CatBoost模型**：机器学习自动学习权重

**核心功能**：
- 多周期预测（1天/5天/20天）
- 三周期一致预测策略
- 预测历史记录与验证

**三周期一致预测策略** ⭐ 重要发现：

| 场景 | 占比 | 至少一周期正确率 | 建议操作 |
|------|------|-----------------|---------|
| **三周期一致看涨** | 17.4% | **92.00%** | 强烈买入 |
| **三周期一致看跌** | 24.0% | **87.28%** | 强烈卖出 |
| 不一致 | 58.6% | - | 观望 |

**各周期特点**：

| 周期 | 准确率 | AUC | 适用场景 |
|------|--------|-----|---------|
| 1天 | ~46% | 0.52 | ❌ 噪音大，不推荐 |
| 5天 | **~58%** | 0.66 | ✅ 准确率最高 |
| 20天 | ~55% | **0.75** | ✅ 趋势判断能力强 |

**最优持有期**：
- 一致看涨时：持有20天（准确率71.2%）
- 一致看跌时：持有5天（准确率67.1%）

**特征配置**（73个特征）：
- 宏观因子：美债收益率、VIX恐慌指数
- 港股通资金：南向资金净流入/净买入
- 技术指标：MA、RSI、MACD、布林带、ATR、ADX等
- RS信号：相对强度信号

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

### Walk-forward 验证结果（2026-04-13 修正后）

| 指标 | 数值 | 业界标准 | 评估 |
|------|------|---------|------|
| **夏普比率** | 0.97 | >1.0 | 接近标准 |
| **最大回撤** | -0.55% | <-20% | 优秀 |
| **平均收益率** | 9.42% | - | 良好 |
| **胜率** | 49.36% | 52%+ | 略低 |
| **索提诺比率** | 2.52 | >1.0 | 优秀 |

**实用性评估**：80/100，⭐⭐⭐⭐⭐ 强烈推荐实盘

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
| `hsi-prediction.yml` | 恒生指数预测（双模型） | 周一到周五 06:00 |
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

## 🔧 开发规范要点

### 代码修改原则（来自 programmer_skill.md）

1. **修改完即测试**：每次代码修改后立即验证
   - 语法检查：`python3 -m py_compile <文件路径>`
   - 功能测试：验证修改是否符合预期
   - 回归测试：确保没有破坏现有功能

2. **避免硬编码路径**：使用相对路径
   ```python
   # ✅ 正确
   script_dir = os.path.dirname(os.path.abspath(__file__))
   data_dir = os.path.join(script_dir, 'data')
   ```

3. **HTTP API超时处理**：调用API时必须设置超时时间

4. **公共代码提取**：识别可复用逻辑，创建通用函数

### 数据泄漏防护

高风险特征必须使用 `.shift(1)` 避免使用当日数据：
- 所有 `.rolling()` 计算的特征
- BB_Position、Price_Percentile、Intraday_Amplitude
- Price_Ratio_MA5/20/50、Support_120d、Resistance_120d

### 测试文件结构

```
tests/
├── test_zscore_detector.py      # Z-Score检测器测试
├── test_isolation_forest.py     # Isolation Forest测试
├── test_feature_extractor.py    # 特征提取器测试
├── test_anomaly_integrator.py   # 异常整合器测试
└── test_cache.py                # 缓存测试
```

---

**最后更新**：2026-04-16（新增恒生指数预测系统、三周期一致预测策略）
