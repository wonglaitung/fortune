# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法、异常检测等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：所有关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：规范化开发流程、系统设计决策、测试验证要求请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)

---

## ⚡ 常用命令

### 测试与验证

```bash
# 语法检查（每次修改后必须执行）
python3 -m py_compile <文件路径>

# 运行所有测试
python3 -m pytest tests/ -v

# 运行单个测试文件
python3 -m pytest tests/test_feature_extractor.py -v
```

### 核心功能命令

| 任务 | 命令 |
|------|------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **恒指预测验证** | `python3 hsi_prediction.py --verify` |
| **恒指Walk-forward验证** | `python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20` |
| **三周期关系分析** | `python3 ml_services/analyze_three_horizon_relationships.py` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` |
| **风险回报率分析** | `python3 ml_services/risk_reward_analyzer.py --stocks watchlist --style moderate` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **个股Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost` |
| **板块验证** | `python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20` |
| **模拟交易** | `python3 simulation_trader.py --duration-days 90 --investor-type moderate` |
| **板块轮动分析** | `python3 analyze_sector_rotation.py && python3 verify_sector_rotation.py` |
| **性能监控** | `python3 ml_services/performance_monitor.py --mode all --no-email` |
| **因果链分析** | `python3 ml_services/analyze_causal_chain.py` |

| **网络分析** | `python3 ml_services/stock_network_analysis.py` |
| **超参数调优** | `python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30` |
| **密度预警** | 综合分析自动计算，历史数据存储在 `data/network_density_history.json` |

### 语言与代码规范
- 对话、代码解释、文档注释使用 **简体中文**，技术术语可括号标注英文
- 变量名和函数名使用英文，注释使用中文
- 遵循 PEP8 规范

---

## ⚠️ 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward准确率 >65% 通常是数据泄漏信号（个股）或 >80%（恒指） |
| **CatBoost 1天模型** | 噪音大，仅供参考 |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **恒指 vs 个股** | 因果链传导完全反向，个股概率高反而预示反转（见 lessons.md） |
| **特征冗余清理** | 清理后夏普比率可能下降 15-20%，需对比验证后再决定 |
| **特征缓存版本** | 新增特征后必须清除缓存（`rm -rf data/feature_cache/*.pkl`） |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测的预处理必须一致 |
| **网络特征** | 网络特征不适合放入个股预测模型，应作为独立风险监控工具 |
| **波动率网络密度** | 密度高→市场"同涨同跌"→选股失效→降仓位；动态阈值（均值+1σ/1.5σ/2σ），历史数据存储在 `data/network_density_history.json` |

### 可用策略（恒指增强模型验证，2026-04-29，33特征）

| 策略 | 胜率 | 操作 |
|------|------|------|
| **假突破(101)做多** | **93.10%** | ⭐⭐⭐⭐⭐ 最优策略 |
| **下跌中继(001)做多** | **89.19%** | ⭐⭐⭐⭐⭐ 次优策略 |
| 一致看跌(000)做空 | 82.05% | ⭐⭐⭐⭐ 样本最多 |
| 一致看涨(111)买入 | 77.86% | ⭐⭐⭐⭐ 准确率回升 |

---

## 📐 数据流架构

```
外部数据源 → data_services/ → 分析层 → ml_services/ → 输出
    ↓              ↓              ↓            ↓          ↓
腾讯财经      技术指标计算    异常检测     CatBoost    邮件报告
yfinance     基本面数据      综合分析     Walk-forward  JSON文件
AKShare      南向资金        主力追踪     性能监控
```

**关键依赖关系**：
- `comprehensive_analysis.py` 整合：大模型建议 + CatBoost预测 + 异常检测 + 板块分析
- `hsi_prediction.py` 调用 `ml_services/hsi_ml_model.py` 进行CatBoost预测
- `detect_stock_anomalies.py` 使用 `anomaly_detector/` 模块的双层检测（Z-Score + Isolation Forest）
- `config.py` 定义股票板块映射 `STOCK_SECTOR_MAPPING` 和自选股列表 `WATCHLIST`

**新增特征模块**（2026-04-27~28）：
- `data_services/calendar_features.py` - 日历效应（22个特征）
- `data_services/volatility_model.py` - GARCH 波动率（4个特征）
- `data_services/regime_detector.py` - HMM 市场状态检测（10个特征，含 Tier 1 增强）
- `data_services/multiscale_features.py` - 跨尺度关联（5个特征，待优化）
- `data_services/info_decay_analyzer.py` - 信息衰减分析（5个特征，待实施）

**数据存储**：
- `data/hsi_models/` - 恒指CatBoost模型（.cbm）和特征配置（.json）
- `data/stock_cache/` - 原始数据缓存（股票、恒指数据，7天有效期）
- `data/feature_cache/` - 特征缓存（730个特征，7天有效期，170x加速）
- `data/prediction_history.json` - 预测历史记录
- `output/` - 分析报告和回测结果

---

## 🤖 机器学习模型

### 模型可信度（Walk-forward 验证）

**恒指增强模型**（2026-04-29，33特征）：

| 周期 | 准确率 | 推荐度 |
|------|--------|--------|
| **20天** | **81.24%** | ⭐⭐⭐⭐⭐ 推荐 |
| 5天 | 60.26% | ⭐⭐⭐ 趋势确认 |
| 1天 | 50.11% | ⚠️ 噪音大 |

**个股完整模型**（2026-04-29 验证，12 folds，59只股票，730特征）：

| 指标 | 数值 | 评估 |
|------|------|------|
| 综合评分 | 80/100 | 优秀 |
| 平均准确率 | 54.93% | ✅ |
| 平均夏普比率 | 0.9059 | ✅ 接近目标 |
| 平均最大回撤 | -0.20% | ✅ 极佳 |
| 夏普标准差 | 0.4770 | ✅ 稳定性优 |

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨，≤ 0.5 预测下跌 |
| 置信度分级阈值 | 0.65 / 0.55 | 用于判断信号强弱，不影响方向 |
| 特征数量 | 730 个 | 精简特征（2026-04-27 清理冗余后） |
| 随机种子 | 42（固定） | 确保可重现性 |

**20天模型参数（适配 730 特征，2026-04-29 优化版）**：

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 600 | 树数量（增加） |
| depth | 7 | 树深度（增加） |
| learning_rate | 0.03 | 学习率（降低） |
| l2_leaf_reg | 2 | L2 正则化（降低） |
| subsample | 0.75 | 行采样（增加） |
| colsample_bylevel | 0.75 | 列采样（增加） |

**优化效果**：夏普比率 0.9059（+11.5%），最大回撤 -0.20%（+62%）

**验证方法说明**：
- **训练时CV准确率**：模型训练时的5折交叉验证准确率（用于文档展示）
- **Walk-forward准确率**：独立时序验证（12 folds），更真实反映预测能力

### 特征重要性（个股20天模型，2026-04-27）

| 排名 | 特征 | 重要性 | 类别 |
|------|------|--------|------|
| 1 | US_10Y_Yield | 5.28 | 宏观类 |
| 2 | **HSI_Regime_Duration** | 3.95 | **市场状态** |
| 3 | **HSI_Regime_Prob_1** | 2.44 | **市场状态** |
| 10 | **HSI_Regime_Prob_0** | 1.44 | **市场状态** |

**关键发现**：新增的 HSI 市场状态特征进入 Top 10，证明其预测价值。

### 板块模型（Walk-forward验证）

| 板块 | 夏普比率 | 胜率 | 推荐度 |
|------|---------|------|--------|
| **消费股** | 0.7445 | 54.80% | ⭐⭐⭐⭐⭐ |
| 银行股 | 0.1546 | 50.44% | ⭐⭐⭐⭐ |
| 半导体股 | 0.1260 | 49.87% | ⭐⭐⭐⭐ |

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

### 主要依赖

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm` `hmmlearn` `arch`

---

## 🚀 自动化调度（.github/workflows/）

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| `hsi-prediction.yml` | 恒生指数预测 | 周一到周五 06:00 |
| `comprehensive-analysis.yml` | 综合分析 | 周一到周五 16:00 |
| `stock-anomaly-detection.yml` | 港股异常检测 | 每天凌晨2点 |
| `hourly-stock-monitor.yml` | 港股异常检测（交易时段） | 10:00-15:00 每小时 |
| `performance-monitor.yml` | 预测性能监控（三时间窗口） | 每个工作日 HK 0:00 |

---

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

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
- BB_Position、Price_Percentile、动量分析特征

```python
# ❌ 错误：使用当日数据
future_return = returns.rolling(5).sum()

# ✅ 正确：使用未来数据
future_return = returns.rolling(5).sum().shift(-5)
```

### CatBoost 分类特征处理

训练和预测时必须一致处理分类特征 NaN：
```python
# 训练时（ml_trading_model.py:4445）
df[col] = df[col].fillna('unknown').astype(str)
encoder = LabelEncoder()
df[col] = encoder.fit_transform(df[col])

# 预测时（ml_trading_model.py:4937-4948）
for col in self.categorical_encoders.keys():
    test_df[col] = test_df[col].fillna('unknown').astype(str)
    encoder = self.categorical_encoders[col]
    test_df[col] = test_df[col].apply(
        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
    )
```

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **详细文档**：[docs/](docs/) - 特征工程、验证方法等
- **数据挖掘计划**：[docs/DATA_MINING_TIER2_PLAN.md](docs/DATA_MINING_TIER2_PLAN.md) - Tier 1/2 特征增强方案
- **板块轮动交易法则**：[docs/SECTOR_ROTATION_TRADING_RULES.md](docs/SECTOR_ROTATION_TRADING_RULES.md)
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **特征重要性分析**：[docs/FEATURE_IMPORTANCE_ANALYSIS.md](docs/FEATURE_IMPORTANCE_ANALYSIS.md)
- **经典交易理论**：[docs/CLASSIC_TRADING_THEORIES.md](docs/CLASSIC_TRADING_THEORIES.md)
- **网络分析详解**：[docs/STOCK_NETWORK_ANALYSIS.md](docs/STOCK_NETWORK_ANALYSIS.md)

---

**最后更新**：2026-04-30（新增波动率网络密度预警功能）
