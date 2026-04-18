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
```

### 核心功能命令

| 任务 | 命令 |
|------|------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **恒指预测验证** | `python3 hsi_prediction.py --verify` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost` |
| **板块验证** | `python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20` |
| **模拟交易** | `python3 simulation_trader.py --duration-days 90 --investor-type moderate` |
| **板块轮动分析** | `python3 analyze_sector_rotation.py && python3 verify_sector_rotation.py` |

### 语言与代码规范
- 对话、代码解释、文档注释使用 **简体中文**，技术术语可括号标注英文
- 变量名和函数名使用英文，注释使用中文
- 遵循 PEP8 规范

---

## ⚠️ 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | 准确率 >65% 通常是数据泄漏信号 |
| **CatBoost 1天模型** | 严重过拟合，**不推荐使用** |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **Walk-forward验证** | **唯一可信**的模型验证方法 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **动量/反转策略** | 收益仅0.3%-0.5%，**不推荐单独使用** |

### 可用策略（两年数据验证）

| 策略 | 收益 | 胜率 | 操作 |
|------|------|------|------|
| **三周期一致(110)做空** | +4.25% | **90%** | ⭐ 最优 |
| **Z-Score+当日下跌抄底** | +4.12% | **72%** | 🟢 买入 |
| **IF high异常** | -3.04% | 43% | 🔴 减仓 |
| **周期/防御轮动** | - | 74.3%/65.9% | ⭐⭐⭐⭐ 有效 |

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

**数据存储**：
- `data/hsi_models/` - 恒指CatBoost模型（.cbm）和特征配置（.json）
- `data/stock_cache/` - 原始数据缓存（股票、恒指数据，7天有效期）
- `data/feature_cache/` - 特征缓存（计算好的892个特征，7天有效期，170x加速）
- `data/prediction_history.json` - 预测历史记录
- `output/` - 分析报告和回测结果

---

## 🤖 机器学习模型

### 模型可信度

| 模型 | 准确率 | 夏普比率 | 推荐度 |
|------|--------|---------|--------|
| **CatBoost 20天** | 61.44% (±1.75%) | 0.97 | ⭐⭐⭐⭐⭐ 推荐 |
| CatBoost 5天 | 56.8% | - | ⭐⭐⭐ 谨慎使用 |
| CatBoost 1天 | 过拟合 | - | ❌ 不推荐 |
| LSTM/Transformer | ~51% | F1≈0 | ❌ 不推荐 |

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨，≤ 0.5 预测下跌 |
| 置信度分级阈值 | 0.65 / 0.55 | 用于判断信号强弱，不影响方向 |
| 特征数量 | 918 个 | 技术指标、基本面、美股市场、情感指标等 |
| 随机种子 | 42（固定） | 确保可重现性 |
| Walk-forward folds | 12 | 业界标准验证方法 |

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

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm`

---

## 🚀 自动化调度（.github/workflows/）

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| `hsi-prediction.yml` | 恒生指数预测 | 周一到周五 06:00 |
| `comprehensive-analysis.yml` | 综合分析 | 周一到周五 16:00 |
| `stock-anomaly-detection.yml` | 港股异常检测 | 每天凌晨2点 |
| `hourly-stock-monitor.yml` | 港股异常检测（交易时段） | 10:00-15:00 每小时 |
| `performance-monitor.yml` | 性能月度报告 | 每月1号 |

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

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **详细文档**：[docs/](docs/) - 特征工程、验证方法等
- **板块轮动交易法则**：[docs/SECTOR_ROTATION_TRADING_RULES.md](docs/SECTOR_ROTATION_TRADING_RULES.md)
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)

---

**最后更新**：2026-04-18
