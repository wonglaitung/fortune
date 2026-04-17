# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法、异常检测等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：所有关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：规范化开发流程、系统设计决策、测试验证要求请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)

---

## ⚡ 快速参考

### 测试与验证

```bash
# 语法检查（每次修改后必须执行）
python3 -m py_compile <文件路径>

# 运行所有测试
python3 -m pytest tests/ -v

# 运行单个测试
python3 -m pytest tests/test_zscore_detector.py -v
```

### 常用命令

| 任务 | 命令 |
|------|------|
| **训练模型** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost` |
| **生成预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **恒指预测验证** | `python3 hsi_prediction.py --verify` |
| **Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **预测性能监控** | `python3 ml_services/performance_monitor.py --mode all --horizon 20` |
| **模拟交易** | `python3 simulation_trader.py --duration-days 90 --investor-type moderate` |
| **板块分析** | `./scripts/run_sector_analysis.sh` |

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
| **深度学习模型** | LSTM/Transformer 表现远不如 CatBoost，**不推荐** |
| **Walk-forward验证** | **唯一可信**的模型验证方法 |
| **交易时段检测** | 交易时段数据不完整，应在收盘后检测 |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |

### 异常策略（两年数据验证）

| 异常类型 | 5日收益 | 胜率 | 策略 |
|---------|---------|------|------|
| **IF high** | -3.04% | 43% | 🔴 减仓 |
| **价格异常+当日下跌** | +4.12% | **72%** | 🟢 抄底 |

### 三周期一致预测策略 ⭐

| 场景 | 占比 | 至少一周期正确率 | 操作 |
|------|------|-----------------|------|
| **三周期一致看涨** | 12.3% | **92%** | 强烈买入 |
| **三周期一致看跌** | 23.4% | **93%** | 强烈卖出 |
| 不一致 | 64.4% | - | 观望 |

> **数据来源**: Walk-forward回测验证（2020-2024年，938个样本）
> **详细说明**: 参阅 [lessons.md](lessons.md)

---

## 📂 项目架构

```
fortune/
├── 核心入口
│   ├── comprehensive_analysis.py       # 综合分析（每日信号生成）
│   ├── detect_stock_anomalies.py       # 港股异常检测
│   ├── hsi_prediction.py               # 恒生指数预测（双模型）
│   ├── hk_smart_money_tracker.py       # 主力资金追踪
│   ├── simulation_trader.py            # 模拟交易系统
│   └── crypto_email.py                 # 加密货币监控
├── anomaly_detector/                   # 异常检测模块
│   ├── zscore_detector.py              # Z-Score 检测器
│   ├── isolation_forest_detector.py    # Isolation Forest 检测器
│   └── anomaly_integrator.py           # 异常整合器
├── data_services/                      # 数据服务层
│   ├── technical_analysis.py           # 技术分析工具
│   ├── fundamental_data.py             # 基本面数据
│   ├── southbound_data.py              # 南向资金数据
│   └── tencent_finance.py              # 腾讯财经数据源
├── ml_services/                        # 机器学习模块
│   ├── ml_trading_model.py             # ML 模型训练/预测
│   ├── hsi_ml_model.py                 # 恒指 CatBoost 模型
│   ├── walk_forward_validation.py      # Walk-forward 验证
│   ├── walk_forward_by_sector.py       # 板块验证
│   └── performance_monitor.py          # 预测性能监控
├── llm_services/                       # 大模型服务
│   ├── qwen_engine.py                  # 通义千问接口
│   └── sentiment_analyzer.py           # 情感分析
├── tests/                              # 测试脚本
├── docs/                               # 文档
├── output/                             # 输出报告
├── data/                               # 数据文件
└── scripts/                            # 运行脚本
```

---

## 🎯 核心功能

### 恒生指数预测系统

**双模型架构**：
- **评分模型**：基于特征重要性加权的规则模型
- **CatBoost模型**：机器学习自动学习权重

**多周期预测**：1天（噪音大，不推荐）、5天（准确率最高 ~58%）、20天（趋势判断能力强）

**特征配置**：宏观因子（美债收益率、VIX）、港股通资金、技术指标（MA、RSI、MACD等）

---

## 🤖 机器学习模型

### 模型可信度

| 模型 | 可信度 | 说明 |
|------|--------|------|
| **CatBoost 20天** | ⭐⭐⭐⭐⭐ | **推荐使用**，准确率 61.44% |
| CatBoost 5天 | ⭐⭐⭐ | 谨慎使用 |
| CatBoost 1天 | ⭐ | **不推荐**，严重过拟合 |
| LSTM/Transformer | ⭐ | **不推荐**，表现远不如 CatBoost |

### CatBoost 配置（推荐）

| 参数 | 值 |
|------|-----|
| 准确率 | 61.44%（±1.75%） |
| 置信度阈值 | 0.65 |
| 特征数量 | 918 个全量特征 |
| 随机种子 | 42（固定） |

### Walk-forward 验证结果

| 指标 | 数值 | 业界标准 | 评估 |
|------|------|---------|------|
| 夏普比率 | 0.97 | >1.0 | 接近标准 |
| 最大回撤 | -0.55% | <-20% | 优秀 |
| 胜率 | 49.36% | 52%+ | 略低 |
| 索提诺比率 | 2.52 | >1.0 | 优秀 |

**实用性评估**：80/100，⭐⭐⭐⭐⭐ 强烈推荐实盘

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

### 依赖管理

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖
yfinance    # 金融数据获取
catboost    # 机器学习模型（主要模型）
akshare     # 中文财经数据
pandas      # 数据处理
```

---

## 🚀 自动化调度

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| `stock-anomaly-detection.yml` | 港股异常检测 | 每天凌晨2点 |
| `hourly-stock-monitor.yml` | 港股异常检测（交易时段） | 10:00-15:00 每小时 |
| `hsi-prediction.yml` | 恒生指数预测 | 周一到周五 06:00 |
| `comprehensive-analysis.yml` | 综合分析 | 周一到周五 16:00 |
| `performance-monitor.yml` | 性能月度报告 | 每月1号 |

---

## 📝 Session Workflow

**会话开始时**：
1. 读取 `progress.txt`，了解项目当前进展
2. 审查 `lessons.md`，检查是否有错误需要纠正

**功能更新后**：
1. 更新 `progress.txt`，记录新的进展
2. 如有新的学习心得，更新 `lessons.md`

---

## 🔧 开发规范要点

### 代码修改原则

1. **修改完即测试**：每次代码修改后立即验证
   - 语法检查：`python3 -m py_compile <文件路径>`
   - 功能测试：验证修改是否符合预期
   - 回归测试：确保没有破坏现有功能

2. **避免硬编码路径**：
   ```python
   # ✅ 正确
   script_dir = os.path.dirname(os.path.abspath(__file__))
   data_dir = os.path.join(script_dir, 'data')
   ```

3. **HTTP API 超时处理**：调用 API 时必须设置超时时间

4. **公共代码提取**：识别可复用逻辑，创建通用函数

### 数据泄漏防护

高风险特征必须使用 `.shift(1)` 避免使用当日数据：
- 所有 `.rolling()` 计算的特征
- BB_Position、Price_Percentile、Intraday_Amplitude
- Price_Ratio_MA5/20/50、Support_120d、Resistance_120d

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase` 或 `|| true` 容错

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md)
- **进度跟踪**：[progress.txt](progress.txt)
- **详细文档**：[docs/](docs/)

---

**最后更新**：2026-04-17
