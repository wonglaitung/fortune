# <img src="assets/icon.svg" width="40" height="48" alt="金融智能分析" style="vertical-align: middle; margin-right: 10px;"> 金融资产和港股智能分析与交易系统

**⭐ 如果您觉得这个项目有用，请先给项目Star再Fork，以支持项目发展！⭐**

实践**人机混合智能**的理念，开发具备变现能力的金融资产智能量化分析助手。系统整合**大模型智能决策**与**机器学习预测模型**，实时监控加密货币、港股、黄金等金融市场。

---

## 📄 效果文档

- [恒生指数及港股交易信号提醒](output/恒生指数及港股交易信号提醒.pdf) - 恒生指数及自选股智能分析报告
- [【综合分析】港股买卖建议](output/【综合分析】港股买卖建议.pdf) - 综合买卖建议报告

---

## 核心功能

### 恒生指数三周期预测系统

**多周期预测**：同时预测1天、5天、20天三个周期

| 周期 | 准确率 | 特点 |
|------|--------|------|
| 1天 | 51.31% | 噪音大，仅供参考 |
| 5天 | 57.17% | 趋势确认，辅助判断 |
| **20天** | **80.73%** | **最可靠，主要决策依据** |

**八大交易模式**（Walk-forward验证，938样本）：

| 模式 | 描述 | 20天准确率 | 策略 |
|------|------|-----------|------|
| **110** | 震荡回调（短涨长跌） | **90.00%** | ⭐ 最优做空 |
| 000 | 一致看跌 | 79.19% | 强烈卖出 |
| 111 | 一致看涨 | 70.31% | 强烈买入 |
| 100 | 冲高回落 | 87.4% | 短线看空 |
| 011 | 探底回升 | 81.4% | 抄底机会 |

**四条交易法则**：

| 法则 | 条件 | 胜率 | 平均收益 |
|------|------|------|----------|
| 一致看涨买入 | 三周期全看涨(111) | 70.31% | +3.55% |
| 一致看跌做空 | 三周期全看跌(000) | 79.19% | +3.21% |
| **震荡回调做空** | 1-5天涨，20天跌(110) | **90.00%** | **+4.25%** |
| 阶梯验证 | 1+5天都对→20天 | 84.1% | - |

### 个股预测 CatBoost 机器学习模型

**性能指标（训练时CV验证）**：

| 周期 | 准确率 | 标准差 |
|------|--------|--------|
| 1天 | 61.24% | ±6.51% |
| 5天 | 62.90% | ±4.61% |
| 20天 | 64.67% | ±4.32% |

**模型配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨 |
| 置信度分级 | 0.65 / 0.55 | 判断信号强弱 |
| 特征数量 | 918 个 | 技术指标、基本面、情感指标等 |
| 特征缓存 | 7天有效期 | **170x 加速** |

### 港股异常检测

**双层检测**：Z-Score + Isolation Forest

**验证策略（两年数据）**：

| 异常类型 | 策略 | 5日收益 | 胜率 |
|---------|------|---------|------|
| **Z-Score + 当日下跌** | 🟢 抄底 | +4.12% | **72%** |
| IF high 异常 | 🔴 减仓 | -3.04% | 43% |

⚠️ **重要**：股票异常策略**不适用于加密货币市场**

### 大模型智能决策

- **六层分析框架**：风险控制 → 市场环境 → 基本面 → 技术面 → 信号识别 → 综合决策
- **板块分析**：16个板块排名和龙头股识别
- **主力资金追踪**：识别建仓和出货信号

### 模拟交易系统

- 支持进取型、稳健型、保守型三种风险偏好
- 自动止损跟踪
- 决策一致性保护（3小时/24小时窗口）

---

## 快速开始

```bash
# 恒生指数预测
python3 hsi_prediction.py --no-email

# 综合分析（含三周期预测）
./scripts/run_comprehensive_analysis.sh

# 港股异常检测
python3 detect_stock_anomalies.py --mode standalone --mode-type deep

# 模型训练
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# Walk-forward验证
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20
```

---

## 技术架构

```
外部数据源 → data_services/ → 分析层 → ml_services/ → 输出
    ↓              ↓              ↓            ↓          ↓
腾讯财经      技术指标计算    异常检测     CatBoost    邮件报告
yfinance     基本面数据      综合分析     Walk-forward  JSON文件
AKShare      南向资金        主力追踪     性能监控
```

**缓存机制**：

| 缓存类型 | 位置 | 有效期 | 加速效果 |
|---------|------|--------|---------|
| 原始数据 | `data/stock_cache/` | 7天 | - |
| 特征缓存 | `data/feature_cache/` | 7天 | **170x** |

---

## 项目结构

```
fortune/
├── 核心脚本
│   ├── comprehensive_analysis.py       # 综合分析
│   ├── hsi_prediction.py               # 恒指三周期预测
│   ├── detect_stock_anomalies.py       # 异常检测
│   └── simulation_trader.py            # 模拟交易
├── ml_services/                        # 机器学习模块
│   ├── ml_trading_model.py             # CatBoost模型
│   └── walk_forward_validation.py      # Walk-forward验证
├── data_services/                      # 数据服务
├── anomaly_detector/                   # 异常检测
├── llm_services/                       # 大模型服务
├── docs/                               # 详细文档
└── data/                               # 数据和缓存
    ├── stock_cache/                    # 原始数据缓存
    └── feature_cache/                  # 特征缓存
```

---

## 自动化调度

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| `hsi-prediction.yml` | 恒生指数预测 | 工作日 06:00 |
| `comprehensive-analysis.yml` | 综合分析 | 工作日 16:00 |
| `stock-anomaly-detection.yml` | 港股异常检测 | 每天凌晨2点 |
| `hourly-stock-monitor.yml` | 交易时段监控 | 10:00-15:00 每小时 |
| `performance-monitor.yml` | 性能报告 | 每月1号 |

---

## 安装部署

```bash
# 1. 克隆项目
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp set_key.sh.sample set_key.sh
# 编辑 set_key.sh，填写邮箱和API密钥
source set_key.sh

# 4. 验证安装
python hsi_email.py --no-email
```

**必填环境变量**：

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP服务器地址 |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱授权码 |
| `RECIPIENT_EMAIL` | 收件人邮箱 |
| `QWEN_API_KEY` | 通义千问API密钥 |

---

## 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | 准确率 >65% 通常有数据泄漏 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **CatBoost 1天** | 过拟合，不推荐 |
| **深度学习** | LSTM/Transformer F1≈0，不推荐 |
| **Walk-forward** | 唯一可信的验证方法 |
| **加密货币** | 股票策略不适用 |

---

## 文档

- **[CLAUDE.md](CLAUDE.md)** - 快速参考指南
- **[lessons.md](lessons.md)** - 经验教训
- **[progress.txt](progress.txt)** - 项目进展
- **[docs/](docs/)** - 详细文档
  - [THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md) - 三周期分析
  - [SECTOR_ROTATION_TRADING_RULES.md](docs/SECTOR_ROTATION_TRADING_RULES.md) - 板块轮动
  - [programmer_skill.md](docs/programmer_skill.md) - 开发规范

---

## 依赖项

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm` `jieba`

---

## 许可证

MIT License

---

## 联系方式

- Issues: https://github.com/wonglaitung/fortune/issues
- Email: wonglaitung@gmail.com

---

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=wonglaitung/fortune&type=Date)
