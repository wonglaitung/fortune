# 📊 金融信息监控与智能交易系统

<div align="center">

一个基于 Python 的综合性金融分析系统，集成多数据源、技术分析工具和大模型智能判断，为投资者提供全面的市场分析和交易策略验证工具。

[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-automated-brightgreen.svg)](https://github.com/features/actions)

[English](README_EN.md) | 简体中文

</div>

---

## 📖 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [功能概览](#功能概览)
- [快速开始](#快速开始)
- [详细功能](#详细功能)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)

---

## 🎯 项目简介

本项目是一个功能强大的金融信息监控与智能交易系统，旨在帮助投资者：

- 📊 **实时监控**：加密货币、港股、黄金等金融市场
- 🔍 **智能分析**：识别主力资金动向和交易信号
- 🤖 **AI决策**：基于大模型进行智能投资决策和持仓分析
- 📈 **策略验证**：验证交易策略的有效性
- 💰 **股息追踪**：获取股息信息和基本面数据
- 📧 **自动通知**：邮件提醒重要信息和交易信号
- 📉 **中期评估**：提供数周至数月投资周期的技术分析

---

## ✨ 核心特性

### 🤖 AI 智能分析
- **大模型集成**：集成 Qwen 大模型，提供智能投资建议
- **持仓分析**：自动分析现有持仓，提供专业的投资建议
- **信号识别**：智能识别买卖信号，减少人工判断
- **策略生成**：基于技术面和基本面生成交易策略
- **多风格分析**：支持进取型短期、稳健型短期、稳健型中期、保守型中期四种投资风格

### 📈 技术分析
- **多指标支持**：RSI、MACD、布林带、ATR、CCI、OBV 等
- **VaR 风险价值**：1日、5日、20日 VaR 计算
- **TAV 评分系统**：加权评分提供精准交易信号
- **趋势分析**：自动识别多头、空头、震荡趋势
- **中期分析**：均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分

### 💹 模拟交易
- **真实模拟**：基于大模型建议的模拟交易系统
- **风险控制**：自动止损机制
- **详细记录**：完整的交易日志和持仓分析
- **多种策略**：支持保守型、平衡型、进取型投资偏好

### 📊 数据获取
- **多数据源**：yfinance、腾讯财经、AKShare 等
- **基本面数据**：财务指标、利润表、资产负债表、现金流量表
- **股息信息**：自动获取股息和除净日信息
- **智能缓存**：7天缓存机制，提高性能

### 🤖 自动化
- **GitHub Actions**：定时自动执行分析任务
- **邮件通知**：重要信息自动邮件提醒
- **定时任务**：支持本地定时执行

---

## 🚀 功能概览

### 数据获取与监控

| 功能 | 脚本 | 频率 | 说明 |
|------|------|------|------|
| 加密货币监控 | `crypto_email.py` | 每小时 | 比特币、以太坊价格和技术分析 |
| 港股 IPO 信息 | `hk_ipo_aastocks.py` | 每天 | 最新 IPO 信息 |
| 黄金市场分析 | `gold_analyzer.py` | 每小时 | 黄金价格和投资建议 |
| 恒生指数监控 | `hsi_email.py` | 交易时段 | 价格、技术指标、交易信号、基本面指标、中期评估指标、AI持仓分析 |

### 智能分析

| 功能 | 脚本 | 说明 |
|------|------|------|
| 主力资金追踪 | `hk_smart_money_tracker.py` | 识别建仓和出货信号，集成基本面分析 |
| 恒生指数策略 | `hsi_llm_strategy.py` | 大模型生成交易策略 |
| 48小时信号分析 | `analyze_last_48_hours_email.py` | 连续交易信号识别 |
| AI 交易分析 | `ai_trading_analyzer.py` | 复盘 AI 推荐策略有效性 |

### 机器学习

| 功能 | 脚本 | 说明 |
|------|------|------|
| 机器学习交易模型 | `ml_trading_model.py` | 基于LightGBM预测次日涨跌，平均准确率57.16% |
| 策略对比分析 | `compare_strategies.py` | 对比ML模型与手工信号的预测准确性 |

### 模拟交易

| 功能 | 脚本 | 说明 |
|------|------|------|
| 模拟交易系统 | `simulation_trader.py` | 基于大模型的自动交易 |

### 辅助功能

| 功能 | 脚本 | 说明 |
|------|------|------|
| 批量新闻获取 | `batch_stock_news_fetcher.py` | 自选股新闻 |
| 基本面数据 | `fundamental_data.py` | 财务指标、利润表等 |
| 技术分析工具 | `technical_analysis.py` | 通用技术指标计算，含中期分析指标系统 |

---

## 🏁 快速开始

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器

### 1. 克隆项目

```bash
git clone https://github.com/wonglaitung/fortune.git
cd fortune
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

编辑 `set_key.sh` 文件：

```bash
# 邮件配置
export YAHOO_EMAIL=your_email@163.com
export YAHOO_APP_PASSWORD=your_app_password
export YAHOO_SMTP=smtp.163.com
export RECIPIENT_EMAIL=recipient1@email.com,recipient2@email.com

# 大模型 API 配置
export QWEN_API_KEY=your_qwen_api_key
```

### 4. 运行示例

```bash
# 加载环境变量
source set_key.sh

# 监控加密货币价格
python crypto_email.py

# 追踪港股主力资金
python hk_smart_money_tracker.py

# 启动模拟交易
python simulation_trader.py

# 恒生指数价格监控（含基本面指标、中期评估指标和 AI 持仓分析）
python hsi_email.py
```

---

## 📖 详细功能

### 1. 加密货币价格监控器

**功能**：
- 实时获取比特币、以太坊价格（USD/HKD）
- 24小时价格变化、市值、交易量
- 技术指标分析（RSI、MACD、均线、布林带等）
- 自动邮件通知（每小时执行）

**使用方法**：
```bash
python crypto_email.py
```

### 2. 港股主力资金追踪器

**功能**：
- 批量扫描自选股，识别建仓和出货信号
- 结合股价位置、成交量比率、南向资金流向
- 集成基本面数据分析（财务指标、利润表、资产负债表、现金流量表）
- 大模型智能分析和投资建议
- 生成可视化图表和 Excel 报告

**使用方法**：
```bash
# 分析当天数据
python hk_smart_money_tracker.py

# 分析指定日期数据
python hk_smart_money_tracker.py --date 2025-10-25
```

### 3. 恒生指数价格监控器（含基本面指标、中期评估指标和 AI 持仓分析）

**功能**：
- 实时获取恒生指数价格和交易数据
- 技术指标计算和信号识别
- 只在有交易信号时才发送邮件
- 支持历史数据分析
- 集成股息信息追踪功能
- VaR 风险价值计算（1日、5日、20日）
- **基本面指标**：基本面评分、PE、PB等估值指标
- **中期评估指标**：均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分、中期趋势健康度、中期可持续性、中期建议
- **AI 智能持仓分析**：读取持仓数据，提供专业投资建议
- **大模型多风格分析**：支持进取型短期、稳健型短期、稳健型中期、保守型中期四种投资风格

**使用方法**：
```bash
# 分析当天数据
python hsi_email.py

# 分析指定日期数据
python hsi_email.py --date 2025-10-25
```

**基本面指标功能**：
- 在交易信号总结表中显示基本面评分、PE、PB
- 在单个股票分析表格中显示基本面评分、PE、PB
- 根据估值水平自动评分和分类（优秀/一般/较差）
- 结合技术指标提供综合判断

**中期评估指标功能**：
- 均线排列状态判断（多头/空头/混乱排列）
- 均线斜率计算（MA20/MA50斜率和角度，判断趋势强度）
- 均线乖离率（评估价格与均线的偏离程度，识别超买超卖状态）
- 支撑阻力位识别（基于近期局部高低点识别关键价格水平）
- 相对强弱指标（计算股票相对于恒生指数的表现）
- 中期趋势评分系统（综合趋势、动量、支撑阻力、相对强弱四维度评分）
- 中期趋势健康度、可持续性评估
- 中期投资建议（强烈买入/买入/持有/卖出/强烈卖出）

**AI 持仓分析功能**：
- 读取 `data/actual_porfolio.csv` 持仓数据
- 使用大模型进行综合投资分析
- 提供整体风险评估
- 各股投资建议（持有/加仓/减仓/清仓）
- 止损位和目标价建议
- 仓位管理建议
- 风险控制措施

**大模型多风格分析功能**：
- 进取型短期分析（日内/数天）：捕捉价格波动机会，高风险高收益
- 稳健型短期分析（日内/数天）：风险收益平衡，稳健收益
- 稳健型中期分析（数周-数月）：基本面和技术面结合，中长期价值投资
- 保守型中期分析（数周-数月）：注重长期价值投资，资产保值和稳健增长
- 支持通过配置开关切换生成全部四种或部分分析风格

### 4. 港股模拟交易系统

**功能**：
- 基于大模型判断进行模拟交易
- 支持保守型、平衡型、进取型投资者偏好
- 严格遵循大模型建议，无随机操作
- 自动止损机制
- 交易记录和状态持久化
- 详细持仓展示和每日总结
- 邮件通知系统
- 大模型多风格分析功能

**使用方法**：
```bash
# 持续运行
python simulation_trader.py

# 运行指定天数
python simulation_trader.py --duration-days 30
```

**投资者类型**：
- **保守型**：偏好低风险、稳定收益的股票，如高股息银行股，注重资本保值
- **平衡型**：平衡风险与收益，兼顾价值与成长，追求稳健增长
- **进取型**：偏好高风险、高收益的股票，如科技成长股，追求资本增值

### 5. AI 交易分析器

**功能**：
- 基于交易记录复盘 AI 推荐策略
- 计算已实现盈亏和未实现盈亏
- 支持多时间维度分析（1天、5天、1个月）
- 显示建议买卖次数和实际执行次数
- 生成详细分析报告

**使用方法**：
```bash
# 分析指定日期
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 分析 5 天数据
python ai_trading_analyzer.py --start-date 2025-12-31 --end-date 2026-01-05

# 分析 1 个月数据
python ai_trading_analyzer.py --start-date 2025-12-05 --end-date 2026-01-05

# 不发送邮件
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05 --no-email
```

### 6. 机器学习交易模型

**功能**：
- 基于LightGBM的二分类模型，预测次日涨跌
- 整合24个特征（技术指标、市场环境、资金流向、基本面）
- 时间序列交叉验证（5折）
- 平均验证准确率57.16%
- 特征重要性分析

**特征类型**：
- **技术指标特征**（15个）：RSI、MACD、布林带、ATR、成交量比率、价格相对均线、涨跌幅
- **市场环境特征**（3个）：恒生指数收益率、股票相对恒指表现
- **资金流向特征**（5个）：价格位置、成交量信号、动量信号
- **基本面特征**（8个）：PE、PB、ROE、ROA、股息率等（未使用，API限制）

**特征重要性 Top 10**：
1. HSI_Return_5d（408）- 恒生指数5日收益率
2. HSI_Return（336）- 恒生指数日收益率
3. Vol_Ratio（185）- 成交量比率
4. BB_width（139）- 布林带宽度
5. ATR（133）- 平均真实波幅
6. Price_Ratio_MA50（130）- 价格相对50日均线
7. MACD_histogram（117）- MACD柱状图
8. RSI（113）- 相对强弱指标
9. Return_20d（109）- 20日收益率
10. Price_Ratio_MA5（104）- 价格相对5日均线

**使用方法**：
```bash
# 训练模型
python ml_trading_model.py --mode train

# 预测股票
python ml_trading_model.py --mode predict

# 指定日期范围训练
python ml_trading_model.py --mode train --start-date 2024-01-01 --end-date 2025-12-31
```

**输出文件**：
- `data/ml_trading_model.pkl` - 训练好的模型
- `data/ml_trading_model_importance.csv` - 特征重要性排名
- `data/ml_trading_model_predictions.csv` - 预测结果

**使用建议**：
- 作为手工信号的补充参考
- 只在信号一致时交易（保守策略）
- 定期重新训练（每周或每月）
- 跟踪实际预测准确率

### 7. 策略对比分析

**功能**：
- 对比机器学习模型预测与主力资金追踪器手工信号
- 分析预测结果、置信度、信号一致性
- 提供组合使用建议

**对比维度**：
- ML模型预测（上涨/下跌）
- ML预测概率
- 手工建仓信号
- 手工出货信号
- 信号一致性

**使用方法**：
```bash
# 对比ML模型和手工信号
python compare_strategies.py

# 指定日期范围对比
python compare_strategies.py --start-date 2025-01-01 --end-date 2025-12-31
```

**输出文件**：
- `data/strategy_comparison.csv` - 策略对比结果

**分析结果**：
- 信号一致：ML预测下跌 + 无建仓信号
- 信号不一致：ML预测上涨 + 无建仓信号
- 一致性统计：百分比和数量

**使用建议**：
- 只在信号一致时交易（保守策略）
- ML模型作为参考，手工信号作为确认
- 关注高置信度预测（概率 > 60%）
- 定期验证ML模型实际准确率

### 8. 通用技术分析工具

**支持的指标**：
- 移动平均线（MA）
- 相对强弱指数（RSI）
- MACD
- 布林带（Bollinger Bands）
- 随机振荡器（Stochastic）
- ATR（平均真实波幅）
- CCI（商品通道指数）
- OBV（能量潮）
- VaR（风险价值）
- TAV（技术分析价值）加权评分系统
- **中期分析指标系统**：
  - 均线排列状态判断（多头/空头/混乱排列）
  - 均线斜率计算（MA20/MA50斜率和角度）
  - 均线乖离率（价格与均线的偏离程度）
  - 支撑阻力位识别（基于近期局部高低点）
  - 相对强弱指标（相对恒生指数的表现）
  - 中期趋势评分系统（综合趋势、动量、支撑阻力、相对强弱四维度评分）

### 7. 基本面数据获取器

**支持的数据**：
- **财务指标**：PE、PB、ROE、ROA、EPS、股息率、市值
- **利润表**：营业收入、净利润、增长率
- **资产负债表**：资产、负债、权益、流动比率
- **现金流量表**：经营、投资、筹资现金流

**使用方法**：
```python
from fundamental_data import get_comprehensive_fundamental_data

# 获取综合基本面数据
data = get_comprehensive_fundamental_data("00700")
print(data)
```

---

## ⚙️ 配置说明

### 环境变量

在 `set_key.sh` 中配置：

```bash
# 邮件配置
export YAHOO_EMAIL=your_email@163.com
export YAHOO_APP_PASSWORD=your_app_password
export YAHOO_SMTP=smtp.163.com
export RECIPIENT_EMAIL=recipient1@email.com,recipient2@email.com

# 大模型 API 配置
export QWEN_API_KEY=your_qwen_api_key

# 可选配置
export MAX_LOSS_PCT=0.2  # 最大亏损百分比（20%）
export DEFAULT_TICK_SIZE=0.01  # 默认最小价格变动单位
```

### 持仓数据配置

创建 `data/actual_porfolio.csv` 文件：

```csv
股票号码,一手股数,成本价,持有手数
1299.HK,200,85.90,2
2800.HK,500,26.48,5
9988.HK,100,174.145,2
```

### 主力资金追踪器参数

在 `hk_smart_money_tracker.py` 中调整：

```python
WATCHLIST = ["0700.HK", "0941.HK", "0939.HK", ...]  # 自选股票列表
DAYS_ANALYSIS = 20  # 分析窗口天数
VOL_WINDOW = 20  # 成交量分析窗口
PRICE_WINDOW = 20  # 价格分析窗口
BUILDUP_MIN_DAYS = 3  # 建仓信号最小确认天数
DISTRIBUTION_MIN_DAYS = 3  # 出货信号最小确认天数

# 阈值参数
PRICE_LOW_PCT = 10  # 建仓信号价格百分位阈值
PRICE_HIGH_PCT = 90  # 出货信号价格百分位阈值
VOL_RATIO_BUILDUP = 1.5  # 建仓信号量比阈值
VOL_RATIO_DISTRIBUTION = 1.5  # 出货信号量比阈值
SOUTHBOUND_THRESHOLD = 10000  # 南向资金阈值
```

### 大模型分析风格配置

在 `hsi_email.py` 中调整：

```python
# True = 生成全部四种（进取型短期、稳健型短期、稳健型中期、保守型中期）
# False = 只生成两种（稳健型短期、稳健型中期）
ENABLE_ALL_ANALYSIS_STYLES = False
```

---

## 📁 项目结构

```
fortune/
├── 📄 核心脚本
│   ├── ai_trading_analyzer.py          # AI 交易分析器
│   ├── analyze_last_48_hours_email.py  # 48 小时信号分析器
│   ├── batch_stock_news_fetcher.py     # 批量新闻获取器
│   ├── compare_strategies.py           # 策略对比分析
│   ├── crypto_email.py                 # 加密货币监控器
│   ├── fundamental_data.py             # 基本面数据获取器
│   ├── gold_analyzer.py                # 黄金市场分析器
│   ├── hk_ipo_aastocks.py              # IPO 信息获取器
│   ├── hk_smart_money_tracker.py       # 主力资金追踪器
│   ├── hsi_email.py                    # 恒生指数价格监控器
│   ├── hsi_llm_strategy.py             # 恒生指数策略分析器
│   ├── ml_trading_model.py             # 机器学习交易模型
│   ├── simulation_trader.py            # 模拟交易系统
│   ├── technical_analysis.py           # 通用技术分析工具
│   └── tencent_finance.py              # 腾讯财经接口
│   ├── analyze_last_48_hours_email.py  # 48 小时信号分析器
│   ├── batch_stock_news_fetcher.py     # 批量新闻获取器
│   ├── crypto_email.py                 # 加密货币监控器
│   ├── fundamental_data.py             # 基本面数据获取器
│   ├── gold_analyzer.py                # 黄金市场分析器
│   ├── hk_ipo_aastocks.py              # IPO 信息获取器
│   ├── hk_smart_money_tracker.py       # 主力资金追踪器
│   ├── hsi_email.py                    # 恒生指数价格监控器
│   ├── hsi_llm_strategy.py             # 恒生指数策略分析器
│   ├── simulation_trader.py            # 模拟交易系统
│   ├── technical_analysis.py           # 通用技术分析工具
│   └── tencent_finance.py              # 腾讯财经接口
│
├── 🔧 配置和脚本
│   ├── send_alert.sh                   # 本地定时执行脚本
│   ├── update_data.sh                  # 数据更新脚本
│   ├── set_key.sh                      # 环境变量配置
│   ├── requirements.txt                # 项目依赖
│   ├── README.md                       # 项目说明文档
│   └── IFLOW.md                        # iFlow 代理上下文
│
├── 🤖 大模型服务
│   └── llm_services/
│       └── qwen_engine.py              # Qwen 大模型接口
│
├── 📊 数据文件
│   ├── actual_porfolio.csv             # 实际持仓数据
│   ├── all_stock_news_records.csv      # 股票新闻记录
│   ├── all_dividends.csv               # 所有股息信息记录
│   ├── recent_dividends.csv            # 最近除净的股息信息
│   ├── upcoming_dividends.csv          # 即将除净的股息信息（未来90天）
│   ├── hsi_strategy_latest.txt         # 恒生指数策略分析
│   ├── ml_trading_model.pkl            # 机器学习训练好的模型文件
│   ├── ml_trading_model_importance.csv # 机器学习特征重要性排名
│   ├── ml_trading_model_predictions.csv # 机器学习模型预测结果
│   ├── simulation_state.json           # 模拟交易状态
│   ├── simulation_transactions.csv     # 交易历史记录
│   ├── simulation_portfolio.csv        # 投资组合价值变化记录
│   ├── simulation_trade_log_*.txt      # 交易日志（按日期分割）
│   ├── southbound_data_cache.pkl       # 南向资金数据缓存
│   ├── strategy_comparison.csv         # 策略对比分析结果
│   └── fundamental_cache/               # 基本面数据缓存
│   ├── all_stock_news_records.csv      # 股票新闻记录
│   ├── all_dividends.csv               # 所有股息信息记录
│   ├── recent_dividends.csv            # 最近除净的股息信息
│   ├── upcoming_dividends.csv          # 即将除净的股息信息（未来90天）
│   ├── hsi_strategy_latest.txt         # 恒生指数策略分析
│   ├── simulation_state.json           # 模拟交易状态
│   ├── simulation_transactions.csv     # 交易历史记录
│   ├── simulation_portfolio.csv        # 投资组合价值变化记录
│   ├── simulation_trade_log_*.txt      # 交易日志（按日期分割）
│   ├── southbound_data_cache.pkl       # 南向资金数据缓存
│   └── fundamental_cache/               # 基本面数据缓存
│
├── 📈 图表输出
│   └── hk_smart_charts/                # 主力资金追踪图表
│
└── 🚀 GitHub Actions
    └── .github/workflows/
        ├── crypto-alert.yml              # 加密货币监控
        ├── gold-analyzer.yml             # 黄金分析
        ├── ipo-alert.yml                 # IPO 信息
        ├── hsi-email-alert.yml           # 恒生指数监控
        ├── smart-money-alert.yml         # 主力资金追踪
        └── ai-trading-analysis-daily.yml  # AI 交易分析
```

---

## 🏗️ 技术架构

```
金融信息监控与智能交易系统
│
├── 📥 数据获取层
│   ├── 加密货币数据 (CoinGecko)
│   ├── 港股数据 (yfinance, 腾讯财经, AKShare)
│   ├── 黄金数据 (yfinance)
│   ├── 基本面数据 (AKShare)
│   ├── 股息数据 (AKShare)
│   └── IPO 数据 (AAStocks)
│
├── 🔍 分析层
│   ├── 技术分析 (technical_analysis.py)
│   │   ├── 短期技术指标（RSI、MACD、布林带、ATR等）
│   │   ├── TAV加权评分系统
│   │   └── 中期分析指标系统
│   │       ├── 均线排列状态判断
│   │       ├── 均线斜率计算
│   │       ├── 均线乖离率
│   │       ├── 支撑阻力位识别
│   │       ├── 相对强弱指标
│   │       └── 中期趋势评分系统
│   ├── 主力资金追踪 (hk_smart_money_tracker.py)
│   ├── 48小时信号分析 (analyze_last_48_hours_email.py)
│   ├── AI 交易分析 (ai_trading_analyzer.py)
│   ├── 恒生指数策略 (hsi_llm_strategy.py)
│   ├── 新闻过滤 (batch_stock_news_fetcher.py)
│   ├── 基本面分析 (fundamental_data.py)
│   ├── 机器学习交易模型 (ml_trading_model.py)
│   │   ├── 特征工程（24个特征）
│   │   ├── LightGBM二分类模型
│   │   ├── 时间序列交叉验证
│   │   └── 特征重要性分析
│   └── 策略对比分析 (compare_strategies.py)
│       ├── ML模型预测 vs 手工信号对比
│       └── 信号一致性分析
│
├── 💹 交易层
│   └── 模拟交易系统 (simulation_trader.py)
│
└── 🤖 服务层
    ├── 大模型服务 (qwen_engine.py)
    ├── 邮件服务 (SMTP)
    └── 数据缓存 (fundamental_cache/, southbound_data_cache.pkl)
```

---

## 🤖 大模型集成

项目集成了 Qwen 大模型，提供智能分析和决策支持：

| 功能模块 | 应用场景 | 说明 |
|---------|---------|------|
| 主力资金追踪 | 股票分析 | 识别建仓和出货信号，集成基本面分析 |
| 模拟交易 | 交易决策 | 生成买卖信号和原因，支持多风格分析 |
| 新闻过滤 | 相关性判断 | 过滤相关新闻 |
| 黄金分析 | 市场分析 | 深度分析和投资建议 |
| 恒生指数策略 | 策略生成 | 生成交易策略 |
| **持仓分析** | **投资建议** | **分析现有持仓，提供专业建议** |
| **多风格分析** | **策略多样性** | **支持四种投资风格和周期的分析报告** |

---

## 📧 邮件通知

系统支持自动邮件通知，包括：

- 💰 加密货币价格更新
- 📋 港股 IPO 信息
- 📊 主力资金追踪报告
- 📈 恒生指数交易信号
- 🥇 黄金市场分析报告
- 🔄 模拟交易通知（买入、卖出、止损等）
- 📊 AI 交易分析报告
- 💼 **AI 持仓投资分析**
- 📊 **基本面指标展示**
- 📊 **中期评估指标展示**

邮件采用统一的表格化样式，清晰易读。

---

## 📊 数据文件说明

| 文件 | 说明 | 更新频率 |
|------|------|---------|
| `actual_porfolio.csv` | 实际持仓数据 | 手动更新 |
| `all_stock_news_records.csv` | 股票新闻记录 | 按需 |
| `all_dividends.csv` | 所有股息信息记录 | 按需 |
| `recent_dividends.csv` | 最近除净的股息信息 | 按需 |
| `upcoming_dividends.csv` | 即将除净的股息信息（未来90天） | 按需 |
| `hsi_strategy_latest.txt` | 恒生指数策略分析 | 每日 |
| `simulation_state.json` | 模拟交易状态 | 实时 |
| `simulation_transactions.csv` | 交易历史记录 | 实时 |
| `simulation_portfolio.csv` | 投资组合价值变化 | 实时 |
| `simulation_trade_log_*.txt` | 详细交易日志 | 每日 |
| `fundamental_cache/` | 基本面数据缓存 | 7天有效期 |
| `southbound_data_cache.pkl` | 南向资金数据缓存 | 按需 |

---

## 📦 依赖项

```txt
yfinance        # 金融数据获取
requests        # HTTP 请求
pandas          # 数据处理
numpy           # 数值计算
akshare         # 中文财经数据
matplotlib      # 数据可视化
beautifulsoup4  # HTML 解析
openpyxl        # Excel 文件处理
scipy           # 科学计算
schedule        # 定时任务
markdown        # Markdown 转 HTML
lightgbm        # 机器学习模型（LightGBM）
scikit-learn    # 机器学习工具库
```

---

## ⚠️ 注意事项

1. **数据源限制**：部分数据源可能有访问频率限制
2. **缓存机制**：基本面数据缓存 7 天，可手动清除
3. **交易时间**：模拟交易系统遵循港股交易时间（9:30-16:00）
4. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
5. **API 密钥**：请妥善保管 API 密钥，不要提交到版本控制
6. **时区注意**：系统默认使用香港时间进行日期计算
7. **持仓数据**：持仓数据文件 `actual_porfolio.csv` 已提交到 git 仓库，如包含敏感信息请使用 GitHub Secrets

---

## ❓ 常见问题

### Q1: 如何获取 163 邮箱的应用专用密码？

**A**: 
1. 登录 163 邮箱
2. 进入设置 → POP3/SMTP/IMAP
3. 开启 SMTP 服务
4. 生成授权码（应用专用密码）

### Q2: 如何获取大模型 API 密钥？

**A**: 请访问大模型服务商官网注册并申请 API 密钥

### Q3: 支持哪些港股？

**A**: 支持所有在腾讯财经有数据的港股，请在配置文件中添加股票代码

### Q4: 如何清除基本面数据缓存？

**A**: 删除 `data/fundamental_cache/` 目录下的所有文件

### Q5: 模拟交易会真实下单吗？

**A**: 不会，模拟交易系统只在本地记录，不会进行真实交易

### Q6: 如何配置 GitHub Secrets？

**A**: 
1. 进入 GitHub 仓库
2. Settings → Secrets and variables → Actions
3. 点击 New repository secret
4. 添加以下 Secrets：
   - `YAHOO_EMAIL`
   - `YAHOO_APP_PASSWORD`
   - `RECIPIENT_EMAIL`
   - `YAHOO_SMTP`
   - `QWEN_API_KEY`

### Q7: 如何使用 AI 持仓分析功能？

**A**: 
1. 创建 `data/actual_porfolio.csv` 文件
2. 添加持仓数据（股票代码、一手股数、成本价、持有手数）
3. 运行 `python hsi_email.py`
4. 系统会自动读取持仓数据并使用 AI 进行分析
5. 分析结果会显示在邮件的"💼 持仓投资分析（AI智能分析）"部分

### Q8: AI 持仓分析会提供什么信息？

**A**: 
- 整体持仓风险评估（低/中/高）
- 各股投资建议（持有/加仓/减仓/清仓）
- 止损位和目标价建议
- 仓位管理建议
- 风险控制措施
- 交易执行建议
- 基本面指标分析
- 中期评估指标分析

### Q9: 什么是中期评估指标？

**A**: 中期评估指标是用于分析数周至数月投资周期的技术指标，包括：
- **均线排列**：识别多头/空头/混乱排列，判断趋势方向
- **均线斜率**：通过线性回归计算MA20/MA50斜率和角度，判断趋势强度
- **乖离率**：评估价格与均线的偏离程度，识别超买超卖状态
- **支撑阻力位**：基于近期局部高低点识别关键价格水平
- **相对强弱**：计算股票相对于恒生指数的表现
- **中期趋势评分**：综合趋势、动量、支撑阻力、相对强弱四维度评分（0-100分）

### Q10: 如何配置大模型分析风格？

**A**: 在 `hsi_email.py` 文件中修改 `ENABLE_ALL_ANALYSIS_STYLES` 参数：
- `True`：生成全部四种分析（进取型短期、稳健型短期、稳健型中期、保守型中期）
- `False`：只生成两种分析（稳健型短期、稳健型中期）

### Q11: 什么是机器学习交易模型？

**A**: 机器学习交易模型是一个基于LightGBM的二分类模型，用于预测股票次日涨跌。它整合了24个特征，包括技术指标、市场环境和资金流向，平均验证准确率为57.16%。

### Q12: 如何训练机器学习模型？

**A**: 运行以下命令：
```bash
python ml_trading_model.py --mode train
```
模型会使用24只自选股的2年历史数据进行训练，并通过5折时间序列交叉验证。

### Q13: 机器学习模型的准确率如何？

**A**: 平均验证准确率为57.16%（±2.55%），显著高于随机猜测（50%）。5折交叉验证结果分别为：54.50%、54.33%、59.41%、56.91%、60.66%。

### Q14: 如何使用机器学习模型的预测结果？

**A**: 
1. 运行 `python ml_trading_model.py --mode predict` 获取预测结果
2. 查看预测概率，关注高置信度预测（概率 > 60%）
3. 与手工信号对比，只在信号一致时交易（保守策略）
4. 定期跟踪实际准确率，验证模型有效性

### Q15: 策略对比分析有什么作用？

**A**: 策略对比分析用于对比机器学习模型预测与主力资金追踪器手工信号的一致性。通过对比，可以：
- 识别信号一致的股票（更可靠）
- 了解两种方法的差异
- 提供组合使用建议
- 降低投资风险

### Q16: 机器学习模型需要多久重新训练一次？

**A**: 建议每周或每月重新训练一次，以适应市场变化。市场风格切换时，模型可能需要更频繁的更新。

### Q17: 机器学习模型支持哪些股票？

**A**: 目前支持配置在 `ml_trading_model.py` 中的24只自选股。可以在代码中添加或修改股票列表。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交 Issue

如果您发现 bug 或有功能建议，请：
1. 搜索现有的 Issues
2. 创建新的 Issue，详细描述问题
3. 提供复现步骤和环境信息

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加必要的注释和文档字符串
- 确保代码通过测试
- 更新相关文档

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 📮 联系方式

如有问题或建议，请通过 [GitHub Issues](https://github.com/wonglaitung/fortune/issues) 联系。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

Made with ❤️ by [wonglaitung](https://github.com/wonglaitung)

</div>

---

**最后更新**: 2026-01-18