# 金融信息监控与模拟交易系统

一个基于 Python 的综合性金融分析系统，集成多种数据源、技术分析工具和大模型智能判断，为投资者提供全面的市场分析和交易策略验证工具。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-automated-brightgreen.svg)](https://github.com/features/actions)

## 📋 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [核心特性](#核心特性)
- [大模型集成](#大模型集成)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目旨在帮助投资者：
- 📊 **实时监控**：加密货币、港股、黄金等金融市场
- 🔍 **智能分析**：识别主力资金动向和交易信号
- 🤖 **AI决策**：基于大模型进行智能投资决策
- 📈 **策略验证**：验证交易策略的有效性
- 💰 **股息追踪**：获取股息信息和基本面数据
- 📧 **自动通知**：邮件提醒重要信息和交易信号

## ✨ 核心功能

### 1. 📈 数据获取与监控

#### 加密货币价格监控器 (`crypto_email.py`)
- 实时获取比特币、以太坊价格（USD/HKD）
- 24小时价格变化、市值、交易量
- 技术指标分析（RSI、MACD、均线、布林带等）
- 自动邮件通知（每小时执行）

#### 港股IPO信息获取器 (`hk_ipo_aastocks.py`)
- 爬取AAStocks网站获取最新IPO信息
- 提取公司名称、上市日期、招股价格、入场费等
- 自动邮件通知（每天北京时间10:00）

#### 黄金市场分析器 (`gold_analyzer.py`)
- 获取黄金相关资产和宏观经济数据
- 技术指标分析和趋势判断
- 大模型深度分析和投资建议
- 自动邮件通知（每小时执行）

#### 恒生指数价格监控器 (`hsi_email.py`)
- 实时获取恒生指数价格和交易数据
- 技术指标计算和信号识别
- 只在有交易信号时发送邮件
- 支持历史数据分析
- 集成股息信息追踪功能
- VaR风险价值计算（1日、5日、20日）

### 2. 🧠 智能分析

#### 港股主力资金追踪器 (`hk_smart_money_tracker.py`)
- 批量扫描自选股，识别建仓和出货信号
- 结合股价位置、成交量比率、南向资金流向
- 集成基本面数据分析（财务指标、利润表、资产负债表、现金流量表）
- 大模型智能分析和投资建议
- 生成可视化图表和Excel报告

#### 恒生指数大模型策略分析器 (`hsi_llm_strategy.py`)
- 获取恒生指数数据
- 计算多种技术指标
- 分析市场趋势（强势多头、多头趋势、弱势空头、空头趋势、震荡整理）
- 大模型生成交易策略建议

#### 通用技术分析工具 (`technical_analysis.py`)
- 移动平均线（MA）
- 相对强弱指数（RSI）
- MACD
- 布林带（Bollinger Bands）
- 随机振荡器（Stochastic）
- ATR（平均真实波幅）
- CCI（商品通道指数）
- OBV（能量潮）
- VaR（风险价值）计算

#### 人工智能股票交易盈利能力分析器 (`ai_trading_analyzer.py`)
- 基于交易记录复盘AI推荐策略
- 计算已实现盈亏和未实现盈亏
- 支持多时间维度分析（1天、5天、1个月）
- 显示建议买卖次数和实际执行次数
- 生成详细分析报告

#### 最近48小时连续交易信号分析器 (`analyze_last_48_hours_email.py`)
- 分析连续买入/卖出同一只股票的信号
- 识别连续3次或以上的交易建议
- 智能过滤和建议

### 3. 💹 模拟交易

#### 港股模拟交易系统 (`simulation_trader.py`)
- 基于大模型判断进行模拟交易
- 支持保守型、平衡型、进取型投资者偏好
- 严格遵循大模型建议，无随机操作
- 自动止损机制
- 交易记录和状态持久化
- 详细持仓展示和每日总结
- 邮件通知系统

### 4. 🔧 辅助功能

#### 批量获取自选股新闻 (`batch_stock_news_fetcher.py`)
- 获取自选股最新新闻
- 大模型过滤相关性
- 保存到CSV文件

#### 港股基本面数据获取器 (`fundamental_data.py`)
- 财务指标（PE、PB、ROE、ROA、EPS、股息率等）
- 利润表数据（营业收入、净利润、增长率等）
- 资产负债表数据（资产、负债、权益等）
- 现金流量表数据（经营、投资、筹资现金流）
- 智能缓存机制（7天有效期）

#### 腾讯财经数据接口 (`tencent_finance.py`)
- 提供稳定的港股和恒生指数数据源

## 🏗️ 技术架构

```
金融信息监控与模拟交易系统
├── 数据获取层
│   ├── 加密货币价格监控器 (@crypto_email.py)
│   ├── 港股IPO信息获取器 (@hk_ipo_aastocks.py)
│   ├── 黄金市场分析器 (@gold_analyzer.py)
│   ├── 港股基本面数据获取器 (@fundamental_data.py)
│   └── 港股股息信息追踪器 (@hsi_email.py)
├── 分析层
│   ├── 港股主力资金追踪器 (@hk_smart_money_tracker.py)
│   ├── 批量获取自选股新闻 (@batch_stock_news_fetcher.py)
│   ├── 通用技术分析工具 (@technical_analysis.py)
│   ├── 恒生指数大模型策略分析器 (@hsi_llm_strategy.py)
│   ├── 恒生指数价格监控器 (@hsi_email.py)
│   ├── 最近48小时连续交易信号分析器 (@analyze_last_48_hours_email.py)
│   └── 人工智能股票交易盈利能力分析器 (@ai_trading_analyzer.py)
├── 交易层
│   └── 港股模拟交易系统 (@simulation_trader.py)
└── 服务层
    ├── 大模型服务 (@llm_services/qwen_engine.py)
    └── 腾讯财经数据接口 (@tencent_finance.py)
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- pip

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并编辑环境变量配置文件：

```bash
cp set_key.sh.example set_key.sh
# 编辑 set_key.sh 填入你的配置
```

需要配置的环境变量：

```bash
# 邮件配置
YAHOO_EMAIL=your_email@163.com
YAHOO_APP_PASSWORD=your_app_password
YAHOO_SMTP=smtp.163.com
RECIPIENT_EMAIL=recipient1@email.com,recipient2@email.com

# 大模型API配置
QWEN_API_KEY=your_qwen_api_key
```

### 3. 运行示例

```bash
# 监控加密货币价格
python crypto_email.py

# 追踪港股主力资金
python hk_smart_money_tracker.py

# 启动模拟交易
python simulation_trader.py
```

## 📖 详细使用指南

### 本地运行

#### 加密货币价格监控
```bash
python crypto_email.py
```

#### 港股主力资金追踪
```bash
# 分析当天数据
python hk_smart_money_tracker.py

# 分析指定日期数据
python hk_smart_money_tracker.py --date 2025-10-25
```

#### 港股模拟交易
```bash
# 持续运行
python simulation_trader.py

# 运行指定天数
python simulation_trader.py --duration-days 30
```

#### 黄金市场分析
```bash
# 分析默认周期
python gold_analyzer.py

# 指定分析周期
python gold_analyzer.py --period 6mo
```

#### 恒生指数价格监控
```bash
# 分析当天数据
python hsi_email.py

# 分析指定日期数据
python hsi_email.py --date 2025-10-25
```

#### AI交易分析
```bash
# 分析指定日期
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 分析5天数据
python ai_trading_analyzer.py --start-date 2025-12-31 --end-date 2026-01-05

# 分析1个月数据
python ai_trading_analyzer.py --start-date 2025-12-05 --end-date 2026-01-05

# 不发送邮件
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05 --no-email
```

### 定时执行

#### 使用 send_alert.sh（本地定时执行）
```bash
# 编辑 crontab
crontab -e

# 添加定时任务（每天早上6点执行）
0 6 * * * /data/fortune/send_alert.sh
```

#### GitHub Actions 自动化

项目已配置多个 GitHub Actions 工作流：

| 工作流 | 执行时间 | 功能 |
|--------|---------|------|
| `crypto-alert.yml` | 每小时 | 加密货币价格监控 |
| `gold-analyzer.yml` | 每小时 | 黄金市场分析 |
| `ipo-alert.yml` | 每天 UTC 2:00 | IPO信息获取 |
| `hsi-email-alert.yml` | 港股交易日交易时段 | 恒生指数监控 |
| `smart-money-alert.yml` | 每天 UTC 22:00 | 主力资金追踪 |
| `ai-trading-analysis-daily.yml` | 每个交易日收盘后 | AI交易分析 |

## ⚙️ 配置说明

### 港股主力资金追踪器参数

在 `hk_smart_money_tracker.py` 中可调整：

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

### 模拟交易系统参数

在 `simulation_trader.py` 中可调整：

```python
investor_type = "平衡型"  # 投资者风险偏好（保守型/平衡型/进取型）
initial_capital = 1000000  # 初始资金（默认100万港元）
analysis_frequency = 60  # 执行频率（默认60分钟）
```

**投资者类型说明**：
- **保守型**：偏好低风险、稳定收益的股票，如高股息银行股，注重资本保值
- **平衡型**：平衡风险与收益，兼顾价值与成长，追求稳健增长
- **进取型**：偏好高风险、高收益的股票，如科技成长股，追求资本增值

## 📁 项目结构

```
fortune/
├── ai_trading_analyzer.py          # AI交易分析器
├── analyze_last_48_hours_email.py  # 48小时信号分析器
├── batch_stock_news_fetcher.py     # 批量新闻获取器
├── crypto_email.py                 # 加密货币监控器
├── fundamental_data.py             # 基本面数据获取器
├── gold_analyzer.py                # 黄金市场分析器
├── hk_ipo_aastocks.py              # IPO信息获取器
├── hk_smart_money_tracker.py       # 主力资金追踪器
├── hsi_email.py                    # 恒生指数价格监控器
├── hsi_llm_strategy.py             # 恒生指数策略分析器
├── simulation_trader.py            # 模拟交易系统
├── technical_analysis.py           # 通用技术分析工具
├── tencent_finance.py              # 腾讯财经接口
├── send_alert.sh                   # 本地定时执行脚本
├── update_data.sh                  # 数据更新脚本
├── set_key.sh                      # 环境变量配置
├── requirements.txt                # 项目依赖
├── README.md                       # 项目说明文档
├── IFLOW.md                        # iFlow代理上下文
├── llm_services/                   # 大模型服务
│   └── qwen_engine.py
├── data/                           # 数据文件目录
│   ├── all_dividends.csv
│   ├── recent_dividends.csv
│   ├── upcoming_dividends.csv
│   ├── all_stock_news_records.csv
│   ├── hsi_strategy_latest.txt
│   ├── simulation_state.json
│   ├── simulation_transactions.csv
│   ├── simulation_portfolio.csv
│   ├── simulation_trade_log_*.txt  # 交易日志（按日期分割）
│   └── fundamental_cache/          # 基本面数据缓存
│       ├── 0005_financial_indicator.pkl
│       ├── 0005_income_statement.pkl
│       └── ...
├── hk_smart_charts/                # 主力资金追踪图表
│   ├── 0700.HK_腾讯控股.png
│   └── ...
└── .github/workflows/              # GitHub Actions工作流
    ├── crypto-alert.yml
    ├── gold-analyzer.yml
    ├── ipo-alert.yml
    ├── hsi-email-alert.yml
    ├── smart-money-alert.yml
    └── ai-trading-analysis-daily.yml
```

## 🎨 核心特性

### 智能量价分析

**反转型信号**：
- 前一日下跌 + 当日上涨 + 成交量放大（≥1.5倍）

**延续型信号**：
- 前一日上涨 + 当日继续上涨 + 成交量放大（≥1.2倍）

**成交量强度分级**：
- 强：≥2.0倍
- 中：≥1.5倍
- 弱：≥1.2倍

### 颜色编码系统

| 颜色 | 含义 | 用途 |
|------|------|------|
| 🟢 绿色 | 多头趋势/买入信号 | 上涨趋势、买入建议 |
| 🔴 红色 | 空头趋势/卖出信号 | 下跌趋势、卖出建议 |
| 🟠 橙色 | 震荡整理/中性信号 | 横盘整理、观望 |

### VaR风险价值计算

| VaR类型 | 适用场景 | 时间周期 |
|---------|---------|---------|
| 1日VaR | 超短线交易 | 日内/隔夜 |
| 5日VaR | 波段交易 | 数天至数周 |
| 20日VaR | 中长期投资 | 1个月以上 |

### 基本面数据分析

**财务指标**：
- 估值指标：PE、PB、市值
- 盈利能力：ROE、ROA、净利率、毛利率
- 成长性：EPS、营业收入增长率、净利润增长率
- 收益性：股息率

**利润表数据**：
- 营业收入、营业利润、利润总额、净利润
- 归属于母公司所有者的净利润
- 营业收入增长率、净利润增长率

**资产负债表数据**：
- 资产总计、负债合计、所有者权益合计
- 流动资产合计、流动负债合计
- 资产负债率、流动比率

**现金流量表数据**：
- 经营活动现金流量净额
- 投资活动现金流量净额
- 筹资活动现金流量净额
- 现金及现金等价物净增加额

## 🤖 大模型集成

项目集成了大模型服务，用于智能分析和交易决策：

| 模块 | 功能 | 应用场景 |
|------|------|----------|
| `hk_smart_money_tracker.py` | 股票分析和投资建议 | 主力资金追踪 |
| `simulation_trader.py` | 交易决策和信号生成 | 模拟交易 |
| `batch_stock_news_fetcher.py` | 新闻相关性过滤 | 新闻获取 |
| `gold_analyzer.py` | 黄金市场深度分析 | 黄金分析 |
| `hsi_llm_strategy.py` | 恒生指数策略分析 | 指数策略 |

大模型服务通过 `llm_services/qwen_engine.py` 提供，支持聊天和嵌入功能。

## 📧 邮件通知

系统支持自动邮件通知，包括：

- 💰 加密货币价格更新
- 📋 港股IPO信息
- 📊 主力资金追踪报告
- 📈 恒生指数交易信号
- 🥇 黄金市场分析报告
- 🔄 模拟交易通知（买入、卖出、止损等）
- 📊 AI交易分析报告

邮件采用统一的表格化样式，清晰易读。

## 📊 数据文件说明

| 文件 | 说明 | 更新频率 |
|------|------|---------|
| `all_dividends.csv` | 所有股息信息记录 | 按需 |
| `recent_dividends.csv` | 最近股息信息记录 | 按需 |
| `upcoming_dividends.csv` | 即将除净的股息信息 | 每日 |
| `all_stock_news_records.csv` | 所有股票相关新闻记录 | 按需 |
| `hsi_strategy_latest.txt` | 恒生指数策略分析报告 | 每日 |
| `simulation_state.json` | 模拟交易状态 | 实时 |
| `simulation_transactions.csv` | 交易历史记录 | 实时 |
| `simulation_portfolio.csv` | 投资组合价值变化 | 实时 |
| `simulation_trade_log_*.txt` | 详细交易日志（按日期分割） | 每日 |
| `fundamental_cache/` | 基本面数据缓存 | 7天有效期 |

## ⚠️ 注意事项

1. **数据源限制**：部分数据源可能有访问频率限制
2. **缓存机制**：基本面数据缓存7天，可手动清除
3. **交易时间**：模拟交易系统遵循港股交易时间（9:30-16:00）
4. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
5. **API密钥**：请妥善保管API密钥，不要提交到版本控制
6. **时区注意**：系统默认使用香港时间进行日期计算

## 📦 依赖项

```
yfinance        # 金融数据获取
requests        # HTTP请求
pandas          # 数据处理
numpy           # 数值计算
akshare         # 中文财经数据
matplotlib      # 数据可视化
beautifulsoup4  # HTML解析
openpyxl        # Excel文件处理
scipy           # 科学计算
schedule        # 定时任务
```

## ❓ 常见问题

### Q1: 如何获取163邮箱的应用专用密码？
A: 登录163邮箱 → 设置 → POP3/SMTP/IMAP → 开启SMTP服务 → 生成授权码

### Q2: 如何获取大模型API密钥？
A: 请访问大模型服务商官网注册并申请API密钥

### Q3: 支持哪些港股？
A: 支持所有在腾讯财经有数据的港股，请在配置文件中添加股票代码

### Q4: 如何清除基本面数据缓存？
A: 删除 `data/fundamental_cache/` 目录下的所有文件

### Q5: 模拟交易会真实下单吗？
A: 不会，模拟交易系统只在本地记录，不会进行真实交易

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题或建议，请通过 [GitHub Issues](https://github.com/wonglaitung/fortune/issues) 联系。

---

**最后更新**: 2026-01-07