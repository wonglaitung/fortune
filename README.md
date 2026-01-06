# 金融信息监控与模拟交易系统

一个基于 Python 的综合性金融分析系统，集成多种数据源、技术分析工具和大模型智能判断，为投资者提供全面的市场分析和交易策略验证工具。

## 项目概述

本项目旨在帮助投资者：
- 实时监控加密货币、港股、黄金等金融市场
- 识别主力资金动向和交易信号
- 基于大模型进行智能投资决策
- 验证交易策略的有效性
- 获取股息信息和基本面数据

## 核心功能

### 1. 数据获取与监控

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

### 2. 智能分析

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
- 生成详细分析报告

#### 最近48小时连续交易信号分析器 (`analyze_last_48_hours_email.py`)
- 分析连续买入/卖出同一只股票的信号
- 识别连续3次或以上的交易建议
- 智能过滤和建议

### 3. 模拟交易

#### 港股模拟交易系统 (`simulation_trader.py`)
- 基于大模型判断进行模拟交易
- 支持保守型、平衡型、进取型投资者偏好
- 严格遵循大模型建议，无随机操作
- 自动止损机制
- 交易记录和状态持久化
- 详细持仓展示和每日总结
- 邮件通知系统

### 4. 辅助功能

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

## 技术架构

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

## 安装与配置

### 环境要求
- Python 3.10+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境变量配置

创建 `.env` 文件或设置以下环境变量：

```bash
# 邮件配置
YAHOO_EMAIL=your_email@163.com
YAHOO_APP_PASSWORD=your_app_password
YAHOO_SMTP=smtp.163.com
RECIPIENT_EMAIL=recipient1@email.com,recipient2@email.com

# 大模型API配置
QWEN_API_KEY=your_qwen_api_key
```

## 使用方法

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

- **crypto-alert.yml**: 每小时执行加密货币价格监控
- **gold-analyzer.yml**: 每小时执行黄金市场分析
- **ipo-alert.yml**: 每天 UTC 2:00 执行IPO信息获取
- **hsi-email-alert.yml**: 港股交易日交易时段执行恒生指数监控
- **smart-money-alert.yml**: 每天 UTC 22:00 执行主力资金追踪
- **ai-trading-analysis-daily.yml**: 每个交易日收盘后执行AI交易分析

## 项目结构

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
├── llm_services/                   # 大模型服务
│   └── qwen_engine.py
├── data/                           # 数据文件目录
│   ├── all_dividends.csv
│   ├── all_stock_news_records.csv
│   ├── hsi_strategy_latest.txt
│   ├── simulation_state.json
│   ├── simulation_transactions.csv
│   ├── simulation_portfolio.csv
│   └── fundamental_cache/          # 基本面数据缓存
└── .github/workflows/              # GitHub Actions工作流
    ├── crypto-alert.yml
    ├── gold-analyzer.yml
    ├── ipo-alert.yml
    ├── hsi-email-alert.yml
    ├── smart-money-alert.yml
    └── ai-trading-analysis-daily.yml
```

## 核心特性

### 智能量价分析
- **反转型信号**：前一日下跌+当日上涨+成交量放大（≥1.5倍）
- **延续型信号**：前一日上涨+当日继续上涨+成交量放大（≥1.2倍）
- 成交量强度分级：强（≥2.0倍）、中（≥1.5倍）、弱（≥1.2倍）

### 颜色编码系统
- 🟢 绿色：多头趋势/买入信号
- 🔴 红色：空头趋势/卖出信号
- 🟠 橙色：震荡整理/中性信号

### VaR风险价值计算
- 1日VaR：超短线交易（日内/隔夜）
- 5日VaR：波段交易（数天至数周）
- 20日VaR：中长期投资（1个月以上）

### 基本面数据分析
- 财务指标：PE、PB、ROE、ROA、EPS、股息率等
- 盈利能力：营业收入、净利润、增长率
- 资产质量：资产、负债、权益结构
- 现金流：经营、投资、筹资现金流

## 数据文件说明

- `all_dividends.csv`: 所有股息信息记录
- `recent_dividends.csv`: 最近股息信息记录
- `upcoming_dividends.csv`: 即将除净的股息信息
- `all_stock_news_records.csv`: 所有股票相关新闻记录
- `hsi_strategy_latest.txt`: 恒生指数策略分析报告
- `simulation_state.json`: 模拟交易状态
- `simulation_transactions.csv`: 交易历史记录
- `simulation_portfolio.csv`: 投资组合价值变化
- `simulation_trade_log_*.txt`: 详细交易日志（按日期分割）
- `fundamental_cache/`: 基本面数据缓存（7天有效期）

## 配置参数

### 港股主力资金追踪器参数
在 `hk_smart_money_tracker.py` 中可调整：
- `WATCHLIST`: 自选股票列表
- `DAYS_ANALYSIS`: 分析窗口天数
- `VOL_WINDOW`: 成交量分析窗口
- `PRICE_WINDOW`: 价格分析窗口
- `BUILDUP_MIN_DAYS`: 建仓信号最小确认天数
- `DISTRIBUTION_MIN_DAYS`: 出货信号最小确认天数
- 各种阈值参数（价格百分位、量比、南向资金等）

### 模拟交易系统参数
在 `simulation_trader.py` 中可调整：
- `investor_type`: 投资者风险偏好（保守型/平衡型/进取型）
- `initial_capital`: 初始资金（默认100万港元）
- `analysis_frequency`: 执行频率（默认60分钟）

投资者类型说明：
- **保守型**：偏好低风险、稳定收益的股票，如高股息银行股
- **平衡型**：平衡风险与收益，兼顾价值与成长
- **进取型**：偏好高风险、高收益的股票，如科技成长股

## 大模型集成

项目集成了大模型服务，用于智能分析和交易决策：
- 股票分析和投资建议（`hk_smart_money_tracker.py`）
- 交易决策和信号生成（`simulation_trader.py`）
- 新闻相关性过滤（`batch_stock_news_fetcher.py`）
- 黄金市场深度分析（`gold_analyzer.py`）
- 恒生指数策略分析（`hsi_llm_strategy.py`）

## 邮件通知

系统支持自动邮件通知，包括：
- 加密货币价格更新
- 港股IPO信息
- 主力资金追踪报告
- 恒生指数交易信号
- 黄金市场分析报告
- 模拟交易通知（买入、卖出、止损等）
- AI交易分析报告

邮件采用统一的表格化样式，清晰易读。

## 注意事项

1. **数据源限制**：部分数据源可能有访问频率限制
2. **缓存机制**：基本面数据缓存7天，可手动清除
3. **交易时间**：模拟交易系统遵循港股交易时间（9:30-16:00）
4. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
5. **API密钥**：请妥善保管API密钥，不要提交到版本控制

## 依赖项

```
yfinance
requests
pandas
numpy
akshare
matplotlib
beautifulsoup4
openpyxl
scipy
schedule
```

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**最后更新**: 2026-01-06