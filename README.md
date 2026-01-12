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

---

## ✨ 核心特性

### 🤖 AI 智能分析
- **大模型集成**：集成 Qwen 大模型，提供智能投资建议
- **持仓分析**：自动分析现有持仓，提供专业的投资建议
- **信号识别**：智能识别买卖信号，减少人工判断
- **策略生成**：基于技术面和基本面生成交易策略

### 📈 技术分析
- **多指标支持**：RSI、MACD、布林带、ATR、CCI、OBV 等
- **VaR 风险价值**：1日、5日、20日 VaR 计算
- **TAV 评分系统**：加权评分提供精准交易信号
- **趋势分析**：自动识别多头、空头、震荡趋势

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
| 恒生指数监控 | `hsi_email.py` | 交易时段 | 价格、技术指标、交易信号、AI持仓分析 |

### 智能分析

| 功能 | 脚本 | 说明 |
|------|------|------|
| 主力资金追踪 | `hk_smart_money_tracker.py` | 识别建仓和出货信号 |
| 恒生指数策略 | `hsi_llm_strategy.py` | 大模型生成交易策略 |
| 48小时信号分析 | `analyze_last_48_hours_email.py` | 连续交易信号识别 |
| AI 交易分析 | `ai_trading_analyzer.py` | 复盘 AI 推荐策略有效性 |

### 模拟交易

| 功能 | 脚本 | 说明 |
|------|------|------|
| 模拟交易系统 | `simulation_trader.py` | 基于大模型的自动交易 |

### 辅助功能

| 功能 | 脚本 | 说明 |
|------|------|------|
| 批量新闻获取 | `batch_stock_news_fetcher.py` | 自选股新闻 |
| 基本面数据 | `fundamental_data.py` | 财务指标、利润表等 |
| 技术分析工具 | `technical_analysis.py` | 通用技术指标计算 |

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

# 恒生指数价格监控（含 AI 持仓分析）
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

### 3. 恒生指数价格监控器（含 AI 持仓分析）

**功能**：
- 实时获取恒生指数价格和交易数据
- 技术指标计算和信号识别
- 只在有交易信号时发送邮件
- 支持历史数据分析
- 集成股息信息追踪功能
- VaR 风险价值计算（1日、5日、20日）
- **AI 智能持仓分析**：读取持仓数据，提供专业投资建议

**使用方法**：
```bash
# 分析当天数据
python hsi_email.py

# 分析指定日期数据
python hsi_email.py --date 2025-10-25
```

**AI 持仓分析功能**：
- 读取 `data/actual_porfolio.csv` 持仓数据
- 使用大模型进行综合投资分析
- 提供整体风险评估
- 各股投资建议（持有/加仓/减仓/清仓）
- 止损位和目标价建议
- 仓位管理建议
- 风险控制措施

### 4. 港股模拟交易系统

**功能**：
- 基于大模型判断进行模拟交易
- 支持保守型、平衡型、进取型投资者偏好
- 严格遵循大模型建议，无随机操作
- 自动止损机制
- 交易记录和状态持久化
- 详细持仓展示和每日总结
- 邮件通知系统

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

### 6. 通用技术分析工具

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

---

## 📁 项目结构

```
fortune/
├── 📄 核心脚本
│   ├── ai_trading_analyzer.py          # AI 交易分析器
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
│   └── data/
│       ├── actual_porfolio.csv         # 实际持仓数据
│       ├── all_stock_news_records.csv  # 股票新闻记录
│       ├── hsi_strategy_latest.txt     # 恒生指数策略分析
│       ├── simulation_state.json       # 模拟交易状态
│       ├── simulation_transactions.csv # 交易历史记录
│       ├── simulation_portfolio.csv    # 投资组合价值
│       ├── simulation_trade_log_*.txt  # 交易日志（按日期分割）
│       └── fundamental_cache/          # 基本面数据缓存
│
├── 📈 图表输出
│   └── hk_smart_charts/                # 主力资金追踪图表
│
└── 🚀 GitHub Actions
    └── .github/workflows/
        ├── crypto-alert.yml             # 加密货币监控
        ├── gold-analyzer.yml            # 黄金分析
        ├── ipo-alert.yml                # IPO 信息
        ├── hsi-email-alert.yml          # 恒生指数监控
        ├── smart-money-alert.yml        # 主力资金追踪
        └── ai-trading-analysis-daily.yml # AI 交易分析
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
│   └── IPO 数据 (AAStocks)
│
├── 🔍 分析层
│   ├── 技术分析 (technical_analysis.py)
│   ├── 主力资金追踪 (hk_smart_money_tracker.py)
│   ├── 48小时信号分析 (analyze_last_48_hours_email.py)
│   ├── AI 交易分析 (ai_trading_analyzer.py)
│   ├── 恒生指数策略 (hsi_llm_strategy.py)
│   └── 新闻过滤 (batch_stock_news_fetcher.py)
│
├── 💹 交易层
│   └── 模拟交易系统 (simulation_trader.py)
│
└── 🤖 服务层
    ├── 大模型服务 (qwen_engine.py)
    ├── 邮件服务 (SMTP)
    └── 数据缓存 (fundamental_cache/)
```

---

## 🤖 大模型集成

项目集成了 Qwen 大模型，提供智能分析和决策支持：

| 功能模块 | 应用场景 | 说明 |
|---------|---------|------|
| 主力资金追踪 | 股票分析 | 识别建仓和出货信号 |
| 模拟交易 | 交易决策 | 生成买卖信号和原因 |
| 新闻过滤 | 相关性判断 | 过滤相关新闻 |
| 黄金分析 | 市场分析 | 深度分析和投资建议 |
| 恒生指数策略 | 策略生成 | 生成交易策略 |
| **持仓分析** | **投资建议** | **分析现有持仓，提供专业建议** |

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

邮件采用统一的表格化样式，清晰易读。

---

## 📊 数据文件说明

| 文件 | 说明 | 更新频率 |
|------|------|---------|
| `actual_porfolio.csv` | 实际持仓数据 | 手动更新 |
| `all_stock_news_records.csv` | 股票新闻记录 | 按需 |
| `hsi_strategy_latest.txt` | 恒生指数策略分析 | 每日 |
| `simulation_state.json` | 模拟交易状态 | 实时 |
| `simulation_transactions.csv` | 交易历史记录 | 实时 |
| `simulation_portfolio.csv` | 投资组合价值变化 | 实时 |
| `simulation_trade_log_*.txt` | 详细交易日志 | 每日 |
| `fundamental_cache/` | 基本面数据缓存 | 7天有效期 |

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

**最后更新**: 2026-01-12