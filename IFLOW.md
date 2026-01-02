# iFlow 上下文

## 编码规范
1. 优先检查是否已有实现 
2. 若无，先提议新增公共函数，再在当前上下文中调用
3. 避免内联重复逻辑

## 目录概览

此目录 (`/data/fortune`) 是一个 Python 项目，包含多个金融信息获取、分析和模拟交易功能：
1. 通过 GitHub Actions 自动发送加密货币价格更新邮件
2. 通过爬取AAStocks网站获取香港股市 IPO 信息并发送邮件
3. 港股主力资金追踪器（识别建仓和出货信号）
4. 港股主力资金历史数据分析
5. 基于大模型的港股模拟交易系统
6. 批量获取自选股新闻
7. 黄金市场分析器
8. 恒生指数大模型策略分析器
9. 恒生指数价格监控器
10. 最近48小时连续交易信号分析器
11. 通用技术分析工具
12. 通过腾讯财经接口获取港股数据

## 关键文件

*   `crypto_email.py`: 主脚本，负责获取加密货币价格并通过邮件服务发送邮件。
*   `hk_ipo_aastocks.py`: 通过爬取AAStocks网站获取香港股市IPO信息的脚本。
*   `hk_smart_money_tracker.py`: 港股主力资金追踪器，分析股票的建仓和出货信号。
*   `hk_smart_money_historical_analysis.py`: 港股主力资金历史数据分析器，分析指定时间范围内的历史信号。
*   `simulation_trader.py`: 基于大模型分析的港股模拟交易系统。
*   `batch_stock_news_fetcher.py`: 批量获取自选股新闻脚本。
*   `gold_analyzer.py`: 黄金市场分析器。
*   `hsi_llm_strategy.py`: 恒生指数大模型策略分析器。
*   `hsi_email.py`: 恒生指数价格监控器，基于技术分析指标生成买卖信号，只在有交易信号时发送邮件。
*   `analyze_last_48_hours_email.py`: 最近48小时连续交易信号分析器，分析最近48小时内连续买入/卖出同一只股票的信号。
*   `technical_analysis.py`: 通用技术分析工具，提供多种技术指标计算功能。
*   `tencent_finance.py`: 通过腾讯财经接口获取港股和恒生指数数据。
*   `llm_services/qwen_engine.py`: 大模型服务接口，提供聊天和嵌入功能。
*   `send_alert.sh`: 本地定时执行脚本，按顺序执行个股新闻获取、恒生指数策略分析、主力资金追踪（使用昨天的日期）和黄金分析。
*   `update_data.sh`: 数据更新脚本，将 data 目录下的文件更新到 GitHub。
*   `set_key.sh`: 环境变量配置，包含API密钥和163邮件配置。
*   `requirements.txt`: 项目依赖包列表，包含所有必需的Python库。
*   `.github/workflows/crypto-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `crypto_email.py` 脚本。
*   `.github/workflows/ipo-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `hk_ipo_aastocks.py` 脚本。
*   `.github/workflows/gold-analyzer.yml`: GitHub Actions 工作流文件，用于定时执行 `gold_analyzer.py` 脚本。
*   `.github/workflows/hsi-email-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `hsi_email.py` 脚本。
*   `.github/workflows/smart-money-alert.yml`: 港股主力资金追踪器的GitHub Actions工作流文件，现已整合个股新闻获取和恒生指数策略分析。
*   `.github/workflows/analyze-last-48-hours-email-alert.yml.bak`: 最近48小时连续交易信号分析器的备份工作流文件。
*   `IFLOW.md`: 此文件，提供 iFlow 代理的上下文信息。
*   `README.md`: 项目详细说明文档。

## 项目类型

这是一个 Python 脚本项目，使用 GitHub Actions 进行自动化调度，并包含数据分析、可视化和大模型集成功能，为投资者提供全面的市场分析和交易策略验证工具。

## 依赖项

项目依赖项在 `requirements.txt` 中定义：
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

### 主要功能

#### 加密货币价格监控
1. 从 CoinGecko API 获取比特币 (Bitcoin) 和以太坊 (Ethereum) 的价格信息（美元和港币）、24小时变化率、市值和24小时交易量。
2. 集成通用技术分析工具，计算多种技术指标（移动平均线、RSI、MACD、布林带等）。
3. 识别最近的交易信号（买入/卖出）。
4. 使用 163 邮件服务将获取到的价格信息通过邮件发送给指定收件人。
5. 通过 GitHub Actions 工作流实现定时自动执行（默认每小时执行一次，全天候监控）。
6. **最新修复**：修复了HTML邮件中显示代码片段的问题，确保邮件内容干净整洁。

#### 香港股市 IPO 信息获取
1. 通过爬取 AAStocks 网站获取香港股市 IPO 信息。
2. 提取公司名称、上市日期、行业、招股日期、每手股数、招股价格、入场费、暗盘日期等信息。
3. 将获取到的 IPO 信息通过 163 邮件服务发送给指定收件人。
4. 通过 GitHub Actions 工作流实现定时自动执行（默认每天 UTC 时间 2:00，即北京时间 10:00）。

#### 港股主力资金追踪
1. 批量扫描自选股，分析股票的建仓和出货信号。
2. 结合股价位置、成交量比率、南向资金流向和相对恒生指数的表现进行综合判断。
3. 生成可视化图表和Excel报告。
4. 集成大模型分析股票数据，提供投资建议。
5. 使用腾讯财经接口获取更准确的港股和恒生指数数据。
6. 集成通用技术分析工具，提供全面的技术指标分析。
7. 支持本地定时执行脚本 `send_alert.sh`。

#### 港股主力资金历史数据分析
1. 分析指定时间段内的历史数据。
2. 识别历史上的建仓和出货信号日期。
3. 生成历史信号报告Excel文件。

#### 港股模拟交易系统
1. 基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易。
2. 默认每60分钟执行一次交易分析，频率可配置。
3. 真实调用大模型进行股票分析，并要求以固定JSON格式输出买卖信号。
4. 支持保守型、平衡型、进取型投资者风险偏好设置。
5. 严格按照大模型建议执行交易，无随机操作。
6. 交易记录和状态自动保存，支持中断后继续。
7. 在无法按大模型建议执行交易时（如资金不足或无持仓），会发送邮件通知。
8. 完整的交易日志记录（按日期分割）。
9. 详细的持仓详情展示和每日总结功能。
10. 实现止损机制，根据大模型建议的止损价格自动执行止损操作。
11. **最新功能**：将大模型建议的买卖原因添加到所有通知邮件中，使用户能够更好地理解交易决策的依据。

#### 批量获取自选股新闻
1. 获取自选股的最新新闻。
2. 使用大模型过滤相关新闻，评估新闻与股票的相关性。
3. 按时间排序并保存相关新闻数据到CSV文件。
4. **重要更新**：新闻获取已从 `akshare.stock_news_em` 更改为 `yfinance` 库，以提高可靠性和数据获取成功率。

#### 黄金市场分析器
1. 获取黄金相关资产和宏观经济数据。
2. 进行技术分析，计算各种技术指标（MACD、RSI、均线、布林带等）。
3. 使用大模型进行深度分析，提供投资建议。
4. 通过 GitHub Actions 工作流实现定时自动执行（默认每小时执行一次，全天候监控）。
5. 支持本地定时执行脚本 `send_alert.sh`。
6. 集成通用技术分析工具，提供全面的技术指标分析。

#### 恒生指数大模型策略分析器
1. 通过腾讯财经API获取最新的恒生指数(HSI)数据。
2. 计算多种技术指标（移动平均线、RSI、MACD、布林带、波动率、量比等）。
3. 分析当前市场趋势（强势多头、多头趋势、弱势空头、空头趋势、震荡整理等）。
4. 调用大模型生成明确的交易策略建议。
5. 将策略分析报告保存到`data/hsi_strategy_latest.txt`文件。
6. 通过邮件发送策略分析报告。
7. 支持本地定时执行脚本 `send_alert.sh`。
8. 集成通用技术分析工具，提供全面的技术指标分析。

#### 恒生指数价格监控器
1. 实时获取恒生指数的价格和交易数据。
2. 计算技术指标（RSI、MACD、均线、布林带等）。
3. 识别买卖信号。
4. 只在检测到当天的交易信号时才发送邮件。
5. 通过GitHub Actions自动化调度（港股交易日的交易时段执行）。
6. 包含详细的技术分析指标和市场概览。
7. 发送交易信号提醒邮件，采用统一的表格样式展示。
8. **最新功能**：支持基于指定日期的历史数据分析，所有技术分析都基于截止到指定日期的数据进行计算。
9. **最新功能**：止损价和止盈价显示保留小数点后两位。
10. **最新功能**：在指标说明中增加了ATR(平均真实波幅)指标的解释。
11. **新增功能**：个股分析中每个股票之间增加分割线，提高可读性。
12. **新增功能**：48小时智能建议使用颜色区分（买入绿色，卖出红色）。
13. **新增功能**：将"震荡"趋势颜色统一为橙色。
14. **新增功能**：使用交易记录中的止损价和目标价，而非技术分析计算值。
15. **新增功能**：添加VaR(风险价值)计算，提供1日、5日和20日VaR值。

#### 最近48小时连续交易信号分析器
1. 分析最近48小时内连续买入/卖出同一只股票的信号。
2. 识别连续3次或以上建议买入同一只股票且期间没有卖出建议的情况。
3. 识别连续3次或以上建议卖出同一只股票且期间没有买入建议的情况。
4. 只在检测到符合条件的信号时发送邮件通知。
5. 通过GitHub Actions自动化调度。

#### 通用技术分析工具
1. 实现多种常用技术指标的计算，包括移动平均线、RSI、MACD、布林带、随机振荡器、ATR、CCI、OBV等。
2. 提供趋势分析算法，基于均线排列判断市场趋势。
3. 提供买卖信号生成机制，基于多种技术指标组合判断。
4. 为其他组件提供统一的技术分析接口。
5. 支持多种金融产品（股票、期货、外汇、加密货币等）的技术分析。
6. **新增功能**：VaR(风险价值)计算功能，支持不同投资风格的风险评估：
   - 超短线交易：1日VaR
   - 波段交易：5日VaR
   - 中长期投资：20日VaR

#### 腾讯财经数据接口
1. 通过腾讯财经API获取港股股票数据。
2. 通过腾讯财经API获取恒生指数数据。
3. 提供更稳定和准确的港股数据源。

### 运行和构建

#### 通用依赖安装
在运行任何脚本之前，请确保安装所有依赖：
```bash
pip install -r requirements.txt
```

#### 环境变量配置
所有脚本都需要以下环境变量（在 `set_key.sh` 中配置）：
- `YAHOO_EMAIL`: 163邮箱地址
- `YAHOO_APP_PASSWORD`: 163邮箱应用专用密码
- `YAHOO_SMTP`: 163邮箱SMTP服务器地址（smtp.163.com）
- `RECIPIENT_EMAIL`: 收件人邮箱地址（支持多个收件人，用逗号分隔）
- `QWEN_API_KEY`: 大模型API密钥（部分脚本需要）

#### 加密货币价格监控

##### 本地运行
```bash
python crypto_email.py
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/crypto-alert.yml`
- 执行时间：每小时执行一次（全天候监控）

#### 香港股市 IPO 信息获取

##### 本地运行
```bash
python hk_ipo_aastocks.py
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/ipo-alert.yml`
- 执行时间：每天 UTC 时间 2:00

#### 港股主力资金追踪

##### 本地运行
```bash
# 分析当天数据
python hk_smart_money_tracker.py

# 分析指定日期数据
python hk_smart_money_tracker.py --date 2025-10-25
```

##### 本地定时执行
项目包含 `send_alert.sh` 脚本，可用于本地定时执行：
```bash
# 编辑 crontab
crontab -e

# 添加以下行以每天执行（请根据需要调整时间）
0 6 * * * /data/fortune/send_alert.sh
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/smart-money-alert.yml`
- 执行时间：每天 UTC 时间 22:00（香港时间 06:00）
- 集成脚本：`batch_stock_news_fetcher.py`, `hsi_llm_strategy.py`, `hk_smart_money_tracker.py`

#### 港股主力资金历史数据分析

##### 本地运行
```bash
# 分析默认时间范围
python hk_smart_money_historical_analysis.py

# 分析指定时间范围
python hk_smart_money_historical_analysis.py --start-date 2025-01-01 --end-date 2025-09-30
```

#### 港股模拟交易系统

##### 本地运行
```bash
# 运行模拟交易（默认持续运行）
python simulation_trader.py

# 运行指定天数的模拟交易
python simulation_trader.py --duration-days 30
```

##### 交易执行逻辑
1. 严格按照"先卖后买"的原则执行交易
2. 买入时优先考虑没有持仓的股票，同时支持对已有持仓股票的加仓
3. 严格按照大模型建议的资金分配比例进行投资，避免过度集中投资
4. 根据不同投资者类型（保守型、平衡型、进取型）自动进行盈亏比例交易
5. 根据市场情况自动建议买入股票
6. 每日收盘后生成交易总结报告
7. 支持手工执行卖出操作
8. 实现止损机制，根据大模型建议的止损价格自动执行止损操作
9. **最新功能**：将大模型建议的买卖原因添加到所有通知邮件中

#### 批量获取自选股新闻

##### 本地运行
```bash
# 单次运行
python batch_stock_news_fetcher.py

# 启用定时任务模式
python batch_stock_news_fetcher.py --schedule
```

#### 黄金市场分析器

##### 本地运行
```bash
# 分析默认周期
python gold_analyzer.py

# 指定分析周期
python gold_analyzer.py --period 6mo
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/gold-analyzer.yml`
- 执行时间：每小时执行一次（全天候监控）

#### 恒生指数大模型策略分析器

##### 本地运行
```bash
python hsi_llm_strategy.py
```

#### 恒生指数价格监控器

##### 本地运行
```bash
# 分析当天数据
python hsi_email.py

# 分析指定日期数据
python hsi_email.py --date 2025-10-25
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/hsi-email-alert.yml`
- 执行时间：港股交易日的交易时段（周一到周五，UTC时间 1:30, 3:15, 6:15, 8:15）

#### 最近48小时连续交易信号分析器

##### 本地运行
```bash
python analyze_last_48_hours_email.py
```

#### 通用技术分析工具

##### 本地运行
```bash
python technical_analysis.py
```

### 配置参数

#### 港股主力资金追踪器参数
在`hk_smart_money_tracker.py`代码顶部可以调整以下参数：

- `WATCHLIST`：自选股票列表
- `DAYS_ANALYSIS`：分析窗口天数
- `VOL_WINDOW`：成交量分析窗口
- `PRICE_WINDOW`：价格分析窗口
- `BUILDUP_MIN_DAYS`：建仓信号最小确认天数
- `DISTRIBUTION_MIN_DAYS`：出货信号最小确认天数
- 阈值参数：
  - `PRICE_LOW_PCT`：建仓信号价格百分位阈值
  - `PRICE_HIGH_PCT`：出货信号价格百分位阈值
  - `VOL_RATIO_BUILDUP`：建仓信号量比阈值
  - `VOL_RATIO_DISTRIBUTION`：出货信号量比阈值
  - `SOUTHBOUND_THRESHOLD`：南向资金阈值

#### 模拟交易系统参数
在`simulation_trader.py`文件中可以调整以下参数：

- `investor_type`：投资者风险偏好（"保守型"、"平衡型"、"进取型"）
- `initial_capital`：初始资金（默认100万港元）
- `analysis_frequency`：执行频率（默认每60分钟执行一次交易分析，可根据需要调整）

不同投资者类型的风险偏好设置：
- 保守型：偏好低风险、稳定收益的股票，如高股息银行股，注重资本保值
- 平衡型：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进取型：偏好高风险、高收益的股票，如科技成长股，追求资本增值

#### 个股新闻获取器参数
在`batch_stock_news_fetcher.py`文件中可以使用以下参数：

- `--schedule` 或 `-s`：启用定时任务模式（默认：单次运行）
- 定时任务模式下，程序会在香港时间上午9点和下午1点半各运行一次

### 项目架构

```
金融信息监控与模拟交易系统
├── 数据获取层
│   ├── 加密货币价格监控器 (@crypto_email.py)
│   ├── 港股IPO信息获取器 (@hk_ipo_aastocks.py)
│   └── 黄金市场分析器 (@gold_analyzer.py)
├── 分析层
│   ├── 港股主力资金追踪器 (@hk_smart_money_tracker.py)
│   ├── 港股主力资金历史数据分析 (@hk_smart_money_historical_analysis.py)
│   ├── 批量获取自选股新闻 (@batch_stock_news_fetcher.py)
│   ├── 通用技术分析工具 (@technical_analysis.py)
│   ├── 恒生指数大模型策略分析器 (@hsi_llm_strategy.py)
│   ├── 恒生指数价格监控器 (@hsi_email.py)
│   └── 最近48小时连续交易信号分析器 (@analyze_last_48_hours_email.py)
├── 交易层
│   └── 港股模拟交易系统 (@simulation_trader.py)
└── 服务层
    ├── 大模型服务 (@llm_services/qwen_engine.py)
    └── 腾讯财经数据接口 (@tencent_finance.py)
```

### 大模型集成

项目集成了大模型服务，用于智能分析和交易决策：
- 通过 `llm_services/qwen_engine.py` 提供大模型接口
- 支持聊天和嵌入功能
- 在 `hk_smart_money_tracker.py` 中使用大模型进行股票分析
- 在 `simulation_trader.py` 中使用大模型进行交易决策
- 在 `batch_stock_news_fetcher.py` 中使用大模型过滤相关新闻
- 在 `gold_analyzer.py` 中使用大模型进行黄金市场深度分析
- 在 `hsi_llm_strategy.py` 中使用大模型进行恒生指数策略分析
- **最新功能**：在 `simulation_trader.py` 中，将大模型建议的买卖原因添加到所有通知邮件中

### 数据文件结构

项目生成的数据文件存储在 `data/` 目录中：
- `all_stock_news_records.csv`: 所有股票相关新闻记录
- `hsi_strategy_latest.txt`: 恒生指数策略分析报告
- `simulation_state.json`: 模拟交易状态保存
- `simulation_trade_log_*.txt`: 详细交易日志记录（按日期分割）
- `simulation_transactions.csv`: 交易历史记录
- `simulation_portfolio.csv`: 投资组合价值变化记录

### 项目扩展性

项目目前包含多个独立的功能模块，未来可以：
1. 扩展更多金融信息获取功能
2. 集成更多邮件服务提供商
3. 添加数据存储和历史分析功能
4. 构建 Web 界面展示信息
5. 增加更多的技术分析指标和信号
6. 集成更多大模型服务提供商
7. 增加更多数据源接口（如其他财经网站API）
8. **最新改进**：在模拟交易系统中增加了更详细的交易决策说明，包括买卖原因的邮件通知
9. **新增功能**：集成通用技术分析工具，提供全面的技术指标分析能力
10. **新增功能**：增加恒生指数大模型策略分析器，提供专业的恒生指数交易策略
11. **重要更新**：`batch_stock_news_fetcher.py` 已更新为使用 `yfinance` 库获取新闻
12. **最新更新**：GitHub Actions 工作流整合运行多个脚本，实现更全面的市场分析
13. **新增功能**：新增恒生指数价格监控器和最近48小时连续交易信号分析器
14. **功能增强**：恒生指数价格监控器支持基于指定日期的历史数据分析
15. **配置更新**：邮件服务已统一使用163邮箱，相关配置参数已同步更新
16. **调度优化**：加密货币价格监控和黄金市场分析器已更新为每小时执行一次，提供更及时的市场监控
17. **新增功能**：恒生指数价格监控器增加VaR风险价值计算，提供1日、5日和20日VaR值
18. **UI优化**：个股分析之间增加分割线，提高邮件内容可读性
19. **UI优化**：48小时智能建议使用颜色区分（买入绿色，卖出红色）
20. **UI优化**：统一"震荡"趋势颜色为橙色，保持视觉一致性
21. **功能更新**：使用交易记录中的止损价和目标价，替代技术分析计算值
22. **Bug修复**：修复crypto_email.py中HTML显示代码片段的问题