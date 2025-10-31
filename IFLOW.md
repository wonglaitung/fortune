# iFlow 上下文

## 目录概览

此目录 (`/data/fortune`) 是一个 Python 项目，包含多个金融信息获取、分析和模拟交易功能：
1. 通过 GitHub Actions 自动发送加密货币价格更新邮件
2. 通过爬取AAStocks网站获取香港股市 IPO 信息并发送邮件
3. 港股主力资金追踪器（识别建仓和出货信号）
4. 港股主力资金历史数据分析
5. 基于大模型的港股模拟交易系统
6. 批量获取自选股新闻
7. 黄金市场分析器
8. 通过腾讯财经接口获取港股数据

## 关键文件

*   `crypto_email.py`: 主脚本，负责获取加密货币价格并通过邮件服务发送邮件。
*   `hk_ipo_aastocks.py`: 通过爬取AAStocks网站获取香港股市IPO信息的脚本。
*   `hk_smart_money_tracker.py`: 港股主力资金追踪器，分析股票的建仓和出货信号。
*   `hk_smart_money_historical_analysis.py`: 港股主力资金历史数据分析器，分析指定时间范围内的历史信号。
*   `simulation_trader.py`: 基于大模型分析的港股模拟交易系统。
*   `batch_stock_news_fetcher.py`: 批量获取自选股新闻脚本。
*   `gold_analyzer.py`: 黄金市场分析器。
*   `tencent_finance.py`: 通过腾讯财经接口获取港股和恒生指数数据。
*   `llm_services/qwen_engine.py`: 大模型服务接口，提供聊天和嵌入功能。
*   `send_alert.sh`: 本地定时执行脚本，用于执行主力资金追踪和黄金分析。
*   `.github/workflows/crypto-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `crypto_email.py` 脚本。
*   `.github/workflows/ipo-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `hk_ipo_aastocks.py` 脚本。
*   `.github/workflows/gold-analyzer.yml`: GitHub Actions 工作流文件，用于定时执行 `gold_analyzer.py` 脚本。
*   `.github/workflows/smart-money-alert.yml.bak`: 港股主力资金追踪器的GitHub Actions工作流备份文件。
*   `IFLOW.md`: 此文件，提供 iFlow 代理的上下文信息。

## 项目类型

这是一个 Python 脚本项目，使用 GitHub Actions 进行自动化调度，并包含数据分析、可视化和大模型集成功能。

### 主要功能

#### 加密货币价格监控
1. 从 CoinGecko API 获取比特币 (Bitcoin) 和以太坊 (Ethereum) 的价格信息（美元和港币）、24小时变化率、市值和24小时交易量。
2. 使用 Yahoo 邮件服务将获取到的价格信息通过邮件发送给指定收件人。
3. 通过 GitHub Actions 工作流实现定时自动执行（默认每天 UTC 时间 0:00、8:00 和 16:00，即北京时间 8:00、16:00 和 0:00）。

#### 香港股市 IPO 信息获取
1. 通过爬取 AAStocks 网站获取香港股市 IPO 信息。
2. 将获取到的 IPO 信息通过邮件发送给指定收件人。
3. 通过 GitHub Actions 工作流实现定时自动执行（默认每天 UTC 时间 1:00，即北京时间 9:00）。

#### 港股主力资金追踪
1. 批量扫描自选股，分析股票的建仓和出货信号。
2. 结合股价位置、成交量比率、南向资金流向和相对恒生指数的表现进行综合判断。
3. 生成可视化图表和Excel报告。
4. 集成大模型分析股票数据，提供投资建议。
5. 使用腾讯财经接口获取更准确的港股和恒生指数数据。
6. 支持本地定时执行脚本 `send_alert.sh`。

#### 港股主力资金历史数据分析
1. 分析指定时间段内的历史数据。
2. 识别历史上的建仓和出货信号日期。
3. 生成历史信号报告Excel文件。

#### 港股模拟交易系统
1. 基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易。
2. 每15分钟执行一次交易分析。
3. 真实调用大模型进行股票分析，并要求以固定JSON格式输出买卖信号。
4. 支持保守型、平衡型、进取型投资者风险偏好设置。
5. 严格按照大模型建议执行交易，无随机操作。
6. 交易记录和状态自动保存，支持中断后继续。
7. 在无法按大模型建议执行交易时（如资金不足或无持仓），会发送邮件通知。
8. 完整的交易日志记录（按日期分割）。
9. 详细的持仓详情展示和每日总结功能。

#### 批量获取自选股新闻
1. 获取自选股的最新新闻。
2. 使用大模型过滤相关新闻，评估新闻与股票的相关性。
3. 按时间排序并保存相关新闻数据到CSV文件。

#### 黄金市场分析器
1. 获取黄金相关资产和宏观经济数据。
2. 进行技术分析，计算各种技术指标。
3. 使用大模型进行深度分析，提供投资建议。
4. 通过 GitHub Actions 工作流实现定时自动执行（默认每天 UTC 时间 7:00，即北京时间 15:00）。
5. 支持本地定时执行脚本 `send_alert.sh`。

#### 腾讯财经数据接口
1. 通过腾讯财经API获取港股股票数据。
2. 通过腾讯财经API获取恒生指数数据。
3. 提供更稳定和准确的港股数据源。

### 运行和构建

#### 加密货币价格监控

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install requests
   ```
3. 设置环境变量:
   - `YAHOO_EMAIL`: 你的邮箱地址。
   - `YAHOO_APP_PASSWORD`: 你的邮箱应用专用密码。
   - `RECIPIENT_EMAIL`: 收件人邮箱地址（可选，默认为 `wonglaitung@google.com`）。
4. 运行脚本:
   ```bash
   python crypto_email.py
   ```

##### GitHub Actions 自动化
该项目配置了 GitHub Actions 工作流 (`.github/workflows/crypto-alert.yml`)，它会:
1. 在 Ubuntu 最新版本的 runner 上执行。
2. 检出代码。
3. 设置 Python 3.10 环境。
4. 安装 `requests` 依赖。
5. 使用仓库中设置的 secrets (`YAHOO_EMAIL`, `YAHOO_APP_PASSWORD`, `RECIPIENT_EMAIL`) 运行 `crypto_email.py` 脚本。
   
需要在 GitHub 仓库的 secrets 中配置以下环境变量:
- `YAHOO_EMAIL`
- `YAHOO_APP_PASSWORD`
- `RECIPIENT_EMAIL`

#### 香港股市 IPO 信息获取

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install requests beautifulsoup4 pandas
   ```
3. 设置环境变量:
   - `YAHOO_EMAIL`: 你的邮箱地址。
   - `YAHOO_APP_PASSWORD`: 你的邮箱应用专用密码。
   - `RECIPIENT_EMAIL`: 收件人邮箱地址（可选，默认为 `wonglaitung@google.com`）。
4. 运行脚本:
   ```bash
   python hk_ipo_aastocks.py
   ```

##### GitHub Actions 自动化
该项目配置了 GitHub Actions 工作流 (`.github/workflows/ipo-alert.yml`)，它会:
1. 在 Ubuntu 最新版本的 runner 上执行。
2. 检出代码。
3. 设置 Python 3.10 玎境。
4. 安装 `requests`, `beautifulsoup4`, `pandas` 依赖。
5. 使用仓库中设置的 secrets (`YAHOO_EMAIL`, `YAHOO_APP_PASSWORD`, `RECIPIENT_EMAIL`) 运行 `hk_ipo_aastocks.py` 脚本。
   
需要在 GitHub 仓库的 secrets 中配置以下环境变量:
- `YAHOO_EMAIL`
- `YAHOO_APP_PASSWORD`
- `RECIPIENT_EMAIL`

#### 港股主力资金追踪

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install yfinance akshare pandas matplotlib openpyxl scipy schedule
   ```
3. 修改 `WATCHLIST` 字典以包含你想要跟踪的股票。
4. 运行脚本:
   ```bash
   python hk_smart_money_tracker.py
   ```
5. 运行指定日期的分析:
   ```bash
   python hk_smart_money_tracker.py --date 2025-10-25
   ```
6. 查看生成的Excel报告 `hk_smart_money_report.xlsx` 和图表 `hk_smart_charts/` 目录。

##### 本地定时执行
项目包含 `send_alert.sh` 脚本，可用于本地定时执行:
```bash
# 编辑 crontab
crontab -e

# 添加以下行以每天执行（请根据需要调整时间）
0 6 * * * /data/fortune/send_alert.sh
```

##### GitHub Actions 自动化
该项目配置了 GitHub Actions 工作流 (`.github/workflows/smart-money-alert.yml.bak`)，它会:
1. 在 Ubuntu 最新版本的 runner 上执行。
2. 检出代码。
3. 设置 Python 3.10 玎境。
4. 安装 `yfinance`, `akshare`, `pandas`, `matplotlib`, `openpyxl`, `scipy`, `schedule` 依赖。
5. 使用仓库中设置的 secrets (`YAHOO_EMAIL`, `YAHOO_APP_PASSWORD`, `RECIPIENT_EMAIL`, `QWEN_API_KEY`) 运行 `hk_smart_money_tracker.py` 脚本。
   
需要在 GitHub 仓库的 secrets 中配置以下环境变量:
- `YAHOO_EMAIL`
- `YAHOO_APP_PASSWORD`
- `RECIPIENT_EMAIL`
- `QWEN_API_KEY`

#### 港股主力资金历史数据分析

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install yfinance akshare pandas matplotlib openpyxl scipy
   ```
3. 运行脚本:
   ```bash
   python hk_smart_money_historical_analysis.py
   ```
4. 运行指定时间范围的分析:
   ```bash
   python hk_smart_money_historical_analysis.py --start-date 2025-01-01 --end-date 2025-09-30
   ```
5. 查看生成的Excel报告 `hk_smart_money_historical_report.xlsx`。

#### 港股模拟交易系统

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install yfinance akshare pandas matplotlib openpyxl scipy schedule
   ```
3. 设置环境变量:
   - `QWEN_API_KEY`: 大模型API密钥。
4. 运行脚本:
   ```bash
   python simulation_trader.py
   ```
5. 运行指定天数的模拟交易:
   ```bash
   python simulation_trader.py --duration-days 30
   ```
6. 查看生成的模拟交易文件:
   - `data/simulation_state.json`: 保存交易状态，支持中断后继续
   - `data/simulation_trade_log_*.txt`: 详细交易日志记录（按日期分割）
   - `data/simulation_transactions.csv`: 交易历史记录
   - `data/simulation_portfolio.csv`: 投资组合价值变化记录

##### 交易执行逻辑
1. 严格按照"先卖后买"的原则执行交易
2. 买入时优先考虑没有持仓的股票
3. 在无法按大模型建议执行交易时（如资金不足或无持仓），会发送邮件通知
4. 根据不同投资者类型（保守型、平衡型、进取型）自动进行盈亏比例交易
5. 根据市场情况自动建议买入股票
6. 每日收盘后生成交易总结报告
7. 支持手工执行卖出操作

#### 批量获取自选股新闻

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install akshare yfinance
   ```
3. 运行脚本:
   ```bash
   python batch_stock_news_fetcher.py
   ```
4. 启用定时任务模式:
   ```bash
   python batch_stock_news_fetcher.py --schedule
   ```
5. 查看生成的新闻数据文件:
   - `data/all_stock_news_records.csv`: 所有股票相关新闻记录

#### 黄金市场分析器

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install yfinance
   ```
3. 运行脚本:
   ```bash
   python gold_analyzer.py
   ```
4. 指定分析周期:
   ```bash
   python gold_analyzer.py --period 6mo
   ```

##### 本地定时执行
项目包含 `send_alert.sh` 脚本，可用于本地定时执行:
```bash
# 编辑 crontab
crontab -e

# 添加以下行以每天执行（请根据需要调整时间）
0 6 * * * /data/fortune/send_alert.sh
```

##### GitHub Actions 自动化
该项目配置了 GitHub Actions 工作流 (`.github/workflows/gold-analyzer.yml`)，它会:
1. 在 Ubuntu 最新版本的 runner 上执行。
2. 检出代码。
3. 设置 Python 3.10 玎境。
4. 安装 `yfinance`, `requests`, `pandas`, `numpy` 依赖。
5. 使用仓库中设置的 secrets (`YAHOO_EMAIL`, `YAHOO_APP_PASSWORD`, `RECIPIENT_EMAIL`, `QWEN_API_KEY`) 运行 `gold_analyzer.py` 脚本。
   
需要在 GitHub 仓库的 secrets 中配置以下环境变量:
- `YAHOO_EMAIL`
- `YAHOO_APP_PASSWORD`
- `RECIPIENT_EMAIL`
- `QWEN_API_KEY`

#### 腾讯财经数据接口

##### 本地运行
腾讯财经数据接口被其他脚本调用，无需单独运行。

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
│   └── 批量获取自选股新闻 (@batch_stock_news_fetcher.py)
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

### 项目扩展性

项目目前包含多个独立的功能模块，未来可以：
1. 扩展更多金融信息获取功能
2. 集成更多邮件服务提供商
3. 添加数据存储和历史分析功能
4. 构建 Web 界面展示信息
5. 增加更多的技术分析指标和信号
6. 集成更多大模型服务提供商
7. 增加更多数据源接口（如其他财经网站API）