# iFlow 上下文

## 目录概览

此目录 (`/data/fortune`) 是一个 Python 项目，包含两个主要功能：
1. 通过 GitHub Actions 自动发送加密货币价格更新邮件
2. 获取香港股市 IPO 信息（需要 API key）

## 关键文件

*   `crypto_email.py`: 主脚本，负责获取加密货币价格并通过 Yahoo 邮件服务发送邮件。
*   `hk_ipo_info.py`: 获取香港股市 IPO 信息的脚本（需要聚合数据 API key）。
*   `.github/workflows/crypto-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `crypto_email.py` 脚本。
*   `IFLOW.md`: 此文件，提供 iFlow 代理的上下文信息。

## 项目类型

这是一个 Python 脚本项目，使用 GitHub Actions 进行自动化调度。

### 主要功能

#### 加密货币价格监控
1. 从 CoinGecko API 获取比特币 (Bitcoin) 和以太坊 (Ethereum) 的价格信息（美元和港币）、24小时变化率、市值和24小时交易量。
2. 使用 Yahoo 邮件服务将获取到的价格信息通过邮件发送给指定收件人。
3. 通过 GitHub Actions 工作流实现定时自动执行（默认每天 UTC 时间 0:00 和 8:00，即北京时间 8:00 和 16:00）。

#### 香港股市 IPO 信息获取
1. 从聚合数据 API 获取香港股市 IPO 信息（需要配置 API key）。
2. 目前为命令行工具，可手动执行查看 IPO 信息。

### 运行和构建

#### 加密货币价格监控

##### 本地运行
1. 确保已安装 Python 3.10 或更高版本。
2. 安装依赖:
   ```bash
   pip install requests
   ```
3. 设置环境变量:
   - `YAHOO_EMAIL`: 你的 Yahoo 邮箱地址。
   - `YAHOO_APP_PASSWORD`: 你的 Yahoo 邮箱应用专用密码。
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
   pip install requests
   ```
3. 设置环境变量:
   - `HKEX_API_KEY`: 聚合数据 API key。
4. 运行脚本:
   ```bash
   python hk_ipo_info.py
   ```

### 项目扩展性

项目目前包含两个独立的功能模块，未来可以：
1. 扩展更多金融信息获取功能
2. 集成更多邮件服务提供商
3. 添加数据存储和历史分析功能
4. 构建 Web 界面展示信息