# <img src="assets/icon.svg" width="40" height="48" alt="金融智能分析" style="vertical-align: middle; margin-right: 10px;"> 金融资产和港股智能分析与交易系统

**⭐ 如果您觉得这个项目有用，请先给项目Star再Fork，以支持项目发展！⭐**

实践**人机混合智能**的理念，开发具备变现能力的金融资产智能量化分析助手。系统整合**大模型智能决策**与**机器学习预测模型**，实时监控加密货币、港股、黄金等金融市场。香港股票方面集成11个数据源，以智能副驾的方式为投资者提供全面的市场分析、交易策略验证和买卖建议。

---

## 📄 效果文档

- [恒生指数及港股交易信号提醒](output/恒生指数及港股交易信号提醒.pdf)
- [【综合分析】港股买卖建议](output/【综合分析】港股买卖建议.pdf)

---

## 📋 目录

- [核心理念](#核心理念)
- [核心功能](#核心功能)
- [机器学习模型](#机器学习模型)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [自动化调度](#自动化调度)
- [性能数据](#性能数据)
- [安装和部署](#安装和部署)
- [依赖项](#依赖项)

---

## 核心理念

本项目的核心理念是**实践人机混合智能**，将大模型的推理能力与机器学习的预测能力相结合：

- **大模型智能决策**：深度分析市场环境、技术指标、基本面数据，生成短期和中期投资建议
- **机器学习预测**：CatBoost 20天模型准确率61.44%，预测个股涨跌方向
- **智能副驾角色**：辅助决策而非替代，最终决策权在投资者手中

**人机协同优势**：全面性（定性+定量）、可靠性（双重验证）、灵活性（适应不同投资风格）、可解释性

---

## 核心功能

### 数据获取与监控

系统整合**11个数据源**：加密货币（CoinGecko）、港股（yfinance/腾讯财经/AKShare）、黄金、美股市场、基本面数据、股息信息、IPO信息、股票新闻

**港股异常检测**：
- 双层检测：Z-Score + Isolation Forest
- 时间间隔：每日和每小时两种模式
- 异常策略：价格异常+当日下跌 = 抄底机会（胜率72%）

### 大模型智能决策

**恒生指数及自选股分析**：
- 六层分析框架：风险控制 → 市场环境 → 基本面 → 技术面 → 信号识别 → 综合决策
- 短期投资分析（日内/数天）：止损位3-5%
- 中期投资分析（数周-数月）：止损位8-12%，含筹码分布分析
- 板块分析：16个板块排名和龙头股识别

**主力资金追踪**：1-6层分析框架，识别建仓和出货信号

**综合分析系统**：每日自动执行，整合大模型建议和CatBoost预测，生成买卖建议

### 风险管理

- VaR（风险价值）和ES（预期损失）计算
- 止损止盈计算（基于ATR或百分比）
- 最大回撤计算
- 风险控制检查（止损/止盈/Trailing Stop）

### 自动化调度

- 12个GitHub Actions工作流全自动运行
- 无需服务器，零成本部署
- 覆盖全天候市场监控和智能分析

**⚠️ 重要提示**：CatBoost 20天模型准确率61.44%，仅供参考，不构成投资建议。本系统仅供学习和研究使用。

---

## 机器学习模型

### CatBoost 模型（推荐）

**性能指标（Walk-forward 验证，12 folds）**：
- 准确率：61.44%（±1.75%）
- 夏普比率：0.97（接近业界标准 1.0）
- 最大回撤：-0.55%（风控优秀）
- 胜率：49.36%
- 索提诺比率：2.52
- 特征数量：918个（技术指标、基本面、美股市场、情感指标、异常检测等）
- 训练时间：1-2分钟

**实用性评估**：80/100，⭐⭐⭐⭐⭐ 强烈推荐实盘

**模型优势**：
- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度
- 更好的泛化能力，减少过拟合
- 稳定性显著提升（标准差1.75%）

### 板块模型（Walk-forward验证）

**消费股板块**：夏普比率0.7445，胜率54.80%，强烈推荐

**银行股板块**：夏普比率0.1546，胜率50.44%，推荐

**半导体股板块**：夏普比率0.1260，胜率49.87%，推荐

### 异常策略验证（两年数据）

基于2024-04-01至2026-04-01的938个港股异常样本验证：

**趋势延续性**：趋势延续率49-51%（接近随机），相关系数-0.10 至 -0.14（均值回归）

**最强抄底信号**：价格异常+当日下跌
- 5天收益率：+4.12%，胜率：72%
- 10天收益率：+6.88%，胜率：73%

**风险预警信号**：Isolation Forest high 异常
- 5天收益率：-3.04%，胜率：43%
- 建议：系统性风险，减仓

### 深度学习模型对比（不推荐）

| 模型 | 准确率 | F1分数 | 推荐指数 |
|------|--------|--------|----------|
| **CatBoost** | **60.88%** | **0.6416** | ⭐⭐⭐⭐⭐ |
| LSTM | 51.79% | 0.0000 | ⭐ |
| Transformer | 51.15% | 0.1303 | ⭐ |

**结论**：CatBoost 远优于深度学习模型，继续使用 CatBoost 单模型作为主要预测模型。

---

## 技术架构

```

数据获取层 → 数据服务层 → 分析层 → 交易层 → 服务层

数据获取层：加密货币、港股、黄金、美股市场、基本面、股息、IPO、新闻

数据服务层：技术分析（RSI、MACD、布林带、ATR等）、基本面分析、板块分析、筹码分布分析

分析层：恒生指数分析、主力资金追踪、AI交易分析、综合分析、ML预测（CatBoost）

交易层：模拟交易系统

服务层：大模型服务（Qwen API）、邮件服务

```

---

## 项目结构

```

fortune/

├── 核心脚本
│   ├── comprehensive_analysis.py       # 综合分析（每日自动执行）
│   ├── detect_stock_anomalies.py       # 港股异常检测
│   ├── hsi_email.py                    # 恒生指数监控
│   ├── hk_smart_money_tracker.py       # 主力资金追踪
│   └── crypto_email.py                 # 加密货币监控
├── 数据服务模块 (data_services/)
│   ├── technical_analysis.py           # 技术分析工具
│   ├── fundamental_data.py             # 基本面数据
│   └── hk_sector_analysis.py           # 板块分析
├── 机器学习模块 (ml_services/)
│   ├── ml_trading_model.py             # ML交易模型
│   ├── batch_backtest.py               # 批量回测
│   └── walk_forward_by_sector.py       # Walk-forward验证
├── 大模型服务 (llm_services/)
│   ├── qwen_engine.py                  # Qwen大模型接口
│   └── sentiment_analyzer.py           # 情感分析
├── 异常检测模块 (anomaly_detector/)
│   ├── zscore_detector.py              # Z-Score检测器
│   └── isolation_forest_detector.py    # Isolation Forest检测器
├── 文档目录 (docs/)
├── 输出文件 (output/)
├── 数据文件 (data/)
└── 配置文件
    ├── config.py                       # 全局配置
    ├── requirements.txt                # 项目依赖
    └── .github/workflows/              # GitHub Actions工作流

```

---

## 自动化调度

系统使用 **GitHub Actions** 进行全自动化调度，无需服务器部署，零硬件成本运行。

### GitHub Actions 工作流（12个）

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| hourly-crypto-monitor.yml | 加密货币监控 | 每小时 |
| stock-anomaly-detection.yml | 港股异常检测 | 每天凌晨2点 |
| hourly-gold-monitor.yml | 黄金监控 | 每小时 |
| hsi-prediction.yml | 恒生指数预测 | 周一到周五 06:00 |
| comprehensive-analysis.yml | 综合分析 | 周一到周五 16:00 |
| batch-stock-news-fetcher.yml | 股票新闻获取 | 每天 |
| daily-ipo-monitor.yml | IPO信息监控 | 每天 |
| daily-ai-trading-analysis.yml | AI交易分析 | 周一到周五 |
| weekly-comprehensive-analysis.yml | 周综合分析 | 每周日 |
| bull-bear-analysis.yml | 牛熊市分析 | 每周日 |
| sector-analysis.yml | 板块表现分析 | 每月1号 |
| performance-monitor.yml | 性能月度报告 | 每月1号 |

**运行成本**：公开仓库无限制，私有仓库每月2000分钟免费（本项目约150-300分钟/月）

**推荐工作流**：
- 模型训练：周末或完市后（数据完整）
- 模型预测：完市后（16:00 HKT）
- 恒指预测：开市前（06:00 HKT）
- 综合分析：完市后（16:00 HKT）

---

## 性能数据

### CatBoost 20天模型（Walk-forward 验证）

**验证方法**：12 folds，每 fold 12个月训练 + 1个月测试

| 指标 | 数值 | 业界标准 | 评估 |
|------|------|---------|------|
| 夏普比率 | 0.97 | >1.0 | 接近标准 |
| 最大回撤 | -0.55% | <-20% | 优秀 |
| 平均收益率 | 9.42%/月 | 1-3% | 偏高⚠️ |
| 胜率 | 49.36% | 52%+ | 略低 |
| 索提诺比率 | 2.52 | >1.0 | 优秀 |

**实用性评估**：80/100，⭐⭐⭐⭐⭐ 强烈推荐实盘

**核心优势**：
- ✅ 最大回撤控制优秀（-0.55%）
- ✅ 正收益月份占比高（92%）
- ✅ 索提诺比率高（2.52），下行风险控制良好

**主要风险**：
- ⚠️ 胜率略低（49% vs 52%）
- ⚠️ 稳定性不足，Fold间波动大
- ⚠️ 月收益 9.42% 偏高，实盘预期应降低

**数据合理性问题**：
- avg_return 使用的是未来第20天的单日收益，不是20天累积收益
- 年化收益 94% 远超业界标准，可能因高置信度阈值筛选效应
- 实盘应考虑交易成本、滑点等，预期收益需打折

### 2024-2026年跨年度回测

- 回测时间：2024-01-02 至 2026-01-02
- 总交易机会：13,457
- 整体准确率：81.53%
- 平均收益率：3.05%（20天持有期）

### 加密货币异常策略验证

**重要发现**：股票异常策略**不适用于加密货币市场**

| 异常类型 | 股票市场（港股） | 加密货币市场 |
|---------|----------------|-------------|
| IF high → 减仓 | ✅ 有效 | ❌ 无效 |
| Z-Score抄底 | ✅ 强烈推荐 | ❌ 高风险 |
| 均值回归 | ✅ 存在 | ❌ 不存在 |

**结论**：加密货币异常只作为监控信号，不进行方向性交易

---

## 注意事项

### 模型性能基准

| 性能等级 | 准确率范围 | 说明 |
|---------|-----------|------|
| 随机/平衡基线 | ≈50% | 随机猜测水平 |
| 有意义的改进 | ≈55-60% | 可交易边际 |
| 非常好/罕见 | ≈60-65% | 优秀模型 |
| 异常高（需怀疑） | >65% | 可能存在数据泄漏 |

### 置信度阈值选择

- 保守型投资者：0.60-0.65（风险控制优先）
- 平衡型投资者：0.55（收益与风险平衡）⭐ 推荐
- 进取型投资者：0.50-0.55（追求更高收益）

### 异常策略注意事项

- ⚠️ "升的继续升，跌的继续跌"假设错误，实际是均值回归
- ✅ **价格异常+当日下跌**是最强抄底信号（胜率72%）
- ✅ **Isolation Forest high 异常**是风险预警信号（建议减仓）
- ❌ **股票异常策略不适用于加密货币**

### 其他注意事项

- 数据验证：严格的时间序列交叉验证，无数据泄漏
- 风险提示：本系统仅供学习和研究使用，不构成投资建议
- API密钥：请妥善保管，不要提交到版本控制

---

## 依赖项

```txt
yfinance        # 金融数据获取
requests        # HTTP请求
pandas          # 数据处理
numpy           # 数值计算
akshare         # 中文财经数据
matplotlib      # 数据可视化
lightgbm        # 机器学习模型（LightGBM）
catboost        # 机器学习模型（CatBoost）主要模型
scikit-learn    # 机器学习工具库
jieba           # 中文分词
nltk            # 自然语言处理
torch           # PyTorch深度学习框架（深度学习模型，需要单独安装，不推荐）
```

---

## 使用指南

### 快速开始

```bash
# 恒生指数及自选股智能分析
python3 hsi_email.py

# 港股异常检测
python3 detect_stock_anomalies.py --mode standalone --mode-type deep

# 综合分析（整合大模型建议和CatBoost预测）
./scripts/run_comprehensive_analysis.sh

# 主力资金追踪
python3 hk_smart_money_tracker.py

# 加密货币异常检测
python3 crypto_email.py --mode deep
```

### 模型训练和预测

```bash
# 训练 CatBoost 20天模型（推荐）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 生成预测
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost

# 批量回测（28只股票）
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6

# 板块Walk-forward验证（业界标准方法）
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20
```

### 进阶使用

详细使用指南请参考 [AGENTS.md](AGENTS.md) 和 [docs/](docs/) 目录。

#### 板块表现分析

```bash
# 使用默认参数（上个月之前的一年）
./scripts/run_sector_analysis.sh

# 自定义日期范围
./scripts/run_sector_analysis.sh 2024-01-01 2025-12-31

# 自定义输出格式（csv/json/markdown/all）
./scripts/run_sector_analysis.sh 2024-01-01 2025-12-31 markdown

# 直接运行分析脚本
python3 ml_services/sector_performance_analysis.py
python3 ml_services/sector_performance_analysis.py --start-date 2024-01-01 --end-date 2025-12-31
python3 ml_services/sector_performance_analysis.py --trades-file output/backtest_20d_trades_20260307_002039.csv
python3 ml_services/sector_performance_analysis.py --output-format all
```

**测试结果示例**：
```
关键发现:
  🏆 表现最佳板块: 生物医药股 (准确率: 91.15%)
  ⚠️  表现最差板块: 公用事业股 (准确率: 69.47%)
  📊 板块数量: 14个
  💰 平均收益率最高: 半导体股 (7.90%)
  🎯 胜率最高: 银行股 (86.37%)
```

#### 股票表现TOP 10排名分析

```bash
# 使用默认参数（上个月之前的一年）
./scripts/run_ranking_analysis.sh

# 自定义日期范围
./scripts/run_ranking_analysis.sh 2024-01-01 2025-12-31

# 自定义输出格式（csv/json/markdown/all）
./scripts/run_ranking_analysis.sh 2024-01-01 2025-12-31 markdown

# 直接运行分析脚本
python3 ml_services/ranking_analysis.py
python3 ml_services/ranking_analysis.py --start-date 2024-01-01 --end-date 2025-12-31
python3 ml_services/ranking_analysis.py --trades-file output/backtest_20d_trades_20260307_002039.csv
python3 ml_services/ranking_analysis.py --output-format all
```

**测试结果示例**：
```
关键发现:
  💰 平均收益率最高: 华虹半导体 (11.91%)
  🎯 胜率最高: 汇丰银行 (75.66%)
  🎯 准确率最高: 汇丰银行 (92.48%)
  🏆 综合优秀股票数量: 7
  📊 分析股票总数: 28只
```

#### 预测性能监控

```bash
# 评估预测准确率
python3 ml_services/performance_monitor.py --mode evaluate --horizon 20

# 生成月度报告
python3 ml_services/performance_monitor.py --mode report --horizon 20

# 评估+报告（推荐）
python3 ml_services/performance_monitor.py --mode all --horizon 20

# 不发送邮件
python3 ml_services/performance_monitor.py --mode all --horizon 20 --no-email
```

**输出文件**：
- `data/prediction_history.json`：预测历史记录
- `output/performance_report_YYYY-MM.md`：月度性能报告

---

### 安装和部署

#### 环境要求

- Python 3.10+
- 操作系统：Linux、macOS、Windows（推荐WSL2）
- 内存：4GB+，磁盘：2GB+

#### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp set_key.sh.sample set_key.sh
# 编辑 set_key.sh，填写邮箱和API密钥
source set_key.sh

# 5. 验证安装
python hsi_email.py --no-email
```

#### 环境变量配置

**必填变量**：

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP服务器地址（如smtp.163.com） |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱授权码（不是登录密码） |
| `RECIPIENT_EMAIL` | 收件人邮箱列表（逗号分隔） |
| `QWEN_API_KEY` | 通义千问API密钥 |

**获取方法**：
- 邮箱授权码：163/Gmail/QQ邮箱设置中生成
- 通义千问API密钥：https://dashscope.aliyun.com/

**注意事项**：
- `set_key.sh` 已添加到 `.gitignore`，不会提交到仓库
- GitHub Actions 需要在 Secrets 中配置相同的环境变量

#### GitHub Actions 自动化部署

**优势**：零成本、零运维、全自动运行

**配置步骤**：
1. Fork本项目到你的GitHub账号
2. 进入仓库 → Settings → Secrets and variables → Actions
3. 添加必填Secrets（SMTP_SERVER、EMAIL_SENDER、EMAIL_PASSWORD、RECIPIENT_EMAIL、QWEN_API_KEY）
4. 启用GitHub Actions工作流

**运行成本**：公开仓库无限制，私有仓库每月2000分钟免费

### 🌟 无服务器部署 - GitHub Actions 自动化

> **适用场景：需要全自动运行、不想维护服务器、或希望零成本部署的用户**

> **⚡ 无需部署服务器，即刻拥有功能完整的金融资产智能量化分析助手**

本项目通过 GitHub Actions 实现全自动化运行，**无需购买服务器、无需维护运维**。

**核心优势**：

| 优势 | 说明 |
|------|------|
| **零成本** | GitHub Actions 免费额度充足，每月2000分钟免费运行时间 |
| **零运维** | 无需服务器维护、无需监控、无需备份 |
| **自动化** | 11个工作流自动运行，覆盖全天候市场监控 |
| **稳定性** | GitHub 提供高可用基础设施，99.9%在线率 |
| **可扩展** | 轻松扩展到更多数据源和分析功能 |
| **安全性** | GitHub Secrets 加密存储环境变量 |

**使用方法**：

**方式一：Fork项目后启用（推荐）**

```bash
# 1. Fork本项目到你的GitHub账号
# 2. 进入你Fork的仓库 → Settings → Secrets and variables → Actions
# 3. 添加以下Secrets（必填）：
#    - EMAIL_SENDER: 你的邮箱地址
#    - EMAIL_PASSWORD: 邮箱授权码
#    - SMTP_SERVER: SMTP服务器地址
#    - RECIPIENT_EMAIL: 收件人邮箱列表（逗号分隔）
#    - QWEN_API_KEY: 通义千问API密钥
# 4. 可选添加以下Secrets（使用默认值）：
#    - QWEN_CHAT_URL: Chat API地址（默认：https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions）
#    - QWEN_CHAT_MODEL: Chat模型名称（默认：qwen-plus-2025-12-01）
#    - MAX_TOKENS: 最大token数（默认：32768）
# 5. 启用GitHub Actions工作流
# 6. 完成！系统将自动运行，分析结果会发送到你的邮箱
```

**方式二：克隆到自己的GitHub仓库**

```bash
# 1. 克隆项目
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 2. 推送到你的GitHub仓库
git remote set-url origin https://github.com/YOUR_USERNAME/fortune.git
git push -u origin main

# 3. 在GitHub仓库中配置Secrets（同方式一）
# 4. 启用GitHub Actions工作流
# 5. 完成！
```

**详细配置步骤**：

1. **配置邮箱服务**：
   - **163邮箱**：设置 → POP3/SMTP/IMAP → 开启POP3/SMTP服务 → 生成授权码
   - **Gmail**：Google账户设置 → 安全性 → 两步验证 → 应用密码 → 生成新密码
   - **QQ邮箱**：设置 → 账户 → POP3/IMAP/SMTP服务 → 生成授权码

2. **配置大模型API**：
   - 访问通义千问官网：https://dashscope.aliyun.com/
   - 注册账号并创建API Key
   - 复制API Key用于配置

3. **添加GitHub Secrets**：
   - 进入仓库 → Settings → Secrets and variables → Actions
   - 点击"New repository secret"
   - 逐个添加以下Secrets：
     - `EMAIL_SENDER`: 你的发件人邮箱
     - `EMAIL_PASSWORD`: 邮箱授权码（不是登录密码）
     - `SMTP_SERVER`: SMTP服务器地址（如smtp.163.com）
     - `RECIPIENT_EMAIL`: 收件人邮箱列表，多个邮箱用逗号分隔
     - `QWEN_API_KEY`: 通义千问API Key

4. **启用工作流**：
   - 进入仓库 → Actions
   - 确认所有工作流已启用
   - 可以查看工作流运行日志

5. **手动触发（可选）**：
   - 进入任一工作流 → Run workflow
   - 选择分支并点击"Run workflow"按钮
   - 等待运行完成，查看结果

**工作流状态监控**：

- **查看运行日志**：进入仓库 → Actions → 选择任一工作流查看运行历史
- **接收分析结果**：所有分析结果会自动发送到 `RECIPIENT_EMAIL` 配置的邮箱

**注意事项**：

- **GitHub Actions 免费额度**：每月2000分钟，对于本项目绰绰有余
- **时区配置**：所有工作流已配置为香港时区，确保运行时间准确
- **数据保密**：使用GitHub Secrets加密存储敏感信息，安全可靠
- **运行频率**：可根据需要调整工作流的触发时间和频率
- **错误通知**：如工作流运行失败，GitHub会自动发送通知

---

## 许可证

MIT License

---

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。

联系邮件：wonglaitung@gmail.com

---

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=wonglaitung/fortune&type=Date)
