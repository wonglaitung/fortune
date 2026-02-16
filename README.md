# 金融信息监控与智能交易系统

一个基于 Python 的综合性金融分析系统，集成多数据源、技术分析工具和大模型智能判断，为投资者提供全面的市场分析和交易策略验证工具。

## 项目简介

本项目旨在帮助投资者：
- 📊 实时监控加密货币、港股、黄金等金融市场
- 🔍 识别主力资金动向和交易信号
- 🤖 基于大模型进行智能投资决策和持仓分析
- 📈 验证交易策略的有效性
- 💰 获取股息信息和基本面数据
- 📧 自动邮件通知重要信息

## 核心功能

### 数据获取与监控
- **加密货币监控**：比特币、以太坊价格和技术分析（每小时）
- **港股IPO信息**：最新IPO信息（每天）
- **黄金市场分析**：黄金价格和投资建议（每小时）
- **恒生指数监控**：价格、技术指标、交易信号（交易时段）
- **美股市场数据**：标普500、纳斯达克、VIX、美国国债收益率
- **基本面数据**：财务指标、利润表、资产负债表、现金流量表
- **股息信息**：自动获取股息和除净日信息

### 智能分析
- **主力资金追踪**：识别建仓和出货信号，集成基本面分析
- **板块分析**：16个板块涨跌幅排名、技术趋势分析、龙头识别
- **板块轮动河流图**：可视化板块排名变化
- **恒生指数策略**：大模型生成交易策略
- **AI交易分析**：复盘AI推荐策略有效性
- **综合分析系统**：整合大模型建议和ML预测结果，生成实质买卖建议

### 机器学习
- **多周期预测**：基于LightGBM和GBDT+LR预测1天、5天、20天后的涨跌
- **2936特征**：技术指标、基本面、美股市场、情感指标、板块分析、长期趋势
- **最新准确率**：51.70%（次日）→ 54.64%（一周）→ 58.97%（一个月）
- **特征重要性分析**：提供模型可解释性

### 模拟交易
- **真实模拟**：基于大模型建议的模拟交易系统
- **风险控制**：自动止损机制
- **详细记录**：完整的交易日志和持仓分析
- **多种策略**：支持保守型、平衡型、进取型投资偏好

## 快速开始

### 环境要求
- Python 3.10 或更高版本
- pip 包管理器

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# 编辑 set_key.sh 文件，设置邮件和大模型API密钥
source set_key.sh
```

### 使用示例

```bash
# 监控加密货币价格
python crypto_email.py

# 追踪港股主力资金
python hk_smart_money_tracker.py

# 板块分析
python data_services/hk_sector_analysis.py

# 恒生指数价格监控
python hsi_email.py

# 启动模拟交易
python simulation_trader.py

# 训练机器学习模型
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type both

# 预测股票涨跌
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type both

# 综合分析（整合大模型建议和ML预测）
python comprehensive_analysis.py
```

## 项目结构

```
fortune/
├── 核心脚本
│   ├── ai_trading_analyzer.py          # AI交易分析器
│   ├── crypto_email.py                 # 加密货币监控器
│   ├── gold_analyzer.py                # 黄金市场分析器
│   ├── hk_ipo_aastocks.py              # IPO信息获取器
│   ├── hk_smart_money_tracker.py       # 主力资金追踪器
│   ├── hsi_email.py                    # 恒生指数监控器
│   ├── simulation_trader.py            # 模拟交易系统
│   ├── comprehensive_analysis.py       # 综合分析脚本
│   └── ...
│
├── 数据服务模块 (data_services/)
│   ├── technical_analysis.py           # 通用技术分析工具
│   ├── fundamental_data.py             # 基本面数据获取器
│   ├── hk_sector_analysis.py           # 板块分析器
│   └── ...
│
├── 机器学习模块 (ml_services/)
│   ├── ml_trading_model.py             # 机器学习交易模型
│   ├── ml_prediction_email.py          # 预测邮件发送器
│   ├── us_market_data.py               # 美股市场数据
│   └── ...
│
├── 大模型服务 (llm_services/)
│   ├── qwen_engine.py                  # Qwen大模型接口
│   └── sentiment_analyzer.py           # 情感分析模块
│
├── 配置文件
│   ├── config.py                       # 全局配置
│   ├── requirements.txt                # 项目依赖
│   └── set_key.sh                      # 环境变量配置
│
└── 数据文件 (data/)
    ├── actual_porfolio.csv             # 实际持仓数据
    ├── ml_trading_model_*.pkl          # 机器学习模型
    └── ...
```

## 技术架构

```
金融信息监控与智能交易系统
│
├── 数据获取层
│   ├── 加密货币数据 (CoinGecko)
│   ├── 港股数据 (yfinance, 腾讯财经, AKShare)
│   ├── 黄金数据 (yfinance)
│   ├── 基本面数据 (AKShare)
│   └── 美股市场数据 (yfinance)
│
├── 数据服务层
│   ├── 技术分析 (RSI、MACD、布林带、ATR等)
│   ├── 基本面分析
│   ├── 板块分析
│   └── 新闻过滤
│
├── 分析层
│   ├── 主力资金追踪
│   ├── AI交易分析
│   ├── 机器学习模型
│   └── 综合分析
│
├── 交易层
│   └── 模拟交易系统
│
└── 服务层
    ├── 大模型服务
    └── 邮件服务
```

## 依赖项

```txt
yfinance        # 金融数据获取
requests        # HTTP请求
pandas          # 数据处理
numpy           # 数值计算
akshare         # 中文财经数据
matplotlib      # 数据可视化
lightgbm        # 机器学习模型
scikit-learn    # 机器学习工具库
```

## 自动化

系统使用 GitHub Actions 进行自动化调度：

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| crypto-alert.yml | 加密货币监控 | 每小时 |
| hsi-email-alert-open_message.yml | 恒生指数监控 | 交易时段 |
| smart-money-alert.yml | 主力资金追踪 | 每天 UTC 22:00 |
| ml-prediction-alert.yml | 综合分析 | 每周日 UTC 01:00 |

## 注意事项

1. **数据源限制**：部分数据源可能有访问频率限制
2. **缓存机制**：基本面数据缓存7天，可手动清除
3. **交易时间**：模拟交易系统遵循港股交易时间
4. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
5. **API密钥**：请妥善保管API密钥，不要提交到版本控制

## 项目状态

- ✅ 核心功能完整且稳定运行
- ✅ 模块化架构易于维护和扩展
- ✅ 机器学习模型经过严格验证
- ✅ 自动化调度系统稳定运行
- ⚠️ 风险管理模块可进一步完善

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。