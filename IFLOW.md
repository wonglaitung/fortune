# iFlow 上下文

## ⚠️ 重要警告

> **📖 编程技能规范**：详细的编码规范和开发流程请参见 [`.iflow/commands/programmer_skill.md`](.iflow/commands/programmer_skill.md)，包括"修改完即测试"等核心原则。

**本项目严格遵守以下原则：**
- 修改完即测试，测试通过再继续
- 优先检查是否已有实现
- 公共代码提取优先
- 避免内联重复逻辑

## ⚠️ 机器学习模型验证警告

> **🔴 高准确率不一定意味着好模型，必须严格验证数据泄漏**
>
> 在时间序列预测中，高准确率（>65%）通常是数据泄漏的信号：
> - 检查数据合并时是否使用了 `ignore_index=True`
> - 确保日期索引被保留，数据按时间顺序排列
> - 验证时间序列交叉验证是否严格按时间顺序分割
> - 对比简单基线（恒定预测、动量策略）的准确率
>
> **参考性范围（经验值）：**
> - 随机/平衡二分类基线：≈50%
> - 常见弱信号（简单动量/基准模型）：≈51–55%
> - 有意义的改进/可交易边际：≈55–60%
> - 非常好/罕见：≈60–65%
> - 异常高（需怀疑）：>65%

## 编码规范

> **📖 参考文档**：详细的编程技能规范请参见 [`.iflow/commands/programmer_skill.md`](.iflow/commands/programmer_skill.md)

本项目遵循以下核心编码原则：

1. **🔴 修改完即测试（最高优先级）** - 每次修改后立即验证
   - 使用 `python3 -m py_compile` 进行语法检查
   - 验证修改的功能是否符合预期
   - **只有测试通过后，才能继续下一步**

2. **优先检查是否已有实现** - 搜索项目中是否已有类似功能
3. **公共代码提取优先** - 先新增公共函数，再在当前上下文中调用
4. **避免内联重复逻辑** - 严禁复制粘贴相同或相似的代码
5. **需求分析优先** - 深入理解用户需求，不要急于编码
6. **整体设计思维** - 考虑改动对整个系统的影响

## 目录概览

此目录 (`/data/fortune`) 是一个 Python 项目，包含多个金融信息获取、分析和模拟交易功能：

1. 加密货币价格监控（通过 GitHub Actions 自动发送邮件）
2. 港股 IPO 信息获取（爬取 AAStocks 网站）
3. 港股主力资金追踪器（识别建仓和出货信号）
4. 基于大模型的港股模拟交易系统
5. 批量获取自选股新闻
6. 黄金市场分析器
7. 恒生指数大模型策略分析器
8. 恒生指数价格监控器（含股息信息、基本面、中期评估指标、AI持仓分析）
9. 通用技术分析工具（含中期分析指标系统）
10. 港股基本面数据获取器
11. AI 交易盈利能力分析器
12. **机器学习交易模型**（LightGBM 和 GBDT+LR，支持 1/5/20 天预测）
13. **美股市场数据获取**（标普500、纳斯达克、VIX、美国国债收益率）
14. **港股板块分析模块**（板块涨跌幅排名、技术趋势分析、龙头识别）
15. **板块轮动河流图生成工具**（可视化板块轮动规律）

## 关键文件

### 核心脚本
| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置文件，包含自选股列表（25只股票） |
| `hk_smart_money_tracker.py` | 港股主力资金追踪器 |
| `hsi_email.py` | 恒生指数价格监控器，含AI持仓分析 |
| `simulation_trader.py` | 基于大模型的港股模拟交易系统 |
| `gold_analyzer.py` | 黄金市场分析器 |
| `hsi_llm_strategy.py` | 恒生指数大模型策略分析器 |
| `ai_trading_analyzer.py` | AI 交易盈利能力分析器 |
| `generate_sector_rotation_river_plot.py` | 板块轮动河流图生成工具 |
| `crypto_email.py` | 加密货币价格监控器 |
| `hk_ipo_aastocks.py` | 港股 IPO 信息获取器 |

### 数据服务模块 (`data_services/`)
| 文件 | 说明 |
|------|------|
| `hk_sector_analysis.py` | 港股板块分析模块 |
| `technical_analysis.py` | 通用技术分析工具 |
| `fundamental_data.py` | 港股基本面数据获取器 |
| `tencent_finance.py` | 腾讯财经数据接口 |
| `batch_stock_news_fetcher.py` | 批量获取自选股新闻 |

### 机器学习模块 (`ml_services/`)
| 文件 | 说明 |
|------|------|
| `ml_trading_model.py` | 机器学习交易模型 |
| `ml_prediction_email.py` | 机器学习预测邮件发送器 |
| `us_market_data.py` | 美股市场数据获取模块 |
| `base_model_processor.py` | 模型处理器基类 |
| `compare_models.py` | 模型对比工具 |

### 大模型服务模块 (`llm_services/`)
| 文件 | 说明 |
|------|------|
| `qwen_engine.py` | 大模型服务接口 |
| `sentiment_analyzer.py` | 情感分析模块（四维情感评分） |

### 配置文件
| 文件 | 说明 |
|------|------|
| `requirements.txt` | 项目依赖包列表 |
| `train_and_predict_all.sh` | 完整训练和预测脚本 |
| `send_alert.sh` | 本地定时执行脚本 |
| `update_data.sh` | 数据更新脚本 |

### GitHub Actions 工作流 (`.github/workflows/`)
| 文件 | 说明 | 执行时间 |
|------|------|----------|
| `smart-money-alert.yml` | 主力资金追踪 | 每天 UTC 22:00 |
| `hsi-email-alert-open_message.yml` | 恒生指数邮件 | 周一到周五 UTC 8:00 |
| `crypto-alert.yml` | 加密货币价格 | 每小时 |
| `gold-analyzer.yml` | 黄金市场分析 | 每小时 |
| `ipo-alert.yml` | IPO 信息 | 每天 UTC 2:00 |
| `ai-trading-analysis-daily.yml` | AI 交易分析 | 周一到周五 UTC 8:30 |
| `ml-prediction-alert.yml` | ML 预测 | 周六 UTC 1:00 |

## 项目类型

Python 脚本项目，使用 GitHub Actions 进行自动化调度，包含数据分析、可视化和大模型集成功能。

## 依赖项

```
yfinance, requests, pandas, numpy, akshare, matplotlib,
beautifulsoup4, openpyxl, scipy, schedule, markdown,
lightgbm, scikit-learn
```

## 主要功能

### 港股主力资金追踪
- 批量扫描自选股，分析建仓和出货信号
- 采用业界标准 0-5 层分析框架
- 支持动态投资者类型（进取型/稳健型/保守型）
- 集成 ML 模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- 集成新闻分析和板块分析数据

### 港股板块分析
- 分析 16 个板块（银行、科技、半导体、AI、新能源等）
- 业界标准 MVP 模型识别龙头股
- 支持多周期分析（1日/5日/20日）
- 支持投资风格配置

### 板块轮动河流图
- 可视化展示过去一年板块排名变化
- 含恒生指数对比
- 生成河流图和热力图
- 输出文件：`output/sector_rotation_river_plot.png`

### 恒生指数价格监控器
- 技术分析指标（RSI、MACD、布林带、ATR 等）
- 基本面指标（PE、PB）
- 中期评估指标（均线排列、乖离率、支撑阻力位等）
- AI 智能持仓分析
- 股息信息追踪

### 机器学习交易模型
- **算法**：LightGBM 和 GBDT+LR
- **特征**：2000+ 特征（技术指标、基本面、美股市场、股票类型、情感指标、板块分析、交叉特征）
- **预测周期**：1天、5天、20天
- **性能**：
  - 次日：50.93%-51.51%
  - 一周：53.70%-54.05%
  - 一个月：57.09%-58.00%（接近业界优秀水平 60%）

### 模拟交易系统
- 基于大模型分析的模拟交易
- 支持三种投资者类型
- 止损机制
- 交易记录自动保存

## 配置参数

### 自选股配置（25只）
在 `config.py` 中配置：
- 银行类：0005.HK、0939.HK、1288.HK、1398.HK、3968.HK
- 科技类：0700.HK、1810.HK、3690.HK、9988.HK
- 半导体：0981.HK、1347.HK
- AI：2533.HK、6682.HK、9660.HK
- 能源：0883.HK、1088.HK
- 其他：0388.HK、0728.HK、0941.HK、1138.HK、1211.HK、1299.HK、1330.HK、2269.HK、2800.HK

### 投资者类型
- `aggressive`：进取型，关注动量
- `moderate`：稳健型，平衡分析
- `conservative`：保守型，关注基本面

## 运行命令

### 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 主力资金追踪（默认稳健型）
python hk_smart_money_tracker.py
python hk_smart_money_tracker.py --investor-type aggressive
python hk_smart_money_tracker.py --date 2025-10-25

# 恒生指数监控
python hsi_email.py
python hsi_email.py --date 2025-10-25

# 板块分析
python data_services/hk_sector_analysis.py --period 5 --style moderate

# 板块轮动河流图
python generate_sector_rotation_river_plot.py

# ML 模型训练和预测
./train_and_predict_all.sh
python ml_services/ml_trading_model.py --mode train --horizon 1

# 模拟交易
python simulation_trader.py --investor-type moderate

# AI 交易分析
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 黄金分析
python gold_analyzer.py

# 加密货币监控
python crypto_email.py
```

## 项目架构

```
金融信息监控与智能交易系统
├── 数据获取层
│   ├── 加密货币价格监控器 (crypto_email.py)
│   ├── 港股IPO信息获取器 (hk_ipo_aastocks.py)
│   ├── 黄金市场分析器 (gold_analyzer.py)
│   ├── 美股市场数据获取器 (ml_services/us_market_data.py)
│   └── 腾讯财经数据接口 (data_services/tencent_finance.py)
├── 数据服务层 (data_services/)
│   ├── 基本面数据获取器 (fundamental_data.py)
│   ├── 批量获取自选股新闻 (batch_stock_news_fetcher.py)
│   ├── 港股板块分析器 (hk_sector_analysis.py)
│   ├── 通用技术分析工具 (technical_analysis.py)
│   └── 腾讯财经数据接口 (tencent_finance.py)
├── 分析层
│   ├── 港股主力资金追踪器 (hk_smart_money_tracker.py)
│   ├── 恒生指数大模型策略分析器 (hsi_llm_strategy.py)
│   ├── 恒生指数价格监控器 (hsi_email.py)
│   ├── AI交易盈利能力分析器 (ai_trading_analyzer.py)
│   └── 机器学习模块 (ml_services/)
├── 交易层
│   └── 港股模拟交易系统 (simulation_trader.py)
└── 服务层 (llm_services/)
    ├── 大模型接口 (qwen_engine.py)
    └── 情感分析模块 (sentiment_analyzer.py)
```

## 项目当前状态

**最后更新**: 2026-02-14

**项目成熟度**: 生产就绪

**核心模块状态**:
- ✅ 数据获取层：完整，支持多数据源
- ✅ 数据服务层：完整，模块化架构
- ✅ 分析层：完整，含技术分析、基本面、ML模型
- ✅ 交易层：完整，模拟交易系统正常运行
- ✅ 服务层：完整，大模型服务集成

**ML模型状态**:
- ✅ 次日模型：50.93%-51.51%
- ✅ 一周模型：53.70%-54.05%
- ✅ 一个月模型：57.09%-58.00%（接近业界优秀水平）
- ✅ 数据泄漏检测：已修复

**自动化状态**:
- ✅ GitHub Actions：7个工作流正常运行
- ✅ 邮件通知：163邮箱服务稳定
- ✅ 定时任务：支持本地cron和GitHub Actions

**待优化项**:
- ⚠️ 风险管理模块（VaR、止损止盈、仓位管理）
- ⚠️ 深度学习模型（LSTM、Transformer）
- ⚠️ Web界面

## 大模型集成

- `llm_services/qwen_engine.py` 提供大模型接口
- 支持聊天和嵌入功能
- 集成到主力资金追踪、模拟交易、新闻过滤、黄金分析等模块
- 情感分析模块提供四维情感评分

## 数据文件结构

数据文件存储在 `data/` 目录：
- `actual_porfolio.csv`: 实际持仓数据
- `all_stock_news_records.csv`: 股票新闻记录
- `simulation_transactions.csv`: 交易历史记录
- `simulation_state.json`: 模拟交易状态
- `ml_trading_model_*.pkl`: ML 模型文件（已从 Git 移除）
- `fundamental_cache/`: 基本面数据缓存（已从 Git 移除）

---
最后更新：2026-02-14
