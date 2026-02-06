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

> **📖 参考文档**：详细的编程技能规范请参见 [`.iflow/commands/programmer_skill.md`](.iflow/commands/programmer_skill.md)，包含需求分析、整体设计、公共代码提取、修改完即测试等完整开发流程。

本项目遵循以下核心编码原则：

1. **🔴 修改完即测试（最高优先级）** - 每次修改后立即验证，避免累积错误
   - 使用 `python3 -m py_compile` 进行语法检查
   - 验证修改的功能是否符合预期
   - 确保没有破坏现有功能
   - **只有测试通过后，才能继续下一步**

2. **优先检查是否已有实现** - 在开始编码前，先搜索项目中是否已有类似功能或可复用的代码
3. **公共代码提取优先** - 若无现有实现，先新增公共函数，再在当前上下文中调用
4. **避免内联重复逻辑** - 严禁复制粘贴相同或相似的代码
5. **需求分析优先** - 深入理解用户需求，不要急于编码
6. **整体设计思维** - 考虑改动对整个系统的影响
7. **可维护性优先** - 考虑长期维护和扩展性

## 目录概览

此目录 (`/data/fortune`) 是一个 Python 项目，包含多个金融信息获取、分析和模拟交易功能：
1. 通过 GitHub Actions 自动发送加密货币价格更新邮件
2. 通过爬取AAStocks网站获取香港股市 IPO 信息并发送邮件
3. 港股主力资金追踪器（识别建仓和出货信号）
4. 基于大模型的港股模拟交易系统
5. 批量获取自选股新闻
6. 黄金市场分析器
7. 恒生指数大模型策略分析器
8. 恒生指数价格监控器（含股息信息追踪、基本面指标、中期评估指标和AI持仓分析）
9. 通用技术分析工具（含中期分析指标系统）
10. 通过腾讯财经接口获取港股数据
11. 人工智能股票交易盈利能力分析器
12. 港股基本面数据获取器（财务指标、利润表、资产负债表、现金流量表）
13. **机器学习交易模型**（基于LightGBM和GBDT+LR的多周期涨跌预测模型，支持1天、5天、20天预测，集成股票类型特征、情感指标、技术-基本面交互特征）
14. **美股市场数据获取**（标普500、纳斯达克、VIX恐慌指数、美国国债收益率）
15. **模型可解释性分析**（GBDT决策路径解析、特征重要性分析）
16. **机器学习预测邮件通知**（自动发送ML模型预测结果邮件）
17. **情感分析模块**（llm_services/sentiment_analyzer.py，提供四维情感评分：相关性、影响度、预期差、情感方向）
18. **不同股票类型分析框架对比**（hsi_email.py vs hk_smart_money_tracker.py的详细对比）
19. **港股板块分析模块**（hk_sector_analysis.py，提供板块涨跌幅排名、技术趋势分析、龙头识别、资金流向分析）

## 关键文件

*   `crypto_email.py`: 主脚本，负责获取加密货币价格并通过邮件服务发送邮件。
*   `hk_ipo_aastocks.py`: 通过爬取AAStocks网站获取香港股市IPO信息的脚本。
*   `hk_smart_money_tracker.py`: 港股主力资金追踪器，分析股票的建仓和出货信号。
*   `simulation_trader.py`: 基于大模型分析的港股模拟交易系统。
*   `gold_analyzer.py`: 黄金市场分析器。
*   `hsi_llm_strategy.py`: 恒生指数大模型策略分析器。
*   `hsi_email.py`: 恒生指数价格监控器，基于技术分析指标生成买卖信号，只在有交易信号时发送邮件，集成股息信息追踪功能、基本面指标、中期评估指标和AI持仓分析功能。
*   `ai_trading_analyzer.py`: 人工智能股票交易盈利能力分析器，用于评估AI交易信号的有效性。
*   `config.py`: **全局配置文件**，包含自选股列表（26只股票，包括汇丰银行、腾讯控股、阿里巴巴、比亚迪股份、友邦保险等）。
*   `data_services/__init__.py`: 数据服务模块初始化文件，使 data_services 成为 Python 包。
*   `data_services/batch_stock_news_fetcher.py`: 批量获取自选股新闻脚本。
*   `data_services/fundamental_data.py`: 港股基本面数据获取模块，提供财务指标、利润表、资产负债表、现金流量表等数据获取功能，支持缓存机制。
*   `data_services/hk_sector_analysis.py`: **港股板块分析模块**，提供板块涨跌幅排名、技术趋势分析、龙头识别、资金流向分析。
*   `data_services/technical_analysis.py`: 通用技术分析工具，提供多种技术指标计算功能，包括VaR风险价值计算、TAV加权评分系统和中期分析指标系统。
*   `data_services/tencent_finance.py`: 通过腾讯财经接口获取港股和恒生指数数据。
*   `llm_services/qwen_engine.py`: 大模型服务接口，提供聊天和嵌入功能。
*   `llm_services/sentiment_analyzer.py`: **情感分析模块**，使用大模型对新闻进行四维情感评分（相关性、影响度、预期差、情感方向），支持批量分析和情感统计。
*   `ml_services/ml_trading_model.py`: **机器学习交易模型**，基于LightGBM和GBDT+LR的二分类模型，预测1天、5天、20天后的涨跌，整合技术指标、基本面、资金流向、美股市场、股票类型、情感指标、技术-基本面交互特征。
*   `ml_services/ml_prediction_email.py`: **机器学习预测邮件发送器**，自动发送ML模型预测结果邮件。
*   `ml_services/compare_models.py`: **模型对比工具**，对比LGBM和GBDT+LR两种模型的预测结果。
*   `ml_services/us_market_data.py`: **美股市场数据获取模块**，提供标普500、纳斯达克、VIX恐慌指数、美国10年期国债收益率等数据。
*   `ml_services/base_model_processor.py`: **模型处理器基类**，提供模型训练、特征重要性分析、GBDT决策路径解析等功能。
*   `train_and_predict_all.sh`: **完整训练和预测脚本**，支持1天、5天、20天三个周期的模型训练和预测，支持历史回测。
*   `send_alert.sh`: 本地定时执行脚本，按顺序执行个股新闻获取、恒生指数策略分析、主力资金追踪（使用昨天的日期）和黄金分析。
*   `update_data.sh`: 数据更新脚本，将 data 目录下的文件更新到 GitHub（带重试机制）。
*   `set_key.sh`: 环境变量配置，包含API密钥和163邮件配置。
*   `requirements.txt`: 项目依赖包列表，包含所有必需的Python库（包含lightgbm、scikit-learn、yfinance）。
*   `.github/workflows/crypto-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `crypto_email.py` 脚本。
*   `.github/workflows/ipo-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `hk_ipo_aastocks.py` 脚本。
*   `.github/workflows/gold-analyzer.yml`: GitHub Actions 工作流文件，用于定时执行 `gold_analyzer.py` 脚本。
*   `.github/workflows/hsi-email-alert.yml`: GitHub Actions 工作流文件，用于定时执行 `hsi_email.py` 脚本。
*   `.github/workflows/hsi-email-alert-open_message.yml`: **新增工作流**，强制发送恒生指数邮件（支持 --force 参数），执行时间：周一到周五下午6:00香港时间（UTC 8:00）。
*   `.github/workflows/smart-money-alert.yml`: 港股主力资金追踪器的GitHub Actions工作流文件，现已整合个股新闻获取和恒生指数策略分析。
*   `.github/workflows/ml-prediction-alert.yml`: **机器学习预测警报工作流**，自动执行ML模型预测并发送邮件通知。
*   `.github/workflows/ai-trading-analysis-daily.yml`: AI交易分析日报的GitHub Actions工作流文件，在每个交易日完成后自动运行1天、5天和1个月的分析报告。
*   `IFLOW.md`: 此文件，提供 iFlow 代理的上下文信息。
*   `README.md`: 项目详细说明文档。
*   `不同股票类型分析框架对比.md`: **不同股票类型分析框架对比文档**，详细说明hsi_email.py和hk_smart_money_tracker.py的核心差异、设计目标、分析框架、关键指标、信号生成机制、适用场景和集成策略。
*   `ml_model_analysis_report.md`: **机器学习模型与业界最佳实践差距分析报告**，详细分析当前模型与业界领先机构的差距，提供改进路线图和优先级建议。
*   `hk_smart_charts/`: 港股主力资金追踪器生成的可视化图表目录。
*   `output/`: 输出文件目录，包含模型分析结果、回测报告等。

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
markdown
lightgbm
scikit-learn
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
8. 集成基本面数据分析，结合财务指标进行综合判断。
9. **新增功能**：集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率），提升分析准确性。
10. **新增功能**：采用业界标准的0-5层分析框架（前置检查→风险控制→核心信号识别→技术面分析→宏观环境→辅助信息），提供系统性分析。
11. **新增功能**：在第三层"核心技术指标分析"中集成情感指标分析（情感MA3/MA7/MA14、波动率、变化率），与RSI、MACD、布林带等技术指标协同工作。
12. **新增功能**：在技术指标展示中添加VIX恐慌指数、成交额变化率（1日/5日/20日）、换手率变化率（5日/20日）等关键流动性指标。
13. **最新功能**：添加动态投资者类型支持，支持aggressive（进取型）、moderate（稳健型）、conservative（保守型）三种类型。
14. **最新功能**：集成新闻分析功能，从 `data/all_stock_news_records.csv` 读取最新新闻，在第六层分析框架中辅助决策。
15. **最新功能**：新闻权重根据投资者类型动态调整（进取型10%、稳健型20%、保守型30%）。
16. **情感分析模块**：基于大模型的情感分析（`llm_services/sentiment_analyzer.py`），提供四维度情感评分（相关性、影响度、预期差、情感方向），支持情感MA3/MA7/MA14、波动率、变化率计算。
17. **情感指标集成**：在ML模型中集成情感指标特征（6个基础特征+78个交互特征），从新闻数据中计算情感趋势特征，提升模型预测能力。
18. **技术-基本面交互特征**：在ML模型中添加18个技术指标与基本面的交互特征（如RSI×PE、MACD×PB等），捕捉非线性关系，提高模型预测准确率。

#### 港股板块分析
1. 批量分析13个板块（银行、科技、半导体、AI、新能源、环保、能源、航运、交易所、公用事业、保险、生物医药、指数基金）
2. 计算板块涨跌幅排名，识别强势和弱势板块
3. 分析板块技术趋势（强势上涨、温和上涨、震荡整理、温和下跌、强势下跌）
4. 识别板块龙头股票（按涨跌幅和成交量综合评分）
5. 分析板块资金流向（基于成交量和涨跌幅）
6. 生成板块分析报告，包括强势板块TOP 3和弱势板块BOTTOM 3
7. 使用腾讯财经接口获取板块内股票数据
8. 集成到 hk_smart_money_tracker.py 和 hsi_email.py 中，为大模型分析提供板块背景信息

#### 港股模拟交易系统
1. 基于hk_smart_money_tracker的分析结果和大模型判断进行模拟交易。
2. 默认每60分钟执行一次交易分析，频率可配置。
3. 真实调用大模型进行股票分析，并要求以固定JSON格式输出买卖信号。
4. 支持aggressive（进取型）、moderate（稳健型）、conservative（保守型）投资者风险偏好设置。
5. 严格按照大模型建议执行交易，无随机操作。
6. 交易记录和状态自动保存，支持中断后继续。
7. 在无法按大模型建议执行交易时（如资金不足或无持仓），会发送邮件通知。
8. 完整的交易日志记录（按日期分割）。
9. 详细的持仓详情展示和每日总结功能。
10. 实现止损机制，根据大模型建议的止损价格自动执行止损操作。
11. **最新功能**：将大模型建议的买卖原因添加到所有通知邮件中，使用户能够更好地理解交易决策的依据。
12. **最新功能**：集成大模型多风格分析功能，支持四种投资风格和周期的分析报告（进取型短期、稳健型短期、稳健型中期、保守型中期）。
13. **配置优化**：添加 `ENABLE_ALL_ANALYSIS_STYLES` 配置开关，默认只生成稳健型短期和稳健型中期两种分析，可通过配置切换生成全部四种分析。
14. **最新功能**：统一投资者类型为英文（aggressive/moderate/conservative），与 hk_smart_money_tracker.py 保持一致。
15. **最新功能**：添加类型转换函数，将中文投资者类型转换为英文，确保跨系统兼容性。

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
16. **新增功能**：集成股息信息追踪功能，自动获取自选股的股息和除净日信息，在邮件中展示未来90天内即将除净的股息信息。
17. **新增功能**：集成加权评分系统(TAV)，提供更精准的交易信号判断。
18. **新增功能**：AI智能持仓分析功能，读取 `data/actual_porfolio.csv` 持仓数据，使用大模型进行综合投资分析，在邮件中展示专业的投资建议。
19. **新增功能**：集成基本面指标（基本面评分、PE、PB）到交易信号总结表和单个股票分析表格。
20. **新增功能**：集成中期评估指标（均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分、中期趋势健康度、中期可持续性、中期建议）到单个股票分析表格。
21. **新增功能**：在指标说明中添加基本面指标的详细解释（基本面评分、PE、PB）。
22. **新增功能**：集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率），在技术指标展示中添加这些关键流动性指标。
23. **新增功能**：在交易信号总结表中添加成交额变化1日和换手率变化5日两个关键流动性指标。
24. **新增功能**：在指标说明中添加VIX恐慌指数、成交额变化率、换手率变化率的详细解释。
25. **新增功能**：采用业界标准的0-5层分析框架（前置检查→风险控制→核心信号识别→技术面分析→宏观环境→辅助信息），提供系统性分析。
26. **新增功能**：针对短期和中期投资者提供不同的分析重点调整，短期投资者关注VIX短期变化、成交额1日/5日变化率、止损位3-5%、立即操作；中期投资者关注VIX中期趋势、成交额5日/20日变化率、基本面、止损位8-12%、分批建仓。
27. **功能优化**：删除快速决策参考表和决策检查清单，简化邮件内容，提高可读性。
28. **最新功能**：在LLM分析提示词中集成新闻数据，从 `data/all_stock_news_records.csv` 读取最新新闻（最近3条）。
29. **最新功能**：在持仓分析和买入信号分析中显示新闻摘要，为大模型提供更全面的信息。
30. **最新功能**：在第5层辅助信息中添加新闻分析说明，提供新闻分析原则和投资者类型权重（进取型10%、稳健型20%、保守型30%）。

#### AI交易分析日报
1. 基于模拟交易数据进行多时间维度的盈利能力分析。
2. 支持分析1天、5天和1个月的不同时间周期。
3. 计算已实现盈亏和未实现盈亏，提供详细的持仓分析。
4. 自动排除价格异常的股票，确保数据准确性。
5. 每个交易日收盘后自动执行，并通过邮件发送分析报告。
6. 支持手动触发，可指定任意日期范围进行分析。
7. **最新功能**：集成邮件通知功能，分析完成后自动发送报告到指定邮箱。
8. **最新修复**：修复交易日计算逻辑，将分析目标从前一天改为当天，在收盘后立即分析当日交易数据。
9. **最新修复**：移除周末日期处理逻辑，因为工作流只在周一到周五运行。

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
7. **新增功能**：TAV（Technical Analysis Value）加权评分系统，提供更精准的交易信号判断。
8. **新增功能**：中期分析指标系统，包含：
   - 均线排列状态判断（多头/空头/混乱排列）
   - 均线斜率计算（MA20/MA50斜率和角度，判断趋势强度）
   - 均线乖离率（评估价格与均线的偏离程度，识别超买超卖状态）
   - 支撑阻力位识别（基于近期局部高低点识别关键价格水平）
   - 相对强弱指标（计算股票相对于恒生指数的表现）
   - 中期趋势评分系统（综合趋势、动量、支撑阻力、相对强弱四维度评分）

#### 港股基本面数据获取器
1. 通过AKShare获取港股财务数据，包括财务指标、利润表、资产负债表、现金流量表等。
2. 提供智能缓存机制，数据缓存有效期为7天，避免重复请求。
3. 支持获取以下财务指标：
   - 市盈率(PE)、市净率(PB)、净资产收益率(ROE)、总资产收益率(ROA)
   - 每股收益(EPS)、每股净资产(BPS)
   - 净利率、毛利率、股息率、市值
4. 支持获取利润表数据：
   - 营业总收入、营业收入、利润总额、净利润
   - 归属于母公司所有者的净利润、营业利润
   - 营业收入增长率、净利润增长率
5. 支持获取资产负债表数据：
   - 资产总计、负债合计、所有者权益合计
   - 流动资产合计、流动负债合计
   - 资产负债率、流动比率
6. 支持获取现金流量表数据：
   - 经营活动现金流量净额、投资活动现金流量净额
   - 筹资活动现金流量净额、现金及现金等价物净增加额
7. 提供综合基本面数据获取功能，一次性获取所有财务数据。
8. 支持缓存清除功能，方便强制更新数据。

#### 港股股息信息追踪器
1. 自动获取自选股的股息和除净日信息。
2. 支持获取未来指定天数内即将除净的股票（默认90天）。
3. 提供最近除净的股息信息记录。
4. 股息信息包括：
   - 除净日、分红方案、截至过户日
   - 最新公告日期、财政年度、分配类型、发放日
5. 在恒生指数价格监控器邮件中自动展示即将除净的股息信息。
6. 通过AKShare的 `stock_hk_dividend_payout_em` 接口获取股息数据。

#### 人工智能股票交易盈利能力分析器
1. 基于交易记录复盘AI推荐的股票交易策略的盈利能力。
2. 采用简化复盘规则：买入信号固定买入1000股，卖出信号清仓全部持仓。
3. 支持按日期范围分析，计算已实现盈亏和未实现盈亏。
4. 提供详细的交易记录分析，包括每只股票的投资、回收和盈亏情况。
5. 兼容历史数据，优先使用current_price字段，当为空时使用price字段。
6. 生成清晰的分析报告，展示总体盈亏率和个股表现。
7. **最新功能**：在报告中显示建议的买卖次数和实际执行的买卖次数，帮助用户了解AI建议频率与复盘执行差异。

#### 机器学习交易模型
1. **模型算法**：基于LightGBM和GBDT+LR的二分类模型，预测1天、5天、20天后的涨跌
2. **特征工程**：整合100+个技术指标 + 美股市场特征 + 基本面特征 + 股票类型特征 + 情感指标 + 技术-基本面交互特征
   - 技术指标特征（80+个）：移动平均线、RSI、MACD、布林带、ATR、成交量比率、价格位置、涨跌幅、成交额变化率、换手率变化率等
   - 市场环境特征（3个）：恒生指数收益率、相对表现
   - 资金流向特征（5个）：价格位置、成交量信号、动量信号
   - 基本面特征（2个）：PE、PB（只使用实际可用的数据）
   - **美股市场特征（11个）**：标普500收益率（1日、5日、20日）、纳斯达克收益率（1日、5日、20日）、VIX绝对值、VIX变化率、VIX比率、美国10年期国债收益率及其变化率
   - **股票类型特征（18个）**：股票类型（bank/tech/energy/utility/semiconductor/ai/new_energy/shipping/exchange/insurance/biotech/environmental/index）、防御性评分、成长性评分、周期性评分、流动性评分、风险评分、衍生特征（银行/科技/周期股分析权重）
   - **情感指标特征（6个）**：情感MA3（短期情绪）、情感MA7（中期情绪）、情感MA14（长期情绪）、情感波动率（情绪稳定性）、情感变化率（情绪变化方向）、情感数据天数
   - **技术-基本面交互特征（18个）**：RSI×PE、RSI×PB、MACD×PE、MACD×PB、MACD_Hist×PE、MACD_Hist×PB、Price_Pct_20d×PE、Price_Pct_20d×PB、ATR×PE、ATR×PB、Vol_Ratio×PE、Vol_Ratio×PB、CMF×PE、CMF×PB、Return_5d×PE、Return_5d×PB、Momentum_5d×PE、Momentum_5d×PB
3. **分类特征编码**：
   - 使用LabelEncoder将字符串类型的分类特征（如Stock_Type）转换为整数编码
   - 编码器在训练时保存到模型文件，在预测时加载使用
   - 支持处理训练时未见过的类别（使用默认值0）
   - 确保训练和预测时的特征编码一致性
4. **模型性能**（2026-01-30最新训练结果，包含情感指标和交互特征）：
   - **次日模型（1天）**：平均验证准确率 **52.98%** (GBDT+LR)，**52.94%** (GBDT)
   - **一周模型（5天）**：平均验证准确率 **53.49%** (2026-01-26数据)
   - **一个月模型（20天）**：平均验证准确率 **57.24%** (2026-01-26数据) - **最佳表现**
   - 使用时间序列交叉验证（5折）
   - 特征重要性分析：VIX_Level、Stock_Type、成交额变化率等指标在Top 10中频繁出现
   - **性能提升**：包含情感指标后，次日模型准确率从52.27%提升至52.98%（+0.71%）
5. **正则化增强**（2026-01-26实施）：
   - 减少树的数量：n_estimators 100→50
   - 降低学习率：0.05→0.03
   - 减少树深度：max_depth 6→4
   - 减少叶子节点数：num_leaves 31→15
   - 增加最小子样本数：min_child_samples 20→30（LGBM）、10→20（GBDT）
   - 减少采样率：subsample 0.8→0.7, colsample_bytree 0.8→0.7
   - 添加L1/L2正则化：reg_alpha=0.1, reg_lambda=0.1
   - 添加额外的正则化参数：min_split_gain=0.1, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5
   - 增加早停耐心：stopping_rounds=10（所有模型）
   - **性能改善**：
     * 过拟合显著降低：次日模型-30.5%，一周模型-18.2%，一个月模型-13.5%
     * 泛化能力提升：验证准确率提升（次日+1.18%，一周+1.44%）
6. **数据泄漏问题修复**：
   - **问题根源**：使用 `pd.concat(all_data, ignore_index=True)` 导致日期索引被重置为 0,1,2,3...，数据顺序按股票代码和处理顺序排列，而非时间顺序
   - **修复方案**：
     - 将 `ignore_index=True` 改为 `ignore_index=False`，保留日期索引
     - 在数据合并后添加 `df.sort_index()` 确保时间顺序正确
     - 在 `train` 方法的 `dropna()` 后添加 `sort_index()` 确保顺序
   - **修复结果**：准确率从 64.78%（虚假）降至 52.42%（真实），符合预期范围（51-55%）
7. **训练流程**：
   - 获取26只自选股的2年历史数据
   - 计算所有技术指标、基本面、美股市场、股票类型特征
   - 使用LabelEncoder编码分类特征
   - 创建标签（N天后涨跌）
   - 时间序列交叉验证（5折）
   - 训练LightGBM和GBDT+LR模型（带正则化）
   - 输出特征重要性和模型对比
8. **使用场景**：
   - 作为手工信号的补充参考
   - 提供量化化的预测概率
   - 支持定期重训练适应市场变化
   - 支持历史回测验证模型准确性
   - 根据股票类型特征，为不同类型股票提供差异化分析
9. **注意事项**：
   - 模型基于历史数据，市场结构变化可能导致失效
   - 需要定期验证实际预测准确率
   - 建议与手工信号结合使用，不单独依赖
   - 一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值
   - 股票类型特征基于业界分析框架，不同类型股票应采用不同的分析权重
10. **业界差距分析**：
    - 详见 `ml_model_analysis_report.md` 文档
    - 当前模型准确率：52.98%（次日）、53.49%（一周）、57.24%（一个月）
    - 业界领先水平：65-70%（日内交易）、62-68%（波段交易）
    - 主要差距：特征数量（100+ vs 500-1000）、模型数量（2 vs 5-10）、风险管理（缺失 vs 完善）
    - 改进建议：立即实现风险管理模块（VaR、止损止盈、仓位管理），扩展特征工程，优化集成方法

#### 机器学习预测邮件通知
1. **功能概述**：自动发送机器学习模型预测结果邮件
2. **支持周期**：1天、5天、20天三个预测周期
3. **预测内容**：
   - LightGBM模型预测结果（上涨/下跌）
   - GBDT+LR模型预测结果（上涨/下跌）
   - 预测概率和置信度
   - 平均概率（两种算法的平均值）
   - 当前价格和预测目标价格
4. **邮件格式**：采用统一的表格格式展示预测结果
5. **排序规则**：先按预测一致性分组（一致的排在一起），再按平均概率降序排序
6. **自动化调度**：通过 GitHub Actions 工作流自动执行（工作流文件：`.github/workflows/ml-prediction-alert.yml`）

#### 美股市场数据获取
1. **数据源**：标普500指数 (^GSPC)、纳斯达克指数 (^IXIC)、VIX恐慌指数 (^VIX)、美国10年期国债收益率 (^TNX)
2. **特征计算**：
   - 标普500收益率（1日、5日、20日）
   - 纳斯达克收益率（1日、5日、20日）
   - **VIX绝对值**（VIX_Level，反映市场恐慌程度）
   - VIX变化率、VIX与20日均线比率
   - 美国10年期国债收益率及其变化率
3. **数据缓存**：支持1小时缓存，避免重复请求
4. **时区处理**：统一使用UTC时区，避免与港股数据冲突
5. **使用场景**：
   - 作为港股预测的外部市场环境特征
   - 捕捉美股对港股的影响
   - 提升模型预测准确率
6. **业界重要性**：
   - VIX < 15：过度乐观，需警惕回调
   - VIX 15-20：正常波动，市场情绪平稳
   - VIX 20-30：轻度恐慌
   - VIX > 30：严重恐慌，通常伴随大跌

#### 模型可解释性分析
1. **特征重要性分析**：
   - LightGBM原生特征重要性
   - GBDT+LR模型的特征重要性
   - LR模型叶子节点系数分析
2. **GBDT决策路径解析**：
   - 自动解析每个叶子节点的决策路径
   - 显示具体的特征名称和阈值
   - 提供可解释的决策规则
3. **可视化分析**：
   - ROC曲线绘制
   - 特征重要性排序图表
   - 模型性能对比报告
4. **输出文件**：
   - `output/gbdt_feature_importance.csv` - GBDT特征重要性
   - `output/lr_leaf_coefficients.csv` - LR叶子节点系数
   - `output/roc_curve.png` - ROC曲线图

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
# 分析当天数据（默认投资者类型：稳健型）
python hk_smart_money_tracker.py

# 分析指定日期数据
python hk_smart_money_tracker.py --date 2025-10-25

# 指定投资者类型
python hk_smart_money_tracker.py --investor-type aggressive  # 进取型
python hk_smart_money_tracker.py --investor-type moderate    # 稳健型
python hk_smart_money_tracker.py --investor-type conservative # 保守型
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

#### 港股板块分析

##### 本地运行
```bash
# 生成完整板块分析报告（默认1日涨跌幅）
python data_services/hk_sector_analysis.py

# 指定分析周期
python data_services/hk_sector_analysis.py --period 5

# 分析指定板块
python data_services/hk_sector_analysis.py --sector bank

# 识别板块龙头
python data_services/hk_sector_analysis.py --leaders bank

# 分析板块资金流向
python data_services/hk_sector_analysis.py --flow bank

# 分析板块趋势
python data_services/hk_sector_analysis.py --trend bank
```

##### 功能说明
- **板块涨跌幅排名**：计算各板块平均涨跌幅，按涨幅排序
- **技术趋势分析**：分析板块内股票的技术趋势，识别强势/弱势板块
- **龙头识别**：基于涨跌幅和成交量综合评分，识别板块龙头股票
- **资金流向分析**：基于成交量和涨跌幅分析板块资金流入/流出情况

#### 港股模拟交易系统

##### 本地运行
```bash
# 运行模拟交易（默认持续运行，默认投资者类型：稳健型）
python simulation_trader.py

# 运行指定天数的模拟交易
python simulation_trader.py --duration-days 30

# 指定投资者类型
python simulation_trader.py --investor-type aggressive  # 进取型
python simulation_trader.py --investor-type moderate    # 稳健型
python simulation_trader.py --investor-type conservative # 保守型
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
10. **最新功能**：集成大模型多风格分析功能，支持四种投资风格和周期的分析报告
11. **配置优化**：添加 `ENABLE_ALL_ANALYSIS_STYLES` 配置开关，默认只生成稳健型短期和稳健型中期两种分析

#### 批量获取自选股新闻

##### 本地运行
```bash
# 单次运行
python data_services/batch_stock_news_fetcher.py

# 启用定时任务模式
python data_services/batch_stock_news_fetcher.py --schedule
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

##### 持仓分析功能
- **新增**：自动读取 `data/actual_porfolio.csv` 持仓数据
- **新增**：使用AI大模型进行综合投资分析
- **新增**：在邮件中展示格式化的投资建议（Markdown转HTML）
- **新增**：提供整体风险评估、个股建议、仓位管理、风险控制措施

##### 基本面和中期评估指标
- **新增**：在交易信号总结表中添加基本面指标（基本面评分、PE、PB）
- **新增**：在单个股票分析HTML表格中添加基本面指标和中期评估指标
- **新增**：在单个股票分析文本版本中添加基本面指标和中期评估指标
- **新增**：在指标说明中添加基本面指标的详细解释

#### AI交易分析日报

##### 本地运行
```bash
# 分析指定日期
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 分析5天数据
python ai_trading_analyzer.py --start-date 2025-12-31 --end-date 2026-01-05

# 分析1个月数据
python ai_trading_analyzer.py --start-date 2025-12-05 --end-date 2026-01-05

# 不发送邮件通知
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05 --no-email
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/ai-trading-analysis-daily.yml`
- 执行时间：每个交易日收盘后（UTC时间8:30，香港时间16:30）
- 运行周期：只在周一到周五运行，周六和周日不运行
- 分析周期：自动执行1天、5天和1个月的分析报告
- 邮件通知：每个分析报告完成后自动发送邮件通知
- **最新修复**：将分析目标从前一天改为当天，在收盘后立即分析当日交易数据

#### 人工智能股票交易盈利能力分析器

##### 本地运行
```bash
# 分析所有数据
python ai_trading_analyzer.py

# 分析指定日期范围
python ai_trading_analyzer.py --start-date 2025-12-01 --end-date 2025-12-31

# 指定交易记录文件
python ai_trading_analyzer.py --file data/simulation_transactions.csv
```

##### 命令行参数
- `--start-date` 或 `-s`: 起始日期 (YYYY-MM-DD)，默认为最早交易日期
- `--end-date` 或 `-e`: 结束日期 (YYYY-MM-DD)，默认为最新交易日期
- `--file` 或 `-f`: 交易记录CSV文件路径（默认：data/simulation_transactions.csv）
- `--no-email`: 不发送邮件通知

#### 机器学习交易模型

##### 完整训练和预测（推荐）
```bash
# 训练所有周期模型并预测（当前日期）
./train_and_predict_all.sh

# 训练所有周期模型并预测（指定日期）
./train_and_predict_all.sh --predict-date 2026-01-15

# 限制训练数据范围
./train_and_predict_all.sh --start-date 2024-01-01 --end-date 2024-12-31

# 完整历史回测
./train_and_predict_all.sh --backtest --start-date 2024-01-01 --end-date 2024-12-31 --predict-date 2024-12-31
```

##### 仅预测（使用已训练模型）
```bash
```

##### 单独训练/预测
```bash
# 训练模型（指定周期）
python ml_services/ml_trading_model.py --mode train --horizon 1 --model-type both --model-path data/ml_trading_model.pkl

# 预测股票（指定周期）
python ml_services/ml_trading_model.py --mode predict --horizon 1 --model-type both --model-path data/ml_trading_model.pkl

# 评估模型
python ml_services/ml_trading_model.py --mode evaluate --horizon 1 --model-type both --model-path data/ml_trading_model.pkl

# 指定日期范围训练
python ml_services/ml_trading_model.py --mode train --horizon 1 --start-date 2024-01-01 --end-date 2025-12-31

# 指定预测日期
python ml_services/ml_trading_model.py --mode predict --horizon 1 --predict-date 2026-01-15
```

##### 命令行参数
- `--mode`: 运行模式（train/predict/evaluate）
- `--horizon`: 预测周期（1=次日，5=一周，20=一个月）
- `--model-type`: 模型类型（lgbm/gbdt_lr/both）
- `--model-path`: 模型保存/加载路径（默认：data/ml_trading_model.pkl）
- `--start-date`: 训练开始日期 (YYYY-MM-DD)
- `--end-date`: 训练结束日期 (YYYY-MM-DD)
- `--predict-date`: 预测日期 (YYYY-MM-DD)

##### 输出文件
- `data/ml_trading_model_lgbm_1d.pkl` - LightGBM次日涨跌模型
- `data/ml_trading_model_lgbm_5d.pkl` - LightGBM一周涨跌模型
- `data/ml_trading_model_lgbm_20d.pkl` - LightGBM一个月涨跌模型
- `data/ml_trading_model_gbdt_lr_1d.pkl` - GBDT+LR次日涨跌模型
- `data/ml_trading_model_gbdt_lr_5d.pkl` - GBDT+LR一周涨跌模型
- `data/ml_trading_model_gbdt_lr_20d.pkl` - GBDT+LR一个月涨跌模型
- `data/ml_trading_model_*_importance.csv` - 特征重要性排名
- `data/ml_trading_model_*_predictions_*.csv` - 预测结果
- `data/ml_trading_model_comparison.csv` - 模型对比结果
- `output/gbdt_feature_importance.csv` - GBDT特征重要性
- `output/lr_leaf_coefficients.csv` - LR叶子节点系数
- `output/roc_curve.png` - ROC曲线图

#### 机器学习预测邮件通知

##### 本地运行
```bash
python ml_services/ml_prediction_email.py
```

##### GitHub Actions 自动化
- 工作流文件：`.github/workflows/ml-prediction-alert.yml`
- 执行时间：
  - 每天香港时间 09:00 (UTC 01:00)
  - 每天香港时间 16:30 (UTC 08:30)
- 支持手动触发，可选择预测周期（1天/5天/20天/全部）

#### 模型对比工具

##### 本地运行
```bash
python ml_services/compare_models.py
```

##### 输出
- 控制台输出两种模型的预测对比
- 包括预测一致性、概率差异等统计信息

#### 通用技术分析工具

##### 本地运行
```bash
python data_services/technical_analysis.py
```

#### 港股基本面数据获取器

##### 本地运行
```bash
# 在其他脚本中导入使用
from data_services.fundamental_data import get_comprehensive_fundamental_data

# 获取综合基本面数据
data = get_comprehensive_fundamental_data("00700")
print(data)

# 清除缓存
from data_services.fundamental_data import clear_cache
clear_cache()
```

##### 使用场景
- 在 `hk_smart_money_tracker.py` 中自动调用，获取股票的财务指标
- 支持缓存机制，避免重复请求，提高性能
- 数据缓存有效期7天，可手动清除缓存强制更新

### 配置参数

#### 自选股配置
在 `config.py` 中配置自选股列表（共26只股票）：
- 银行类：0005.HK（汇丰银行）、0939.HK（建设银行）、0941.HK（中国移动）、1288.HK（农业银行）、1398.HK（工商银行）、3968.HK（招商银行）
- 科技类：0700.HK（腾讯控股）、1810.HK（小米集团-W）、3690.HK（美团-W）、9988.HK（阿里巴巴-SW）
- 半导体：0981.HK（中芯国际）、1347.HK（华虹半导体）
- AI：2533.HK（黑芝麻智能）、6682.HK（第四范式）、9660.HK（地平线机器人）
- 能源：0883.HK（中国海洋石油）、1088.HK（中国神华）
- 环保：1330.HK（绿色动力环保）
- 航运：1138.HK（中远海能）
- 交易所：0388.HK（香港交易所）
- 保险：1299.HK（友邦保险）
- 生物医药：2269.HK（药明生物）
- 指数基金：2800.HK（盈富基金）
- 电信：0728.HK（中国电信）
- 新能源：1211.HK（比亚迪股份）

#### 港股主力资金追踪器参数
在`hk_smart_money_tracker.py`代码顶部可以调整以下参数：

- `WATCHLIST`：自选股票列表（从 config.py 导入）
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

**命令行参数**：
- `--date`：指定分析日期（YYYY-MM-DD格式）
- `--investor-type`：投资者类型（aggressive进取型、moderate稳健型、conservative保守型），默认为稳健型

#### 模拟交易系统参数
在`simulation_trader.py`文件中可以调整以下参数：

- `investor_type`：投资者风险偏好（"aggressive"进取型、"moderate"稳健型、"conservative"保守型），默认为"moderate"
- `initial_capital`：初始资金（默认100万港元）
- `analysis_frequency`：执行频率（默认每60分钟执行一次交易分析，可根据需要调整）

**命令行参数**：
- `--duration-days`：模拟交易天数
- `--analysis-frequency`：分析频率（分钟）
- `--investor-type`：投资者风险偏好（aggressive进取型、moderate稳健型、conservative保守型），默认为moderate
- `--manual-sell`：手工卖出股票代码
- `--sell-percentage`：卖出比例（0.0-1.0）

不同投资者类型的风险偏好设置：
- 保守型（conservative）：偏好低风险、稳定收益的股票，如高股息银行股，注重资本保值
- 稳健型（moderate）：平衡风险与收益，兼顾价值与成长，追求稳健增长
- 进取型（aggressive）：偏好高风险、高收益的股票，如科技成长股，追求资本增值

#### 大模型分析风格配置
在`hsi_email.py`文件中可以调整以下参数：

- `ENABLE_ALL_ANALYSIS_STYLES`：是否启用全部四种分析风格（默认False，只生成稳健型短期和稳健型中期两种分析）
  - True：生成全部四种（进取型短期、稳健型短期、稳健型中期、保守型中期）
  - False：只生成两种（稳健型短期、稳健型中期）

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
│   ├── 黄金市场分析器 (@gold_analyzer.py)
│   ├── 港股基本面数据获取器 (@data_services/fundamental_data.py)
│   ├── 港股股息信息追踪器 (@hsi_email.py 中的股息功能)
│   ├── 美股市场数据获取器 (@ml_services/us_market_data.py)
│   ├── 新闻数据源 (@data/all_stock_news_records.csv)
│   └── 腾讯财经数据接口 (@data_services/tencent_finance.py)
├── 数据服务层
│   ├── 港股基本面数据获取器 (@data_services/fundamental_data.py)
│   ├── 批量获取自选股新闻 (@data_services/batch_stock_news_fetcher.py)
│   ├── 港股板块分析器 (@data_services/hk_sector_analysis.py)
│   │   ├── 板块涨跌幅排名
│   │   ├── 技术趋势分析
│   │   ├── 龙头识别
│   │   └── 资金流向分析
│   ├── 通用技术分析工具 (@data_services/technical_analysis.py)
│   │   ├── 短期技术指标（RSI、MACD、布林带、ATR等）
│   │   ├── TAV加权评分系统
│   │   └── 中期分析指标系统（均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分）
│   └── 腾讯财经数据接口 (@data_services/tencent_finance.py)
├── 分析层
│   ├── 港股主力资金追踪器 (@hk_smart_money_tracker.py)
│   ├── 恒生指数大模型策略分析器 (@hsi_llm_strategy.py)
│   ├── 恒生指数价格监控器 (@hsi_email.py)
│   │   ├── 技术分析指标
│   │   ├── 基本面指标（基本面评分、PE、PB）
│   │   └── 中期评估指标（均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分等）
│   ├── 人工智能股票交易盈利能力分析器 (@ai_trading_analyzer.py)
│   ├── 机器学习模块 (@ml_services/)
│   │   ├── 机器学习交易模型 (@ml_trading_model.py)
│   │   │   ├── 特征工程（技术指标、市场环境、资金流向、基本面、美股市场、股票类型特征）
│   │   │   ├── 分类特征编码（LabelEncoder处理字符串类型特征）
│   │   │   ├── LightGBM二分类模型（预测1天、5天、20天后涨跌）
│   │   │   ├── GBDT+LR两阶段模型（预测1天、5天、20天后涨跌）
│   │   │   ├── 时间序列交叉验证（5-Fold）
│   │   │   ├── 正则化增强（L1/L2正则化、早停、树深度控制）
│   │   │   └── 特征重要性分析
│   │   ├── 机器学习预测邮件发送器 (@ml_prediction_email.py)
│   │   ├── 模型对比工具 (@compare_models.py)
│   │   ├── 美股市场数据获取器 (@us_market_data.py)
│   │   └── 模型处理器基类 (@base_model_processor.py)
│   │       ├── 特征重要性分析
│   │       ├── GBDT决策路径解析
│   │       ├── LR叶子节点系数分析
│   │       └── ROC曲线绘制
│   └── 不同股票类型分析框架 (@不同股票类型分析框架对比.md)
│       ├── hsi_email.py vs hk_smart_money_tracker.py对比
│       ├── 设计目标差异
│       ├── 分析框架对比
│       ├── 关键指标对比
│       ├── 适用场景说明
│       └── 集成策略建议
├── 交易层
│   └── 港股模拟交易系统 (@simulation_trader.py)
└── 服务层
    ├── 大模型服务 (@llm_services/)
    │   ├── qwen_engine.py - 大模型接口
    │   └── sentiment_analyzer.py - 情感分析模块
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
- 在 `hk_smart_money_tracker.py` 中，集成基本面数据分析，结合财务指标进行综合判断
- **最新功能**：在 `hsi_email.py` 中，使用大模型进行持仓投资分析，提供专业的投资建议
- **最新功能**：在 `hsi_email.py` 中，集成大模型多风格分析功能，支持四种投资风格和周期的分析报告
- **最新功能**：在 `simulation_trader.py` 中，将大模型建议的买卖原因添加到所有通知邮件中
- **最新功能**：在 `hk_smart_money_tracker.py` 中，集成新闻数据分析，从 `data/all_stock_news_records.csv` 读取最新新闻
- **最新功能**：在 `hsi_email.py` 中，集成新闻数据分析，在持仓分析和买入信号分析中显示新闻摘要
- **最新功能**：新闻分析采用"不进行情感分类，直接提供新闻内容"的方式，由大模型自主判断影响
- **最新功能**：在 `hk_smart_money_tracker.py` 和 `hsi_email.py` 中，集成板块分析数据，从 `hk_sector_analysis.py` 读取板块趋势信息，提供宏观市场背景

### 数据文件结构

项目生成的数据文件存储在 `data/` 目录中：
- `actual_porfolio.csv`: 实际持仓数据文件，包含股票代码、一手股数、成本价、持有手数等信息
- `all_stock_news_records.csv`: 所有股票相关新闻记录（包含股票名称、股票代码、新闻时间、新闻标题、简要内容）
- `all_dividends.csv`: 所有股息信息记录
- `recent_dividends.csv`: 最近除净的股息信息
- `upcoming_dividends.csv`: 即将除净的股息信息（未来90天）
- `hsi_strategy_latest.txt`: 恒生指数策略分析报告
- `simulation_state.json`: 模拟交易状态保存
- `simulation_trade_log_*.txt`: 详细交易日志记录（按日期分割）
- `simulation_transactions.csv`: 交易历史记录
- `simulation_portfolio.csv`: 投资组合价值变化记录
- `southbound_data_cache.pkl`: 南向资金数据缓存
- `fundamental_cache/`: 基本面数据缓存目录（包含财务指标、利润表、资产负债表、现金流量表的缓存文件，已从Git跟踪中移除）
- `ml_trading_model_lgbm_1d.pkl`: LightGBM次日涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_lgbm_5d.pkl`: LightGBM一周涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_lgbm_20d.pkl`: LightGBM一个月涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_gbdt_lr_1d.pkl`: GBDT+LR次日涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_gbdt_lr_5d.pkl`: GBDT+LR一周涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_gbdt_lr_20d.pkl`: GBDT+LR一个月涨跌模型（已从Git跟踪中移除）
- `ml_trading_model_*_importance.csv`: 机器学习特征重要性排名（已从Git跟踪中移除）
- `ml_trading_model_*_predictions_*.csv`: 机器学习模型预测结果（已从Git跟踪中移除）
- `ml_trading_model_comparison.csv`: 模型对比结果（已从Git跟踪中移除）

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
21. **功能更新**：使用交易记录中的止损价和目标价，而非技术分析计算值
22. **Bug修复**：修复crypto_email.py中HTML显示代码片段的问题
23. **新增功能**：添加人工智能股票交易盈利能力分析器，用于评估AI交易信号的有效性
24. **功能优化**：AI交易分析器支持历史数据兼容，自动处理current_price和price字段
25. **新增功能**：添加港股基本面数据获取器，提供财务指标、利润表、资产负债表、现金流量表等数据
26. **新增功能**：添加港股股息信息追踪器，自动获取并展示即将除净的股息信息
27. **功能增强**：港股主力资金追踪器集成基本面数据分析，结合财务指标进行综合判断
28. **最新功能**：AI交易分析器在报告中显示建议的买卖次数和实际执行的买卖次数
29. **最新修复**：修复AI交易分析工作流的交易日计算逻辑，将分析目标从前一天改为当天
30. **最新修复**：简化AI交易分析工作流的周末处理逻辑，因为工作流只在周一到周五运行
31. **新增功能**：在恒生指数价格监控器中集成TAV加权评分系统，提供更精准的交易信号判断
32. **功能优化**：update_data.sh脚本增加重试机制，提高Git操作的稳定性
33. **新增功能**：在恒生指数价格监控器中集成AI智能持仓分析功能，读取持仓数据并提供专业投资建议
34. **新增功能**：添加Markdown到HTML转换功能，优化大模型分析内容的邮件展示效果
35. **依赖更新**：在requirements.txt中添加markdown库，支持完整Markdown转换
36. **工作流优化**：更新GitHub Actions工作流使用requirements.txt统一管理依赖
37. **数据优化**：新增股息数据文件（all_dividends.csv、recent_dividends.csv、upcoming_dividends.csv），提供更完整的股息信息追踪
38. **流程优化**：更新编码规范，强调"修改完即测试"原则，每次修改后立即验证，避免累积错误
39. **新增功能**：在technical_analysis.py中实现中期分析指标系统，包含均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分等6个核心指标
40. **新增功能**：在hsi_email.py中集成基本面指标（基本面评分、PE、PB）到交易信号总结表和单个股票分析表格
41. **新增功能**：在hsi_email.py中集成中期评估指标（均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分、中期趋势健康度、中期可持续性、中期建议）到单个股票分析表格
42. **新增功能**：在hsi_email.py中实现大模型多风格分析功能，支持四种投资风格和周期的分析报告（进取型短期、稳健型短期、稳健型中期、保守型中期）
43. **配置优化**：添加ENABLE_ALL_ANALYSIS_STYLES配置开关，默认只生成稳健型短期和稳健型中期两种分析，可通过配置切换生成全部四种分析
44. **新增功能**：实现机器学习交易模型（ml_services/ml_trading_model.py），基于LightGBM和GBDT+LR预测次日涨跌
45. **依赖更新**：在requirements.txt中添加lightgbm和scikit-learn，支持机器学习功能
46. **功能扩展**：机器学习模型支持34个特征（技术指标、市场环境、资金流向、基本面、美股市场），提供特征重要性分析
47. **新增功能**：实现美股市场数据获取模块（ml_services/us_market_data.py），提供标普500、纳斯达克、VIX、美国国债收益率等数据
48. **新增功能**：机器学习模型集成美股市场特征
49. **新增功能**：实现模型处理器基类（ml_services/base_model_processor.py），提供特征重要性分析、GBDT决策路径解析等功能
50. **新增功能**：实现完整训练和预测脚本（train_and_predict_all.sh），支持1天、5天、20天三个周期的模型训练和预测
51. **新增功能**：实现模型对比工具（ml_services/compare_models.py），对比LGBM和GBDT+LR两种模型的预测结果
52. **功能优化**：模型文件命名支持周期后缀（_1d、_5d、_20d），避免不同周期模型相互覆盖
53. **功能扩展**：机器学习模型支持多周期预测（1天、5天、20天），适应不同投资需求
54. **功能扩展**：支持历史回测功能，可基于任意历史日期进行预测，验证模型历史表现
55. **新增功能**：GBDT决策路径解析，自动解析每个叶子节点的决策路径，显示具体的特征名称和阈值
56. **新增功能**：支持时间序列交叉验证（5-Fold），评估模型在不同时期的稳定性
57. **架构重组**：将 ML 模块整合到 ml_services 目录，提高代码组织性和可维护性
58. **Bug修复**：修复机器学习模型导入错误，将相对导入改为绝对导入
59. **重要修复**：修复机器学习模型数据泄漏问题，将准确率从 64.78%（虚假）降至 52.42%（真实）
60. **新增功能**：实现机器学习预测邮件发送器（ml_services/ml_prediction_email.py），自动发送ML模型预测结果邮件
61. **新增功能**：实现机器学习预测警报工作流（.github/workflows/ml-prediction-alert.yml），自动执行ML模型预测并发送邮件通知
62. **新增功能**：添加关键流动性指标（VIX_Level、成交额变化率、换手率变化率），提升模型预测能力
63. **性能提升**：一个月模型准确率提升至59.16%，接近业界优秀水平（60%）
64. **UI优化**：ML预测邮件添加平均概率列，按一致性和平均概率排序展示
65. **Git优化**：从Git跟踪中移除ML模型文件和输出文件，避免仓库膨胀
66. **Git优化**：从Git跟踪中移除fundamental_cache缓存目录，避免缓存文件被提交
67. **新增功能**：在hk_smart_money_tracker.py中集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率），采用业界标准的0-5层分析框架，提升分析准确性
68. **新增功能**：在hsi_email.py中集成ML模型关键指标和0-5层分析框架，针对短期和中期投资者提供不同的分析重点调整
69. **功能优化**：在hsi_email.py的交易信号总结表中添加成交额变化1日和换手率变化5日两个关键流动性指标
70. **功能优化**：在hsi_email.py的指标说明中添加VIX恐慌指数、成交额变化率、换手率变化率的详细解释
71. **功能优化**：删除hsi_email.py中的快速决策参考表和决策检查清单，简化邮件内容，提高可读性
72. **新增功能**：添加股票类型特征到ML模型，支持13种股票类型分类（bank、tech、energy、utility、semiconductor、ai、new_energy、shipping、exchange、insurance、biotech、environmental、index）
73. **新增功能**：实现分类特征编码（LabelEncoder），解决LightGBM无法直接处理字符串数据的问题
74. **新增功能**：添加股票类型衍生特征（防御性评分、成长性评分、周期性评分、流动性评分、风险评分、银行/科技/周期股分析权重）
75. **性能优化**：实施全面的正则化增强，显著降低过拟合程度（次日-30.5%，一周-18.2%，一个月-13.5%）
76. **性能提升**：通过正则化提升模型泛化能力，验证准确率提升（次日+1.18%，一周+1.44%）
77. **新增文档**：添加不同股票类型分析框架对比文档（不同股票类型分析框架对比.md），详细说明hsi_email.py和hk_smart_money_tracker.py的差异和适用场景
78. **功能扩展**：机器学习模型支持52个特征（新增18个股票类型特征），提供更全面的股票分析能力
79. **最新功能**：在hk_smart_money_tracker.py中添加动态投资者类型支持（aggressive/moderate/conservative），支持 `--investor-type` 命令行参数
80. **最新功能**：在hk_smart_money_tracker.py中集成新闻分析功能，从 `data/all_stock_news_records.csv` 读取最新新闻，添加到第5层辅助信息
81. **最新功能**：在hk_smart_money_tracker.py中修复综合评分权重，符合策略权重（成交量25%、技术指标30%、南向资金15%、价格位置10%、MACD信号10%、RSI指标10%）
82. **最新功能**：在simulation_trader.py中统一投资者类型为英文（aggressive/moderate/conservative），添加类型转换函数
83. **最新功能**：在hsi_email.py的LLM分析提示词中集成新闻数据，在持仓分析和买入信号分析中显示新闻摘要
84. **最新功能**：在hsi_email.py的第5层辅助信息中添加新闻分析说明，提供新闻分析原则和投资者类型权重
85. **最新功能**：实现情感分析模块（llm_services/sentiment_analyzer.py），提供四维情感评分（相关性、影响度、预期差、情感方向），支持批量分析和情感统计
86. **最新功能**：在ML模型中集成情感指标特征（6个基础特征+78个交互特征），从新闻数据中计算情感趋势特征
87. **最新功能**：在ML模型中添加18个技术指标与基本面的交互特征（RSI×PE、MACD×PB等），捕捉非线性关系
88. **最新功能**：修复基本面数据提取逻辑，使用正确的键名（fi_pe_ratio、fi_pb_ratio）
89. **最新功能**：添加情感特征默认值处理，当新闻数据不可用时返回默认值0，避免数据丢失
90. **最新功能**：修复数据泄漏问题，在dropna前删除全为NaN的列，避免删除所有训练数据
91. **性能提升**：包含情感指标和交互特征后，次日模型准确率从52.27%提升至52.98%（+0.71%）
92. **新增功能**：实现港股板块分析模块（hk_sector_analysis.py），提供板块涨跌幅排名、技术趋势分析、龙头识别、资金流向分析
93. **新增功能**：板块分析模块涵盖13个板块（银行、科技、半导体、AI、新能源、环保、能源、航运、交易所、公用事业、保险、生物医药、指数基金）
94. **最新功能**：在hk_smart_money_tracker.py和hsi_email.py中集成板块分析数据，为大模型分析提供宏观市场背景
95. **提示词重构**：重构hk_smart_money_tracker.py的提示词结构，从混乱的"6层"重构为清晰的"0-5层"分析框架
96. **提示词优化**：简化JSON数据结构，移除重复字段，提高提示词可读性和效率
97. **架构重组**：将数据服务模块（batch_stock_news_fetcher.py、fundamental_data.py、hk_sector_analysis.py、technical_analysis.py、tencent_finance.py）移至 data_services/ 目录，提高代码组织性和可维护性
98. **导入更新**：更新所有文件中的导入语句，使用 `from data_services.xxx import` 格式，确保模块导入正确
99. **新增功能**：新增 config.py 全局配置文件，统一管理自选股列表（26只股票）
100. **新增功能**：新增 ml_model_analysis_report.md 文档，详细分析当前ML模型与业界最佳实践的差距
101. **新增功能**：新增 hsi-email-alert-open_message.yml 工作流，支持强制发送恒生指数邮件（--force 参数），执行时间：周一到周五下午6:00香港时间
102. **配置更新**：自选股列表新增 1211.HK（比亚迪股份）和 1299.HK（友邦保险），总数达到26只

---
最后更新：2026-02-03