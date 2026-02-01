# 📊 金融信息监控与智能交易系统

<div align="center">

一个基于 Python 的综合性金融分析系统，集成多数据源、技术分析工具和大模型智能判断，为投资者提供全面的市场分析和交易策略验证工具。

[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-automated-brightgreen.svg)](https://github.com/features/actions)

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
- 📉 **中期评估**：提供数周至数月投资周期的技术分析

---

## ✨ 核心特性

### 🤖 AI 智能分析
- **大模型集成**：集成 Qwen 大模型，提供智能投资建议
- **持仓分析**：自动分析现有持仓，提供专业的投资建议
- **信号识别**：智能识别买卖信号，减少人工判断
- **策略生成**：基于技术面和基本面生成交易策略
- **多风格分析**：支持进取型短期、稳健型短期、稳健型中期、保守型中期四种投资风格

### 📈 技术分析
- **多指标支持**：RSI、MACD、布林带、ATR、CCI、OBV 等
- **VaR 风险价值**：1日、5日、20日 VaR 计算
- **TAV 评分系统**：加权评分提供精准交易信号
- **趋势分析**：自动识别多头、空头、震荡趋势
- **中期分析**：均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分

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

### ⚠️ 数据验证
- **模型验证**：严格的机器学习模型验证机制
- **数据泄漏检测**：自动检测和修复数据泄漏问题
- **时间序列验证**：确保时间序列交叉验证的严格性
- **多维度评估**：准确率、AUC、Log Loss、Precision、Recall、F1 等指标

---

## 🚀 功能概览

### 数据获取与监控

| 功能 | 脚本 | 频率 | 说明 |
|------|------|------|------|
| 加密货币监控 | `crypto_email.py` | 每小时 | 比特币、以太坊价格和技术分析 |
| 港股 IPO 信息 | `hk_ipo_aastocks.py` | 每天 | 最新 IPO 信息 |
| 黄金市场分析 | `gold_analyzer.py` | 每小时 | 黄金价格和投资建议 |
| 恒生指数监控 | `hsi_email.py` | 交易时段 | 价格、技术指标、交易信号、基本面指标、中期评估指标、AI持仓分析 |

### 智能分析

| 功能 | 脚本 | 说明 |
|------|------|------|
| 主力资金追踪 | `hk_smart_money_tracker.py` | 识别建仓和出货信号，集成基本面分析 |
| 板块分析 | `hk_sector_analysis.py` | 板块涨跌幅排名、技术趋势分析、龙头识别、资金流向分析 |
| 恒生指数策略 | `hsi_llm_strategy.py` | 大模型生成交易策略 |
| AI 交易分析 | `ai_trading_analyzer.py` | 复盘 AI 推荐策略有效性 |

### 机器学习

| 功能 | 脚本 | 说明 |
|------|------|------|
| 机器学习交易模型 | `ml_services/ml_trading_model.py` | 基于LightGBM预测次日涨跌，集成股票类型特征，正则化增强后准确率52.27%（次日）、53.49%（一周）、57.24%（一个月） |
| 机器学习预测邮件通知 | `ml_services/ml_prediction_email.py` | 自动发送ML模型预测结果邮件 |
| 模型对比工具 | `ml_services/compare_models.py` | 对比LGBM和GBDT+LR两种模型的预测结果 |

### 模拟交易

| 功能 | 脚本 | 说明 |
|------|------|------|
| 模拟交易系统 | `simulation_trader.py` | 基于大模型的自动交易 |

### 辅助功能

| 功能 | 脚本 | 说明 |
|------|------|------|
| 批量新闻获取 | `batch_stock_news_fetcher.py` | 自选股新闻 |
| 基本面数据 | `fundamental_data.py` | 财务指标、利润表等 |
| 技术分析工具 | `technical_analysis.py` | 通用技术指标计算，含中期分析指标系统 |
| 美股市场数据 | `ml_services/us_market_data.py` | 标普500、纳斯达克、VIX、美国国债收益率 |

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

# 板块分析
python hk_sector_analysis.py

# 启动模拟交易
python simulation_trader.py

# 恒生指数价格监控（含基本面指标、中期评估指标和 AI 持仓分析）
python hsi_email.py

# 训练机器学习模型
python ml_services/ml_trading_model.py --mode train --horizon 1 --model-type both

# 预测股票涨跌
python ml_services/ml_trading_model.py --mode predict --horizon 1 --model-type both

# 发送机器学习预测邮件
python ml_services/ml_prediction_email.py
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
- **新增功能**：集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- **新增功能**：采用业界标准的0-5层分析框架（前置检查→风险控制→核心信号识别→技术面分析→宏观环境→辅助信息）

**使用方法**：
```bash
# 分析当天数据
python hk_smart_money_tracker.py

# 分析指定日期数据
python hk_smart_money_tracker.py --date 2025-10-25

# 指定投资者类型
python hk_smart_money_tracker.py --investor-type aggressive  # 进取型
python hk_smart_money_tracker.py --investor-type moderate    # 稳健型
python hk_smart_money_tracker.py --investor-type conservative # 保守型
```

### 3. 港股板块分析器

**功能**：
- 批量分析13个板块（银行、科技、半导体、AI、新能源、环保、能源、航运、交易所、公用事业、保险、生物医药、指数基金）
- 板块涨跌幅排名，识别强势和弱势板块
- 板块技术趋势分析（强势上涨、温和上涨、震荡整理、温和下跌、强势下跌）
- 板块龙头股票识别（基于涨跌幅和成交量综合评分）
- 板块资金流向分析（基于成交量和涨跌幅）
- 生成板块分析报告，包括强势板块TOP 3和弱势板块BOTTOM 3
- 使用腾讯财经接口获取板块内股票数据
- 集成到 hk_smart_money_tracker.py 和 hsi_email.py 中，为大模型分析提供板块背景信息

**使用方法**：
```bash
# 生成完整板块分析报告（默认1日涨跌幅）
python hk_sector_analysis.py

# 指定分析周期
python hk_sector_analysis.py --period 5

# 分析指定板块
python hk_sector_analysis.py --sector bank

# 识别板块龙头
python hk_sector_analysis.py --leaders bank

# 分析板块资金流向
python hk_sector_analysis.py --flow bank

# 分析板块趋势
python hk_sector_analysis.py --trend bank
```

**板块列表**：
- 银行股：汇丰银行、建设银行、农业银行、工商银行、招商银行
- 科技股：腾讯控股、阿里巴巴-SW、美团-W、小米集团-W
- 半导体：中芯国际、华虹半导体
- 人工智能：第四范式、地平线机器人、黑芝麻智能
- 新能源：比亚迪股份
- 环保：绿色动力环保
- 能源：中国海洋石油、中国神华
- 航运：中远海能
- 交易所：香港交易所
- 公用事业：中国电信、中国移动
- 保险：友邦保险
- 生物医药：药明生物
- 指数基金：盈富基金

### 4. 恒生指数价格监控器（含基本面指标、中期评估指标和 AI 持仓分析）

**功能**：
- 实时获取恒生指数价格和交易数据
- 技术指标计算和信号识别
- 只在有交易信号时才发送邮件
- 支持历史数据分析
- 集成股息信息追踪功能
- VaR 风险价值计算（1日、5日、20日）
- **基本面指标**：基本面评分、PE、PB等估值指标
- **中期评估指标**：均线排列、均线斜率、乖离率、支撑阻力位、相对强弱、中期趋势评分、中期趋势健康度、中期可持续性、中期建议
- **AI 智能持仓分析**：读取持仓数据，提供专业投资建议
- **大模型多风格分析**：支持进取型短期、稳健型短期、稳健型中期、保守型中期四种投资风格
- **新增功能**：在交易信号总结表中添加成交额变化1日和换手率变化5日两个关键流动性指标
- **新增功能**：在指标说明中添加VIX恐慌指数、成交额变化率、换手率变化率的详细解释
- **新增功能**：针对短期和中期投资者提供不同的分析重点调整
- **功能优化**：删除快速决策参考表和决策检查清单，简化邮件内容

**使用方法**：
```bash
# 分析当天数据
python hsi_email.py

# 分析指定日期数据
python hsi_email.py --date 2025-10-25
```

**基本面指标功能**：
- 在交易信号总结表中显示基本面评分、PE、PB
- 在单个股票分析表格中显示基本面评分、PE、PB
- 根据估值水平自动评分和分类（优秀/一般/较差）
- 结合技术指标提供综合判断

**中期评估指标功能**：
- 均线排列状态判断（多头/空头/混乱排列）
- 均线斜率计算（MA20/MA50斜率和角度，判断趋势强度）
- 均线乖离率（评估价格与均线的偏离程度，识别超买超卖状态）
- 支撑阻力位识别（基于近期局部高低点识别关键价格水平）
- 相对强弱指标（计算股票相对于恒生指数的表现）
- 中期趋势评分系统（综合趋势、动量、支撑阻力、相对强弱四维度评分）
- 中期趋势健康度、可持续性评估
- 中期投资建议（强烈买入/买入/持有/卖出/强烈卖出）

**AI 持仓分析功能**：
- 读取 `data/actual_porfolio.csv` 持仓数据
- 使用大模型进行综合投资分析
- 提供整体风险评估
- 各股投资建议（持有/加仓/减仓/清仓）
- 止损位和目标价建议
- 仓位管理建议
- 风险控制措施

**大模型多风格分析功能**：
- 进取型短期分析（日内/数天）：捕捉价格波动机会，高风险高收益
- 稳健型短期分析（日内/数天）：风险收益平衡，稳健收益
- 稳健型中期分析（数周-数月）：基本面和技术面结合，中长期价值投资
- 保守型中期分析（数周-数月）：注重长期价值投资，资产保值和稳健增长
- 支持通过配置开关切换生成全部四种或部分分析风格

**短期和中期投资者差异化分析**：
- **短期投资者**：关注VIX短期变化、成交额1日/5日变化率、止损位3-5%、立即操作
- **中期投资者**：关注VIX中期趋势、成交额5日/20日变化率、基本面、止损位8-12%、分批建仓

### 4. 港股模拟交易系统

**功能**：
- 基于大模型判断进行模拟交易
- 支持保守型、平衡型、进取型投资者偏好
- 严格遵循大模型建议，无随机操作
- 自动止损机制
- 交易记录和状态持久化
- 详细持仓展示和每日总结
- 邮件通知系统
- 大模型多风格分析功能

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

### 6. 机器学习交易模型

**功能**：
- 基于LightGBM和GBDT+LR的二分类模型，预测1天、5天、20天后的涨跌
- 整合52个特征（技术指标、市场环境、资金流向、基本面、美股市场、股票类型特征）
- 时间序列交叉验证（5折）
- **真实验证准确率：52.27%（次日）、53.49%（一周）、57.24%（一个月）**
- 特征重要性分析和可解释性分析
- 分类特征编码（LabelEncoder）
- 正则化增强（L1/L2正则化、早停、树深度控制）

**⚠️ 重要说明**：
- 一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值
- 次日和一周模型准确率在51-55%范围内，符合常见弱信号模型预期
- 建议与手工信号结合使用，不单独依赖

**特征类型**：
- **技术指标特征**（15个）：RSI、MACD、布林带、ATR、成交量比率、价格相对均线、涨跌幅
- **市场环境特征**（3个）：恒生指数收益率、股票相对恒指表现
- **资金流向特征**（5个）：价格位置、成交量信号、动量信号
- **基本面特征**（8个）：PE、PB、ROE、ROA、股息率、EPS、净利率、毛利率
- **美股市场特征**（10个）：标普500收益率、纳斯达克收益率、VIX变化率、VIX比率、美国10年期国债收益率及其变化率
- **股票类型特征**（18个）：股票类型（13种分类）、防御性评分、成长性评分、周期性评分、流动性评分、风险评分、衍生特征（银行/科技/周期股分析权重）

**分类特征编码**：
- 使用LabelEncoder将字符串类型的分类特征（如Stock_Type）转换为整数编码
- 编码器在训练时保存到模型文件，在预测时加载使用
- 支持处理训练时未见过的类别（使用默认值0）
- 确保训练和预测时的特征编码一致性

**正则化增强**（2026-01-26实施）：
- 减少树的数量：n_estimators 100→50
- 降低学习率：0.05→0.03
- 减少树深度：max_depth 6→4
- 减少叶子节点数：num_leaves 31→15
- 增加最小子样本数：min_child_samples 20→30（LGBM）、10→20（GBDT）
- 减少采样率：subsample 0.8→0.7, colsample_bytree 0.8→0.7
- 添加L1/L2正则化：reg_alpha=0.1, reg_lambda=0.1
- 添加额外的正则化参数：min_split_gain=0.1, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5
- 增加早停耐心：stopping_rounds=10（所有模型）

**性能改善**：
- 过拟合显著降低：次日模型-30.5%、一周模型-18.2%、一个月模型-13.5%
- 泛化能力提升：次日+1.18%、一周+1.44%

**模型性能**（2026-01-26最新训练结果，正则化增强后）：
- **次日模型（1天）**：平均验证准确率 **52.27%** (训练/验证差距：25.48%)
- **一周模型（5天）**：平均验证准确率 **53.49%** (训练/验证差距：33.97%)
- **一个月模型（20天）**：平均验证准确率 **57.24%** (训练/验证差距：32.45%) - **最佳表现**

**特征重要性 Top 10**：
1. VIX_Level（恐慌指数绝对值）
2. Stock_Type（股票类型）
3. 成交额变化率1日
4. VIX_Change（恐慌指数变化率）
5. HSI_Return_5d（恒生指数5日收益率）
6. SP500_Return（标普500日收益率）
7. US_10Y_Yield_Change（美国10年期国债收益率变化率）
8. US_10Y_Yield（美国10年期国债收益率）
9. 换手率变化率5日
10. Stock_Growth_Score（股票成长性评分）

**使用方法**：
```bash
# 训练所有周期模型
python ml_services/ml_trading_model.py --mode train --horizon 1 --model-type both

# 预测所有周期
python ml_services/ml_trading_model.py --mode predict --horizon 1 --model-type both

# 指定日期范围训练
python ml_services/ml_trading_model.py --mode train --horizon 1 --start-date 2024-01-01 --end-date 2025-12-31

# 指定预测日期
python ml_services/ml_trading_model.py --mode predict --horizon 1 --predict-date 2026-01-15
```

**输出文件**：
- `data/ml_trading_model_lgbm_1d.pkl` - LightGBM次日涨跌模型
- `data/ml_trading_model_lgbm_5d.pkl` - LightGBM一周涨跌模型
- `data/ml_trading_model_lgbm_20d.pkl` - LightGBM一个月涨跌模型
- `data/ml_trading_model_gbdt_lr_1d.pkl` - GBDT+LR次日涨跌模型
- `data/ml_trading_model_gbdt_lr_5d.pkl` - GBDT+LR一周涨跌模型
- `data/ml_trading_model_gbdt_lr_20d.pkl` - GBDT+LR一个月涨跌模型
- `data/ml_trading_model_*_importance.csv` - 特征重要性排名
- `data/ml_trading_model_*_predictions_*.csv` - 预测结果
- `output/gbdt_feature_importance.csv` - GBDT特征重要性
- `output/lr_leaf_coefficients.csv` - LR叶子节点系数
- `output/roc_curve.png` - ROC曲线图

**使用建议**：
- ✅ **一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值**
- ✅ **次日和一周模型准确率在51-55%范围内，符合常见弱信号模型预期**
- 作为手工信号的补充参考
- 只在信号一致时交易（保守策略）
- 定期重新训练（每周或每月）
- 跟踪实际预测准确率
- 关注高置信度预测（概率 > 60%）
- 根据股票类型特征，为不同类型股票提供差异化分析
- 优先关注一个月模型的预测结果（准确率最高）

### 7. 机器学习预测邮件通知

**功能**：
- 自动发送机器学习模型预测结果邮件
- 支持1天、5天、20天三个预测周期
- 显示 LightGBM 和 GBDT+LR 两种模型的预测结果
- 显示预测概率和置信度
- 显示当前价格和预测目标价格
- 统一的 HTML 表格格式

**使用方法**：
```bash
python ml_services/ml_prediction_email.py
```

**GitHub Actions 自动化**：
- 工作流文件：`.github/workflows/ml-prediction-alert.yml`
- 执行时间：
  - 每天香港时间 09:00 (UTC 01:00)
  - 每天香港时间 16:30 (UTC 08:30)
- 支持手动触发，可选择预测周期（1天/5天/20天/全部）

### 8. 模型对比工具

**功能**：
- 对比 LightGBM 和 GBDT+LR 两种模型的预测结果
- 分析预测一致性
- 计算概率差异统计

**使用方法**：
```bash
python ml_services/compare_models.py
```

**输出**：
- 控制台输出两种模型的预测对比
- 包括预测一致性、概率差异等统计信息

### 9. 通用技术分析工具

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
- **中期分析指标系统**：
  - 均线排列状态判断（多头/空头/混乱排列）
  - 均线斜率计算（MA20/MA50斜率和角度）
  - 均线乖离率（价格与均线的偏离程度）
  - 支撑阻力位识别（基于近期局部高低点）
  - 相对强弱指标（相对恒生指数的表现）
  - 中期趋势评分系统（综合趋势、动量、支撑阻力、相对强弱四维度评分）

### 10. 基本面数据获取器

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

### 11. 美股市场数据获取器

**功能**：
- 获取标普500指数 (^GSPC)、纳斯达克指数 (^IXIC)
- 获取VIX恐慌指数 (^VIX)
- 获取美国10年期国债收益率 (^TNX)
- 计算美股市场特征（收益率、变化率、比率）
- 作为港股预测的外部市场环境特征

**使用方法**：
```python
from ml_services.us_market_data import us_market_data

# 获取美股市场数据
df = us_market_data.get_all_us_market_data(period_days=730)
print(df)
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

### 大模型分析风格配置

在 `hsi_email.py` 中调整：

```python
# True = 生成全部四种（进取型短期、稳健型短期、稳健型中期、保守型中期）
# False = 只生成两种（稳健型短期、稳健型中期）
ENABLE_ALL_ANALYSIS_STYLES = False
```

---

## 📁 项目结构

```
fortune/
├── 📄 核心脚本
│   ├── ai_trading_analyzer.py          # AI 交易分析器
│   ├── batch_stock_news_fetcher.py     # 批量新闻获取器
│   ├── crypto_email.py                 # 加密货币监控器
│   ├── fundamental_data.py             # 基本面数据获取器
│   ├── gold_analyzer.py                # 黄金市场分析器
│   ├── hk_ipo_aastocks.py              # IPO 信息获取器
│   ├── hk_sector_analysis.py           # 板块分析器
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
├── 🤖 机器学习模块
│   └── ml_services/
│       ├── __init__.py                   # 模块初始化
│       ├── ml_trading_model.py          # 机器学习交易模型
│       ├── ml_prediction_email.py       # 机器学习预测邮件发送器
│       ├── compare_models.py             # 模型对比工具
│       ├── us_market_data.py            # 美股市场数据获取
│       └── base_model_processor.py      # 模型处理器基类
│
├── 🤖 大模型服务
│   └── llm_services/
│       └── qwen_engine.py              # Qwen 大模型接口
│
├── 📊 数据文件
│   ├── actual_porfolio.csv             # 实际持仓数据
│   ├── all_stock_news_records.csv      # 股票新闻记录
│   ├── all_dividends.csv               # 所有股息信息记录
│   ├── recent_dividends.csv            # 最近除净的股息信息
│   ├── upcoming_dividends.csv          # 即将除净的股息信息（未来90天）
│   ├── hsi_strategy_latest.txt         # 恒生指数策略分析
│   ├── ml_trading_model_lgbm_1d.pkl  # LightGBM次日涨跌模型
│   ├── ml_trading_model_lgbm_5d.pkl  # LightGBM一周涨跌模型
│   ├── ml_trading_model_lgbm_20d.pkl # LightGBM一个月涨跌模型
│   ├── ml_trading_model_gbdt_lr_1d.pkl  # GBDT+LR次日涨跌模型
│   ├── ml_trading_model_gbdt_lr_5d.pkl  # GBDT+LR一周涨跌模型
│   ├── ml_trading_model_gbdt_lr_20d.pkl # GBDT+LR一个月涨跌模型
│   ├── ml_trading_model_*_importance.csv  # 机器学习特征重要性排名
│   ├── ml_trading_model_*_predictions_*.csv  # 机器学习模型预测结果
│   ├── ml_trading_model_comparison.csv  # 模型对比结果
│   ├── simulation_state.json           # 模拟交易状态
│   ├── simulation_transactions.csv     # 交易历史记录
│   ├── simulation_portfolio.csv        # 投资组合价值变化记录
│   ├── simulation_trade_log_*.txt      # 交易日志（按日期分割）
│   ├── southbound_data_cache.pkl       # 南向资金数据缓存
│   └── fundamental_cache/               # 基本面数据缓存
│
├── 📈 图表输出
│   └── hk_smart_charts/                # 主力资金追踪图表
│
├── 📈 模型输出
│   ├── output/gbdt_feature_importance.csv  # GBDT特征重要性
│   ├── output/lr_leaf_coefficients.csv      # LR叶子节点系数
│   └── output/roc_curve.png                # ROC曲线图
│
└── 🚀 GitHub Actions
    └── .github/workflows/
        ├── crypto-alert.yml              # 加密货币监控
        ├── gold-analyzer.yml             # 黄金分析
        ├── ipo-alert.yml                 # IPO 信息
        ├── hsi-email-alert.yml           # 恒生指数监控
        ├── smart-money-alert.yml         # 主力资金追踪
        ├── ml-prediction-alert.yml     # 机器学习预测警报
        ├── ai-trading-analysis-daily.yml  # AI 交易分析日报
        ├── ml-train-models.yml.bak      # 机器学习模型训练（备份）
        └── hsi-email-alert-open_message.yml  # 恒生指数监控（备用）
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
│   ├── 股息数据 (AKShare)
│   ├── IPO 数据 (AAStocks)
│   └── 美股市场数据 (yfinance)
│
├── 🔍 分析层
│   ├── 技术分析 (technical_analysis.py)
│   │   ├── 短期技术指标（RSI、MACD、布林带、ATR等）
│   │   ├── TAV加权评分系统
│   │   └── 中期分析指标系统
│   │       ├── 均线排列状态判断
│   │       ├── 均线斜率计算
│   │       ├── 均线乖离率
│   │       ├── 支撑阻力位识别
│   │       ├── 相对强弱指标
│   │       └── 中期趋势评分系统
│   ├── 主力资金追踪 (hk_smart_money_tracker.py)
│   ├── 板块分析 (hk_sector_analysis.py)
│   │   ├── 板块涨跌幅排名
│   │   ├── 技术趋势分析
│   │   ├── 龙头识别
│   │   └── 资金流向分析
│   ├── AI 交易分析 (ai_trading_analyzer.py)
│   ├── 恒生指数策略 (hsi_llm_strategy.py)
│   ├── 新闻过滤 (batch_stock_news_fetcher.py)
│   ├── 基本面分析 (fundamental_data.py)
│   ├── 机器学习模块 (ml_services/)
│   │   ├── 机器学习交易模型 (ml_trading_model.py)
│   │   │   ├── 特征工程（52个特征，含18个股票类型特征）
│   │   │   ├── 分类特征编码（LabelEncoder）
│   │   │   ├── LightGBM二分类模型
│   │   │   ├── GBDT+LR两阶段模型
│   │   │   ├── 时间序列交叉验证（5折）
│   │   │   ├── 正则化增强（L1/L2、早停、树深度控制）
│   │   │   ├── 特征重要性分析
│   │   │   └── GBDT决策路径解析
│   │   ├── 机器学习预测邮件发送器 (ml_prediction_email.py)
│   │   ├── 模型对比工具 (compare_models.py)
│   │   ├── 美股市场数据获取器 (us_market_data.py)
│   │   └── 模型处理器基类 (base_model_processor.py)
│   └── 不同股票类型分析框架 (不同股票类型分析框架对比.md)
│       ├── hsi_email.py vs hk_smart_money_tracker.py对比
│       ├── 设计目标差异
│       ├── 分析框架对比
│       ├── 关键指标对比
│       ├── 适用场景说明
│       └── 集成策略建议
│
├── 💹 交易层
│   └── 模拟交易系统 (simulation_trader.py)
│
└── 🤖 服务层
    ├── 大模型服务 (llm_services/qwen_engine.py)
    ├── 邮件服务 (SMTP)
    └── 数据缓存 (fundamental_cache/, southbound_data_cache.pkl)
```

---

## 🤖 大模型集成

项目集成了 Qwen 大模型，提供智能分析和决策支持：

| 功能模块 | 应用场景 | 说明 |
|---------|---------|------|
| 主力资金追踪 | 股票分析 | 识别建仓和出货信号，集成基本面分析 |
| 模拟交易 | 交易决策 | 生成买卖信号和原因，支持多风格分析 |
| 新闻过滤 | 相关性判断 | 过滤相关新闻 |
| 黄金分析 | 市场分析 | 深度分析和投资建议 |
| 恒生指数策略 | 策略生成 | 生成交易策略 |
| **持仓分析** | **投资建议** | **分析现有持仓，提供专业建议** |
| **多风格分析** | **策略多样性** | **支持四种投资风格和周期的分析报告** |

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
- 🤖 **机器学习模型预测结果**
- 💼 **AI 持仓投资分析**
- 📊 **基本面指标展示**
- 📊 **中期评估指标展示**
- 📊 **成交额变化率指标展示**

邮件采用统一的表格化样式，清晰易读。

---

## 📊 数据文件说明

| 文件 | 说明 | 更新频率 |
|------|------|---------|
| `actual_porfolio.csv` | 实际持仓数据 | 手动更新 |
| `all_stock_news_records.csv` | 股票新闻记录 | 按需 |
| `all_dividends.csv` | 所有股息信息记录 | 按需 |
| `recent_dividends.csv` | 最近除净的股息信息 | 按需 |
| `upcoming_dividends.csv` | 即将除净的股息信息（未来90天） | 按需 |
| `hsi_strategy_latest.txt` | 恒生指数策略分析 | 每日 |
| `simulation_state.json` | 模拟交易状态 | 实时 |
| `simulation_transactions.csv` | 交易历史记录 | 实时 |
| `simulation_portfolio.csv` | 投资组合价值变化 | 实时 |
| `simulation_trade_log_*.txt` | 详细交易日志 | 每日 |
| `ml_trading_model_lgbm_1d.pkl` | LightGBM次日涨跌模型 | 按需 |
| `ml_trading_model_lgbm_5d.pkl` | LightGBM一周涨跌模型 | 按需 |
| `ml_trading_model_lgbm_20d.pkl` | LightGBM一个月涨跌模型 | 按需 |
| `ml_trading_model_gbdt_lr_1d.pkl` | GBDT+LR次日涨跌模型 | 按需 |
| `ml_trading_model_gbdt_lr_5d.pkl` | GBDT+LR一周涨跌模型 | 按需 |
| `ml_trading_model_gbdt_lr_20d.pkl` | GBDT+LR一个月涨跌模型 | 按需 |
| `ml_trading_model_*_importance.csv` | 机器学习特征重要性排名 | 按需 |
| `ml_trading_model_*_predictions_*.csv` | 机器学习模型预测结果 | 按需 |
| `ml_trading_model_comparison.csv` | 模型对比结果 | 按需 |
| `fundamental_cache/` | 基本面数据缓存 | 7天有效期 |
| `southbound_data_cache.pkl` | 南向资金数据缓存 | 按需 |
| `output/gbdt_feature_importance.csv` | GBDT特征重要性 | 按需 |
| `output/lr_leaf_coefficients.csv` | LR叶子节点系数 | 按需 |
| `output/roc_curve.png` | ROC曲线图 | 按需 |

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
lightgbm        # 机器学习模型（LightGBM）
scikit-learn    # 机器学习工具库
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
8. **⚠️ 机器学习模型说明**（2026-01-26最新）：
   - ✅ 一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值
   - ✅ 次日和一周模型准确率在51-55%范围内，符合常见弱信号模型预期
   - 通过正则化增强，显著降低了过拟合程度
   - 建议定期验证实际预测准确率，优先关注一个月模型的预测结果
   - 建议与手工信号结合使用，不单独依赖

---

## 💾 数据存储与历史分析功能

系统已实现完整的数据存储和历史分析能力，支持多种存储格式和灵活的历史分析方式。

### 数据存储功能

**1. 模拟交易数据存储**
- **simulation_state.json** - 模拟交易状态持久化（持仓、资金、交易历史）
- **simulation_transactions.csv** - 完整交易记录（买入、卖出、止损等）
- **simulation_portfolio.csv** - 投资组合价值变化记录（每日快照）
- **simulation_trade_log_*.txt** - 详细交易日志（按日期分割，包含所有操作细节）

**2. 基本面数据缓存**
- **fundamental_cache/** 目录 - 智能缓存机制，避免重复请求
  - 财务指标缓存（PE、PB、ROE、ROA等）
  - 利润表缓存（营业收入、净利润等）
  - 资产负债表缓存（资产、负债、权益等）
  - 现金流量表缓存（经营、投资、筹资现金流）
- **7天缓存有效期**，自动过期更新，可手动清除

**3. 其他数据缓存**
- **southbound_data_cache.pkl** - 南向资金数据缓存
- **ml_trading_model_*.pkl** - 机器学习模型文件（LightGBM、GBDT+LR）
- **ml_trading_model_*_importance.csv** - 特征重要性排名
- **ml_trading_model_*_predictions_*.csv** - 预测结果缓存

### 历史分析功能

**1. 基于指定日期的分析**
支持基于任意历史日期进行分析，所有技术指标都基于截止到指定日期的数据进行计算：

```bash
# 分析指定日期的恒生指数
python hsi_email.py --date 2025-10-25

# 分析指定日期的主力资金追踪
python hk_smart_money_tracker.py --date 2025-10-25
```

**2. 日期范围分析**
支持指定日期范围进行多时间维度分析：

```bash
# 分析指定日期范围的AI交易表现
python ai_trading_analyzer.py --start-date 2025-12-01 --end-date 2025-12-31

# 分析1天数据
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 分析5天数据
python ai_trading_analyzer.py --start-date 2025-12-31 --end-date 2026-01-05

# 分析1个月数据
python ai_trading_analyzer.py --start-date 2025-12-05 --end-date 2026-01-05
```

**3. 机器学习模型历史回测**
支持基于历史数据的模型训练和预测回测：

```bash
# 使用指定日期范围训练模型
python ml_services/ml_trading_model.py --mode train --horizon 1 --start-date 2024-01-01 --end-date 2025-12-31

# 预测指定日期的股票涨跌
python ml_services/ml_trading_model.py --mode predict --horizon 1 --predict-date 2026-01-15

# 完整历史回测（使用train_and_predict_all.sh）
./train_and_predict_all.sh --backtest --start-date 2024-01-01 --end-date 2024-12-31 --predict-date 2024-12-31
```

**4. 交易记录回溯**
系统自动保存完整的交易历史记录，支持随时回溯和分析：

```bash
# 查看所有交易记录
cat data/simulation_transactions.csv

# 查看特定日期的交易日志
cat data/simulation_trade_log_20260127.txt

# 使用ai_trading_analyzer.py回溯分析历史交易表现
python ai_trading_analyzer.py --file data/simulation_transactions.csv
```

### 技术特点

1. **持久化存储**：所有关键数据都持久化存储，系统重启后可恢复
2. **智能缓存**：基本面数据使用7天缓存机制，提高性能避免重复请求
3. **时间序列完整**：所有历史数据按时间顺序存储，支持完整的时间序列分析
4. **灵活查询**：支持按日期、日期范围、股票代码等多维度查询
5. **可追溯性**：完整的交易日志和决策历史，支持审计和复盘

### 应用场景

- **策略回测**：基于历史数据验证交易策略的有效性
- **性能分析**：分析AI推荐策略的历史盈利能力
- **模型验证**：验证机器学习模型的历史准确率
- **风险复盘**：回溯分析交易决策的风险控制效果
- **趋势分析**：基于历史数据分析市场趋势变化
- **持仓分析**：跟踪持仓变化和历史盈亏

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
- 基本面指标分析
- 中期评估指标分析

### Q9: 什么是中期评估指标？

**A**: 中期评估指标是用于分析数周至数月投资周期的技术指标，包括：
- **均线排列**：识别多头/空头/混乱排列，判断趋势方向
- **均线斜率**：通过线性回归计算MA20/MA50斜率和角度，判断趋势强度
- **乖离率**：评估价格与均线的偏离程度，识别超买超卖状态
- **支撑阻力位**：基于近期局部高低点识别关键价格水平
- **相对强弱**：计算股票相对于恒生指数的表现
- **中期趋势评分**：综合趋势、动量、支撑阻力、相对强弱四维度评分（0-100分）

### Q10: 如何配置大模型分析风格？

**A**: 在 `hsi_email.py` 文件中修改 `ENABLE_ALL_ANALYSIS_STYLES` 参数：
- `True`：生成全部四种分析（进取型短期、稳健型短期、稳健型中期、保守型中期）
- `False`：只生成两种分析（稳健型短期、稳健型中期）

### Q11: 什么是机器学习交易模型？

**A**: 机器学习交易模型是一个基于LightGBM和GBDT+LR的二分类模型，用于预测股票1天、5天、20天后的涨跌。它整合了52个特征（包括18个股票类型特征），包括技术指标、市场环境、资金流向、基本面、美股市场和股票类型特征，支持分类特征编码和正则化增强。2026-01-26最新训练结果显示：真实验证准确率为52.27%（次日）、53.49%（一周）、57.24%（一个月），其中一个月模型接近业界优秀水平（60%）。

### Q12: 如何训练机器学习模型？

**A**: 运行以下命令：
```bash
# 训练所有周期模型
python ml_services/ml_trading_model.py --mode train --horizon 1 --model-type both

# 训练指定周期模型
python ml_services/ml_trading_model.py --mode train --horizon 5 --model-type both

# 指定日期范围训练
python ml_services/ml_trading_model.py --mode train --horizon 1 --start-date 2024-01-01 --end-date 2025-12-31
```

模型会使用24只自选股的2年历史数据进行训练，并通过5折时间序列交叉验证。

### Q13: 机器学习模型的准确率如何？

**A**: （2026-01-26最新训练结果，正则化增强后）
- **次日模型（1天）**：平均验证准确率 **52.27%** (训练/验证差距：25.48%)
- **一周模型（5天）**：平均验证准确率 **53.49%** (训练/验证差距：33.97%)
- **一个月模型（20天）**：平均验证准确率 **57.24%** (训练/验证差距：32.45%) - **最佳表现**

**重要说明**：
- ✅ **一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值**
- ✅ **次日和一周模型准确率在51-55%范围内，符合常见弱信号模型预期**
- 通过正则化增强，显著降低了过拟合程度（次日-30.5%、一周-18.2%、一个月-13.5%）
- 建议结合多种信号和方法，不要单独依赖单一模型
- 定期跟踪实际预测准确率，验证模型有效性
- 优先关注一个月模型的预测结果（准确率最高）

### Q14: 如何使用机器学习模型的预测结果？

**A**: 
1. 运行 `python ml_services/ml_trading_model.py --mode predict --horizon 1 --model-type both` 获取预测结果
2. 查看预测概率，关注高置信度预测（概率 > 60%）
3. 与手工信号对比，只在信号一致时交易（保守策略）
4. 定期跟踪实际准确率，验证模型有效性
5. 关注 GBDT+LR 模型的预测结果（准确率较高）

### Q15: 机器学习模型需要多久重新训练一次？

**A**: 建议每周或每月重新训练一次，以适应市场变化。市场风格切换时，模型可能需要更频繁的更新。

### Q16: 机器学习模型支持哪些股票？

**A**: 目前支持配置在 `ml_services/ml_trading_model.py` 中的24只自选股。可以在代码中添加或修改股票列表。

### Q17: 机器学习模型的特征有哪些？

**A**: 机器学习模型整合了52个特征：
- **技术指标特征**（15个）：移动平均线、RSI、MACD、布林带、ATR、成交量比率、价格位置、涨跌幅等
- **市场环境特征**（3个）：恒生指数收益率、相对表现
- **资金流向特征**（5个）：价格位置、成交量信号、动量信号
- **基本面特征**（8个）：PE、PB、ROE、ROA、股息率、EPS、净利率、毛利率
- **美股市场特征**（10个）：标普500收益率（1日、5日、20日）、纳斯达克收益率（1日、5日、20日）、VIX变化率、VIX比率、美国10年期国债收益率及其变化率
- **股票类型特征**（18个）：股票类型（13种分类：bank/tech/energy/utility/semiconductor/ai/new_energy/shipping/exchange/insurance/biotech/environmental/index）、防御性评分、成长性评分、周期性评分、流动性评分、风险评分、衍生特征（银行/科技/周期股分析权重）
- **分类特征编码**：使用LabelEncoder将字符串类型的分类特征转换为整数编码
- **正则化增强**：L1/L2正则化、早停、树深度控制等

### Q18: 如何验证机器学习模型的有效性？

**A**: 
1. **多维评估**：不仅看准确率，还要看 AUC、Log Loss、Precision、Recall、F1 等指标
2. **基线对比**：对比简单基线（恒定预测上涨、动量策略）的准确率
3. **时间序列验证**：确保训练集和验证集按时间顺序分割
4. **实际跟踪**：记录模型在真实市场中的实际预测准确率
5. **回测验证**：使用历史数据回测，验证模型历史表现
6. **交叉验证**：使用时间序列交叉验证，避免普通 k-fold 分割

### Q19: 如何发送机器学习预测邮件？

**A**: 
1. 本地运行：`python ml_services/ml_prediction_email.py`
2. GitHub Actions 自动化：工作流会自动执行并发送邮件
3. 邮件包含所有预测周期的预测结果
4. 可以通过环境变量 `PREDICTION_HORIZONS` 控制发送哪些周期的预测

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

**最后更新**: 2026-02-01

## 🎉 最近更新

### 2026-02-01
- **新增功能**：实现港股板块分析模块（hk_sector_analysis.py），提供板块涨跌幅排名、技术趋势分析、龙头识别、资金流向分析
- **新增功能**：板块分析模块涵盖13个板块（银行、科技、半导体、AI、新能源、环保、能源、航运、交易所、公用事业、保险、生物医药、指数基金）
- **新增功能**：在hk_smart_money_tracker.py和hsi_email.py中集成板块分析数据，为大模型分析提供宏观市场背景
- **提示词重构**：重构hk_smart_money_tracker.py的提示词结构，从混乱的"6层"重构为清晰的"0-5层"分析框架（前置检查→风险控制→核心信号识别→技术面分析→宏观环境→辅助信息）
- **提示词优化**：简化JSON数据结构，移除重复字段，提高提示词可读性和效率
- **文档更新**：更新IFLOW.md和README.md，添加板块分析模块和相关使用说明

### 2026-01-26
- **新增功能**：机器学习模型集成股票类型特征（13种股票类型分类：bank/tech/energy/utility/semiconductor/ai/new_energy/shipping/exchange/insurance/biotech/environmental/index）
- **新增功能**：机器学习模型添加18个股票类型特征（股票类型、防御性评分、成长性评分、周期性评分、流动性评分、风险评分、衍生特征）
- **新增功能**：实现分类特征编码（LabelEncoder），解决LightGBM无法直接处理字符串数据的问题
- **性能优化**：实施全面的正则化增强，显著降低过拟合程度（次日-30.5%、一周-18.2%、一个月-13.5%）
- **性能提升**：通过正则化提升模型泛化能力，验证准确率提升（次日+1.18%、一周+1.44%）
- **模型性能**：一个月模型准确率57.24%接近业界优秀水平（60%），具有实际交易价值
- **特征扩展**：机器学习模型支持52个特征（从34个增加到52个）
- **新增文档**：添加不同股票类型分析框架对比文档（不同股票类型分析框架对比.md），详细说明hsi_email.py和hk_smart_money_tracker.py的差异和适用场景

### 2026-01-24
- **新增功能**：在恒生指数价格监控器的交易信号总结表中添加成交额变化1日和换手率变化5日两个关键流动性指标
- **新增功能**：在恒生指数价格监控器的指标说明中添加VIX恐慌指数、成交额变化率、换手率变化率的详细解释
- **功能优化**：删除恒生指数价格监控器中的快速决策参考表和决策检查清单，简化邮件内容，提高可读性
- **功能优化**：在港股主力资金追踪器中集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- **功能优化**：在港股主力资金追踪器中采用业界标准的0-5层分析框架
- **功能优化**：在恒生指数价格监控器中针对短期和中期投资者提供不同的分析重点调整

### 2026-01-23
- **新增功能**：在恒生指数价格监控器中集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- **新增功能**：在恒生指数价格监控器中采用业界标准的0-5层分析框架
- **新增功能**：在恒生指数价格监控器中针对短期和中期投资者提供不同的分析重点调整

### 2026-01-22
- **新增功能**：在港股主力资金追踪器中集成ML模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- **新增功能**：在港股主力资金追踪器中采用业界标准的0-5层分析框架
- **新增功能**：在港股主力资金追踪器中在技术指标展示中添加VIX恐慌指数、成交额变化率（1日/5日/20日）、换手率变化率（5日/20日）等关键流动性指标

### 2026-01-19
- **新增功能**：机器学习预测邮件通知
  - 实现自动发送ML模型预测结果邮件
  - 支持1天、5天、20天三个预测周期
  - 添加 GitHub Actions 自动化工作流
- **架构重组**：将 ML 模块整合到 ml_services 目录
- **功能增强**：机器学习模型支持多周期预测（1天、5天、20天）
- **功能扩展**：机器学习模型集成美股市场特征，提升模型能力
- **功能扩展**：实现模型可解释性分析（GBDT决策路径解析、特征重要性分析）

### 2026-01-18
- **新增功能**：恒生指数价格监控器集成AI智能持仓分析功能
- **新增功能**：恒生指数价格监控器集成基本面指标和中期评估指标
- **新增功能**：恒生指数价格监控器支持大模型多风格分析
- **新增功能**：机器学习交易模型支持多周期预测
- **新增功能**：模型对比工具
- **新增功能**：机器学习预测邮件发送器

### 2026-01-17
- **新增功能**：人工智能股票交易盈利能力分析器
- **新增功能**：机器学习交易模型
- **新增功能**：港股基本面数据获取器
- **新增功能**：美股市场数据获取器
- **新增功能**：模型处理器基类

### 2026-01-16
- **新增功能**：恒生指数价格监控器集成股息信息追踪功能
- **新增功能**：恒生指数价格监控器集成VaR风险价值计算
- **新增功能**：通用技术分析工具集成TAV加权评分系统
- **新增功能**：通用技术分析工具集成中期分析指标系统

### 2026-01-15
- **新增功能**：港股主力资金追踪器集成基本面数据分析
- **新增功能**：批量获取自选股新闻支持 yfinance
- **新增功能**：机器学习预测警报工作流

### 2026-01-10
- **架构重组**：将 ML 模块整合到 ml_services 目录
- **Bug修复**：修复机器学习模型导入错误
- **功能扩展**：增加恒生指数大模型策略分析器

---

## 🚀 未来计划

- [x] 机器学习模型准确率优化（目标：55-60%）- ✅ 一个月模型准确率已达57.24%
- [x] 集成量化回测框架 - ✅ `ai_trading_analyzer.py` 已实现完整回测框架
- [x] 添加风险管理和止损止盈优化功能 - ✅ `hsi_email.py` 已实现完整的风险管理和止损止盈功能
- [x] 添加更多技术分析指标和信号 - ✅ 已实现50+技术分析指标和信号（风险、流动性、基本面、中期、综合评分等）
- [x] 实现异常检测和风险预警系统 - ✅ 已实现市场环境异常预警（VIX>30警报、成交额萎缩预警、系统性崩盘风险评分）和股票行为异常检测（价格暴涨暴跌、成交量异常放大、技术指标异常值）
- [x] 添加数据存储和历史分析功能 - ✅ 已实现完整的持久化存储系统（JSON/CSV/pickle缓存）和历史分析能力（指定日期分析、日期范围分析、历史回测）

### 机器学习模型特征工程优化计划（基于GitHub港股项目分析）

参考 GitHub 上的优秀港股机器学习项目（`crownpku/hk_ipo_prediction`、`hengruiyun/AI-Stock-Master`、`zvtvz/zvt`、`jasperyeoh/Hybrid-Topic-LLM`），计划实现以下特征工程优化：

#### 新特征开发
- [x] **情感趋势特征**：✅ 已实现情感指标的移动平均（MA3、MA7、MA14）、波动率、变化率
- [ ] **主题分布特征**：使用LDA对新闻进行主题建模（10个主题），将主题分布作为特征输入
- [ ] **交互特征**：创建主题与情感的交互特征（10个主题×情感分数）、技术指标与基本面的交互
- [ ] **预期差距特征**：计算新闻相对于市场预期的差距，区分"预期内"和"超预期"新闻
- [x] **多维度情感分析**：✅ 已实现四维情感评分（Relevance/Impact/Expectation_Gap/Sentiment），参考 `jasperyeoh/Hybrid-Topic-LLM`（`llm_services/sentiment_analyzer.py`）
- [ ] **技术动量特征**：实现 `hengruiyun/AI-Stock-Master` 的TMA算法特征（趋势斜率、一致性、置信度、成交量因子）

#### 特征工程优化
- [ ] **特征选择优化**：使用F-test + 互信息混合方法进行特征选择，减少噪声特征
- [ ] **时间序列交叉验证**：强化5折时间序列分割，严格避免前瞻偏差
- [ ] **假阳性优化**：实现F0.5阈值优化策略，降低假阳性率（目标降低46.3%）
- [ ] **正则化增强**：持续优化正则化参数（当前：树深度4、min_child_samples30、L1/L2=0.1）
- [ ] **过拟合控制**：降低训练/验证差距（当前：次日25.48%、一周33.97%、一个月32.45%，目标<15%）

#### 性能目标
- **一个月模型**：从57.24%提升至60%+（业界优秀水平）
- **一周模型**：从53.49%提升至55-58%（有意义改进）
- **次日模型**：从52.27%提升至53-55%（接近弱信号上限）
- **整体目标**：所有模型训练/验证差距<15%

### 其他功能计划
- [ ] 机器学习预测结果可视化优化
- [ ] 新增更多美股市场特征（欧洲股市、商品期货）
- [ ] 集成更多大模型服务提供商
- [ ] 实现 Web 界面展示信息
- [ ] 扩展更多数据源接口
- [ ] 机器学习模型支持实时在线学习和增量更新
- [ ] 实现投资组合优化算法
- [ ] 实现机器学习模型自动超参数调优
- [ ] 集成深度学习模型（LSTM、Transformer等）
- [ ] 实现强化学习交易系统
- [ ] 集成更多市场数据源（期权、期货、外汇等）
- [ ] 实现智能推荐系统
- [ ] 实现实时数据流处理能力
- [ ] 实现分布式训练和推理
- [ ] 实现模型版本管理和A/B测试
- [ ] 实现自定义指标和策略编辑器
- [ ] 实现社区策略分享和订阅功能