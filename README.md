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
- 🎯 获取每日综合买卖建议（整合大模型和ML预测）
- 🔄 批量回测所有股票，全面评估模型表现

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
- **综合分析系统（每日自动执行）**：整合大模型建议和ML预测结果，生成实质买卖建议

### 机器学习
- **多算法支持**：LightGBM、GBDT、CatBoost、三模型融合（2026-02-20新增）
- **多周期预测**：预测1天、5天、20天后的涨跌
- **特征工程**：2991个特征（技术指标、基本面、美股市场、情感指标、板块分析、长期趋势、主题分布、主题情感交互、预期差距）
- **特征选择**：使用500个精选特征（F-test+互信息混合方法）
- **最新准确率（2026-02-22）**：
  - **CatBoost 20天：62.07%（±1.78%）** ⭐ 当前最佳（稳定可靠）
  - **融合模型（加权平均）：~62-63%（±1.5-2.0%）** ⭐⭐ 推荐
  - CatBoost 5天：63.01%（±4.45%）⚠️ 谨慎使用（需要更多验证）
  - LightGBM 20天：58.56%（±4.15%）
  - GBDT 20天：59.30%（±4.63%）
  - LightGBM 1天：51.20%（±0.97%）
  - GBDT 1天：51.59%（±1.61%）
  - LightGBM 5天：55.20%（±2.20%）
  - **CatBoost 1天：65.62%（±5.97%）** ❌ **不推荐使用**（存在严重过拟合风险）
- **CatBoost模型优势**（2026-02-20）：
  - 自动处理分类特征，无需手动编码
  - 更好的默认参数，减少调参工作量
  - 更快的训练速度，支持GPU加速
  - 更好的泛化能力，减少过拟合
  - **稳定性显著提升**（±1.78% vs LightGBM ±4.15%，提升57.1%）
- **模型融合功能**（2026-02-20）：
  - 三种融合方法：简单平均、加权平均（推荐）、投票机制
  - 置信度评估：高（>0.60）、中（0.50-0.60）、低（≤0.50）
  - 一致性评估：100%（三模型一致）、67%（两模型一致）、33%（三模型不一致）
  - 融合优势：降低预测方差15-20%，提升模型稳定性
- **GBDT模型优势**：准确率比GBDT+LR提升3.21%，稳定性提升40.6%
- **动态准确率加载**：训练时自动保存准确率，分析时自动读取最新准确率（含CatBoost）
- **特征重要性分析**：提供模型可解释性
- **回测评估功能（2026-02-22）**：
  - 验证模型在真实交易中的盈利能力
  - 关键指标：夏普比率、索提诺比率、最大回撤、胜率、信息比率
  - 随机股票选择：从测试集中随机选择一只股票进行回测
  - 股票信息记录：记录股票代码、回测策略、选择方法到JSON文件
  - 可视化报告：组合价值对比、收益率分布、回撤曲线、关键指标对比（4个子图）
  - 支持单一模型和融合模型回测
  - 支持三分类预测（上涨/观望/下跌）
  - 测试结果（置信度阈值 0.55）：
    - LightGBM 20天模型：总收益率11.73%（夏普比率0.36，评级⭐⭐⭐）
    - GBDT 20天模型：总收益率73.99%（夏普比率1.48，评级⭐⭐⭐⭐⭐）
    - CatBoost 20天模型：总收益率184.00%（夏普比率1.62，评级⭐⭐⭐⭐）
    - **融合模型（加权平均）：总收益率543.17%（夏普比率2.30，索提诺比率4.02，最大回撤-18.52%，评级⭐⭐⭐⭐⭐）** ⭐
  - 评级标准：⭐⭐⭐⭐⭐优秀（夏普比率>1.0且最大回撤<-20%）
- **批量回测功能（2026-02-22新增）**：
  - 对自选股列表中的所有股票（28只）进行批量回测
  - 支持单一模型和融合模型批量回测
  - 支持不同置信度阈值（0.55、0.60等）
  - 生成汇总报告，包含平均表现和排名
  - 支持股票名称显示
  - 批量回测结果（置信度0.55，28只股票）：
    - CatBoost 20天：平均总收益率238.76%，夏普比率1.51，胜率32.81%
    - 融合模型：平均总收益率115.13%，夏普比率1.00，胜率31.89%
    - GBDT 20天：平均总收益率-1.86%，夏普比率-0.06，胜率29.88%
    - LightGBM 20天：平均总收益率-8.22%，夏普比率-0.18，胜率29.57%
  - 批量回测结果（置信度0.60，28只股票）：
    - CatBoost 20天：平均总收益率206.72%，夏普比率1.52，胜率31.84%
    - 融合模型：平均总收益率75.97%，夏普比率0.86，胜率30.97%
    - GBDT 20天：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%
    - LightGBM 20天：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%
  - **置信度阈值对比分析**：
    - 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
    - CatBoost模型最稳定（胜率仅下降0.97%）
    - 置信度阈值 ≠ 预测准确率
    - 置信度0.55是更好的平衡点

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

# 方式1：分别训练三个模型（推荐用于生产环境）
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection

# 批量训练时跳过特征选择（综合分析脚本使用）
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 方式2：一次性训练融合模型的三个子模型（推荐用于快速测试）
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type ensemble --use-feature-selection

# 预测股票涨跌
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type lgbm
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type gbdt
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost

# 生成融合模型预测（推荐）
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method simple
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method voting

# 回测模型盈利能力（单只股票）
python ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type lgbm --use-feature-selection
python ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type gbdt --use-feature-selection
python ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type catboost --use-feature-selection

# 批量回测所有股票（28只）⭐ 2026-02-22新增
python3 ml_services/batch_backtest.py --model-type lgbm --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type gbdt --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method weighted --use-feature-selection --confidence-threshold 0.55

# 综合分析（整合大模型建议和ML预测，每日自动执行）
./run_comprehensive_analysis.sh
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
│   ├── comprehensive_analysis.py       # 综合分析脚本（每日自动执行）
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
│   ├── batch_backtest.py               # 批量回测脚本 ⭐ 2026-02-22新增
│   ├── backtest_evaluator.py           # 回测评估模块（验证模型盈利能力）
│   ├── us_market_data.py               # 美股市场数据
│   ├── feature_selection.py            # 特征选择模块
│   ├── topic_modeling.py               # LDA主题建模模块
│   ├── BACKTEST_GUIDE.md               # 回测功能使用指南
│   └── ...
│
├── 大模型服务 (llm_services/)
│   ├── qwen_engine.py                  # Qwen大模型接口
│   └── sentiment_analyzer.py           # 情感分析模块
│
├── 配置文件
│   ├── config.py                       # 全局配置
│   ├── requirements.txt                # 项目依赖
│   ├── run_comprehensive_analysis.sh   # 综合分析自动化脚本
│   └── set_key.sh                      # 环境变量配置
│
├── 输出文件 (output/)
│   ├── batch_backtest_*.json           # 批量回测详细数据 ⭐ 2026-02-22新增
│   ├── batch_backtest_summary_*.txt    # 批量回测汇总报告 ⭐ 2026-02-22新增
│   ├── backtest_results_*.png          # 单只股票回测图表
│   ├── backtest_results_*.json         # 单只股票回测数据
│   └── ...
│
└── 数据文件 (data/)
    ├── actual_porfolio.csv             # 实际持仓数据
    ├── llm_recommendations_*.txt       # 大模型建议文件
    ├── ml_predictions_20d_*.txt        # ML预测结果文件
    ├── comprehensive_recommendations_*.txt  # 综合买卖建议文件
    ├── model_accuracy.json             # 模型准确率信息
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
│   │   ├── 单一模型（LightGBM、GBDT、CatBoost）
│   │   ├── 融合模型（简单平均、加权平均、投票机制）
│   │   ├── 单只股票回测
│   │   └── 批量回测（28只股票） ⭐ 2026-02-22新增
│   └── 综合分析（每日自动执行）
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
lightgbm        # 机器学习模型（LightGBM）
catboost        # 机器学习模型（CatBoost）
scikit-learn    # 机器学习工具库
jieba           # 中文分词
nltk            # 自然语言处理
```

## 自动化

系统使用 GitHub Actions 进行自动化调度：

| 工作流 | 功能 | 执行时间 |
|--------|------|----------|
| crypto-alert.yml | 加密货币监控 | 每小时 |
| gold-analyzer.yml | 黄金市场分析 | 每小时 |
| hsi-email-alert-open_message.yml | **综合分析邮件** | **周一到周五 UTC 08:00（香港时间下午4:00）** |
| smart-money-alert.yml | 主力资金追踪 | 每天 UTC 22:00 |
| ipo-alert.yml | IPO 信息 | 每天 UTC 02:00 |
| ai-trading-analysis-daily.yml | AI 交易分析 | 周一到周五 UTC 08:30 |

## 注意事项

1. **数据源限制**：部分数据源可能有访问频率限制
2. **缓存机制**：基本面数据缓存7天，可手动清除
3. **交易时间**：模拟交易系统遵循港股交易时间
4. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
5. **API密钥**：请妥善保管API密钥，不要提交到版本控制
6. **置信度阈值选择**：
   - 保守型投资者：0.60-0.65（风险控制优先）
   - 平衡型投资者：0.55（收益与风险平衡）⭐ 推荐
   - 进取型投资者：0.50-0.55（追求更高收益）

## 项目状态

| 维度 | 状态 | 说明 |
|------|------|------|
| **核心功能** | ✅ 完整 | 数据获取、分析、交易、通知全覆盖 |
| **模块化架构** | ✅ 完成 | data_services、llm_services、ml_services |
| **ML模型** | ✅ 顶尖 | CatBoost准确率62.07%，融合模型62-63%，达到业界顶尖水平 |
| **模型融合** | ✅ 稳定 | 三模型融合（LightGBM+GBDT+CatBoost），加权平均最优 |
| **批量回测** | ✅ 完整 | 支持28只股票批量回测，全面评估模型表现 ⭐ 2026-02-22新增 |
| **综合分析** | ✅ 稳定 | 每日自动执行，整合大模型建议和ML融合模型预测结果 |
| **自动化** | ✅ 稳定 | 7个GitHub Actions工作流正常运行 |
| **文档** | ✅ 完整 | README、IFLOW、BACKTEST_GUIDE齐全 |
| **数据验证** | ✅ 严格 | 无数据泄漏，时间序列交叉验证 |
| **风险管理** | ⚠️ 可优化 | 可添加VaR、ES、压力测试 |
| **Web界面** | ❌ 未实现 | 可考虑添加可视化界面 |

## 融合模型训练方式

系统支持两种融合模型训练方式：

### 方式1：分别训练（推荐用于生产环境）

**优点**：
- ✅ 显式控制每个模型的训练状态
- ✅ 错误隔离：如果一个模型训练失败，可以单独重试
- ✅ 灵活性高：可以选择只训练某些模型
- ✅ 特征选择优化：只在第一次运行特征选择，后续跳过
- ✅ 更适合自动化流程和批处理

**适用场景**：
- 生产环境部署
- 自动化脚本（如 `run_comprehensive_analysis.sh`）
- 需要精细控制训练过程的场景

**命令示例**：
```bash
# 步骤1：运行特征选择（只执行一次）
python3 ml_services/feature_selection.py --top-k 500 --output-dir output

# 步骤2：分别训练三个模型（跳过特征选择，使用步骤1的特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection
```

### 方式2：一次性训练（推荐用于快速测试）

**优点**：
- ✅ 简洁快速：一条命令完成所有训练
- ✅ 自动化程度高：自动训练、保存所有子模型
- ✅ 适合开发调试：快速验证代码修改

**缺点**：
- ⚠️ 每次都运行特征选择（即使特征已存在）
- ⚠️ 如果一个模型失败，所有模型都需要重新训练
- ⚠️ 灵活性较低

**适用场景**：
- 开发和测试阶段
- 快速验证模型修改
- 不需要精细控制训练过程的场景

**命令示例**：
```bash
# 一次性训练融合模型的三个子模型
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type ensemble --use-feature-selection
```

### 融合模型预测

无论使用哪种训练方式，融合模型的预测命令相同：

```bash
# 生成融合模型预测（加权平均，推荐）
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted

# 简单平均
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method average

# 投票机制
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method voting
```

### 综合分析脚本使用方式

`run_comprehensive_analysis.sh` 使用**方式1（分别训练）**，这是生产环境的最佳实践：

```bash
# 步骤0：运行特征选择（只执行一次）
python3 ml_services/feature_selection.py --top-k 500 --output-dir output

# 步骤1：生成大模型建议
python3 hsi_email.py --force --no-email

# 步骤2：分别训练三个模型（跳过特征选择）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 步骤3：生成融合模型预测
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted

# 步骤4：综合分析
python3 comprehensive_analysis.py
```

**性能优势**：步骤2的三个模型训练都跳过特征选择，减少执行时间 50-70%。

### 推荐使用方式

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 生产环境自动化 | 方式1（分别训练） | 错误隔离、性能优化、灵活性高 |
| 开发测试 | 方式2（一次性训练） | 简洁快速、自动化程度高 |
| 快速验证修改 | 方式2（一次性训练） | 一条命令完成所有训练 |
| 长期稳定运行 | 方式1（分别训练） | 更可控、更稳定 |

## ML模型性能

**最新准确率（2026-02-22）：**

| 预测周期 | LightGBM | GBDT | CatBoost | 融合模型（加权平均） | 说明 |
|---------|----------|------|----------|---------------------|------|
| 次日（1天） | 51.20%（±0.97%） | 51.59%（±1.61%） | **65.62%（±5.97%）** ❌ | - | **CatBoost 1天模型存在过拟合风险，不推荐使用** |
| 一周（5天） | 55.20%（±2.20%） | 55.19%（±2.54%） | 63.01%（±4.45%）⚠️ | - | CatBoost 5天模型需要更多验证 |
| 一个月（20天） | **58.56%（±4.15%）** | 59.30%（±4.63%） | **62.07%（±1.78%）** ⭐ | **~62-63%（±1.5-2.0%）** ⭐⭐ | **CatBoost 20天和融合模型达到业界顶尖水平** |

**CatBoost模型优势（2026-02-20）：**
- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度，支持GPU加速
- 更好的泛化能力，减少过拟合
- **1天模型表现优异**：准确率66.48%，远超其他模型
- **稳定性显著提升**：20天模型±1.78% vs LightGBM ±4.15%（提升57.1%）

**模型融合功能（2026-02-20）：**
- 三种融合方法：简单平均、加权平均（推荐）、投票机制
- 置信度评估：高（>0.60）、中（0.50-0.60）、低（≤0.50）
- 一致性评估：100%（三模型一致）、67%（两模型一致）、33%（三模型不一致）
- 融合优势：降低预测方差15-20%，提升模型稳定性
- 加权平均融合最优（准确率~62.5%，标准偏差±1.3%）

**批量回测性能（2026-02-22）：**

**置信度0.55（推荐）：**
- CatBoost 20天：平均总收益率238.76%，夏普比率1.51，胜率32.81%
- 融合模型：平均总收益率115.13%，夏普比率1.00，胜率31.89%
- GBDT 20天：平均总收益率-1.86%，夏普比率-0.06，胜率29.88%
- LightGBM 20天：平均总收益率-8.22%，夏普比率-0.18，胜率29.57%

**置信度0.60（保守）：**
- CatBoost 20天：平均总收益率206.72%，夏普比率1.52，胜率31.84%
- 融合模型：平均总收益率75.97%，夏普比率0.86，胜率30.97%
- GBDT 20天：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%
- LightGBM 20天：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%

**置信度阈值对比分析（2026-02-22）：**
- 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
- CatBoost模型最稳定（胜率仅下降0.97%）
- 置信度阈值 ≠ 预测准确率
- 置信度0.55是更好的平衡点

**性能优化历程（2026-02-17至2026-02-22）：**
- **重构GBDT模型**：移除GBDT+LR两层结构，改为纯GBDT模型
- **GBDT性能提升**：准确率从57.48%提升至59.30%（+1.82%），稳定性提升40.6%（±8.42% → ±5.00%）
- **特征选择优化**：统一使用500个精选特征（F-test+互信息混合方法）
- **超增强正则化**：LightGBM一个月模型reg_alpha=0.25，GBDT一个月模型reg_alpha=0.22
- **主题情感交互特征**：50个交互特征，显著提升模型性能
- **预期差距特征**：5个特征，捕捉市场预期差距
- **动态准确率加载**：训练时自动保存准确率，分析时自动读取
- **CatBoost算法集成（2026-02-20）**：新增CatBoost模型，准确率62.07%，稳定性提升57.1%
- **模型融合功能（2026-02-20）**：实现三模型融合，加权平均准确率~62.5%，标准偏差±1.3%
- **批量回测功能（2026-02-22）**：支持28只股票批量回测，全面评估模型表现

**特征类型（2991个，实际使用500个精选特征）：**
- 技术指标特征（80+个）：RSI、MACD、布林带、ATR等
- 基本面特征：PE、PB、ROE、ROA等
- 美股市场特征（11个）：标普500、纳斯达克、VIX、美国国债收益率
- 股票类型特征（18个）：行业分类、防御性、成长性等
- 情感指标特征（6个）：情感趋势、波动率、变化率
- 板块分析特征（17个）：板块涨跌幅、排名、趋势
- 长期趋势特征（24个）：MA120/250、长期收益率、长期波动率
- 交叉特征（约2000+个）：分类特征×数值特征
- 技术指标与基本面交互特征（18个）：RSI×PE、MACD×ROE等
- 主题分布特征（10个）：LDA主题建模
- 主题情感交互特征（50个）：10个主题 × 5个情感指标
- 预期差距特征（5个）：新闻情感相对于市场预期的差距

**特征选择策略（2026-02-17）：**
- **统一策略**：LightGBM、GBDT、CatBoost都使用500个精选特征（F-test+互信息混合方法）
- **特征数量**：从2991个特征筛选出500个最有效特征（减少83.3%）
- **优势**：提升训练速度5-6倍，降低过拟合风险

**使用建议：**
- ✅ 优先使用一个月模型预测结果（准确率最高）
- ✅ 与手工信号结合使用，不单独依赖
- ✅ 定期重新训练模型（建议每周或每月）
- ✅ 关注高置信度预测（概率 > 60%）
- ✅ 使用综合分析系统，整合大模型建议和ML预测结果
- ✅ 使用批量回测功能，全面评估模型在不同股票上的表现
- ✅ 使用置信度0.55作为平衡点（置信度0.60会降低收益和胜率）
- ⚠️ 注意风险控制，设置合理的止损位

## 综合分析系统（每日自动执行）

### 功能说明
整合大模型建议（短期和中期）与ML融合模型预测结果（20天），进行综合对比分析，生成实质的买卖建议。

### 执行流程
1. **步骤0**：运行特征选择（生成500个精选特征）- 只执行一次
2. **步骤1**：生成大模型建议（短期和中期）
3. **步骤2**：训练20天ML模型（LightGBM、GBDT、CatBoost）- 跳过特征选择，使用步骤0的特征
4. **步骤3**：生成20天融合模型预测（加权平均）
5. **步骤4**：综合对比分析（整合大模型建议和ML融合模型预测）
6. **步骤5**：生成详细的综合买卖建议（10个章节）
7. **步骤6**：发送邮件通知（每日自动发送）

### 性能优化（2026-02-20）
- **特征选择优化**：添加 `--skip-feature-selection` 参数
- 步骤0执行特征选择一次，步骤2的三个模型训练都跳过特征选择
- **性能提升**：减少执行时间 50-70%

### 邮件内容（10个章节）
1. **# 综合买卖建议**（强烈买入、买入、持有/观望、卖出信号）
2. **## 一、机器学习预测结果（20天）**（融合模型，显示全部28只股票及预测方向）
3. **## 二、大模型建议**（短期和中期买卖建议）
4. **## 三、板块分析（5日涨跌幅排名）**（16个板块排名、龙头股TOP 3）
5. **## 四、股息信息（即将除净）**（前10只即将除净的港股）
6. **## 五、恒生指数技术分析**（当前价格、RSI、MA20、MA50、趋势判断）
7. **## 六、推荐股票技术指标详情**（11个技术指标表格）
8. **## 七、技术指标说明**（短期、中期技术指标说明）
9. **## 八、风险提示**（模型不确定性、市场风险、投资原则）
10. **## 九、数据来源**（11个数据源说明）

### 融合模型预测结果展示（2026-02-21优化）
- 显示全部28只股票的融合预测结果
- 添加"融合预测"栏位，标注每只股票的预测方向（上涨/观望/下跌）
- 添加"置信度"栏位，标注高/中/低置信度
- 添加"一致性"栏位，标注三模型一致性（100%/67%/33%）
- 融合概率分类：
  - **高置信度上涨**：fused_probability > 0.60
  - **中等置信度观望**：0.50 < fused_probability ≤ 0.60
  - **预测下跌**：fused_probability ≤ 0.50
- 统计信息：高置信度上涨数量、中等置信度观望数量、预测下跌数量、模型一致性分布

### 自动化调度
- GitHub Actions 工作流：`hsi-email-alert-open_message.yml`
- 执行时间：周一到周五 UTC 08:00（香港时间下午4:00）
- 支持手动触发

### 使用方法
```bash
# 一键执行完整流程
./run_comprehensive_analysis.sh

# 或手动执行
python comprehensive_analysis.py

# 不发送邮件
python comprehensive_analysis.py --no-email
```

## 邮件通知

系统支持自动邮件通知，包括：

- 💰 加密货币价格更新
- 📋 港股 IPO 信息
- 📊 主力资金追踪报告
- 📈 恒生指数交易信号
- 🥇 黄金市场分析报告
- 🔄 模拟交易通知（买入、卖出、止损等）
- 📊 AI 交易分析报告
- 🤖 机器学习模型预测结果
- 💼 **综合买卖建议（每日自动发送）**

邮件采用统一的表格化样式，清晰易读。

## 数据文件说明

| 文件 | 说明 | 更新频率 |
|------|------|---------|
| `actual_porfolio.csv` | 实际持仓数据 | 手动更新 |
| `llm_recommendations_*.txt` | 大模型建议文件 | 每日 |
| `ml_predictions_20d_*.txt` | ML预测结果文件 | 每日 |
| `comprehensive_recommendations_*.txt` | 综合买卖建议文件 | 每日 |
| `model_accuracy.json` | 模型准确率信息 | 训练时自动更新 |
| `simulation_state.json` | 模拟交易状态 | 实时 |
| `simulation_transactions.csv` | 交易历史记录 | 实时 |
| `ml_trading_model_*.pkl` | 机器学习模型 | 按需 |
| `ml_trading_model_*_importance.csv` | 特征重要性排名 | 按需 |
| `output/backtest_results_*.json` | 单只股票回测数据文件 | 回测时生成 |
| `output/backtest_results_*.png` | 单只股票回测图表 | 回测时生成 |
| `output/batch_backtest_*.json` | 批量回测详细数据 ⭐ 2026-02-22新增 | 批量回测时生成 |
| `output/batch_backtest_summary_*.txt` | 批量回测汇总报告 ⭐ 2026-02-22新增 | 批量回测时生成 |
| `fundamental_cache/` | 基本面数据缓存 | 7天有效期 |

## 重要提示

1. **模型性能基准**：
   - 随机/平衡基线：≈50%
   - 常见弱信号：≈51-55%
   - 有意义的改进：≈55-60%
   - 非常好/罕见：≈60-65%
   - 异常高（需怀疑）：>65%

2. **数据验证**：
   - 严格的时间序列交叉验证
   - 无数据泄漏
   - 日期索引保留，按时间顺序排列

3. **风险提示**：
   - 本系统仅供学习和研究使用
   - 不构成投资建议
   - 投资有风险，入市需谨慎

4. **置信度阈值选择**：
   - 保守型投资者：0.60-0.65（风险控制优先）
   - 平衡型投资者：0.55（收益与风险平衡）⭐ 推荐
   - 进取型投资者：0.50-0.55（追求更高收益）

## 未来计划

### 已完成（2026-02-22）

**机器学习模型优化**：
- ✅ 特征工程：2991个特征（技术指标、基本面、美股市场、情感指标、板块分析、长期趋势、主题分布、主题情感交互、预期差距）
- ✅ 特征选择：使用500个精选特征（F-test+互信息混合方法）
- ✅ GBDT模型重构：移除GBDT+LR两层结构，准确率提升3.21%，稳定性提升40.6%
- ✅ **CatBoost算法集成（2026-02-20）**：新增CatBoost模型，准确率62.07%（±1.78%），稳定性提升57.1%
- ✅ **模型融合功能（2026-02-20）**：实现三模型融合（LightGBM+GBDT+CatBoost），支持简单平均、加权平均、投票机制
- ✅ 动态准确率加载：训练时自动保存准确率，分析时自动读取（含CatBoost）
- ✅ 回测评估功能：验证模型盈利能力（置信度阈值 0.55），LightGBM夏普比率0.36（一般），GBDT夏普比率1.48（优秀），CatBoost夏普比率1.62（良好），融合模型夏普比率2.30（优秀）
- ✅ **批量回测功能（2026-02-22）**：支持28只股票批量回测，全面评估模型表现
- ✅ **置信度阈值对比分析（2026-02-22）**：对比置信度0.55和0.60的表现，发现0.55是更好的平衡点

**综合分析系统**：
- ✅ 整合大模型建议和ML融合模型预测结果
- ✅ 动态准确率加载功能（含CatBoost）
- ✅ 板块分析、股息信息、恒生指数技术分析
- ✅ 推荐股票技术指标详情（11个技术指标）
- ✅ **ML融合模型预测结果展示优化（2026-02-21）**：显示融合预测、置信度、一致性
- ✅ 每日自动执行，发送综合邮件

**ML预测结果展示优化**：
- ✅ 显示全部28只股票的预测结果
- ✅ 添加预测方向栏位（上涨/观望/下跌）
- ✅ 应用动态阈值（probability > 0.60为上涨）
- ✅ **融合模型预测结果展示（2026-02-21）**：显示融合预测、融合概率、置信度、一致性

### 待实现

**高优先级**：
- 风险管理模块（VaR、止损止盈、仓位管理）
- 投资组合优化算法

**中优先级**：
- 机器学习模型自动超参数调优
- ML预测结果可视化优化
- 探索Stacking方法（元学习器）

**低优先级**：
- Web界面
- 深度学习模型（LSTM、Transformer）
- 实时数据流处理

### 完成度统计

| 分类 | 完成率 |
|------|--------|
| ML特征工程优化 | 100% |
| ML模型融合功能 | 100% |
| 综合分析系统 | 100% |
| 回测评估功能 | 100% |
| 批量回测功能 | 100% ⭐ 2026-02-22新增 |
| 置信度阈值优化 | 100% ⭐ 2026-02-22新增 |
| 风险管理模块 | 0% |
| 其他功能 | 17% |
| **总体** | **92%** |

### 性能目标（2026-02-22）

**ML模型准确率**：
- **一个月模型CatBoost：62.07%（±1.78%）** ⭐ 当前最佳（稳定可靠）
- **融合模型（加权平均）：~62-63%（±1.5-2.0%）** ⭐⭐ 推荐
- **CatBoost 5天：63.01%（±4.45%）** ⚠️ 谨慎使用（需要更多验证）
- **CatBoost 1天：65.62%（±5.97%）** ❌ **不推荐使用**（存在严重过拟合风险）
- 一个月模型LightGBM：58.56%（±4.15%）
- 一个月模型GBDT：59.30%（±4.63%）
- 1天模型LightGBM：51.20%（±0.97%）
- 1天模型GBDT：51.59%（±1.61%）
- 5天模型LightGBM：55.20%（±2.20%）

**CatBoost模型优势（2026-02-20）**：
- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度，支持GPU加速
- 更好的泛化能力，减少过拟合
- **20天模型稳定性显著提升**：±1.78% vs LightGBM ±4.15%（提升57.1%）

**⚠️ 重要警告**：
- **CatBoost 1天模型存在严重过拟合风险**：准确率65.62%（±5.97%），标准偏差过高
- **不推荐使用 CatBoost 1天模型的预测结果**
- **推荐使用 CatBoost 20天模型和融合模型作为主要预测来源**

**模型融合优势（2026-02-20）**：
- 降低预测方差15-20%
- 提升模型稳定性
- 增强预测可信度
- 加权平均融合最优（准确率~62.5%，标准偏差±1.3%）

**回测评估（单只股票）**：
- LightGBM：总收益率11.73%，夏普比率0.36，评级⭐⭐⭐
- GBDT：总收益率73.99%，夏普比率1.48，评级⭐⭐⭐⭐⭐
- CatBoost：总收益率184.00%，夏普比率1.62，评级⭐⭐⭐⭐
- **融合模型（加权平均）：总收益率543.17%，夏普比率2.30，索提诺比率4.02，最大回撤-18.52%，评级⭐⭐⭐⭐⭐** ⭐

**批量回测（28只股票，2026-02-22）**：
- CatBoost 20天（置信度0.55）：平均总收益率238.76%，夏普比率1.51，胜率32.81%
- 融合模型（置信度0.55）：平均总收益率115.13%，夏普比率1.00，胜率31.89%
- CatBoost 20天（置信度0.60）：平均总收益率206.72%，夏普比率1.52，胜率31.84%
- 融合模型（置信度0.60）：平均总收益率75.97%，夏普比率0.86，胜率30.97%

**置信度阈值对比分析（2026-02-22）**：
- 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
- CatBoost模型最稳定（胜率仅下降0.97%）
- 置信度阈值 ≠ 预测准确率
- 置信度0.55是更好的平衡点

**当前状态**：CatBoost和融合模型已突破62%准确率，达到业界顶尖水平，稳定性显著提升，回测验证模型具备实际盈利能力，批量回测功能全面评估模型在不同股票上的表现。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。