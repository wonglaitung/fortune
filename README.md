# 港股智能分析与交易系统

**⭐ 如果您觉得这个项目有用，请先给项目Star再Fork，以支持项目发展！⭐**

实践人机混合智能的理念，以实际获利为目地的金融资产智能量化分析项目，实时监控加密货币、港股、黄金等金融市场。香港股票方面集成11个数据源、机器学习模型和大模型智能决策。模型回测年化收益率79.54%，支持实时技术指标、板块轮动分析、综合买卖建议和自动化调度，以智能副驾的方式为投资者提供全面的市场分析和交易策略验证。

## 📋 目录

- [核心优势](#核心优势)
- [核心功能](#核心功能)
- [机器学习模型](#机器学习模型)
- [技术架构](#技术架构)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [自动化调度](#自动化调度)
- [性能数据](#性能数据)
- [项目状态](#项目状态)
- [注意事项](#注意事项)
- [依赖项](#依赖项)
- [未来计划](#未来计划)
- [快速开始](#快速开始)

---

## 核心优势

### 🏆 业界领先的模型性能

| 指标 | CatBoost 20天 | 说明 |
|------|---------------|------|
| **准确率** | 61.88% (±2.56%) | 业界顶尖水平 |
| **F1分数** | 0.6748 (±0.0379) | 综合评估优异 |
| **年化收益率** | 79.54% | 实际回测表现卓越 |
| **夏普比率** | 1.14 | 风险调整后收益优秀 |
| **优秀股票占比** | 71% (20/28) | 模型稳定性强 |

### 🚀 关键特性

- **CatBoost 单模型策略**：经过严格验证，CatBoost 单模型表现远优于融合模型和深度学习模型
- **深度学习模型对比实验**：LSTM、Transformer与CatBoost对比评估，验证CatBoost优势
- **综合分析系统**：每日自动执行，整合大模型建议和CatBoost预测结果，生成实质买卖建议
- **批量回测功能**：支持29只股票批量回测，全面评估模型表现
- **实时技术指标**：集成恒生指数及自选股实时技术指标
- **模拟交易记录**：最近48小时模拟交易记录以表格格式展示
- **恒生指数涨跌预测**：基于特征重要性的加权评分模型
- **月度趋势分析**：2024-2026年跨年度回测月度分析，识别季节性规律
- **股票月度趋势对比**：相关性分析、波动性分析、异常值检测
- **牛熊市分析自动化**：每周一自动执行，分析市场环境和股票表现
- **筹码分布分析**：基于成交量的简单分箱法，计算筹码集中度、拉升阻力，辅助判断股票上涨潜力

### ⚠️ 重要警告

1. **CatBoost 1天模型存在严重过拟合风险**：准确率63.09%（±4.33%），标准偏差过高，**不推荐使用**
2. **融合模型表现不如CatBoost单模型**：所有融合方法年化收益率均低于5%，建议优先使用CatBoost单模型
3. **深度学习模型（LSTM、Transformer）表现远不如CatBoost**：F1分数极低，回测无交易，**不推荐用于实际交易**
4. **推荐使用 CatBoost 20天模型**：准确率61.88%，标准偏差仅±2.56%，稳定性最强

---

## 核心功能

### 数据获取与监控

- **加密货币监控**：比特币、以太坊价格和技术分析（每小时）
- **港股IPO信息**：最新IPO信息（每天）
- **黄金市场分析**：黄金价格和投资建议（每小时）
- **恒生指数监控**：价格、技术指标、交易信号（交易时段）
- **恒生指数涨跌预测**：基于特征重要性的加权评分模型，预测短期走势
- **美股市场数据**：标普500、纳斯达克、VIX、美国国债收益率
- **基本面数据**：财务指标、利润表、资产负债表、现金流量表
- **股息信息**：自动获取股息和除净日信息

### 智能分析

- **主力资金追踪**：识别建仓和出货信号，集成基本面分析和筹码分布分析
- **筹码分布分析**：
  - 基于成交量的简单分箱法（默认20个价格区间）
  - 计算筹码集中度（HHI指数）：高>0.3，中0.15-0.3，低<0.15
  - 计算上方筹码比例（拉升阻力）：高>60%（困难），中30-60%（注意），低<30%（容易）
  - 阻力标识系统：✅低阻力、⚠️中等阻力、🔴高阻力
  - 集成到主力资金追踪器，影响建仓/出货评分
  - 集成到综合分析系统，添加到ML预测表格
- **板块分析**：16个板块涨跌幅排名、技术趋势分析、龙头识别
- **板块轮动河流图**：可视化板块排名变化
- **恒生指数策略**：大模型生成交易策略
- **AI交易分析**：复盘AI推荐策略有效性
- **综合分析系统（每日自动执行）**：整合大模型建议和CatBoost预测结果，生成实质买卖建议
- **模拟交易记录展示**：最近48小时模拟交易记录（表格格式）
- **模型对比回测**：定期回测3种基本模型和5种融合方法，生成汇总对比报告
- **深度学习模型对比实验**：LSTM、Transformer与CatBoost对比评估
- **月度趋势分析**：2024-2026年跨年度回测月度分析，分析收益率季节性规律
- **股票月度趋势对比**：单个股票与总体趋势相关性分析、波动性分析、异常值检测

### 模拟交易

- **真实模拟**：基于大模型建议的模拟交易系统
- **风险控制**：自动止损机制
- **详细记录**：完整的交易日志和持仓分析
- **多种策略**：支持保守型、平衡型、进取型投资偏好

---

## 机器学习模型

### 模型架构

- **CatBoost 单模型**：CatBoost 20天模型（主要使用）⭐
- **多周期预测**：预测1天、5天、20天后的涨跌
- **特征工程**：2991个特征（技术指标、基本面、美股市场、情感指标、板块分析、长期趋势、主题分布、主题情感交互、预期差距）
- **特征选择**：使用500个精选特征（statistical方法：F-test+互信息混合）

### CatBoost 模型优势

- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度，支持GPU加速
- 更好的泛化能力，减少过拟合
- 稳定性显著提升（±2.56% vs LightGBM ±3.87%，提升33.9%）
- 实际回测表现优异：年化收益率79.54%，夏普比率1.14

### 预测概率与实际涨幅相关性分析

基于6,328条回测交易记录的实证分析，验证了预测概率与实际涨幅的强正相关关系。

**关键数据**：

| 指标 | 数值 | 解释 |
|------|------|------|
| 相关系数 | 0.6289 | 强正相关 |
| R²值 | 0.3956 | 概率能解释39.56%的涨幅变化 |
| P值 | <0.0001 | 极显著相关 |
| 回归方程 | 涨幅 = 0.3566 × 概率 - 0.1760 | |

**不同概率区间的表现**：

| 概率区间 | 交易次数 | 平均涨幅 | 胜率 | 标准差 |
|---------|---------|---------|------|--------|
| <0.50 | 2,358 | -6.69% | 12.6% | 9.28% |
| 0.50-0.55 | 485 | -0.24% | 40.6% | 8.66% |
| 0.55-0.60 | 537 | 3.47% | 60.0% | 10.34% |
| 0.60-0.65 | 613 | 6.14% | 79.8% | 10.14% |
| 0.65-0.70 | 644 | 8.15% | 89.1% | 9.58% |
| 0.70-0.75 | 553 | 9.25% | 93.9% | 9.74% |
| 0.75-0.80 | 449 | 10.43% | 98.0% | 9.98% |
| 0.80-0.85 | 329 | 10.77% | 96.7% | 8.20% |
| 0.85-0.90 | 269 | 11.66% | 99.3% | 8.82% |
| >0.90 | 91 | 11.73% | 100.0% | 6.28% |

**核心发现**：

- ✅ **理论正确**：预测概率越大，实际涨幅也越大
- ✅ **阶梯式增长**：从概率0.50到0.90，平均涨幅从-0.24%增长到11.73%
- ✅ **胜率同步提升**：概率0.50-0.55胜率40.6%，概率>0.90胜率100%
- ✅ **高质量信号**：概率>0.70的信号，平均涨幅9.25%，胜率93.9%

**应用建议**：可基于概率建立涨幅映射表，设置预期涨幅>0.5%（对应概率约0.57）作为买入阈值，有效过滤低质量信号。

### 深度学习模型对比实验

**实验结论**：经过严格测试，**CatBoost 远优于深度学习模型**，推荐继续使用 CatBoost 作为主要预测模型。

| 模型 | 准确率 | F1分数 | 标准差 | 训练时间 | 回测收益率 | 推荐指数 |
|------|--------|--------|--------|----------|-----------|----------|
| **CatBoost** | **63.09%** ⭐ | **0.6022** ⭐ | **±4.33%** | 快（1-2分钟） | **79.54%** | ⭐⭐⭐⭐⭐ |
| **LSTM** | 51.79% | **0.0000** ❌ | ±0.78% | 慢（1-2分钟） | 0.00% | ⭐ |
| **Transformer** | 51.15% | 0.1303 | ±4.00% | 慢（2-3分钟） | 0.00% | ⭐ |

**关键发现**：
- ✅ CatBoost 绝对优势：准确率领先，F1分数优秀，实际回测表现优异
- ❌ LSTM 表现最差：F1分数为0，完全无法识别上涨信号，预测概率偏低（0.48-0.49）
- ❌ Transformer 表现略好于LSTM，但仍远不如CatBoost：准确率仅51.15%，F1分数0.1303

**根本原因**：
1. **数据量不足**：单个股票约700个样本，深度学习需要>10,000个样本（仅为需求的7%）
2. **信噪比极低**：股价数据信噪比<10%，深度学习容易学习噪声
3. **特征工程不足**：深度学习主要使用原始价格序列，CatBoost使用500个精选特征
4. **序列依赖假设错误**：股价主要是随机游走（有效市场假说）
5. **过拟合风险高**：深度学习参数/样本比>140:1（严重过拟合），CatBoost约7:1

**最终建议**：
- ✅ 继续使用 **CatBoost 单模型**作为主要预测模型
- ❌ 放弃 LSTM 和 Transformer 用于实际交易
- ⚠️ 深度学习模型仅用于对比研究和学术探索

### 融合模型方法

系统支持5种融合方法，用于对比研究：

| 融合方法 | 平均总收益率 | 年化收益率 | 夏普比率 | 买入信号胜率 | 说明 |
|---------|-----------|-----------|---------|------|------|
| CatBoost 单模型 | 185.49% | 79.54% | 1.20 | 30.02% | 主要使用 ⭐ |
| 加权平均 | 0.07% | 3.16% | 0.32 | 22.75% | 基于模型准确率加权 |
| 简单平均 | 2.69% | 4.01% | 0.40 | 23.63% | 三个模型概率平均 |
| 投票机制 | 1.63% | 2.31% | 0.36 | 22.58% | 多数投票决定方向 |
| 动态市场 | 0.66% | 2.40% | 0.33 | 22.96% | 根据市场状态动态选择 |
| 高级动态 | 0.66% | 2.40% | 0.33 | 22.96% | CatBoost主导（90%权重） |

**注意**：根据回测结果，所有融合方法的表现均不如CatBoost单模型，建议优先使用CatBoost单模型。融合模型仅用于对比研究。

### 批量回测功能

- 对自选股列表中的所有股票（29只）进行批量回测
- 支持单一模型和融合模型批量回测
- 支持不同置信度阈值（0.55、0.60等）
- 生成汇总报告，包含平均表现和排名
- 支持股票名称显示

---

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
│   │   └── 筹码分布分析（HHI指数、拉升阻力分析）
│   ├── 基本面分析
│   ├── 板块分析
│   └── 新闻过滤
│
├── 分析层
│   ├── 主力资金追踪
│   ├── AI交易分析
│   ├── 机器学习模型
│   │   ├── CatBoost 单模型（主要使用）⭐
│   │   ├── LightGBM 模型
│   │   ├── GBDT 模型
│   │   ├── LSTM 模型（对比实验，不推荐）
│   │   ├── Transformer 模型（对比实验，不推荐）
│   │   ├── 融合模型（5种方法）
│   │   ├── 批量回测（29只股票）
│   │   ├── 月度趋势分析（跨年度）
│   │   └── 股票月度趋势对比
│   └── 综合分析（每日自动执行）
│
├── 交易层
│   └── 模拟交易系统
│
├── 工具脚本层
│   ├── 数据诊断工具
│   ├── 特征评估工具
│   └── 训练工具
│
└── 服务层
    ├── 大模型服务
    └── 邮件服务
```

---

## 使用示例

### 快速体验

```bash
# 监控加密货币价格
python crypto_email.py

# 追踪港股主力资金
python hk_smart_money_tracker.py

# 恒生指数价格监控
python hsi_email.py

# 综合分析（一键执行）
./run_comprehensive_analysis.sh
```

### 模型训练和预测

```bash
# 训练 CatBoost 模型（推荐）
python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 生成 CatBoost 预测
python ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost
```

### 批量回测

```bash
# CatBoost 批量回测（推荐）
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55

# 其他模型回测
python3 ml_services/batch_backtest.py --model-type lgbm --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type gbdt --horizon 20 --use-feature-selection --confidence-threshold 0.55

# 回测结果会保存到：
# - output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json（详细数据）
# - output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt（汇总报告）
```

### 模型对比回测

```bash
# 模型对比回测（3个基本模型 + 5个融合方法）
./run_model_comparison.sh
python3 run_model_comparison.sh --force-train  # 强制重新训练所有模型
```

### 月度趋势分析

```bash
# 月度趋势分析（2024-2026年跨年度）
python3 ml_services/backtest_monthly_analysis.py

# 股票月度趋势对比
python3 ml_services/stock_monthly_trend_analysis.py
```

### 深度学习模型对比实验（不推荐）

```bash
# LSTM模型对比实验（实验性，不推荐）
python3 ml_services/lstm_experiment.py --horizon 1  # 1天预测
python3 ml_services/lstm_experiment.py --horizon 5  # 5天预测
python3 ml_services/lstm_experiment.py --horizon 20 --stocks 0700.HK 0939.HK 1347.HK  # 20天预测

# Transformer模型对比实验（实验性，不推荐）
python3 ml_services/transformer_experiment.py --horizon 1  # 1天预测
python3 ml_services/transformer_experiment.py --use-feature-selection  # 使用特征选择
python3 ml_services/transformer_experiment.py --stocks 0700.HK 0939.HK  # 自定义测试股票
```

### 综合分析

```bash
# 一键执行完整流程
./run_comprehensive_analysis.sh

# 或手动执行
python comprehensive_analysis.py

# 不发送邮件
python comprehensive_analysis.py --no-email
```

---

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
│   ├── hsi_prediction.py               # 恒生指数涨跌预测器
│   ├── simulation_trader.py            # 模拟交易系统
│   ├── comprehensive_analysis.py       # 综合分析脚本（每日自动执行）
│   └── ...
│
├── 数据服务模块 (data_services/)
│   ├── technical_analysis.py           # 通用技术分析工具
│   │   └── get_chip_distribution()     # 筹码分布分析
│   ├── fundamental_data.py             # 基本面数据获取器
│   ├── hk_sector_analysis.py           # 板块分析器
│   └── ...
│
├── 机器学习模块 (ml_services/)
│   ├── ml_trading_model.py             # 机器学习交易模型
│   │   ├── LightGBMModel               # LightGBM模型
│   │   ├── GBDTModel                   # GBDT模型
│   │   ├── CatBoostModel               # CatBoost模型 ⭐
│   │   ├── EnsembleModel               # 融合模型
│   │   ├── LSTMModel                   # LSTM模型（不推荐）
│   │   └── TransformerModel            # Transformer模型（不推荐）
│   ├── batch_backtest.py               # 批量回测脚本
│   ├── backtest_evaluator.py           # 回测评估模块
│   ├── backtest_monthly_analysis.py    # 月度趋势分析脚本
│   ├── stock_monthly_trend_analysis.py # 股票月度趋势对比脚本
│   ├── us_market_data.py               # 美股市场数据
│   ├── feature_selection.py            # 特征选择模块
│   ├── topic_modeling.py               # LDA主题建模模块
│   ├── lstm_experiment.py              # LSTM对比实验脚本
│   ├── transformer_experiment.py       # Transformer对比实验脚本
│   ├── BACKTEST_GUIDE.md               # 回测功能使用指南
│   └── ...
│
├── 大模型服务 (llm_services/)
│   ├── qwen_engine.py                  # Qwen大模型接口
│   └── sentiment_analyzer.py           # 情感分析模块
│
├── 工具脚本 (scripts/)
│   ├── data_diagnostic.py              # 数据诊断工具
│   ├── feature_evaluation.py           # 特征评估工具
│   └── train_with_feature_selection.py # 训练工具
│
├── 配置文件
│   ├── config.py                       # 全局配置
│   ├── requirements.txt                # 项目依赖
│   ├── run_comprehensive_analysis.sh   # 综合分析自动化脚本
│   ├── run_model_comparison.sh         # 模型对比自动化脚本
│   ├── set_key.sh                      # 环境变量配置（已加入.gitignore）
│   ├── set_key.sh.sample               # 环境变量配置模板
│   └── .github/workflows/              # GitHub Actions工作流配置
│
├── 文档目录 (docs/)
│   ├── CATBOOST_SIGNAL_QUALITY_ANALYSIS.md  # CatBoost信号质量分析
│   ├── model_importance_std_analysis.md     # 模型重要性标准差分析
│   ├── feature_selection_methods_comparison.md # 特征选择方法对比
│   ├── feature_selection_summary.md         # 特征选择总结
│   ├── backtest_results_report.md           # 回测结果报告
│   ├── backtest_horizon_explanation.md      # 回测周期说明
│   ├── DEEP_LEARNING_COMPARISON_README.md   # 深度学习模型对比实验指南
│   ├── TIME_SERIES_LEAKAGE_ANALYSIS.md      # 时间序列泄漏分析
│   ├── 不同股票类型分析框架对比.md
│   └── IMPROVEMENT_POINTS_FROM_DAILY_STOCK_ANALYSIS.md # 从daily_stock_analysis项目学到的提升点
│
├── 输出文件 (output/)
│   ├── batch_backtest_*.json           # 批量回测详细数据
│   ├── batch_backtest_summary_*.txt    # 批量回测汇总报告
│   ├── model_comparison_report_*.txt   # 模型对比汇总报告
│   ├── lstm_experiment_*.json          # LSTM对比实验详细数据
│   ├── transformer_experiment_*.json   # Transformer对比实验详细数据
│   └── ...
│
└── 数据文件 (data/)
    ├── actual_porfolio.csv             # 实际持仓数据
    ├── llm_recommendations_*.txt       # 大模型建议文件
    ├── ml_trading_model_catboost_predictions_20d.csv  # CatBoost预测结果
    ├── comprehensive_recommendations_*.txt  # 综合买卖建议文件
    ├── model_accuracy.json             # 模型准确率信息
    └── ...
```

---

## 自动化调度

### GitHub Actions 工作流

系统使用 **GitHub Actions** 进行全自动化调度，无需服务器部署，零硬件成本运行。目前有9个工作流正常运行，覆盖全天候市场监控和智能分析。

| 工作流 | 功能 | 执行时间 | 说明 |
|--------|------|----------|------|
| **hourly-crypto-monitor.yml** | 每小时加密货币监控 | 每小时 | 监控比特币、以太坊价格和技术分析 |
| **hourly-gold-monitor.yml** | 每小时黄金监控 | 每小时 | 监控黄金价格和投资建议 |
| **hsi-prediction.yml** | 恒生指数涨跌预测 | 周一到周五 UTC 22:00（香港时间上午6:00） | 预测恒生指数短期走势 |
| **comprehensive-analysis.yml** | 综合分析邮件 | 周一到周五 UTC 08:00（香港时间下午4:00） | 整合大模型建议和CatBoost预测结果 |
| **batch-stock-news-fetcher.yml** | 批量股票新闻获取 | 每天 UTC 22:00 | 批量获取自选股新闻，用于情感分析 |
| **daily-ipo-monitor.yml** | IPO 信息监控 | 每天 UTC 02:00 | 获取最新IPO信息 |
| **daily-ai-trading-analysis.yml** | AI 交易分析日报 | 周一到周五 UTC 08:30 | AI驱动的交易策略分析 |
| **weekly-comprehensive-analysis.yml** | 周综合交易分析 | 每周日 UTC 01:00（香港时间上午9:00） | 全面周度分析 |
| **bull-bear-analysis.yml** | 牛熊市分析自动化 | 每周日 UTC 17:00（香港时间周一上午1:00） | 分析市场环境和股票表现 |

**配置说明**：详细的配置步骤请参考文档末尾的[快速开始](#快速开始)章节中的"🌟 无服务器部署 - GitHub Actions 自动化"部分。

### 运行成本

**GitHub Actions 免费额度**：
- 公开仓库：无限制
- 私有仓库：每月2000分钟免费
- 每个工作流运行时间通常在1-5分钟
- 本项目总运行时间每月约150-300分钟
- **结论**：免费额度充足，完全够用

---

## 性能数据

### 最新模型准确率

| 预测周期 | 模型 | 准确率 | F1分数 | 推荐指数 |
|---------|------|--------|--------|----------|
| **20天（推荐）** | **CatBoost** | **61.88%** ⭐ | **0.6748** ⭐ | ⭐⭐⭐⭐⭐ |
| 20天 | GBDT | 59.99% | 0.7301 | ⭐⭐⭐ |
| 20天 | LightGBM | 58.98% | 0.7179 | ⭐⭐ |
| 5天 | CatBoost | 63.96% | 0.6785 | ⚠️ 需要更多验证 |
| 5天 | GBDT | 55.31% | 0.6943 | ⭐⭐ |
| 5天 | LightGBM | 55.12% | 0.6924 | ⭐⭐ |
| **1天（不推荐）** | **CatBoost** | **63.09%** ❌ | **0.6022** ❌ | ⭐ 过拟合风险 |
| 1天 | GBDT | 53.56% | 0.5151 | ⭐ |
| 1天 | LightGBM | 50.91% | 0.4995 | ⭐ |

**重要提示**：
- ⭐ **推荐使用 CatBoost 20天模型**：准确率61.88%，标准偏差仅±2.56%，稳定性最强
- ❌ **不推荐使用 CatBoost 1天模型**：存在严重过拟合风险，标准偏差±4.33%过高

### 批量回测性能（置信度0.55，28只股票）

| 模型类型 | 年化收益率 | 夏普比率 | 买入信号胜率 | 优秀股票占比 |
|---------|-----------|---------|------|-------------|
| **CatBoost 20天** ⭐ | **79.54%** | **1.20** | **30.02%** | **71% (20/28)** |
| LightGBM 20天 | 13.56% | 0.26 | 32.41% | 11% (3/28) |
| GBDT 20天 | 7.67% | 0.17 | 29.53% | 4% (1/28) |
| 融合模型（加权平均） | 3.16% | 0.32 | 22.75% | 7% (2/28) |

**CatBoost 批量回测详细表现**：
- 平均总收益率：185.49%
- 最高收益率：1353.20%（1347.HK 华虹半导体）
- 最低收益率：9.45%（0941.HK 中国移动）
- 收益率中位数：160.12%
- 收益率标准差：259.76%

### 2024-2026年跨年度回测月度分析

**总体性能指标**：
- 回测时间范围：2024-01-02 至 2026-01-02
- 总交易机会：13,457
- 买入信号数：7,554（占比56.15%）
- 整体准确率：81.53%
- 平均收益率：3.05%（20天持有期）

**关键发现**：
1. **季节性规律明显**：上半年平均收益率3.79% > 下半年2.74%
2. **最佳月份**：2025-01（收益率16.58%）
3. **最差月份**：2025-03（收益率-7.78%）
4. **投资建议**：优先投资第一季度（1-3月）和第三季度（7-9月）

---

## 综合分析系统

### 功能说明
整合大模型建议（短期和中期）与 CatBoost 单模型预测结果（20天），进行综合对比分析，生成实质的买卖建议。

### 执行流程
1. **步骤0**：运行特征选择（statistical方法，生成500个精选特征）- 只执行一次
2. **步骤1**：训练 CatBoost 20天模型（使用步骤0的特征，跳过特征选择）
3. **步骤2**：生成 CatBoost 单模型预测
4. **步骤3**：生成大模型建议（短期和中期）
5. **步骤4**：综合对比分析（整合大模型建议和CatBoost预测）
6. **步骤5**：生成详细的综合买卖建议
7. **步骤6**：发送邮件通知（每日自动发送）

### 邮件内容结构
1. **# 信息参考**
2. **## 一、机器学习预测结果（20天）**（CatBoost 单模型，显示全部29只股票及预测方向）
3. **## 二、大模型建议**（短期和中期买卖建议）
4. **## 三、实时技术指标**（恒生指数及自选股实时技术指标）
5. **## 四、最近48小时模拟交易记录**（表格格式）
6. **## 五、板块分析（5日涨跌幅排名）**
7. **## 六、股息信息（即将除净）**
8. **## 七、恒生指数技术分析**
9. **## 八、股票技术指标详情**
10. **## 九、恒生指数涨跌预测**
11. **## 十、技术指标说明**
12. **## 十一、决策框架**
13. **## 十二、风险提示**
14. **## 十三、数据来源**
15. **## 十四、深度学习模型对比实验**

### CatBoost 预测结果展示
- 显示全部29只股票的 CatBoost 预测结果
- 添加"预测方向"栏位（上涨/下跌）
- 添加"预测概率"栏位
- 添加"置信度"栏位（高/中/低）
- 添加"阻力标识"栏位：
  - ✅：低阻力（上方筹码 < 30%），拉升容易
  - ⚠️：中等阻力（30-60%），注意风险
  - 🔴：高阻力（> 60%），拉升困难
  - N/A：无法计算（数据不足）
- 筹码分布摘要：
  - 低/中/高阻力股票数量统计
  - 高阻力股票列表（股票代码、名称、上方筹码比例、拉升难度）
  - 阻力标识说明
- 预测概率分类：
  - **高置信度上涨**：prediction_probability > 0.60
  - **中等置信度上涨**：0.50 < prediction_probability ≤ 0.60
  - **预测下跌**：prediction_probability ≤ 0.50

---

## 项目状态

| 维度 | 状态 | 说明 |
|------|------|------|
| **核心功能** | ✅ 完整 | 数据获取、分析、交易、通知全覆盖 |
| **恒生指数预测** | ✅ 完整 | 基于特征重要性的加权评分模型 |
| **深度学习对比实验** | ✅ 完整 | LSTM、Transformer与CatBoost对比评估 |
| **F1分数指标** | ✅ 完整 | 模型性能评估中加入F1分数指标 |
| **模型对比回测** | ✅ 完整 | 支持3个基本模型和5种融合方法的批量回测 |
| **月度趋势分析** | ✅ 完整 | 2024-2026年跨年度回测月度分析 |
| **股票月度趋势对比** | ✅ 完整 | 相关性分析、波动性分析、异常值检测 |
| **牛熊市分析自动化** | ✅ 完整 | 每周一自动执行，分析市场环境和股票表现 |
| **筹码分布分析** | ✅ 完整 | 基于成交量的简单分箱法，计算筹码集中度和拉升阻力 |
| **模块化架构** | ✅ 完成 | data_services、llm_services、ml_services |
| **ML模型** | ✅ 顶尖 | CatBoost 20天准确率61.88%，达到业界顶尖水平 |
| **批量回测** | ✅ 完整 | 支持29只股票批量回测 |
| **综合分析** | ✅ 稳定 | 每日自动执行，整合大模型建议和CatBoost预测 |
| **交易记录展示** | ✅ 完整 | 最近48小时模拟交易记录以表格格式展示 |
| **实时指标集成** | ✅ 完整 | 集成 hsi_email.py 的实时技术指标 |
| **自动化** | ✅ 稳定 | 9个GitHub Actions工作流正常运行 |
| **文档** | ✅ 完整 | README、AGENTS、BACKTEST_GUIDE齐全 |
| **数据验证** | ✅ 严格 | 无数据泄漏，时间序列交叉验证 |
| **风险管理** | ⚠️ 可优化 | 可添加VaR、ES、压力测试 |
| **Web界面** | ❌ 未实现 | 可考虑添加可视化界面 |

---

## 注意事项

### 模型性能基准

| 性能等级 | 准确率范围 | 说明 |
|---------|-----------|------|
| 随机/平衡基线 | ≈50% | 随机猜测水平 |
| 常见弱信号 | ≈51-55% | 简单动量/基准模型 |
| 有意义的改进 | ≈55-60% | 可交易边际 |
| 非常好/罕见 | ≈60-65% | 优秀模型 |
| 异常高（需怀疑） | >65% | 可能存在数据泄漏 |

### 重要警告

1. **CatBoost 1天模型存在严重过拟合风险**：
   - 准确率63.09%（±4.33%），标准偏差过高
   - **不推荐使用** CatBoost 1天模型的预测结果
   - **推荐使用** CatBoost 20天模型作为主要预测来源

2. **融合模型表现不如CatBoost单模型**：
   - 所有融合方法年化收益率均低于5%
   - 融合模型仅用于对比研究，不建议用于实际交易

3. **深度学习模型（LSTM、Transformer）表现远不如CatBoost**：
   - LSTM F1分数为0，完全无法识别上涨信号
   - Transformer F1分数0.1303，表现很差
   - **不推荐用于实际交易**

4. **置信度阈值选择**：
   - 保守型投资者：0.60-0.65（风险控制优先）
   - 平衡型投资者：0.55（收益与风险平衡）⭐ 推荐
   - 进取型投资者：0.50-0.55（追求更高收益）

### 其他注意事项

5. **数据验证**：严格的时间序列交叉验证，无数据泄漏，日期索引保留，按时间顺序排列
6. **数据源限制**：部分数据源可能有访问频率限制
7. **缓存机制**：基本面数据缓存7天，可手动清除
8. **交易时间**：模拟交易系统遵循港股交易时间
9. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
10. **API密钥**：请妥善保管API密钥，不要提交到版本控制

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

## 未来计划

### 待实现功能

**高优先级**：
- 风险管理模块：VaR（风险价值）、止损止盈策略、仓位管理

**中优先级**：
- 机器学习模型自动超参数调优
- ML预测结果可视化优化

**低优先级**：
- Web界面
- 实时数据流处理

### 项目状态

- ✅ 核心功能完成度：95%+
- ✅ CatBoost 20天模型：年化收益率79.54%
- ✅ 综合分析系统：每日自动执行
- ✅ 自动化调度：9个GitHub Actions工作流
- ⚠️ 风险管理：待实现

---

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
# 复制配置模板：cp set_key.sh.sample set_key.sh
# 编辑 set_key.sh 文件，设置邮件和大模型API密钥
source set_key.sh

# 4.（可选）安装PyTorch用于深度学习模型对比实验（不推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 环境变量配置

`set_key.sh` 脚本用于配置系统运行所需的环境变量。

**必填变量**：

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `YAHOO_SMTP` | SMTP服务器地址 | `smtp.163.com` |
| `YAHOO_EMAIL` | 发件人邮箱 | `your-email@163.com` |
| `YAHOO_APP_PASSWORD` | 邮箱应用密码 | 从邮箱设置中生成的授权码 |
| `RECIPIENT_EMAIL` | 收件人邮箱列表（逗号分隔） | `user1@gmail.com,user2@yahoo.com.hk` |
| `QWEN_API_KEY` | 通义千问大模型API密钥 | `sk-xxxxxxxxxxxxxxxxxxxx` |

**可选变量（大模型配置）**：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `QWEN_CHAT_URL` | 通义千问chat API地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` |
| `QWEN_CHAT_MODEL` | 通义千问chat模型名称 | `qwen-plus-2025-12-01` |
| `MAX_TOKENS` | 最大token数 | `32768` |

**配置步骤**：

1. 复制配置模板：`cp set_key.sh.sample set_key.sh`
2. 编辑 `set_key.sh` 文件，填写配置信息
3. 激活配置：`source set_key.sh`
4. 验证配置：`echo $YAHOO_EMAIL`

**注意事项**：
- `set_key.sh` 已添加到 `.gitignore`，不会提交到仓库
- 敏感信息请妥善保管，不要泄露
- 未设置可选变量时，系统将使用默认值
- GitHub Actions 需要在 Secrets 中配置相同的环境变量

**邮箱授权码获取方法**：

- **163邮箱**：设置 → POP3/SMTP/IMAP → 开启POP3/SMTP服务 → 生成授权码
- **Gmail**：Google账户设置 → 安全性 → 两步验证 → 应用密码 → 生成新密码
- **QQ邮箱**：设置 → 账户 → POP3/IMAP/SMTP服务 → 生成授权码

### 快速体验

```bash
# 监控加密货币价格
python crypto_email.py

# 追踪港股主力资金
python hk_smart_money_tracker.py

# 恒生指数价格监控
python hsi_email.py

# 综合分析（一键执行）
./run_comprehensive_analysis.sh
```

### 🌟 无服务器部署 - GitHub Actions 自动化

> **⚡ 无需部署服务器，即刻拥有功能完整的金融资产智能量化分析助手**

本项目通过 GitHub Actions 实现全自动化运行，**无需购买服务器、无需维护运维**。

**核心优势**：

| 优势 | 说明 |
|------|------|
| **零成本** | GitHub Actions 免费额度充足，每月2000分钟免费运行时间 |
| **零运维** | 无需服务器维护、无需监控、无需备份 |
| **自动化** | 9个工作流自动运行，覆盖全天候市场监控 |
| **稳定性** | GitHub 提供高可用基础设施，99.9%在线率 |
| **可扩展** | 轻松扩展到更多数据源和分析功能 |
| **安全性** | GitHub Secrets 加密存储环境变量 |

**使用方法**：

**方式一：Fork项目后启用（推荐）**

```bash
# 1. Fork本项目到你的GitHub账号
# 2. 进入你Fork的仓库 → Settings → Secrets and variables → Actions
# 3. 添加以下Secrets（必填）：
#    - YAHOO_EMAIL: 你的邮箱地址
#    - YAHOO_APP_PASSWORD: 邮箱授权码
#    - YAHOO_SMTP: SMTP服务器地址
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
     - `YAHOO_EMAIL`: 你的发件人邮箱
     - `YAHOO_APP_PASSWORD`: 邮箱授权码（不是登录密码）
     - `YAHOO_SMTP`: SMTP服务器地址（如smtp.163.com）
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

---

**最后更新**: 2026-03-13
