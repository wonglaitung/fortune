# 金融资产和港股智能分析与交易系统

**⭐ 如果您觉得这个项目有用，请先给项目Star再Fork，以支持项目发展！⭐**

实践**人机混合智能**的理念，开发具备变现能力的金融资产智能量化分析助手。系统整合**大模型智能决策**与**机器学习预测模型**，实时监控加密货币、港股、黄金等金融市场。香港股票方面集成11个数据源，以智能副驾的方式为投资者提供全面的市场分析、交易策略验证和买卖建议。

---

## 📋 目录

- [核心理念](#核心理念)
- [核心优势](#核心优势)
- [核心功能](#核心功能)
- [大模型智能决策](#大模型智能决策)
- [机器学习预测模型](#机器学习预测模型)
- [综合分析系统](#综合分析系统)
- [技术架构](#技术架构)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [自动化调度](#自动化调度)
- [性能数据](#性能数据)
- [项目状态](#项目状态)
- [注意事项](#注意事项)
- [依赖项](#依赖项)
- [快速开始](#快速开始)

---

## 核心理念

### 🤝 人机混合智能

本项目的核心理念是**实践人机混合智能**，将大模型的推理能力与机器学习的预测能力相结合，为投资者提供更全面、更可靠的决策支持。

**人机协作模式**：
- **大模型（人类智能的延伸）**：
  - 深度分析市场环境、技术指标、基本面数据
  - 生成短期和中期投资建议（具体到股票、操作、价格指引、风险提示）
  - 提供板块分析和龙头股识别
  - 进行策略复盘和AI交易分析
  
- **机器学习（数据驱动的预测）**：
  - CatBoost 20天模型准确率60.88%，预测个股涨跌方向
  - 板块特定模型评估不同行业股票的预测准确度
  - 为大模型建议提供量化支撑

- **智能副驾角色**：
  - 不是替代投资者，而是辅助决策
  - 提供数据驱动的分析和建议
  - 帮助投资者识别机会和控制风险
  - 最终决策权在投资者手中

**人机协同优势**：
- ✅ **全面性**：大模型提供定性分析，机器学习提供定量预测
- ✅ **可靠性**：双重验证，大模型建议和ML预测相互印证
- ✅ **灵活性**：适应不同投资风格（保守型、稳健型、进取型）
- ✅ **可解释性**：大模型提供解释性建议，机器学习提供概率预测

---

## 核心优势

### 🏆 业界领先的性能

**大模型智能决策**：
- **多维度分析**：综合技术指标、基本面、市场环境、板块轮动等多维度信息
- **智能投资建议**：生成短期（日内/数天）和中期（数周-数月）投资建议，包含推荐理由、操作建议、价格指引、风险提示
- **动态调整**：根据市场环境变化，动态调整投资策略和风险控制措施
- **人机协同**：与机器学习预测结果相互印证，提高决策可靠性

**机器学习预测模型**：
- **CatBoost 20天模型**：准确率60.88%，F1分数0.6416，标准差2.06%
- **板块特定模型**：为16个板块训练独立模型，评估不同行业股票的预测准确度
- **真实预测能力**：通过Walk-forward验证，评估模型的真实预测能力和泛化能力

### 🚀 关键特性

**大模型智能决策功能**：
- ✅ **恒生指数及自选股分析**：六层分析框架（风险控制、市场环境、基本面、技术面、信号识别、综合评分）
- ✅ **短期投资分析**（日内/数天）：关注动量、成交量、突破信号，止损位3-5%
- ✅ **中期投资分析**（数周-数月）：关注趋势持续性、均线排列、资金流向，止损位8-12%，含筹码分布分析
- ✅ **板块分析和龙头识别**：16个板块排名，MVP模型识别龙头股
- ✅ **主力资金追踪**：0-5层分析框架，识别建仓和出货信号
- ✅ **AI交易分析**：复盘AI推荐策略有效性
- ✅ **大模型建议自动保存**：保存短期和中期建议到文件，方便综合对比分析

**综合分析系统（每日自动执行）**：
- ✅ 整合大模型建议和CatBoost预测结果
- ✅ 生成详细的综合买卖建议
- ✅ 包含实时技术指标、模拟交易记录、板块分析等14个部分

**数据获取与监控**：
- ✅ 11个数据源：加密货币、港股、黄金、美股、基本面、股息、IPO等
- ✅ 实时技术指标：恒生指数及自选股的RSI、MACD、布林带、ATR等
- ✅ 模拟交易记录：最近48小时模拟交易记录以表格格式展示

**市场分析功能**：
- ✅ **牛熊市分析自动化**：每周一自动执行，分析市场环境和股票表现
- ✅ **板块表现分析自动化**：每月1号自动执行，按股票类型分析准确度
- ✅ **股票表现TOP 10排名分析**：每月1号自动执行，按不同指标排名
- ✅ **月度趋势分析**：2024-2026年跨年度回测月度分析，识别季节性规律
- ✅ **筹码分布分析**：基于成交量的分箱法，计算筹码集中度和拉升阻力

**风险管理功能**：
- ✅ VaR（风险价值）和ES（预期损失）计算
- ✅ 止损止盈计算（基于ATR或百分比）
- ✅ 最大回撤计算
- ✅ 风险控制检查（止损/止盈/Trailing Stop）

**自动化调度**：
- ✅ 11个GitHub Actions工作流全自动运行
- ✅ 无需服务器，零成本部署
- ✅ 覆盖全天候市场监控和智能分析

### ⚠️ 重要提示

- **机器学习模型仅供参考**：CatBoost 20天模型准确率60.88%，仅供参考，不构成投资建议
- **风险提示**：本系统仅供学习和研究使用，不构成投资建议
- **人机协作**：系统是智能副驾，最终决策权在投资者手中
- **市场波动**：市场存在不确定性，预测模型无法保证100%准确

---

## 核心功能

### 数据获取与监控

系统整合**11个数据源**，为大模型智能决策提供全面的数据支撑：

- **加密货币监控**：比特币、以太坊价格和技术分析（每小时）
- **港股IPO信息**：最新IPO信息（每天）
- **黄金市场分析**：黄金价格和投资建议（每小时）
- **恒生指数监控**：价格、技术指标、交易信号（交易时段）
- **美股市场数据**：标普500、纳斯达克、VIX、美国国债收益率
- **基本面数据**：财务指标、利润表、资产负债表、现金流量表
- **股息信息**：自动获取股息和除净日信息
- **股票新闻**：批量获取自选股新闻，用于情感分析

### 大模型智能决策

#### 恒生指数及自选股分析（hsi_email.py）

**六层分析框架**：
1. **第一层：风险控制检查**（止损/止盈/Trailing Stop）
2. **第二层：市场环境评估**（VIX、成交额、换手率、系统性风险）
3. **第三层：基本面质量评估**（基本面评分、估值水平）
4. **第四层：技术面分析**（多周期趋势、相对强度、技术指标协同）
5. **第五层：信号识别**（建仓/出货信号筛选）
6. **第六层：综合评分与决策**（最终判断）

**风险管理**：
- VaR（风险价值）计算：1日VaR（超短线）、5日VaR（波段交易）、20日VaR（中长期）
- ES（预期损失）计算：尾部风险量化
- 最大回撤计算
- 止损止盈计算（基于ATR或百分比）

**投资建议类型**：
- **短期投资分析**（日内/数天）：
  - 进取型短期分析（可选）
  - 稳健型短期分析
  - 关注短期动量、成交量变化、突破信号
  - 止损位设置较紧（3-5%），快速止损保护本金

- **中期投资分析**（数周-数月）：
  - 稳健型中期分析
  - 保守型中期分析（可选）
  - 关注趋势持续性、均线排列、资金流向
  - 止损位设置较宽（8-12%），允许中期波动
  - **筹码分布分析**：
    - 上方筹码比例：评估拉升阻力（低/中/高）
    - 筹码集中度：评估主力控盘程度（高/中/低）
    - 阻力等级分类：提供突破难度的直观判断

**板块分析**：
- 16个板块排名和龙头股识别
- 业界标准MVP模型（动量+成交量+基本面）
- 支持小市值板块的龙头股识别

**模拟交易**：
- 最近48小时模拟交易记录
- 止损止盈建议
- 连续信号跟踪

**大模型建议自动保存**：短期和中期建议保存到 `data/llm_recommendations_YYYY-MM-DD.txt`

#### 主力资金追踪（hk_smart_money_tracker.py）

**1-6层分析框架**：
- 第1层：数据准备和基础指标计算
- 第2层：建仓信号识别
- 第3层：出货信号识别
- 第4层：信号强度评估
- 第5层：风险评估
- 第6层：综合决策

**投资风格支持**：
- **进取型**：关注动量，快速进出
- **稳健型**：平衡分析，风险可控
- **保守型**：关注基本面，长期持有

**筹码分布分析集成**：
- 为每只股票计算筹码分布
- 识别拉升阻力
- 影响建仓/出货评分

#### AI交易分析（ai_trading_analyzer.py）

- 复盘AI推荐策略有效性
- 分析交易记录和收益率
- 生成改进建议

#### 恒生指数策略（hsi_llm_strategy.py）

- 大模型生成恒生指数交易策略
- 支持多种技术指标
- 提供进场/出场信号

### 综合分析系统

**每日自动执行**，整合大模型建议和CatBoost预测结果，生成实质买卖建议，包含14个部分的内容（详见独立章节）。

### 模拟交易系统

- **真实模拟**：基于大模型建议的模拟交易系统
- **风险控制**：自动止损机制
- **详细记录**：完整的交易日志和持仓分析
- **多种策略**：支持保守型、平衡型、进取型投资偏好
- **大模型提示词优化**：确保大模型提取所有买卖建议，不受策略建议影响

---

## 机器学习预测模型

> **机器学习预测模型作为大模型智能决策的补充，提供数据驱动的量化预测**

### 模型架构

- **CatBoost 20天模型**：主要使用的预测模型（准确率60.88%，F1分数0.6416）
- **板块特定模型**：为16个板块训练独立模型（银行、半导体、科技等）
- **多周期预测**：预测1天、5天、20天后的涨跌
- **特征工程**：520+个特征（技术指标、基本面、美股市场、情感指标等）
- **特征选择**：使用500个精选特征（statistical方法：F-test+互信息混合）

### CatBoost 模型优势

- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度（1-2分钟）
- 更好的泛化能力，减少过拟合
- 稳定性显著提升（标准差2.06%）

### 板块特定模型性能（Walk-forward验证）

> **Walk-forward验证是业界标准的模型验证方法，每个fold重新训练模型，评估真实预测能力**

**银行股板块**（6只股票，12-fold验证）：
- 买入胜率：**50.72%**（突破50%盈亏线）
- 平均收益率：2.98%（20天持有期）
- 年化收益率：35.82%
- 夏普比率：**3.04** ⭐⭐⭐⭐⭐（业界优秀标准，基于Fold收益率计算）
- 最大回撤：-13.30%
- 稳定性评级：中（良好）

**关键发现**：
- ✅ 银行股夏普比率3.04是**优秀的**，远超业界标准（>1.0）
- ✅ 银行股回撤仅-13.30%，符合防御性资产特性
- ✅ 半导体股夏普比率0.126较低，需继续优化
- ✅ 银行股和半导体股表现最佳，推荐继续使用板块模型

### 预测概率与实际涨幅相关性

基于6,328条回测交易记录的实证分析，验证了预测概率与实际涨幅的强正相关关系：

- **相关系数**：0.6289（强正相关）
- **R²值**：0.3956（概率能解释39.56%的涨幅变化）
- **关键发现**：预测概率越大，实际涨幅也越大；概率>0.70的信号，20天平均涨幅9.25%（年化116.55%），胜率93.9%

### 深度学习模型对比（实验性）

经过严格测试，**CatBoost 远优于深度学习模型**：

| 模型 | 准确率 | F1分数 | 推荐指数 |
|------|--------|--------|----------|
| **CatBoost** | **60.88%** | **0.6416** | ⭐⭐⭐⭐⭐ |
| **LSTM** | 51.79% | 0.0000 | ⭐ |
| **Transformer** | 51.15% | 0.1303 | ⭐ |

**结论**：继续使用 CatBoost 单模型作为主要预测模型，深度学习模型仅用于对比研究。

### 融合模型方法（对比研究）

系统支持5种融合方法，用于对比研究：

| 融合方法 | 年化收益率 | 夏普比率 | 买入信号胜率 |
|---------|-----------|---------|------|
| CatBoost 单模型 | 79.54% | 1.20 | 30.02% |
| 加权平均 | 3.16% | 0.32 | 22.75% |
| 简单平均 | 4.01% | 0.40 | 23.63% |
| 投票机制 | 2.31% | 0.36 | 22.58% |
| 动态市场 | 2.40% | 0.33 | 22.96% |

**注意**：根据历史回测结果，所有融合方法的表现均不如CatBoost单模型，建议优先使用CatBoost单模型。融合模型仅用于对比研究。

---

## 综合分析系统

> **整合大模型智能决策与机器学习预测模型，生成实质买卖建议**

### 功能说明

综合分析系统每日自动执行，整合大模型建议和CatBoost预测结果，进行综合对比分析，生成实质的买卖建议。

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
2. **## 一、机器学习预测结果（20天）**（CatBoost单模型，显示全部28只股票及预测方向）
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

- 显示全部28只股票的 CatBoost 预测结果
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
  - 高阻力股票列表
  - 阻力标识说明

### 决策框架

**买入策略**：
- 强烈买入信号且预测概率>0.60
- 大模型建议买入且CatBoost预测上涨

**持有策略**：
- 预测概率>0.50且无卖出信号
- 大模型建议持有

**卖出策略**：
- 预测概率≤0.50且大模型建议卖出
- 止损位被触发

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
├── 分析层（大模型智能决策）
│   ├── 恒生指数及自选股分析（六层分析框架）
│   ├── 主力资金追踪（0-5层分析框架）
│   ├── AI交易分析
│   ├── 恒生指数策略
│   └── 综合分析（每日自动执行）
│
├── 机器学习层（预测模型）
│   ├── CatBoost 单模型（主要使用）⭐
│   ├── LightGBM 模型
│   ├── GBDT 模型
│   ├── LSTM 模型（对比实验，不推荐）
│   ├── Transformer 模型（对比实验，不推荐）
│   ├── 融合模型（5种方法）
│   ├── 板块特定模型
│   ├── Walk-forward验证
│   └── 批量回测（28只股票）
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
    ├── 大模型服务（Qwen API）
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
./scripts/run_comprehensive_analysis.sh
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
# CatBoost 批量回测（推荐，使用阈值0.6）
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.6

# 其他模型回测
python3 ml_services/batch_backtest.py --model-type lgbm --horizon 20 --use-feature-selection --confidence-threshold 0.6
python3 ml_services/batch_backtest.py --model-type gbdt --horizon 20 --use-feature-selection --confidence-threshold 0.6

# 回测结果会保存到：
# - output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json（详细数据）
# - output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt（汇总报告）
```

### 模型对比回测

```bash
# 模型对比回测（3个基本模型 + 5个融合方法）
./scripts/run_model_comparison.sh
python3 scripts/run_model_comparison.sh --force-train  # 强制重新训练所有模型
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

### 板块Walk-forward验证（推荐）

**什么是Walk-forward验证？**
业界标准的模型验证方法，每个fold重新训练模型，评估真实预测能力，避免数据泄漏。

```bash
# 验证所有板块（16个板块，推荐）
python3 ml_services/walk_forward_by_sector.py --all-sectors --horizon 20 --use-feature-selection --start-date 2024-01-01 --end-date 2025-12-31 --output-dir output

# 验证单个板块（如银行股）
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20 --use-feature-selection --start-date 2024-01-01 --end-date 2025-12-31 --output-dir output

# 验证多个板块
python3 ml_services/walk_forward_by_sector.py --sectors tech bank consumer --horizon 20 --use-feature-selection --start-date 2024-01-01 --end-date 2025-12-31 --output-dir output

# 自定义参数（推荐使用阈值0.6）
python3 ml_services/walk_forward_by_sector.py --all-sectors --train-window 12 --test-window 1 --step-window 1 --horizon 20 --confidence-threshold 0.6 --use-feature-selection --start-date 2024-01-01 --end-date 2025-12-31 --output-dir output
```

**支持板块**：`bank`（银行）、`semiconductor`（半导体）、`tech`（科技）、`ai`（人工智能）、`exchange`（交易所）等16个板块

**Walk-forward验证说明**：
- **目的**：业界标准的验证方法，评估模型的真实预测能力和泛化能力
- **方法**：12个月训练窗口，1个月测试窗口，滚动步长1个月
- **优势**：每个fold重新训练模型，避免数据泄漏，更准确评估泛化能力
- **输出**：
  - 每个板块生成：JSON、CSV、Markdown格式报告
  - 汇总报告：`output/walk_forward_sectors_summary_{timestamp}.md`
  - 包含年化收益率数据（符合业界标准）
- **预计时间**：约4小时（16个板块）

**关键指标**：
- 夏普比率：风险调整后收益指标（必须使用年化收益率计算）
- 最大回撤：极端风险指标
- 稳定性评级：收益率标准差（高<2%，中<5%，低>=5%）
- 20天收益率：持有20天的收益率
- 年化收益率：20天收益率 × 12.6（252个交易日/20天）

### 板块表现分析

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

### 股票表现TOP 10排名分析

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

### 综合分析

```bash
# 一键执行完整流程
./scripts/run_comprehensive_analysis.sh

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
│   ├── walk_forward_by_sector.py       # 板块Walk-forward验证 ⭐
│   ├── train_sector_model.py           # 板块模型训练
│   ├── evaluate_sector_model.py        # 板块模型评估
│   ├── us_market_data.py               # 美股市场数据
│   ├── feature_selection.py            # 特征选择模块
│   ├── topic_modeling.py               # LDA主题建模模块
│   ├── lstm_experiment.py              # LSTM对比实验脚本
│   ├── transformer_experiment.py       # Transformer对比实验脚本
│   ├── sector_performance_analysis.py  # 板块表现分析脚本
│   ├── ranking_analysis.py             # 股票表现排名分析脚本
│   ├── BACKTEST_GUIDE.md               # 回测功能使用指南
│   └── ...
│
├── 大模型服务 (llm_services/)
│   ├── qwen_engine.py                  # Qwen大模型接口
│   └── sentiment_analyzer.py           # 情感分析模块
│
├── 自动化脚本 (scripts/)
│   ├── run_comprehensive_analysis.sh   # 综合分析自动化脚本
│   ├── run_model_comparison.sh         # 模型对比自动化脚本
│   ├── run_bull_bear_analysis.sh      # 牛熊市分析自动化脚本
│   ├── run_sector_analysis.sh         # 板块表现分析自动化脚本
│   ├── run_ranking_analysis.sh        # 股票表现排名分析自动化脚本
│   ├── train_and_predict_all.sh       # 训练和预测自动化脚本
│   ├── data_diagnostic.py             # 数据诊断工具
│   ├── feature_evaluation.py          # 特征评估工具
│   └── train_with_feature_selection.py # 训练工具
│
├── 配置文件
│   ├── config.py                       # 全局配置
│   ├── requirements.txt                # 项目依赖
│   ├── set_key.sh                      # 环境变量配置（已加入.gitignore）
│   ├── set_key.sh.sample               # 环境变量配置模板
│   ├── update_data.sh                  # 数据更新工具
│   ├── send_alert.sh                   # 定时任务发送工具
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
│   ├── BULL_BEAR_ANALYSIS_GUIDE.md          # 牛熊市分析使用指南
│   ├── 不同股票类型分析框架对比.md
│   └── IMPROVEMENT_POINTS_FROM_DAILY_STOCK_ANALYSIS.md # 从daily_stock_analysis项目学到的提升点
│
├── 输出文件 (output/)
│   ├── batch_backtest_*.json           # 批量回测详细数据
│   ├── batch_backtest_summary_*.txt    # 批量回测汇总报告
│   ├── model_comparison_report_*.txt   # 模型对比汇总报告
│   ├── lstm_experiment_*.json          # LSTM对比实验详细数据
│   ├── transformer_experiment_*.json   # Transformer对比实验详细数据
│   ├── sector_performance_analysis_*.csv   # 板块表现分析CSV数据
│   ├── sector_performance_analysis_*.json  # 板块表现分析JSON数据
│   ├── sector_performance_analysis_*.md     # 板块表现分析Markdown报告
│   ├── ranking_analysis_*.csv              # 股票表现排名分析CSV数据
│   ├── ranking_analysis_*.json             # 股票表现排名分析JSON数据
│   ├── ranking_analysis_*.md                # 股票表现排名分析Markdown报告
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

### 模型训练和预测时机

**训练时机**：
- **最佳时机**：周末或完市后（数据完整且不影响交易）
- **避免时机**：开市中（数据不稳定，资源占用）、开市前（数据不足）
- **训练频率**：
  - CatBoost 20天模型：每月1次或当市场环境发生重大变化时
  - 板块模型：每季度1次
  - 特征选择：只执行一次，除非特征工程发生重大变化

**预测/使用时机**：
- **最佳预测时机**：完市后（16:00 HKT），使用当日完整数据预测次日
- **恒生指数预测**：开市前（06:00 HKT），预测当日走势
- **避免预测时机**：开市前（数据不足，预测质量低）、开市中（数据不完整）

**推荐工作流**：

| 操作 | 时机 | 说明 |
|------|------|------|
| **模型训练** | 周末或完市后 | 数据完整，不影响交易 |
| **模型预测** | 完市后（16:00 HKT） | 使用当日完整数据预测次日 |
| **恒指预测** | 开市前（06:00 HKT） | 预测当日走势 |
| **综合分析** | 完市后（16:00 HKT） | 生成次日买卖建议 |
| **避免开市中操作** | 09:30-16:00 | 数据不稳定，资源占用 |

**时间线参考**：
- 06:00 HKT：查看恒指预测邮件
- 09:30-16:00：开市中监控（可选）
- 16:00 HKT：综合分析邮件自动发送，查看次日买卖建议
- 20:00 HKT：更新交易记录

---

### GitHub Actions 工作流

系统使用 **GitHub Actions** 进行全自动化调度，无需服务器部署，零硬件成本运行。目前有11个工作流正常运行，覆盖全天候市场监控和智能分析。

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
| **sector-analysis.yml** | 板块表现分析自动化 | 每月1号 UTC 19:00（香港时间上午2:00） | 按股票类型分析模型准确度 |
| **ranking-analysis.yml** | 股票表现TOP 10排名分析 | 每月1号 UTC 19:00（香港时间上午3:00） | 按不同指标排名 |

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

**CatBoost 20天模型（推荐）**：
- 准确率：60.88%
- F1分数：0.6416
- 标准差：2.06%
- 训练时间：1-2分钟
- 特征数量：500+

**其他模型（对比研究）**：
- GBDT 20天：准确率57.94%，F1分数0.7007
- LightGBM 20天：准确率58.93%，F1分数0.7102
- CatBoost 1天：准确率63.09%（⚠️ 不推荐，存在过拟合风险）

### 2024-2026年跨年度回测月度分析

**总体性能指标**：
- 回测时间范围：2024-01-02 至 2026-01-02
- 总交易机会：13,457
- 买入信号数：7,554（占比56.15%）
- 整体准确率：81.53%
- 平均收益率：3.05%（20天持有期）

**季节性规律**：
- 上半年平均收益率：3.79%
- 下半年平均收益率：2.74%
- 最佳月份：2025-01（收益率16.58%）
- 最差月份：2025-03（收益率-7.78%）

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
| **板块表现分析自动化** | ✅ 完整 | 每月1号自动执行，按股票类型分析模型准确度 |
| **股票表现TOP 10排名分析** | ✅ 完整 | 每月1号自动执行，按不同指标排名 |
| **筹码分布分析** | ✅ 完整 | 基于成交量的简单分箱法，计算筹码集中度和拉升阻力 |
| **板块Walk-forward验证** | ✅ 完整 | 业界标准的板块模型验证，评估真实预测能力 |
| **市场环境自适应过滤** | ✅ 完整 | 基于ADX+波动率双因子，动态调整过滤条件 |
| **数据泄漏修正** | ✅ 完整 | 系统性修正10+个特征，使用.shift(1)确保滞后数据 |
| **模块化架构** | ✅ 完成 | data_services、llm_services、ml_services |
| **ML模型** | ✅ 顶尖 | CatBoost 20天准确率60.88%，F1分数0.6416 |
| **批量回测** | ✅ 完整 | 支持28只股票批量回测 |
| **综合分析** | ✅ 稳定 | 每日自动执行，整合大模型建议和CatBoost预测 |
| **交易记录展示** | ✅ 完整 | 最近48小时模拟交易记录以表格格式展示 |
| **实时指标集成** | ✅ 完整 | 集成 hsi_email.py 的实时技术指标 |
| **自动化** | ✅ 稳定 | 11个GitHub Actions工作流正常运行 |
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

### 置信度阈值详细分析

基于12个Fold的Walk-forward验证结果（银行股板块，2024-01-01至2025-12-31）：

**整体性能对比**：

| 指标 | 阈值0.55 | 阈值0.6 | 阈值0.65 |
|------|---------|---------|---------|
| 年化收益率 | 37.59% | 39.58% | 40.15% |
| 买入信号胜率 | 50.72% | 50.40% | 50.88% |
| 平均准确率 | 62.97% | 61.30% | 63.10% |
| 夏普比率 | 0.0245 | 0.0338 | 0.0346 |
| 索提诺比率 | 0.8369 | 0.9287 | 0.9391 |
| 最大回撤 | -13.30% | -13.30% | -12.97% |
| 总买入信号数 | 706 | 695 | 615 |

**关键发现**：
- ✅ 阈值0.6的综合评分最高（78.7分 vs 0.55: 73.9分）
- ⚠️ 阈值0.65的边际效益递减（仅比0.6提升0.9分）
- ⚠️ 阈值0.65的信号减少显著（-12.9%）
- ✅ 阈值0.65在50%的Fold（6/12）表现最佳

**12个Fold级别推荐**：

┌──────┬──────────┬──────────┬───────────────────────────────────────────┐
│ Fold │ 市场环境 │ 推荐阈值 │ 理由                                      │
├──────┼──────────┼──────────┼───────────────────────────────────────────┤
│ 1    │ 牛市     │ 0.6/0.65 │ 收益率最高（6.54%）                       │
│ 2    │ 牛市     │ 0.65     │ 胜率最高（54.05%）                        │
│ 3    │ 震荡市   │ 0.6      │ 收益率改善最大（+0.87%）                  │
│ 4    │ 震荡市   │ 0.55     │ 收益率最高（0.26%），胜率最高（71.74%）   │
│ 5    │ 牛市     │ 0.6      │ 胜率最高（68.00%）                        │
│ 6    │ 牛市     │ 0.55     │ 收益率最高（5.06%），胜率最高（60.29%）   │
│ 7    │ 牛市     │ 0.6      │ 收益率最高（6.32%）                       │
│ 8    │ 震荡市   │ 0.65     │ 从负转正（0.04%）⭐                       │
│ 9    │ 震荡市   │ 0.65     │ 胜率最高（58.06%）                        │
│ 10   │ 牛市     │ 0.6/0.65 │ 收益率几乎相同                            │
│ 11   │ 牛市     │ 0.65     │ 收益率最高（5.31%），准确率最高（74.00%） │
│ 12   │ 牛市     │ 0.55     │ 收益率最高（-0.85%），胜率最高（59.26%）  │
└──────┴──────────┴──────────┴───────────────────────────────────────────┘

💡 **关键洞察**：

1. **市场环境是关键**：
   - 牛市：三个阈值相似，0.6交易机会更多
   - 震荡市：0.65表现最佳（Fold 3、8改善显著）

2. **边际效益递减**：
   - 0.55→0.6：改善显著
   - 0.6→0.65：改善有限

3. **稳定性改善持续**：
   - 阈值0.65的胜率标准差最低（9.31%）

4. **交易机会代价显著**：
   - 阈值0.65的信号减少12.9%

**最终推荐**：阈值0.6（综合评分提升显著，代价可接受）

详细分析报告：
- 12个Fold详细对比：`output/walk_forward_12_folds_detailed_analysis_055_vs_060_vs_065.md`
- 综合对比分析：`output/threshold_optimization_analysis_060_vs_065_vs_055.md`

---

### 其他注意事项

1. **置信度阈值选择**：
   - 保守型投资者：0.60-0.65（风险控制优先）
   - 平衡型投资者：0.55（收益与风险平衡）⭐ 推荐
   - 进取型投资者：0.50-0.55（追求更高收益）

2. **数据验证**：严格的时间序列交叉验证，无数据泄漏，日期索引保留，按时间顺序排列
3. **数据源限制**：部分数据源可能有访问频率限制
4. **缓存机制**：基本面数据缓存7天，可手动清除
5. **交易时间**：模拟交易系统遵循港股交易时间
6. **风险提示**：本系统仅供学习和研究使用，不构成投资建议
7. **API密钥**：请妥善保管API密钥，不要提交到版本控制

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
./scripts/run_comprehensive_analysis.sh
```

### 🌟 无服务器部署 - GitHub Actions 自动化

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

**最后更新**: 重构内容结构，突出大模型智能决策
