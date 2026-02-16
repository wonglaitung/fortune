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
7. **避免硬编码路径** - 使用相对路径基于脚本目录构建路径
   - **十二要素应用原则**：配置应该外化，不应硬编码
   - **跨环境兼容性**：代码应能在不同环境中运行，不依赖特定路径
   - **使用相对路径**：基于脚本所在目录构建路径，而非绝对路径
   - **正确做法**：`script_dir = os.path.dirname(os.path.abspath(__file__)); data_dir = os.path.join(script_dir, 'data')`

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
16. **大模型建议保存功能**（自动保存短期和中期建议到文本文件）
17. **ML预测结果保存功能**（自动保存20天预测结果到文本文件）
18. **综合分析系统**（整合大模型建议和ML预测结果，生成实质买卖建议）

## 关键文件

### 核心脚本（11个）
| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置文件，包含自选股列表（28只股票） |
| `hk_smart_money_tracker.py` | 港股主力资金追踪器 |
| `hsi_email.py` | 恒生指数价格监控器，含AI持仓分析、大模型建议保存、--no-email参数 |
| `hsi_llm_strategy.py` | 恒生指数大模型策略分析器 |
| `simulation_trader.py` | 基于大模型的港股模拟交易系统 |
| `ai_trading_analyzer.py` | AI 交易盈利能力分析器 |
| `crypto_email.py` | 加密货币价格监控器 |
| `gold_analyzer.py` | 黄金市场分析器 |
| `hk_ipo_aastocks.py` | 港股 IPO 信息获取器 |
| `generate_sector_rotation_river_plot.py` | 板块轮动河流图生成工具 |
| **`comprehensive_analysis.py`** | **综合分析脚本，整合大模型建议和ML预测结果** |

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
| `ml_trading_model.py` | 机器学习交易模型，含预测结果保存功能 |
| `ml_prediction_email.py` | 机器学习预测邮件发送器 |
| `us_market_data.py` | 美股市场数据获取模块 |
| `base_model_processor.py` | 模型处理器基类 |
| `compare_models.py` | 模型对比工具 |
| `test_regularization.py` | 正则化策略验证脚本 |
| `feature_selection.py` | **特征选择模块（F-test+互信息混合方法）** |

### 大模型服务模块 (`llm_services/`)
| 文件 | 说明 |
|------|------|
| `qwen_engine.py` | 大模型服务接口 |
| `sentiment_analyzer.py` | 情感分析模块（四维情感评分） |

### 配置文件（5个）
| 文件 | 说明 |
|------|------|
| `requirements.txt` | 项目依赖包列表 |
| `train_and_predict_all.sh` | 完整训练和预测脚本（1天、5天、20天） |
| **`run_comprehensive_analysis.sh`** | **综合分析自动化脚本** |
| `send_alert.sh` | 本地定时执行脚本 |
| `update_data.sh` | 数据更新脚本 |
| `set_key.sh` | 环境变量配置脚本 |

### GitHub Actions 工作流 (`.github/workflows/`)
| 文件 | 说明 | 执行时间 |
|------|------|----------|
| `smart-money-alert.yml` | 主力资金追踪 | 每天 UTC 22:00 |
| `hsi-email-alert-open_message.yml` | 恒生指数邮件 | 周一到周五 UTC 8:00 |
| `crypto-alert.yml` | 加密货币价格 | 每小时 |
| `gold-analyzer.yml` | 黄金市场分析 | 每小时 |
| `ipo-alert.yml` | IPO 信息 | 每天 UTC 2:00 |
| `ai-trading-analysis-daily.yml` | AI 交易分析 | 周一到周五 UTC 8:30 |
| **`ml-prediction-alert.yml`** | **综合分析** | **每周日 UTC 1:00** |

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
- **大模型建议自动保存**：短期和中期建议保存到 `data/llm_recommendations_YYYY-MM-DD.txt`
- **--no-email 参数**：支持禁用邮件发送，仅生成分析报告

### 机器学习交易模型
- **算法**：LightGBM 和 GBDT+LR
- **特征**：2936 特征（技术指标、基本面、美股市场、股票类型、情感指标、板块分析、交叉特征、长期趋势特征）
- **预测周期**：1天、5天、20天
- **性能**（2026-02-16最新，经过超增强正则化优化和特征选择优化）：
  - **次日**：LightGBM 51.70%（±2.41%），GBDT+LR 52.28%（±1.66%）
  - **一周**：LightGBM 54.64%（±2.82%），GBDT+LR 53.75%（±2.94%）
  - **一个月**：LightGBM 59.11%（±6.90%），GBDT+LR 58.32%（±7.31%）
- **超增强正则化（2026-02-16）**：
  - **LightGBM一个月模型**：reg_alpha=0.25, reg_lambda=0.25（超强正则化，降低波动）
  - **GBDT+LR一个月模型**：reg_alpha=0.22, reg_lambda=0.22（强正则化）
  - **其他模型**：reg_alpha=0.15, reg_lambda=0.15
- **特征选择优化（2026-02-16）**：
  - **LightGBM**：使用500个精选特征（F-test+互信息混合方法，从2936个特征筛选）
  - **GBDT+LR**：跳过特征选择，使用全部2936个特征（差异化策略）
  - **实现方式**：通过model_type属性区分，自动应用差异化策略
- **优化效果**：
  - LightGBM一个月模型：准确率57.63% → 59.11%（+1.48%）
  - GBDT+LR一个月模型：准确率55.94% → 58.32%（+2.38%）
  - 标准偏差：LightGBM ±6.90%，GBDT+LR ±7.31%
- **预测结果自动保存**：20天预测结果保存到 `data/ml_predictions_20d_YYYY-MM-DD.txt`

### 模拟交易系统
- 基于大模型分析的模拟交易
- 支持三种投资者类型
- 止损机制
- 交易记录自动保存

### 综合分析系统（新增）
**功能说明**：整合大模型建议（短期和中期）与ML预测结果（20天），进行综合对比分析，生成实质的买卖建议

**执行流程**：
1. 调用 `hsi_email.py --force --no-email` 生成大模型建议（不发送邮件）
2. 训练20天ML模型
3. 生成20天ML预测
4. 提取大模型建议中的买卖信息（包含推荐理由、操作建议、价格指引、风险提示）
5. 提取ML预测结果中的上涨概率信息
6. 提交给大模型进行综合分析
7. 生成详细的综合买卖建议，包含：
   - 强烈买入信号（2-3只）
   - 买入信号（3-5只）
   - 持有/观望
   - 卖出信号（如有）
   - 风险控制建议
8. 发送邮件通知，包含完整信息参考章节

**运行方式**：
```bash
# 一键执行完整流程
./run_comprehensive_analysis.sh

# 或手动执行各步骤
python3 hsi_email.py --force --no-email
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type both
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type both
python3 comprehensive_analysis.py
python3 comprehensive_analysis.py --no-email  # 不发送邮件
```

**输出文件**：
- `data/llm_recommendations_YYYY-MM-DD.txt`：大模型建议（短期和中期）
- `data/ml_predictions_20d_YYYY-MM-DD.txt`：ML 20天预测结果
- `data/comprehensive_recommendations_YYYY-MM-DD.txt`：综合买卖建议

**邮件内容**：
- **第一部分：综合买卖建议**
  - 强烈买入信号
  - 买入信号
  - 持有/观望
  - 卖出信号
  - 风险控制建议
- **第二部分：信息参考**
  - 大模型短期买卖建议（日内/数天）
  - 大模型中期买卖建议（数周-数月）
  - 机器学习预测结果（20天）
    - LightGBM模型
    - GBDT+LR模型

**自动化调度**：
- GitHub Actions 工作流：`ml-prediction-alert.yml`
- 执行时间：每周日 UTC 01:00（香港时间上午9点）
- 支持手动触发

## 配置参数

### 自选股配置（28只）
在 `config.py` 中配置：
- 银行类：0005.HK、0939.HK、1288.HK、1398.HK、3968.HK
- 科技类：0700.HK、1810.HK、3690.HK、9988.HK
- 半导体：0981.HK、1347.HK
- AI：2533.HK、6682.HK、9660.HK
- 能源：0883.HK、1088.HK
- 房地产：0012.HK、0016.HK、1109.HK（新增）
- 其他：0388.HK、0728.HK、0941.HK、1138.HK、1211.HK、1299.HK、1330.HK、2269.HK、2800.HK

**⚠️ 重要配置说明（新增股票时必读）**：

如果在 `config.py` 中增加新的股票代码，必须同时在 `ml_services/ml_trading_model.py` 的 `create_stock_type_features` 方法中补充股票类型信息：

1. **在 `stock_type_mapping` 字典中添加股票类型**：
   ```python
   '股票代码.HK': {'type': '类型', 'name': '股票名称', 'defensive': XX, 'growth': XX, 'cyclical': XX, 'liquidity': XX, 'risk': XX},
   ```
   - **type**: 股票类型（bank/tech/utility/semiconductor/ai/energy/shipping/exchange/insurance/biotech/new_energy/environmental/real_estate/index）
   - **defensive**: 防御性评分（0-100）
   - **growth**: 成长性评分（0-100）
   - **cyclical**: 周期性评分（0-100）
   - **liquidity**: 流动性评分（0-100）
   - **risk**: 风险评分（0-100）

2. **在 `stock_info_mapping` 字典中添加相同的股票类型信息**

3. **在衍生特征权重部分添加该股票类型的权重特征**：
   - 必须在 `'Cyclical_Style_Flow_Weight': 0.2 if stock_info['type'] in ['energy', 'shipping', 'exchange'] else 0.0,` 这一行后面添加
   - 格式示例：
   ```python
   # 房地产股：基本面权重20%，技术分析权重60%，资金流向权重20%
   'RealEstate_Style_Fundamental_Weight': 0.2 if stock_info['type'] == 'real_estate' else 0.0,
   'RealEstate_Style_Technical_Weight': 0.6 if stock_info['type'] == 'real_estate' else 0.0,
   'RealEstate_Style_Flow_Weight': 0.2 if stock_info['type'] == 'real_estate' else 0.0,
   ```

**如果不配置股票类型信息，会出现"⚠️ 未找到股票 XXXX.HK 的类型信息"警告，并且该股票将无法生成股票类型相关特征。**

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

# 恒生指数监控（自动保存大模型建议，可选不发送邮件）
python hsi_email.py
python hsi_email.py --date 2025-10-25
python hsi_email.py --no-email  # 仅生成报告，不发送邮件

# 板块分析
python data_services/hk_sector_analysis.py --period 5 --style moderate

# 板块轮动河流图
python generate_sector_rotation_river_plot.py

# ML 模型训练和预测（自动保存预测结果）
./train_and_predict_all.sh
python ml_services/ml_trading_model.py --mode train --horizon 1
python ml_services/ml_trading_model.py --mode predict --horizon 20
python ml_services/ml_trading_model.py --mode train --horizon 20 --use-feature-selection  # 使用特征选择（LightGBM 500特征，GBDT+LR 2936特征）

# 模拟交易
python simulation_trader.py --investor-type moderate

# AI 交易分析
python ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 黄金分析
python gold_analyzer.py

# 加密货币监控
python crypto_email.py

# 综合分析（一键执行）
./run_comprehensive_analysis.sh
python comprehensive_analysis.py
python comprehensive_analysis.py --no-email  # 不发送邮件
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
│   │   └── 大模型建议保存功能 (save_llm_recommendations)
│   │   └── --no-email 参数（禁用邮件发送）
│   ├── AI交易盈利能力分析器 (ai_trading_analyzer.py)
│   ├── **综合分析脚本** (comprehensive_analysis.py)
│   │   ├── 提取大模型建议（extract_llm_recommendations）
│   │   ├── 提取ML预测结果（extract_ml_predictions）
│   │   ├── 综合对比分析（run_comprehensive_analysis）
│   │   └── 邮件发送功能（send_email）
│   └── 机器学习模块 (ml_services/)
│       ├── 机器学习交易模型 (ml_trading_model.py)
│   │   ├── 特征工程（2936个特征，含54个新增特征）
│   │   │   ├── 滚动统计特征（偏度、峰度、多周期波动率）
│   │   │   ├── 价格形态特征（日内振幅、影线比例、缺口）
│   │   │   ├── 量价关系特征（背离、OBV、成交量波动率）
│   │   │   └── **长期趋势特征**（MA120/250、长期收益率、长期波动率、长期ATR、长期成交量、长期支撑阻力位、长期RSI）
│   │   ├── **特征选择模块**（feature_selection.py）
│   │   │   ├── F-test特征选择（统计显著性）
│   │   │   ├── 互信息特征选择（关联强度）
│   │   │   ├── 混合方法（交集+综合得分）
│   │   │   └── 差异化策略（LightGBM 500特征，GBDT+LR 2936特征）
│   │   ├── 分类特征编码（LabelEncoder）
│   │   ├── **超增强正则化（2026-02-16）**
│       │   │   ├── LightGBM一个月模型：reg_alpha=0.25, reg_lambda=0.25
│       │   │   ├── GBDT+LR一个月模型：reg_alpha=0.22, reg_lambda=0.22
│       │   │   └── 其他模型：reg_alpha=0.15, reg_lambda=0.15
│       │   ├── 正则化增强（L1/L2正则化、早停、树深度控制）
│       │   ├── 特征重要性分析
│       │   └── **预测结果保存功能** (save_predictions_to_text)
│       ├── 机器学习预测邮件发送器 (ml_prediction_email.py)
│       ├── 美股市场数据获取模块 (us_market_data.py)
│       ├── 模型处理器基类 (base_model_processor.py)
│       ├── 模型对比工具 (compare_models.py)
│       └── **正则化策略验证脚本** (test_regularization.py)
├── 交易层
│   └── 港股模拟交易系统 (simulation_trader.py)
└── 服务层 (llm_services/)
    ├── 大模型接口 (qwen_engine.py)
    └── 情感分析模块 (sentiment_analyzer.py)
```

## 数据保存功能

### 大模型建议保存
**功能说明**：自动保存短期和中期大模型建议到文本文件，方便后续提取和对比分析

**保存位置**：`data/llm_recommendations_YYYY-MM-DD.txt`

**保存时机**：`hsi_email.py` 生成大模型分析后立即保存

**文件格式**：
```
================================================================================
大模型买卖建议报告
日期: 2026-02-14
生成时间: 2026-02-14 22:07:47
================================================================================

【中期建议】持仓分析
--------------------------------------------------------------------------------
[大模型持仓分析内容...]

【短期建议】买入信号分析
--------------------------------------------------------------------------------
[大模型买入信号分析内容...]
```

**使用方法**：
```bash
# 运行恒生指数监控，自动保存大模型建议
python hsi_email.py

# 建议内容会保存到 data/llm_recommendations_YYYY-MM-DD.txt
```

### ML预测结果保存
**功能说明**：自动保存20天预测结果到文本文件，包含详细的预测结果和统计信息

**保存位置**：`data/ml_predictions_20d_YYYY-MM-DD.txt`

**保存时机**：`ml_services/ml_trading_model.py` 预测20天周期时自动保存

**文件格式**：
```
================================================================================
机器学习20天预测结果
预测日期: 2026-02-14
生成时间: 2026-02-14 22:07:48
================================================================================

【预测结果】
--------------------------------------------------------------------------------
股票代码       股票名称         预测方向       上涨概率         当前价格         数据日期            预测目标日期         
--------------------------------------------------------------------------------
0700.HK    腾讯控股         上涨         0.6500       380.50       2026-02-14      2026-03-14     

--------------------------------------------------------------------------------
【统计信息】
--------------------------------------------------------------------------------
预测上涨: 3 只
预测下跌: 2 只
总计: 5 只
上涨比例: 60.0%

两个模型一致性: 3/5 (60.0%)
平均上涨概率: 0.5200
```

**使用方法**：
```bash
# 运行20天预测，自动保存预测结果
python ml_services/ml_trading_model.py --mode predict --horizon 20

# 或使用完整训练和预测脚本
./train_and_predict_all.sh

# 预测结果会保存到 data/ml_predictions_20d_YYYY-MM-DD.txt
```

### 综合分析结果保存
**功能说明**：自动保存综合买卖建议到文本文件，包含详细的推荐理由、操作建议、价格指引和风险提示

**保存位置**：`data/comprehensive_recommendations_YYYY-MM-DD.txt`

**保存时机**：`comprehensive_analysis.py` 综合分析完成后自动保存

**文件格式**：
```
================================================================================
综合买卖建议
生成时间: 2026-02-14 23:19:13
分析日期: 2026-02-14
================================================================================

# 综合买卖建议

## 强烈买入信号（2-3只）
1. [股票代码] [股票名称] 
   - 推荐理由：[详细的推荐理由，包含技术面、基本面、资金面等分析]
   - 操作建议：买入/卖出/持有/观望
   - 建议仓位：[X]%
   - 价格指引：
     * 建议买入价：HK$XX.XX
     * 止损位：HK$XX.XX（-X.X%）
     * 目标价：HK$XX.XX（+X.X%）
   - 操作时机：[具体的操作时机说明]
   - 风险提示：[主要风险因素]

## 买入信号（3-5只）
[同上格式]

## 持有/观望
[同上格式]

## 卖出信号（如有）
[同上格式]

## 风险控制建议
- 当前市场整体风险：[高/中/低]
- 建议仓位百分比：[X]%
- 止损位设置：[策略]
- 组合调整建议：[具体的组合调整建议]
```

**使用方法**：
```bash
# 一键执行完整综合分析流程
./run_comprehensive_analysis.sh

# 或手动执行
python comprehensive_analysis.py

# 综合建议会保存到 data/comprehensive_recommendations_YYYY-MM-DD.txt
```

## 综合对比分析流程

### 完整流程
1. **步骤1**：调用 `hsi_email.py --force --no-email` 生成大模型建议
   - 不发送邮件，避免重复通知
   - 保存到 `data/llm_recommendations_YYYY-MM-DD.txt`

2. **步骤2**：训练20天ML模型
   - 使用 LightGBM 和 GBDT+LR 算法
   - 应用超增强正则化配置

3. **步骤3**：生成20天ML预测
   - 保存到 `data/ml_predictions_20d_YYYY-MM-DD.txt`

4. **步骤4**：综合分析
   - 提取大模型建议中的买卖信息（推荐理由、操作建议、价格指引、风险提示）
   - 提取ML预测结果中的上涨概率信息
   - 提交给大模型进行综合对比分析
   - 生成详细的综合买卖建议
   - 发送邮件通知（可通过 `--no-email` 参数禁用）

### 对比维度
- 大模型短期建议 vs ML 20天预测（概率高的一致信号优先）
- 大模型中期建议 vs 股票基本面的匹配度
- 技术分析信号与大模型、ML预测的一致性

## 项目当前状态

**最后更新**: 2026-02-16

**项目成熟度**: 生产就绪

**核心模块状态**:
- ✅ 数据获取层：完整，支持多数据源
- ✅ 数据服务层：完整，模块化架构
- ✅ 分析层：完整，含技术分析、基本面、ML模型
- ✅ **综合分析系统**：完整，整合大模型建议和ML预测结果
- ✅ 交易层：完整，模拟交易系统正常运行
- ✅ 服务层：完整，大模型服务集成

**ML模型状态**（2026-02-16最新，经过超增强正则化优化和特征选择优化）:
- ✅ 次日模型：LightGBM 51.70%（±2.41%），GBDT+LR 52.28%（±1.66%）
- ✅ 一周模型：LightGBM 54.64%（±2.82%），GBDT+LR 53.75%（±2.94%）
- ✅ **一个月模型**：LightGBM 59.11%（±6.90%），GBDT+LR 58.32%（±7.31%）
- ✅ **超增强正则化（2026-02-16）**：
  - LightGBM一个月模型：reg_alpha=0.25, reg_lambda=0.25（标准偏差-11.7%）
  - GBDT+LR一个月模型：reg_alpha=0.22, reg_lambda=0.22（标准偏差-14.3%）
- ✅ **特征选择优化（2026-02-16）**：
  - LightGBM：使用500个精选特征（准确率+1.48%）
  - GBDT+LR：跳过特征选择，使用全部2936个特征（准确率+0.52%）
  - 差异化策略：通过model_type属性区分，自动应用
- ✅ **数据泄漏检测**：已修复
- ✅ **预测结果保存**：自动保存20天预测结果到文本文件

**大模型功能状态**:
- ✅ 恒生指数监控集成大模型分析
- ✅ AI智能持仓分析
- ✅ 大模型建议自动保存（短期和中期建议）
- ✅ 情感分析模块（四维情感评分）
- ✅ 多风格分析支持（进取型短期、稳健型短期、稳健型中期、保守型中期）
- ✅ **--no-email 参数**：支持禁用邮件发送，仅生成分析报告

**综合分析系统状态**:
- ✅ 大模型建议提取功能（提取推荐理由、操作建议、价格指引、风险提示）
- ✅ ML预测结果提取功能（提取预测上涨的股票）
- ✅ 综合对比分析功能（整合两种信息源）
- ✅ 邮件发送功能（SMTP_SSL + 重试机制）
- ✅ 自动化脚本（run_comprehensive_analysis.sh）
- ✅ GitHub Actions 工作流（每周日自动执行）

**自动化状态**:
- ✅ GitHub Actions：7个工作流正常运行
- ✅ 邮件通知：163邮箱服务稳定
- ✅ 定时任务：支持本地cron和GitHub Actions
- ✅ 数据保存：大模型建议、ML预测结果、综合建议自动保存
- ✅ 综合分析：每周日自动执行，生成实质买卖建议

**待优化项**:
- ⚠️ **一个月模型波动性仍需优化**（±6.90% / ±7.31%，目标±5%）
- ⚠️ **风险管理模块**（VaR、止损止盈、仓位管理）
- ⚠️ **深度学习模型**（LSTM、Transformer）
- ⚠️ **Web界面**

## 大模型集成

- `llm_services/qwen_engine.py` 提供大模型接口
- 支持聊天和嵌入功能
- 集成到主力资金追踪、模拟交易、新闻过滤、黄金分析等模块
- 情感分析模块提供四维情感评分
- **大模型建议自动保存**：短期和中期建议保存到文本文件，方便综合对比分析
- **综合分析**：整合大模型建议和ML预测结果，生成实质买卖建议

## 数据文件结构

数据文件存储在 `data/` 目录：
- `actual_porfolio.csv`: 实际持仓数据
- `all_stock_news_records.csv`: 股票新闻记录
- `simulation_transactions.csv`: 交易历史记录
- `simulation_state.json`: 模拟交易状态
- `llm_recommendations_YYYY-MM-DD.txt`: **大模型建议文件**
- `ml_predictions_20d_YYYY-MM-DD.txt`: **ML预测结果文件**
- `comprehensive_recommendations_YYYY-MM-DD.txt`: **综合买卖建议文件（新增）**
- `ml_trading_model_*.pkl`: ML 模型文件（已从 Git 移除）
- `fundamental_cache/`: 基本面数据缓存（已从 Git 移除）

## ML模型优化经验

### 2026-02-14 特征工程优化

#### 优化背景
- 目标：提升机器学习模型（次日、一周、一个月）的预测准确率，达到业界优秀水平
- 初始性能：次日50.93%、一周53.70%、一个月57.09%（LightGBM）
- 最终性能：次日51.70%、一周54.64%、一个月58.97%（LightGBM）

#### 新增特征总结

**批次1：高优先级和中优先级特征（约30个）**

| 类别 | 特征数 | 说明 |
|------|--------|------|
| 滚动统计特征 | 9 | 均线偏离度、多周期波动率、偏度、峰度 |
| 价格形态特征 | 12 | 高低点位置、日内振幅、影线比例、开盘缺口 |
| 量价关系特征 | 7 | 量价背离、OBV趋势、成交量波动率 |

**批次2：长期趋势特征（24个）**

| 类别 | 特征数 | 说明 |
|------|--------|------|
| 长期均线特征 | 7 | MA120、MA250、均线排列、趋势斜率 |
| 长期收益率特征 | 4 | 120日/250日收益率、动量、动量加速度 |
| 长期乖离率 | 2 | MA120/MA250乖离率 |
| 长期波动率 | 2 | 60日/120日波动率 |
| 长期ATR | 4 | 60日/120日ATR均值、相对ATR |
| 长期成交量 | 3 | 120日/250日成交量、成交活跃度 |
| 长期支撑阻力位 | 3 | 120日高低点、距离支撑阻力位 |
| 长期RSI | 1 | 120日RSI |

#### 分周期优化

##### 次日模型（horizon=1）- 强正则化
- n_estimators: 50 → 40
- learning_rate: 0.03 → 0.02
- max_depth: 4 → 3
- num_leaves: 15 → 12
- min_child_samples: 30 → 40
- reg_alpha/reg_lambda: 0.1 → 0.2

**效果**：次日模型从50.08%提升至51.66%（+1.58%）

##### 一周模型（horizon=5）- 防过拟合
- num_leaves: 32 → 24
- stopping_rounds: 10 → 15
- min_child_samples: 20 → 30

**效果**：一周模型GBDT+LR从51.21%提升至52.34%（+1.13%）

##### 一个月模型（horizon=20）- 差异化配置
- **LightGBM**: reg_alpha/reg_lambda: 0.15 → 0.18（降低波动）
- **GBDT+LR**: reg_alpha/reg_lambda: 0.18 → 0.15（恢复准确率）

**效果**：
- LightGBM：57.96% → 58.97%（+1.01%）
- GBDT+LR：55.66% → 58.52%（+2.86%，完全恢复）

#### 特征验证结果

| 特征 | 重要性排名 | 重要性 | 说明 |
|------|-----------|--------|------|
| `Kurtosis_20d` | Top 10 | 3.53% | 进入一个月模型Top 10 |
| `Skewness_20d` | Top 10 | 1.58% | 进入一个月模型Top 10 |
| `Volatility_120d` | Top 10 | 1.47% | 进入一个月模型Top 10 |
| `HSI_Return_60d` | Top 1 | 17.57% | 核心特征 |
| `US_10Y_Yield` | Top 2 | 8.35% | 美股特征 |
| `VIX_Level` | Top 3 | 6.98% | 恐慌指数 |

#### 关键经验总结

1. **特征工程的重要性**
   - 新增54个特征（批次1约30个，批次2 24个）
   - 总特征数从2530增至2936（+16%）
   - 多个新特征进入Top 10/20，证明有效性

2. **分周期优化策略**
   - 不同周期需要不同的正则化策略
   - 次日、一周模型保持0.15配置
   - 一个月模型采用差异化配置（LightGBM=0.18, GBDT+LR=0.15）

3. **长期趋势特征的价值**
   - 专门针对一个月模型添加24个长期特征
   - 长期均线、收益率、波动率对一个月预测至关重要
   - 长期特征能捕捉大周期趋势，减少短期噪音

4. **正则化差异化配置**
   - LightGBM一个月模型可承受更强正则化（0.18）
   - GBDT+LR一个月模型对正则化更敏感（0.15最优）
   - 配置差异化获得最佳综合性能

5. **业界最佳实践**
   - 偏度、峰度是风险管理的核心指标
   - 120日/250日均线是趋势分析的生命线
   - 量价背离是经典反转信号

6. **LR算法probability含义（2026-02-15修正）**
   - **关键理解**：probability字段始终代表上涨概率P(y=1|x)，不会根据prediction改变含义
   - **prediction=1时**：probability > 0.5（上涨概率高）
   - **prediction=0时**：probability <= 0.5（上涨概率低，即下跌概率高）
   - **强烈上涨信号**：prediction=1且probability > 0.65
   - **强烈下跌信号**：prediction=0且probability < 0.40（即下跌概率 > 60%）
   - **中性信号**：probability在0.40-0.60之间（上涨或下跌概率都不超过60%）
   - **实际案例**：1299.HK友邦保险，prediction=0, probability=0.474，不是强烈下跌信号，而是观望信号（下跌概率52.6%，不强烈）
   - **常见误区**：错误认为prediction=0时probability代表下跌概率，这会导致判断标准错误

### 2026-02-16 超增强正则化优化

#### 优化背景
- 目标：进一步降低一个月模型的训练/验证差距（过拟合）
- 初始状态：LightGBM ±7.17%，GBDT+LR ±7.07%
- 目标状态：降低至 <15%（业界可接受范围）

#### 优化策略
- **超增强正则化**：大幅提升L1/L2正则化系数
- **早停机制增强**：增加early stopping patience
- **树结构简化**：减少树深度和叶子节点数
- **样本和特征采样**：降低采样比例

#### 优化配置

##### LightGBM一个月模型（horizon=20）
```python
lgb_params = {
    'n_estimators': 40,           # 45→40
    'learning_rate': 0.02,         # 0.025→0.02
    'max_depth': 3,                # 4→3
    'num_leaves': 11,              # 13→11
    'min_child_samples': 40,       # 35→40
    'subsample': 0.6,              # 0.65→0.6
    'colsample_bytree': 0.6,       # 0.65→0.6
    'reg_alpha': 0.25,             # 0.18→0.25 (+39%)
    'reg_lambda': 0.25,            # 0.18→0.25 (+39%)
    'min_split_gain': 0.15,        # 0.12→0.15
    'feature_fraction': 0.6,       # 0.65→0.6
    'bagging_fraction': 0.6,       # 0.65→0.6
    'stopping_rounds': 15,         # 10→15
}
```

##### GBDT+LR一个月模型（horizon=20）
```python
n_estimators = 28           # 32→28
num_leaves = 20              # 24→20
stopping_rounds = 18         # 12→18
min_child_samples = 35       # 30→35
reg_alpha = 0.22             # 0.15→0.22 (+47%)
reg_lambda = 0.22            # 0.15→0.22 (+47%)
subsample = 0.6              # 0.65→0.6
colsample_bytree = 0.6       # 0.65→0.6
learning_rate = 0.025        # 0.03→0.025
min_split_gain = 0.12        # 0.1→0.12
feature_fraction = 0.6       # 0.7→0.6
bagging_fraction = 0.6       # 0.7→0.6
```

#### 优化效果

| 模型 | 优化前准确率 | 优化后准确率 | 变化 | 优化前标准偏差 | 优化后标准偏差 | 改善 |
|------|-------------|-------------|------|--------------|--------------|------|
| LightGBM (1个月) | 58.97% | 57.63% | -1.34% | ±7.17% | ±6.33% | -11.7% |
| GBDT+LR (1个月) | 58.52% | 57.80% | -0.72% | ±7.07% | ±6.06% | -14.3% |

**后续特征选择优化（2026-02-16）**：
- LightGBM一个月模型：57.63% → 59.11%（+1.48%），标准偏差±6.90%
- GBDT+LR一个月模型：57.80% → 58.32%（+0.52%），标准偏差±7.31%

#### 关键经验总结

1. **正则化强化的权衡**
   - 准确率略微下降（<1.4%）
   - 稳定性显著提升（11.7-14.3%）
   - 实际交易中稳定性更重要

2. **差异化策略的重要性**
   - LightGBM可承受更强正则化（0.25）
   - GBDT+LR对正则化稍敏感（0.22）
   - 配置差异化获得最佳平衡

3. **早停机制的作用**
   - stopping_rounds从10增加到15-18
   - 有效防止过拟合
   - 提升模型泛化能力

4. **持续优化的必要性**
   - 一个月模型波动性仍需优化（当前±6.90%/±7.31%，目标±5%）
   - 后续可考虑特征选择、深度学习模型
   - **特征选择优化（2026-02-16）**：已实现，LightGBM准确率提升至59.11%，GBDT+LR准确率提升至58.32%

### 2026-02-16 特征选择优化

#### 优化背景
- 目标：降低特征数量，提升模型训练速度，减少过拟合风险
- 初始状态：2936个特征
- 目标状态：筛选出500个最有效特征

#### 优化策略
- **F-test特征选择**：筛选统计显著性高的特征（ANOVA F-value）
- **互信息特征选择**：筛选与目标变量关联强的特征（Mutual Information）
- **混合方法**：取交集+综合得分排序，确保特征质量

#### 优化配置

```python
# F-test选择
from sklearn.feature_selection import SelectKBest, f_classif
f_selector = SelectKBest(f_classif, k=1000)
X_f_selected = f_selector.fit_transform(X, y)

# 互信息选择
from sklearn.feature_selection import SelectKBest, mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=1000)
X_mi_selected = mi_selector.fit_transform(X, y)

# 混合选择（交集+综合得分）
f_scores = f_selector.scores_
mi_scores = mi_selector.scores_
selected_features = select_top_features(f_scores, mi_scores, top_k=500)
```

#### 优化效果

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| **特征数量** | 2,936 | 500 | -83.0% |
| **训练速度** | 基准 | 预期5-6倍提升 | +400-500% |
| **准确率**（LightGBM） | 57.63% | 59.11% | +1.48% |
| **准确率**（GBDT+LR） | 55.94% | 58.32% | +2.38% |
| **标准偏差**（LightGBM） | ±6.33% | ±6.90% | +0.57% |
| **标准偏差**（GBDT+LR） | ±7.36% | ±7.31% | -0.05% |
| **过拟合风险** | 高 | 显著降低 | - |
| **交集特征占比** | - | 52.4% | - |
| **平均综合得分** | - | 0.1943 | - |

#### Top 10核心特征

| 排名 | 特征名称 | 综合得分 | 说明 |
|------|---------|---------|------|
| 1 | Volume_MA250 | 0.500 | 250日成交量 |
| 2 | 60d_Trend_Resistance_120d | 0.454 | 60日趋势+120日阻力位 |
| 3 | 60d_RS_Signal_Volume_MA250 | 0.451 | 60日相对强弱信号+250日成交量 |
| 4 | 60d_RS_Signal_Resistance_120d | 0.443 | 60日相对强弱信号+120日阻力位 |
| 5 | 60d_Trend_MA250 | 0.442 | 60日趋势+250日均线 |
| 6 | 60d_Trend_Volume_MA250 | 0.441 | 60日趋势+250日成交量 |
| 7 | Support_120d | 0.438 | 120日支撑位 |
| 8 | 60d_RS_Signal_MA250 | 0.431 | 60日相对强弱信号+250日均线 |
| 9 | Resistance_120d | 0.427 | 120日阻力位 |
| 10 | MA250 | 0.404 | 250日均线 |

#### 关键发现

1. **长期趋势特征占据主导地位**
   - MA250（250日均线）出现在多个高得分特征中
   - 60日、20日、10日多周期趋势特征表现优异
   - 支撑阻力位（Support_120d、Resistance_120d）重要性高

2. **美股市场特征重要性验证**
   - SP500_Return_5d（标普500 5日收益率）- 排名15
   - NASDAQ_Return_20d（纳斯达克20日收益率）- 排名16
   - 美股市场数据对港股预测具有重要作用

3. **特征工程的有效性**
   - 长期趋势特征（24个）在Top 20中占据多数
   - 证明特征工程方向正确，长期趋势对一个月预测至关重要
   - 量价关系特征（Volume_MA250）得分最高

5. **训练效率显著提升**
   - 特征减少83%，训练速度预期提升5-6倍
   - 内存占用降低，可支持更多实验
   - 模型复杂度降低，过拟合风险减少

6. **模型差异化特征选择策略**
   - LightGBM：使用500个精选特征（F-test+互信息混合）
   - GBDT+LR：跳过特征选择，使用全部2936个特征
   - 差异化原因：GBDT+LR的LR层需要更多输入维度，对特征选择敏感（初始准确率下降2.20%）
   - 实现方式：通过model_type属性区分，LightGBM应用特征选择，GBDT+LR自动跳过

7. **GBDT+LR特征选择问题发现与解决**
   - **问题发现**：GBDT+LR使用特征选择后准确率从57.80%降至55.60%（-2.20%）
   - **根本原因**：GBDTLRModel的prepare_data方法缺少`create_technical_fundamental_interactions`调用
   - **特征差异**：缺少18个技术指标与基本面交互特征 + 234个相关交叉特征，共计252个特征
   - **修复方案**：在GBDTLRModel.prepare_data中添加`create_technical_fundamental_interactions`调用
   - **修复效果**：准确率从55.94%恢复到58.32%（+2.38%），特征数量从2684恢复到2936

#### 下一步计划
1. 将特征选择集成到训练流程中
2. 使用选择的500个特征重新训练模型
3. 验证准确率提升效果（目标+0.5-1.0%）
4. 评估模型稳定性是否进一步提升

## 提交记录
- commit 6d7168e: update: 更新comprehensive_analysis.py中的ML模型性能数据
- commit 5f6f5d2: fix(ml): 修复GBDT+LR缺少技术指标与基本面交互特征的问题
- commit 6265a84: feat(ml): 实现GBDT+LR跳过特征选择策略
- commit b4c8b51: fix(ml): 修复特征选择特征索引不匹配问题
- commit e2eac5e: chore: 更新综合分析脚本，使用特征选择功能
- commit 316308d: feat(ml): 集成特征选择功能到训练流程
- commit 2ece211: docs: 更新README和IFLOW.md，添加特征选择优化完成记录
- commit 26e1ede: feat(ml): 添加特征选择优化功能，使用F-test+互信息混合方法
- commit cf881d9: refactor: 简化提示词，删除冗余的优化日期信息
- commit 8e20075: docs: 更新IFLOW.md，添加2026-02-16超增强正则化优化记录
- commit 766cb36: perf(ml): 超增强一个月模型正则化以降低过拟合
- commit 05eb45d: docs: 添加计划完成度分析到README
- commit 898c40f: docs: 补充未来计划到README
- commit 935b0ff: docs: 补充README重要内容
- commit 327fa84: docs: 重写README为简洁版本，添加房地产股票到自选股
- commit e5f1423: feat: 邮件中添加完整信息参考章节
- commit ffcd794: fix: 修复硬编码路径问题，使用相对路径替代绝对路径
- commit 8143a36: refactor: 采用方案A（短期触发+中期确认）并优化提示词，使用正则表达式替代脆弱字符串匹配
- commit 2644341: fix: 修正LR算法判断标准并关闭思考模式，删除邮件信息参考章节
- commit 0239abb: refactor: 优化综合分析提示词，建立量化判断标准和明确信息源优先级
- commit 0e84c3b: feat: 综合分析分离短期/中期建议和LightGBM/GBDT+LR预测，提升大模型决策透明度
- commit 81eed22: refactor: 使用Markdown库替代正则表达式生成HTML邮件，提升格式兼容性
- commit ce691db: feat: 综合分析邮件添加完整信息参考，包含大模型建议和ML预测结果
- commit b7f74fa: chore: 添加综合分析生成的数据文件
- commit e4f7bce: feat: 美化综合分析邮件样式，添加HTML格式支持
- commit cdb9218: chore: 更新综合分析生成的数据文件
- commit f56e5f5: docs: 更新IFLOW.md，添加综合分析系统功能说明
- commit e4e08c9: feat: 添加--no-email参数到hsi_email.py，并在综合分析脚本中使用
- commit 76fd611: feat: 综合分析输出增加详细信息，包括推荐理由、操作建议、价格指引和风险提示
- commit 00224ba: feat: 提取完整的操作信息，包括操作建议、价格指引和风险提示
- commit 9686799: fix: 提取完整的推荐理由，不再限制为50字符
- commit 2ea5d83: fix: 修复LLM和ML预测结果提取函数，支持实际文件格式
- commit 8c0bca7: chore: 添加综合分析生成的数据文件
- commit 39cab47: fix: 修复邮件发送功能，参考其他代码使用SMTP_SSL和重试机制
- commit 8d7097a: fix: 修复save_predictions_to_text函数中total_count未定义的错误
- commit f6b70ec: feat: 添加综合分析邮件发送功能
- commit 76dbf79: chore: 修改综合分析工作流执行时间为每周日上午9点
- commit 03f8e19: refactor: 修改ML预测工作流为综合分析工作流，调用run_comprehensive_analysis.sh
- commit 0b5bb3e: revert: 恢复1天、5天和20天的训练和预测功能
- commit 2f90fda: fix: 在生成20天预测前先训练模型
- commit 0e6135f: feat: 添加综合分析脚本，整合大模型建议和ML预测结果生成实质买卖建议
- commit 1c14d1a: feat: 添加大模型建议保存和ML预测结果保存功能
- commit 409e026: refactor: 简化train_and_predict_all.sh，只训练和预测20天模型
- commit 1792716: docs: 更新IFLOW.md，添加大模型建议保存和ML预测结果保存功能的文档说明
- commit 094a0a1: docs: 更新README.md中的ML模型优化经验章节，添加2026-02-14最新性能数据和特征工程总结
- commit 025a004: docs: 更新IFLOW.md中的ML模型优化信息
- commit 976df17: perf(ml): GBDT+LR一个月模型恢复0.15正则化配置
- commit c277f3e: perf(ml): 应用strong正则化配置到一个月模型
- commit 2705b61: feat(ml): 添加正则化策略验证脚本
- commit f7d0fac: docs: 添加ML模型优化经验章节到README
- commit cea7cc9: perf(ml): 针对一个月模型增强L1/L2正则化参数
- commit 60cd56c: feat(ml): 添加长期趋势特征优化一个月模型
- commit f809898: perf(ml): 分周期优化模型正则化参数
- commit 6179bfb: feat(ml): 添加高优先级和中优先级特征工程

---
最后更新：2026-02-16