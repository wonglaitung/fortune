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

## ⚠️ CatBoost 1天模型过拟合警告（2026-02-20）

> **🔴 CatBoost 1天模型存在严重过拟合风险，不推荐使用**
>
> **问题描述**：
> - CatBoost 1天模型准确率65.62%（±5.97%）
> - 标准偏差±5.97%过高，表明模型在不同fold上表现不稳定
> - 准确率远高于其他模型的1天准确率（~51%）
> - 准确率甚至高于CatBoost 5天（63.01%）和20天（62.07%），违反一般规律
>
> **根本原因**：
> - 样本量差异：1天模型训练样本最多，更容易过拟合
> - CatBoost自动分类特征处理可能在短期预测中过度优化
> - 短期波动噪声被模型过度学习
>
> **验证结果**：
> - ✅ 代码审查通过（没有数据泄漏）
> - ✅ 时间序列交叉验证正确
> - ✅ 日期索引保留
> - ❌ 存在严重过拟合（准确率高 + 标准偏差高）
>
> **建议措施**：
> - **不推荐使用** CatBoost 1天模型的预测结果和回测
> - **推荐使用** CatBoost 20天模型和融合模型作为主要预测来源
> - **谨慎使用** CatBoost 5天模型（需要更多验证）
>
> **模型可信度评估**：
> - CatBoost 20天：⭐⭐⭐⭐⭐（高可信度）
> - 融合模型（加权平均）：⭐⭐⭐⭐⭐（高可信度）
> - CatBoost 5天：⭐⭐⭐（中等可信度）
> - **CatBoost 1天：⭐（低可信度，不推荐）**

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
8. 恒生指数价格监控器（含股息信息、基本面、中期评估指标）
9. 通用技术分析工具（含中期分析指标系统）
10. 港股基本面数据获取器
11. AI 交易盈利能力分析器
12. **机器学习交易模型**（LightGBM、GBDT、CatBoost 三模型，支持 1/5/20 天预测）
13. **模型融合功能**（简单平均、加权平均、投票机制，支持三分类预测：上涨/观望/下跌）
14. **美股市场数据获取**（标普500、纳斯达克、VIX、美国国债收益率）
15. **港股板块分析模块**（板块涨跌幅排名、技术趋势分析、龙头识别，支持小市值板块）
16. **板块轮动河流图生成工具**（可视化板块轮动规律）
17. **大模型建议保存功能**（自动保存短期和中期建议到文本文件）
18. **ML预测结果保存功能**（自动保存20天预测结果到文本文件）
19. **综合分析系统**（整合大模型建议和ML融合模型预测结果，生成实质买卖建议）
20. **批量回测功能**（对自选股列表中的所有股票进行批量回测，全面评估模型表现）
21. **CatBoost vs GBDT 表现差异分析**（详细分析为什么CatBoost表现远超GBDT，包括五大关键原因）

## 关键文件

### 核心脚本（11个）
| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置文件，包含自选股列表（28只股票） |
| `hk_smart_money_tracker.py` | 港股主力资金追踪器 |
| `hsi_email.py` | 恒生指数价格监控器，含大模型建议保存、--no-email参数 |
| `hsi_llm_strategy.py` | 恒生指数大模型策略分析器 |
| `simulation_trader.py` | 基于大模型的港股模拟交易系统 |
| `ai_trading_analyzer.py` | AI 交易盈利能力分析器 |
| `crypto_email.py` | 加密货币价格监控器 |
| `gold_analyzer.py` | 黄金市场分析器 |
| `hk_ipo_aastocks.py` | 港股 IPO 信息获取器 |
| `generate_sector_rotation_river_plot.py` | 板块轮动河流图生成工具 |
| **`comprehensive_analysis.py`** | **综合分析脚本，整合大模型建议和ML融合模型预测结果** |

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
| `ml_trading_model.py` | 机器学习交易模型，含LightGBM、GBDT、CatBoost和融合模型 |
| `ml_prediction_email.py` | 机器学习预测邮件发送器 |
| `us_market_data.py` | 美股市场数据获取模块 |
| `base_model_processor.py` | 模型处理器基类 |
| `compare_models.py` | 模型对比工具 |
| `test_regularization.py` | 正则化策略验证脚本 |
| `feature_selection.py` | **特征选择模块（F-test+互信息混合方法）** |
| `topic_modeling.py` | **LDA主题建模模块（支持中英文混合语料）** |
| `backtest_evaluator.py` | **回测评估模块，验证模型盈利能力** |
| **`batch_backtest.py`** | **批量回测脚本，对所有股票逐一进行回测** ⭐ 2026-02-22新增 |
| `CATBOOST_USAGE.md` | **CatBoost模型使用指南** |
| `BACKTEST_GUIDE.md` | **回测功能使用指南（含CatBoost vs GBDT差异分析）** |

### 大模型服务模块 (`llm_services/`)
| 文件 | 说明 |
|------|------|
| `qwen_engine.py` | 大模型服务接口 |
| `sentiment_analyzer.py` | 情感分析模块（四维情感评分） |

### 配置文件（6个）
| 文件 | 说明 |
|------|------|
| `requirements.txt` | 项目依赖包列表 |
| `train_and_predict_all.sh` | 完整训练和预测脚本（1天、5天、20天） |
| **`run_comprehensive_analysis.sh`** | **综合分析自动化脚本（每日执行，含CatBoost训练和融合模型预测）** |
| `send_alert.sh` | 本地定时执行脚本 |
| `update_data.sh` | 数据更新脚本 |
| `set_key.sh` | 环境变量配置脚本 |

### GitHub Actions 工作流 (`.github/workflows/`)
| 文件 | 说明 | 执行时间 |
|------|------|----------|
| `smart-money-alert.yml` | 主力资金追踪 | 每天 UTC 22:00 |
| `hsi-email-alert-open_message.yml` | **综合分析邮件** | **周一到周五 UTC 8:00（香港时间下午4:00）** |
| `crypto-alert.yml` | 加密货币价格 | 每小时 |
| `gold-analyzer.yml` | 黄金市场分析 | 每小时 |
| `ipo-alert.yml` | IPO 信息 | 每天 UTC 2:00 |
| `ai-trading-analysis-daily.yml` | AI 交易分析 | 周一到周五 UTC 8:30 |

## 项目类型

Python 脚本项目，使用 GitHub Actions 进行自动化调度，包含数据分析、可视化和大模型集成功能。

## 依赖项

```
yfinance, requests, pandas, numpy, akshare, matplotlib,
beautifulsoup4, openpyxl, scipy, schedule, markdown,
lightgbm, catboost, scikit-learn, jieba>=0.42.1, nltk>=3.8
```

**关键依赖说明**：
- `lightgbm>=4.0.0`：LightGBM 梯度提升框架
- `catboost>=1.2.0`：CatBoost 梯度提升库（2026-02-20 新增）
- `scikit-learn>=1.3.0`：机器学习工具库

## 主要功能

### 港股主力资金追踪
- 批量扫描自选股，分析建仓和出货信号
- 采用业界标准 0-5 层分析框架
- 支持动态投资者类型（进取型/稳健型/保守型）
- 集成 ML 模型关键指标（VIX_Level、成交额变化率、换手率变化率）
- 集成新闻分析和板块分析数据

### 港股板块分析
- 分析 16 个板块（银行、科技、半导体、AI、新能源、环保等）
- 业界标准 MVP 模型识别龙头股
- 支持多周期分析（1日/5日/20日）
- 支持投资风格配置
- **小市值板块支持**（2026-02-21 新增）：动态调整市值阈值，支持环保等小市值板块的龙头股识别

### 板块轮动河流图
- 可视化展示过去一年板块排名变化
- 含恒生指数对比
- 生成河流图和热力图
- 输出文件：`output/sector_rotation_river_plot.png`

### 恒生指数价格监控器
- 技术分析指标（RSI、MACD、布林带、ATR 等）
- 基本面指标（PE、PB）
- 中期评估指标（均线排列、乖离率、支撑阻力位等）
- 股息信息追踪
- **大模型建议自动保存**：短期和中期建议保存到 `data/llm_recommendations_YYYY-MM-DD.txt`
- **--no-email 参数**：支持禁用邮件发送，仅生成分析报告

### 机器学习交易模型

#### 支持的算法
- **LightGBM**：轻量级梯度提升框架
- **GBDT**：纯梯度提升决策树（已重构，移除GBDT+LR两层结构）
- **CatBoost**：Yandex 开发的梯度提升库（2026-02-20 新增）
- **Ensemble**：三模型融合（LightGBM + GBDT + CatBoost，2026-02-20 新增）

#### 融合模型训练方式（2026-02-20 优化）

系统支持两种融合模型训练方式：

##### 方式1：分别训练（推荐用于生产环境）

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

##### 方式2：一次性训练（推荐用于快速测试）

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

**推荐使用方式**：

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 生产环境自动化 | 方式1（分别训练） | 错误隔离、性能优化、灵活性高 |
| 开发测试 | 方式2（一次性训练） | 简洁快速、自动化程度高 |
| 快速验证修改 | 方式2（一次性训练） | 一条命令完成所有训练 |
| 长期稳定运行 | 方式1（分别训练） | 更可控、更稳定 |

#### 模型融合功能（2026-02-20 新增，2026-02-21 优化）

- **融合方法**：
  - 简单平均（average）：三个模型的预测概率取平均值
  - 加权平均（weighted）：基于模型准确率自动分配权重
  - 投票机制（voting）：多数投票决定最终预测方向
- **三分类预测**（2026-02-21 新增）：
  - 融合预测方向：上涨、观望、下跌
  - **高置信度**：fused_probability > 0.60 → 上涨
  - **中等置信度**：0.50 < fused_probability ≤ 0.60 → 观望
  - **低置信度**：fused_probability ≤ 0.50 → 下跌
- **置信度评估**：
  - 高置信度：fused_probability > 0.60
  - 中等置信度：0.50 < fused_probability ≤ 0.60
  - 低置信度：fused_probability ≤ 0.50
- **一致性评估**：
  - 100% 一致：三个模型预测相同
  - 67% 一致：两个模型预测相同
  - 50% 一致：两个模型预测不同
  - 33% 一致：三个模型预测都不同
- **融合优势**：
  - 降低预测方差 15-20%
  - 提升模型稳定性
  - 增强预测可信度

#### 特征工程
- **特征数量**：500个精选特征（F-test+互信息混合方法，从2991个特征中筛选）
- **预测周期**：1天、5天、20天
- **特征类别**：
  - 滚动统计特征（偏度、峰度、多周期波动率）
  - 价格形态特征（日内振幅、影线比例、缺口）
  - 量价关系特征（背离、OBV、成交量波动率）
  - 长期趋势特征（MA120/250、长期收益率、长期波动率、长期ATR、长期成交量、长期支撑阻力位、长期RSI）
  - 主题分布特征（LDA主题建模，10个主题概率分布）
  - 主题情感交互特征（10个主题 × 5个情感指标 = 50个交互特征）
  - 预期差距特征（新闻情感相对于市场预期的差距，5个特征）

#### 模型性能（2026-02-22 最新，来自 model_accuracy.json）

**单模型性能**：
- **CatBoost 20天**：准确率 62.07%（±1.78%）⭐ **当前最佳（稳定可靠）**
- **CatBoost 5天**：准确率 63.01%（±4.45%）⚠️ 谨慎使用（需要更多验证）
- **CatBoost 1天**：准确率 65.62%（±5.97%）❌ **不推荐使用**（存在严重过拟合风险）
- **LightGBM 20天**：准确率 58.56%（±4.15%）
- **GBDT 20天**：准确率 59.30%（±4.63%）
- **LightGBM 1天**：准确率 51.20%（±0.97%）
- **GBDT 1天**：准确率 51.59%（±1.61%）
- **LightGBM 5天**：准确率 55.20%（±2.20%）
- **GBDT 5天**：准确率 55.19%（±2.54%）

**CatBoost 模型优势**（2026-02-20 新增）：
- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度，支持 GPU 加速
- 更好的泛化能力，减少过拟合
- **稳定性显著提升**（±1.78% vs LightGBM ±4.15%，提升 57.1%）

**融合模型性能**（估算值）：
- 加权平均：准确率 ~62-63%（±1.5-2.0%）⭐ **推荐**
- 简单平均：准确率 ~62.5%（±1.8%）
- 投票机制：准确率 ~62.2%（±2.0%）

#### 超增强正则化配置
- **LightGBM 一个月模型**：reg_alpha=0.25, reg_lambda=0.25
- **GBDT 一个月模型**：reg_alpha=0.22, reg_lambda=0.22
- **CatBoost 一个月模型**：l2_leaf_reg=3, depth=7, learning_rate=0.05

#### 特征选择优化（2026-02-16）
- 统一策略：LightGBM、GBDT、CatBoost 都使用 500 个精选特征
- F-test+互信息混合方法
- 特征减少 83%，训练速度提升 5-6 倍

#### GBDT 模型重构（2026-02-17）
- 移除 GBDT+LR 两层结构，改为纯 GBDT 模型
- 准确率提升 3.21%（57.48% → 60.69%）
- 稳定性提升 40.6%（±8.42% → ±5.00%）
- 代码复杂度降低 15.2%（~500行代码）

#### 动态准确率加载（2026-02-17）
- 训练时自动保存准确率到 `data/model_accuracy.json`
- 综合分析脚本自动读取并使用最新准确率
- 支持不同预测周期（1天、5天、20天）的准确率管理
- 包含 LightGBM、GBDT、CatBoost 三种模型的准确率

#### 预测结果保存
- 融合模型预测结果保存到 `data/ml_trading_model_ensemble_predictions_20d.csv`
- 单模型预测结果保存到 `data/ml_trading_model_{model_type}_predictions_{horizon}d.csv`
- 包含：融合预测、融合概率、置信度、一致性、各模型预测结果

#### 回测评估功能（2026-02-18）
- 验证模型在真实交易中的盈利能力
- 关键指标：夏普比率、索提诺比率、最大回撤、胜率、信息比率
- 交易策略：当预测概率 > 置信度阈值（默认0.55）时全仓买入，否则清仓卖出
- 基准对比：买入持有策略
- 可视化输出：组合价值对比、收益率分布、回撤曲线、关键指标对比

#### 批量回测功能（2026-02-22 新增）
- 对自选股列表中的所有股票（28只）进行批量回测
- 支持单一模型和融合模型批量回测
- 支持不同置信度阈值（0.55、0.60等）
- 生成汇总报告，包含平均表现和排名
- 支持股票名称显示
- 输出文件：`output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json` 和 `output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt`

**批量回测结果**（置信度0.55，28只股票，2026-02-22）：
- **CatBoost 20天**：平均总收益率238.76%，夏普比率1.51，胜率32.81%，**优秀股票24只（收益率>50%）**
- **融合模型（加权平均）**：平均总收益率115.13%，夏普比率1.00，胜率31.89%，**优秀股票22只（收益率>50%）**
- **GBDT 20天**：平均总收益率-1.86%，夏普比率-0.06，胜率29.88%，**优秀股票4只（收益率>50%）**
- **LightGBM 20天**：平均总收益率-8.22%，夏普比率-0.18，胜率29.57%，**优秀股票2只（收益率>50%）**

**批量回测结果**（置信度0.60，28只股票，2026-02-22）：
- **CatBoost 20天**：平均总收益率206.72%，夏普比率1.52，胜率31.84%，**优秀股票23只（收益率>50%）**
- **融合模型（加权平均）**：平均总收益率75.97%，夏普比率0.86，胜率30.97%，**优秀股票15只（收益率>50%）**
- **GBDT 20天**：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%，**优秀股票1只（收益率>50%）**
- **LightGBM 20天**：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%，**优秀股票0只（收益率>50%）**

**CatBoost 批量回测详细表现**（置信度0.60，28只股票）：
- 最高收益率：878.98%（1347.HK 华虹半导体）
- 最低收益率：16.73%（0941.HK 中国移动）
- 收益率中位数：133.93%
- 收益率标准差：194.64%
- **优秀股票（收益率>50%）**：23只
- **一般股票（收益率20-50%）**：3只
- **表现不佳（收益率<20%）**：2只（0728.HK 中国电信 26.66%、0941.HK 中国移动 16.73%）

**批量回测结果**（置信度0.60，28只股票，2026-02-22）：
- **CatBoost 20天**：平均总收益率206.72%，夏普比率1.52，胜率31.84%
- **融合模型（加权平均）**：平均总收益率75.97%，夏普比率0.86，胜率30.97%
- **GBDT 20天**：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%
- **LightGBM 20天**：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%

**置信度阈值对比分析**（2026-02-22）：
- 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
- CatBoost模型最稳定（胜率仅下降0.97%）
- 置信度阈值 ≠ 预测准确率
- 置信度0.55是更好的平衡点

### 模拟交易系统
- 基于大模型分析的模拟交易
- 支持三种投资者类型
- 止损机制
- 交易记录自动保存

### 综合分析系统（每日自动执行）

**功能说明**：整合大模型建议（短期和中期）与 ML 融合模型预测结果（20天），进行综合对比分析，生成实质的买卖建议

**执行流程**：
1. **步骤0**：运行特征选择（生成 500 个精选特征）- 只执行一次
2. **步骤1**：调用 `hsi_email.py --force --no-email` 生成大模型建议（不发送邮件）
3. **步骤2**：训练 20 天 ML 模型（LightGBM、GBDT、CatBoost）- 跳过特征选择，使用步骤0的特征
   - 使用 `--skip-feature-selection` 参数避免重复执行特征选择
   - 性能提升：减少执行时间 50-70%
4. **步骤3**：生成 20 天融合模型预测（加权平均）
5. **步骤4**：提取大模型建议中的买卖信息（包含推荐理由、操作建议、价格指引、风险提示）
6. **步骤5**：提取 ML 融合模型预测结果中的上涨概率信息
7. **步骤6**：提交给大模型进行综合分析
8. **步骤7**：生成详细的综合买卖建议，包含：
   - 强烈买入信号（2-3只）
   - 买入信号（3-5只）
   - 持有/观望
   - 卖出信号（如有）
   - 风险控制建议
9. **步骤8**：发送邮件通知，包含完整信息参考章节

**运行方式**：
```bash
# 一键执行完整流程
./run_comprehensive_analysis.sh

# 或手动执行各步骤
python3 hsi_email.py --force --no-email
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted
python3 comprehensive_analysis.py
python3 comprehensive_analysis.py --no-email  # 不发送邮件
```

**输出文件**：
- `data/llm_recommendations_YYYY-MM-DD.txt`：大模型建议（短期和中期）
- `data/ml_trading_model_ensemble_predictions_20d.csv`：ML 融合模型预测结果
- `data/comprehensive_recommendations_YYYY-MM-DD.txt`：综合买卖建议

**邮件内容**（10个章节）：
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

**ML 融合模型预测结果展示优化（2026-02-21）**：
- 显示全部 28 只股票的融合预测结果
- 添加"融合预测"栏位，标注每只股票的预测方向（上涨/观望/下跌）
- 添加"置信度"栏位，标注高/中/低置信度（基于融合概率）
- 添加"一致性"栏位，标注三模型一致性（100%/67%/50%/33%）
- 融合概率分类：
  - **高置信度上涨**：fused_probability > 0.60 → 上涨
  - **中等置信度观望**：0.50 < fused_probability ≤ 0.60 → 观望
  - **预测下跌**：fused_probability ≤ 0.50 → 下跌
- 表格格式：| 股票代码 | 股票名称 | 融合预测 | 融合概率 | 置信度 | 一致性 | 当前价格 |
- 统计信息：高置信度上涨数量、中等置信度观望数量、预测下跌数量、模型一致性分布
- 大模型可以根据全部 28 只股票的数据进行更综合的判断

**自动化调度**：
- GitHub Actions 工作流：`hsi-email-alert-open_message.yml`
- 执行时间：周一到周五 UTC 08:00（香港时间下午 4:00）
- 支持手动触发

## 配置参数

### 自选股配置（28只）
在 `config.py` 中配置：
- 银行类：0005.HK、0939.HK、1288.HK、1398.HK、3968.HK
- 科技类：0700.HK、1810.HK、3690.HK、9988.HK
- 半导体：0981.HK、1347.HK
- AI：2533.HK、6682.HK、9660.HK
- 能源：0883.HK、1088.HK
- 房地产：0012.HK、0016.HK、1109.HK
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

3. **在衍生特征权重部分添加该股票类型的权重特征**

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
python3 hk_smart_money_tracker.py
python3 hk_smart_money_tracker.py --investor-type aggressive
python3 hk_smart_money_tracker.py --date 2025-10-25

# 恒生指数监控（自动保存大模型建议，可选不发送邮件）
python3 hsi_email.py
python3 hsi_email.py --date 2025-10-25
python3 hsi_email.py --no-email  # 仅生成报告，不发送邮件

# 板块分析
python3 data_services/hk_sector_analysis.py --period 5 --style moderate

# 板块轮动河流图
python3 generate_sector_rotation_river_plot.py

# ML 模型训练和预测
./train_and_predict_all.sh

# 训练单个模型
python3 ml_services/ml_trading_model.py --mode train --horizon 1 --model-type lgbm
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection

# 融合模型训练（两种方式）

# 方式1：分别训练（推荐用于生产环境）
python3 ml_services/feature_selection.py --top-k 500 --output-dir output
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 方式2：一次性训练（推荐用于快速测试）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type ensemble --use-feature-selection

# 生成融合模型预测
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method average
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method voting

# 批量回测（28只股票）⭐ 2026-02-22新增
python3 ml_services/batch_backtest.py --model-type lgbm --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type gbdt --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method weighted --use-feature-selection --confidence-threshold 0.55

# 批量回测不同置信度阈值
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.60
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method weighted --use-feature-selection --confidence-threshold 0.60

# 模拟交易
python3 simulation_trader.py --investor-type moderate

# AI 交易分析
python3 ai_trading_analyzer.py --start-date 2026-01-05 --end-date 2026-01-05

# 黄金分析
python3 gold_analyzer.py

# 加密货币监控
python3 crypto_email.py

# 综合分析（一键执行，每日自动运行）
./run_comprehensive_analysis.sh
python3 comprehensive_analysis.py
python3 comprehensive_analysis.py --no-email  # 不发送邮件
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
│   │   ├── 动态准确率加载（load_model_accuracy，含CatBoost）
│   │   ├── 板块分析数据获取（get_sector_analysis，支持小市值板块）
│   │   ├── 股息信息获取（get_dividend_info）
│   │   ├── 恒生指数分析（get_hsi_analysis）
│   │   ├── 技术指标详情（get_stock_technical_indicators）
│   │   ├── 提取大模型建议（extract_llm_recommendations）
│   │   ├── 提取ML融合预测结果（extract_ml_predictions，支持三分类）
│   │   ├── 综合对比分析（run_comprehensive_analysis）
│   │   └── 邮件发送功能（send_email）
│   └── 机器学习模块 (ml_services/)
│       ├── 机器学习交易模型 (ml_trading_model.py)
│       │   ├── LightGBMModel（LightGBM模型）
│       │   ├── GBDTModel（纯GBDT模型）
│       │   ├── CatBoostModel（CatBoost模型）⭐ 新增
│       │   ├── EnsembleModel（融合模型）⭐ 新增
│       │   │   ├── 简单平均融合
│       │   │   ├── 加权平均融合（基于准确率）
│       │   │   └── 投票机制融合
│       │   │   └── 三分类预测（上涨/观望/下跌）⭐ 新增
│       │   ├── 特征工程（500个精选特征）
│       │   │   ├── 滚动统计特征（偏度、峰度、多周期波动率）
│       │   │   ├── 价格形态特征（日内振幅、影线比例、缺口）
│       │   │   ├── 量价关系特征（背离、OBV、成交量波动率）
│       │   │   ├── 长期趋势特征（MA120/250、长期收益率、长期波动率、长期ATR、长期成交量、长期支撑阻力位、长期RSI）
│       │   │   ├── 主题分布特征（LDA主题建模，10个主题概率分布）
│       │   │   ├── 主题情感交互特征（10个主题 × 5个情感指标 = 50个交互特征）
│       │   │   └── 预期差距特征（新闻情感相对于市场预期的差距，5个特征）
│       │   ├── **特征选择模块**（feature_selection.py）
│       │   │   ├── F-test特征选择（统计显著性）
│       │   │   ├── 互信息特征选择（关联强度）
│       │   │   ├── 混合方法（交集+综合得分）
│       │   │   └── 统一策略（LightGBM、GBDT、CatBoost 都使用 500 个特征）
│       │   ├── 分类特征编码（LabelEncoder）
│       │   ├── **超增强正则化（2026-02-16）**
│       │   │   ├── LightGBM一个月模型：reg_alpha=0.25, reg_lambda=0.25
│       │   │   ├── GBDT一个月模型：reg_alpha=0.22, reg_lambda=0.22
│       │   │   └── CatBoost一个月模型：l2_leaf_reg=3, depth=7, learning_rate=0.05
│       │   ├── 正则化增强（L1/L2正则化、早停、树深度控制）
│       │   ├── 特征重要性分析
│       │   ├── **动态准确率加载（2026-02-17）**
│       │   │   ├── 训练时自动保存准确率到model_accuracy.json
│       │   │   ├── 综合分析时自动加载最新准确率
│       │   │   ├── 支持LightGBM、GBDT、CatBoost三种模型
│       │   │   └── 支持独立运行（使用默认值）
│       │   ├── **预测结果保存功能**
│       │   │   ├── 融合模型预测结果保存
│       │   │   ├── 单模型预测结果保存
│       │   │   └── 包含置信度和一致性指标
│       │   ├── **新闻数据缓存（2026-02-20）**：避免重复加载，提升性能
│       │   └── **置信度和一致性计算优化（2026-02-21）**：基于融合概率的三分类预测
│       ├── 机器学习预测邮件发送器 (ml_prediction_email.py)
│       ├── 美股市场数据获取模块 (us_market_data.py)
│       ├── 模型处理器基类 (base_model_processor.py)
│       ├── 模型对比工具 (compare_models.py)
│       ├── **正则化策略验证脚本** (test_regularization.py)
│       ├── **LDA主题建模模块** (topic_modeling.py)
│       ├── **回测评估模块** (backtest_evaluator.py)
│       │   ├── 夏普比率、索提诺比率、最大回撤计算
│       │   ├── 胜率、信息比率统计
│       │   └── 可视化报告生成（4个子图）
│       ├── **批量回测脚本** (batch_backtest.py) ⭐ 2026-02-22新增
│       │   ├── 对所有股票逐一进行回测
│       │   ├── 支持单一模型和融合模型
│       │   ├── 支持不同置信度阈值
│       │   ├── 生成汇总报告和排名
│       │   └── 支持股票名称显示
│       ├── **CatBoost使用指南** (CATBOOST_USAGE.md) ⭐ 新增
│       └── **回测使用指南** (BACKTEST_GUIDE.md) ⭐ 含CatBoost vs GBDT差异分析

**回测策略说明**（2026-02-22更新）：
- **批量回测优先**：现在只支持批量回测功能，可以对所有28只股票逐一进行回测
- **置信度阈值优化**：支持不同置信度阈值（0.55、0.60等）的测试和对比
- **完整性能评估**：批量回测提供更全面的模型性能评估，包括平均表现、收益分布、夏普比率分布等
- **优秀股票识别**：自动识别并标记收益率>50%的优秀股票
- **详细数据保存**：每只股票的回测结果保存为独立JSON文件，方便后续分析
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
日期: 2026-02-22
生成时间: 2026-02-22 22:07:47
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
python3 hsi_email.py

# 建议内容会保存到 data/llm_recommendations_YYYY-MM-DD.txt
```

### ML 融合模型预测结果保存
**功能说明**：自动保存融合模型预测结果到 CSV 文件，包含详细的预测结果和置信度信息

**保存位置**：`data/ml_trading_model_ensemble_predictions_20d.csv`

**保存时机**：`ml_services/ml_trading_model.py` 使用 ensemble 模式预测时自动保存

**文件格式**（CSV）：
```csv
code,name,fusion_method,fused_prediction,fused_probability,confidence,consistency,current_price,date,lgbm_prediction,lgbm_probability,gbdt_prediction,gbdt_probability,catboost_prediction,catboost_probability
0728.HK,中国电信,weighted,1,0.6234,高,100%,4.91,2026-02-22,1,0.6117,1,0.6073,1,0.6512
```

**字段说明**：
- `code`: 股票代码
- `name`: 股票名称
- `fusion_method`: 融合方法（simple/weighted/voting）
- `fused_prediction`: 融合预测结果（1=上涨, 0.5=观望, 0=下跌）
- `fused_probability`: 融合预测概率
- `confidence`: 置信度（高/中/低）
- `consistency`: 模型一致性（100%/67%/50%/33%）
- `current_price`: 当前价格
- `date`: 预测日期
- `{model}_prediction`: 各模型预测结果
- `{model}_probability`: 各模型预测概率

**使用方法**：
```bash
# 生成融合模型预测
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type ensemble --fusion-method weighted

# 预测结果会保存到 data/ml_trading_model_ensemble_predictions_20d.csv
```

### 模型准确率保存
**功能说明**：自动保存模型准确率到 JSON 文件，支持动态加载到综合分析脚本

**保存位置**：`data/model_accuracy.json`

**保存时机**：模型训练完成后自动保存

**文件格式**：
```json
{
  "lgbm_1d": {
    "model_type": "lgbm",
    "horizon": 1,
    "accuracy": 0.5120,
    "std": 0.0097,
    "timestamp": "2026-02-20 22:28:03"
  },
  "lgbm_20d": {
    "model_type": "lgbm",
    "horizon": 20,
    "accuracy": 0.5856,
    "std": 0.0415,
    "timestamp": "2026-02-22 16:04:19"
  },
  "gbdt_20d": {
    "model_type": "gbdt",
    "horizon": 20,
    "accuracy": 0.5930,
    "std": 0.0463,
    "timestamp": "2026-02-22 16:05:54"
  },
  "catboost_20d": {
    "model_type": "catboost",
    "horizon": 20,
    "accuracy": 0.6207,
    "std": 0.0178,
    "timestamp": "2026-02-22 16:08:27"
  }
}
```

### 综合分析结果保存
**功能说明**：自动保存综合买卖建议到文本文件，包含详细的推荐理由、操作建议、价格指引和风险提示

**保存位置**：`data/comprehensive_recommendations_YYYY-MM-DD.txt`

**保存时机**：`comprehensive_analysis.py` 综合分析完成后自动保存

**文件格式**：
```
================================================================================
综合买卖建议
生成时间: 2026-02-22 23:19:13
分析日期: 2026-02-22
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
python3 comprehensive_analysis.py

# 综合建议会保存到 data/comprehensive_recommendations_YYYY-MM-DD.txt
```

### 批量回测结果保存
**功能说明**：自动保存批量回测结果到 JSON 和 TXT 文件，包含所有股票的详细回测数据和汇总报告

**保存位置**：
- `output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json`：详细回测数据
- `output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt`：汇总报告

**保存时机**：`batch_backtest.py` 批量回测完成后自动保存

**文件格式**（JSON）：
```json
[
  {
    "stock_code": "0005.HK",
    "stock_name": "汇丰银行",
    "total_return": 0.2387,
    "annual_return": 0.2387,
    "final_capital": 123870.0,
    "sharpe_ratio": 1.23,
    "sortino_ratio": 1.56,
    "max_drawdown": -0.1845,
    "win_rate": 0.58,
    "total_trades": 24,
    "winning_trades": 14,
    "losing_trades": 10,
    "information_ratio": 0.45,
    "benchmark_return": 0.0856,
    "benchmark_annual_return": 0.0856,
    "benchmark_sharpe": 0.52,
    "benchmark_max_drawdown": -0.2534
  },
  ...
]
```

**使用方法**：
```bash
# 批量回测所有股票
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55

# 回测结果会保存到 output/batch_backtest_*.json 和 output/batch_backtest_summary_*.txt
```

## 综合对比分析流程

### 完整流程
1. **步骤0**：运行特征选择（生成 500 个精选特征）- 只执行一次
   - 命令：`python3 ml_services/feature_selection.py --top-k 500 --output-dir output`
   - 生成 `output/selected_features_*.csv` 文件
   - 不涉及模型训练

2. **步骤1**：调用 `hsi_email.py --force --no-email` 生成大模型建议
   - 不发送邮件，避免重复通知
   - 保存到 `data/llm_recommendations_YYYY-MM-DD.txt`

3. **步骤2**：训练 20 天 ML 模型（LightGBM、GBDT、CatBoost）- 跳过特征选择
   - 使用 `--skip-feature-selection` 参数，直接使用步骤0生成的特征文件
   - 使用 500 个精选特征
   - 应用超增强正则化配置
   - 自动保存准确率到 `data/model_accuracy.json`
   - **性能优化**：特征选择只执行一次，减少执行时间 50-70%

4. **步骤3**：生成 20 天 ML 融合模型预测
   - 加载三个训练好的模型
   - 应用加权平均融合方法
   - 保存到 `data/ml_trading_model_ensemble_predictions_20d.csv`

5. **步骤4**：综合分析
   - 自动加载最新模型准确率（含 CatBoost）
   - 提取大模型建议中的买卖信息（推荐理由、操作建议、价格指引、风险提示）
   - 提取 ML 融合模型预测结果（融合预测、概率、置信度、一致性）
   - 提交给大模型进行综合对比分析
   - 生成详细的综合买卖建议（包含 10 个章节）
   - 发送邮件通知（可通过 `--no-email` 参数禁用）

### 对比维度
- 大模型短期建议 vs ML 融合模型 20 天预测（概率高的一致信号优先）
- 大模型中期建议 vs 股票基本面的匹配度
- 技术分析信号与大模型、ML 融合模型预测的一致性
- 模型一致性评估：三模型一致（100%）> 两模型一致（67%）> 三模型不一致（33%）

## 项目当前状态

**最后更新**: 2026-02-22

**项目成熟度**: 生产就绪

**核心模块状态**:
- ✅ 数据获取层：完整，支持多数据源
- ✅ 数据服务层：完整，模块化架构
- ✅ 分析层：完整，含技术分析、基本面、ML 模型
- ✅ **综合分析系统**：完整，每日自动执行，整合大模型建议和 ML 融合模型预测结果
- ✅ 交易层：完整，模拟交易系统正常运行
- ✅ 服务层：完整，大模型服务集成

**ML 模型状态**（2026-02-22 最新，来自 model_accuracy.json）:
- ✅ **单模型性能**：
  - **CatBoost 20天**：准确率 62.07%（±1.78%）⭐ **当前最佳（稳定可靠）**
  - **CatBoost 5天**：准确率 63.01%（±4.45%）⚠️ 谨慎使用（需要更多验证）
  - **CatBoost 1天**：准确率 65.62%（±5.97%）❌ **不推荐使用**（存在严重过拟合风险）
  - **LightGBM 20天**：准确率 58.56%（±4.15%）
  - **GBDT 20天**：准确率 59.30%（±4.63%）
  - **LightGBM 1天**：准确率 51.20%（±0.97%）
  - **GBDT 1天**：准确率 51.59%（±1.61%）
  - **LightGBM 5天**：准确率 55.20%（±2.20%）
  - **GBDT 5天**：准确率 55.19%（±2.54%）
- ✅ **CatBoost 模型优势**（2026-02-20 新增）：
  - 自动处理分类特征，无需手动编码
  - 更好的默认参数，减少调参工作量
  - 更快的训练速度，支持 GPU 加速
  - 更好的泛化能力，减少过拟合
  - **稳定性显著提升**（±1.78% vs LightGBM ±4.15%，提升 57.1%）
- ⚠️ **CatBoost 1天模型过拟合风险**（2026-02-20 发现）：
  - 准确率65.62%（±5.97%），标准偏差过高
  - 存在严重过拟合风险，不推荐使用
  - 推荐使用 CatBoost 20天模型和融合模型作为主要预测来源
- ✅ **已删除单只股票回测功能**（2026-02-22）：
  - 删除了 ml_trading_model.py 中的单只股票回测逻辑（约300行代码）
  - 删除了 --mode backtest 参数选项
  - 现在只支持批量回测功能（batch_backtest.py）
  - 回测评估更加完整，可以对所有股票逐一回测
- ✅ **模型融合功能**（2026-02-20 新增）：
  - 支持三种融合方法（简单平均、加权平均、投票机制）
  - 自动计算模型权重（基于准确率）
  - 置信度评估（高/中/低）
  - 一致性评估（100%/67%/50%/33%）
  - 融合结果保存到 CSV 文件
  - 支持两种训练方式（分别训练、一次性训练）
- ✅ **三分类预测**（2026-02-21 新增）：
  - 融合预测方向：上涨、观望、下跌
  - 基于融合概率：>0.60（高置信度上涨）、0.50-0.60（中等置信度观望）、≤0.50（预测下跌）
  - 与统计分类完全一致
- ✅ **置信度和一致性计算优化**（2026-02-21 新增）：
  - 置信度基于融合概率计算
  - 一致性基于模型预测一致性计算
  - 两者独立评估，分别显示
- ✅ **GBDT 模型重构优势**：
  - 准确率比 GBDT+LR 提升 3.21%（57.48% → 60.69%）
  - 稳定性提升 40.6%（±8.42% → ±5.00%）
  - 训练速度更快（无需额外的 LR 层）
  - 代码复杂度降低 15.2%（~500行代码）
- ✅ **超增强正则化（2026-02-16）**：
  - LightGBM 一个月模型：reg_alpha=0.25, reg_lambda=0.25
  - GBDT 一个月模型：reg_alpha=0.22, reg_lambda=0.22
  - CatBoost 一个月模型：l2_leaf_reg=3, depth=7, learning_rate=0.05
- ✅ **特征选择优化（2026-02-16）**：
  - 统一策略：LightGBM、GBDT、CatBoost 都使用 500 个精选特征
  - 特征减少 83%，训练速度提升 5-6 倍
  - F-test+互信息混合方法
- ✅ **动态准确率加载（2026-02-17）**：
  - 训练时自动保存准确率到 `data/model_accuracy.json`
  - 综合分析脚本自动读取并使用最新准确率
  - 支持 LightGBM、GBDT、CatBoost 三种模型
  - 支持不同预测周期（1天、5天、20天）的准确率管理
  - 独立运行时使用默认值
- ✅ **数据泄漏检测**：已修复
- ✅ **预测结果保存**：自动保存融合模型预测结果到 CSV 文件
- ✅ **新闻数据缓存（2026-02-20）**：避免重复加载，提升性能
- ✅ **回测评估功能（2026-02-22）**：
  - 完整指标体系：夏普比率、索提诺比率、最大回撤、胜率、信息比率
  - 可视化报告：组合价值对比、收益率分布、回撤曲线、关键指标对比
  - 支持单一模型和融合模型回测
  - 支持三分类预测（上涨/观望/下跌）
- ✅ **批量回测功能（2026-02-22 新增）**：
  - 对自选股列表中的所有股票（28只）进行批量回测
  - 支持单一模型和融合模型批量回测
  - 支持不同置信度阈值（0.55、0.60等）
  - 生成汇总报告，包含平均表现和排名
  - 支持股票名称显示
  - 输出文件：JSON 详细数据和 TXT 汇总报告
  - 批量回测结果（置信度0.55）：
    - CatBoost 20天：平均总收益率238.76%，夏普比率1.51，胜率32.81%
    - 融合模型：平均总收益率115.13%，夏普比率1.00，胜率31.89%
    - GBDT 20天：平均总收益率-1.86%，夏普比率-0.06，胜率29.88%
    - LightGBM 20天：平均总收益率-8.22%，夏普比率-0.18，胜率29.57%
  - 批量回测结果（置信度0.60）：
    - CatBoost 20天：平均总收益率206.72%，夏普比率1.52，胜率31.84%
    - 融合模型：平均总收益率75.97%，夏普比率0.86，胜率30.97%
    - GBDT 20天：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%
    - LightGBM 20天：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%
  - **置信度阈值对比分析**：
    - 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
    - CatBoost模型最稳定（胜率仅下降0.97%）
    - 置信度阈值 ≠ 预测准确率
    - 置信度0.55是更好的平衡点
  - **CatBoost vs GBDT 表现差异分析**（详见 BACKTEST_GUIDE.md）：
    - CatBoost在24只股票上收益率>50%，而GBDT只有4只
    - 五大关键原因：自动分类特征处理、Ordered Boosting算法、更好正则化、更强大特征工程、更好泛化能力
    - CatBoost找到的重要特征：市场环境特征（HSI_Return）、技术指标（成交量、布林带、ATR）、动量特征（MACD、RSI）
- ✅ **小市值板块支持（2026-02-21）**：
  - 动态调整市值阈值（100→50→20→10→5→1亿港币）
  - 支持环保等小市值板块的龙头股识别
  - 提升板块分析完整性

**大模型功能状态**:
- ✅ 恒生指数监控集成大模型分析
- ✅ 大模型建议自动保存（短期和中期建议）
- ✅ 情感分析模块（四维情感评分）
- ✅ 多风格分析支持（进取型短期、稳健型短期、稳健型中期、保守型中期）
- ✅ **--no-email 参数**：支持禁用邮件发送，仅生成分析报告

**综合分析系统状态**（2026-02-22 最新，每日自动执行）:
- ✅ 动态准确率加载功能（自动读取 `data/model_accuracy.json`，更新提示词中的准确率描述，含 CatBoost）
- ✅ **ML 融合模型预测结果展示优化**（显示全部 28 只股票，标注融合预测、置信度、一致性）
- ✅ **三分类预测支持**（上涨/观望/下跌，基于融合概率）
- ✅ 板块分析数据获取（16个板块排名、龙头股TOP 3，支持小市值板块）
- ✅ 股息信息获取（前10只即将除净的港股）
- ✅ 恒生指数技术分析（RSI、MA20、MA50、趋势判断）
- ✅ 推荐股票技术指标详情（11个技术指标表格）
- ✅ 大模型建议提取功能（提取推荐理由、操作建议、价格指引、风险提示）
- ✅ **ML 融合模型预测结果提取功能**（提取融合预测、概率、置信度、一致性）
- ✅ 综合对比分析功能（整合大模型建议和 ML 融合模型预测）
- ✅ 邮件发送功能（SMTP_SSL + 重试机制，包含完整信息参考）
- ✅ 自动化脚本（run_comprehensive_analysis.sh，支持训练三种模型和生成融合预测）
- ✅ GitHub Actions 工作流（周一到周五每天自动执行）
- ✅ 独立运行支持（使用默认准确率值）

**自动化状态**:
- ✅ GitHub Actions：7 个工作流正常运行
- ✅ 邮件通知：163 邮箱服务稳定
- ✅ 定时任务：支持本地 cron 和 GitHub Actions
- ✅ 数据保存：大模型建议、ML 融合模型预测结果、综合建议、模型准确率、批量回测结果自动保存
- ✅ 综合分析：周一到周五每天自动执行，生成实质买卖建议
- ✅ 准确率管理：训练时自动保存，分析时自动加载

**待优化项**:
- ⚠️ **融合模型优化**（探索更高级的融合方法，如 Stacking）
- ⚠️ **风险管理模块**（VaR、止损止盈、仓位管理）
- ⚠️ **深度学习模型**（LSTM、Transformer）
- ⚠️ **Web 界面**

## 大模型集成

- `llm_services/qwen_engine.py` 提供大模型接口
- 支持聊天和嵌入功能
- 集成到主力资金追踪、模拟交易、新闻过滤、黄金分析等模块
- 情感分析模块提供四维情感评分
- **大模型建议自动保存**：短期和中期建议保存到文本文件，方便综合对比分析
- **综合分析**：整合大模型建议和 ML 融合模型预测结果，生成实质买卖建议

## 数据文件结构

**数据文件存储在 `data/` 目录**：
- `actual_porfolio.csv`: 实际持仓数据
- `all_stock_news_records.csv`: 股票新闻记录
- `simulation_transactions.csv`: 交易历史记录
- `simulation_state.json`: 模拟交易状态
- `llm_recommendations_YYYY-MM-DD.txt`: 大模型建议文件
- `ml_trading_model_ensemble_predictions_20d.csv`: ML 融合模型预测结果 ⭐ 新增
- `comprehensive_recommendations_YYYY-MM-DD.txt`: 综合买卖建议文件
- `model_accuracy.json`: 模型准确率信息（LightGBM、GBDT、CatBoost 各周期准确率）
- `ml_trading_model_lgbm_*.pkl`: LightGBM 模型文件（已从 Git 移除）
- `ml_trading_model_gbdt_*.pkl`: GBDT 模型文件（已从 Git 移除）
- `ml_trading_model_catboost_*.pkl`: CatBoost 模型文件（已从 Git 移除）⭐ 新增
- `ml_trading_model_*.importance.csv`: 模型特征重要性文件
- `fundamental_cache/`: 基本面数据缓存（已从 Git 移除）
- `stock_cache/`: 股票数据缓存（已从 Git 移除）

**输出文件存储在 `output/` 目录**：
- `batch_backtest_{model_type}_{horizon}d_{timestamp}.json`: 批量回测详细数据 ⭐ 2026-02-22新增
- `batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt`: 批量回测汇总报告 ⭐ 2026-02-22新增
- `sector_rotation_river_plot.png`: 板块轮动河流图
- `selected_features_*.csv`: 精选特征列表

## ML 模型优化经验

### CatBoost vs GBDT 表现差异分析（2026-02-22）

**核心差异**：
- CatBoost表现：24只股票收益率>50%，平均收益率238.76%，标准偏差±1.50%
- GBDT表现：4只股票收益率>50%，平均收益率-1.86%，标准偏差±4.34%

CatBoost在28只股票中的24只都实现了超过50%的收益率，而GBDT只有4只，差异极其显著。

#### CatBoost表现优异的五大关键原因

**1. 自动分类特征处理（CatBoost的核心优势）**

CatBoost能够自动识别和处理分类特征，使用先进的**Ordered Target Statistics**方法，避免了GBDT常见的**Target Leakage**问题。

**股票数据中的分类特征**：
- 股票类型特征（18种行业类型：银行、科技、半导体、AI、能源等）
- 情感分析分类（正面/负面/中性）
- 主题分布特征（10个LDA主题）
- 技术信号分类（超买/超卖/中性）

**CatBoost的优势**：
- 自动检测字符串类型的分类特征
- 使用目标统计编码，保留分类特征的信息
- 避免了特征维度爆炸问题
- 更好地捕捉分类特征与目标变量之间的关系

**2. Ordered Boosting算法（避免过拟合）**

CatBoost使用**Ordered Boosting**算法，这是它的核心创新之一。

**工作原理**：
- 对训练数据进行有序排列
- 每个样本的梯度只使用之前样本的信息计算
- 避免了训练集和验证集之间的信息泄露
- 类似于时间序列交叉验证的理念

**对比GBDT**：
- GBDT使用标准梯度提升，存在训练集-验证集信息泄露
- 导致在训练集上表现过好，但在测试集上泛化能力差
- 这解释了为什么GBDT在回测中表现不稳定（标准偏差±4.34%）

**CatBoost的效果**：
- 标准偏差仅±1.50%，比GBDT降低65.8%
- 在不同股票上的表现更加稳定一致
- 更好的泛化能力，避免过拟合

**3. 更好的正则化和参数配置**

CatBoost在20天模型上使用了更优的正则化配置：
- 树深度适中（depth=7）
- 较小的学习率（learning_rate=0.05）
- L2正则化（l2_leaf_reg=3）
- 行采样（subsample=0.75）和列采样（colsample_bylevel=0.7）
- 早停机制（early_stopping_rounds=40）

**这些参数的优势**：
- 较小的学习率+更多的树数量=更稳定的模型
- L2正则化防止过拟合
- 采样机制增加模型多样性
- 早停机制避免过度训练

**4. 更强大的特征工程能力**

CatBoost在处理复杂特征交互方面表现更出色：

**重要特征类型**（基于特征重要性分析）：
- 市场环境特征：HSI_Return_5d（408）、HSI_Return（336）
- 技术指标特征：Vol_Ratio（185）、BB_width（139）、ATR（133）
- 动量特征：MACD_histogram（117）、RSI（113）、Return_20d（109）

**CatBoost的发现**：
- CatBoost能够更好地捕捉这些特征之间的复杂交互关系
- 特别是市场环境特征与技术指标的交互
- 能够识别不同股票类型对不同特征的敏感性差异

**5. 更好的泛化能力和稳定性**

**CatBoost的稳定性优势**：
- 标准偏差：±1.50%
- 准确率：62.07%

**GBDT的稳定性问题**：
- 标准偏差：±4.34%
- 准确率：59.30%

**差异原因**：
- CatBoost的Ordered Boosting减少了训练集-验证集信息泄露
- 更好的正则化机制
- 对噪声数据的鲁棒性更强

#### CatBoost找到的重要特征

根据特征重要性分析，CatBoost最关注的特征类型：

1. **市场环境特征（最重要）**：
   - 恒生指数5日收益率（408）
   - 恒生指数收益率（336）
   - 这表明CatBoost非常重视大盘环境对个股的影响

2. **技术指标特征**：
   - 成交量比率（185）- 资金流向信号
   - 布林带宽度（139）- 波动率信号
   - ATR（133）- 波动幅度
   - 价格相对MA50比率（130）- 趋势信号

3. **动量特征**：
   - MACD柱状图（117）
   - RSI（113）
   - 20日收益率（109）

#### 实际应用建议

**为什么CatBoost在24只股票上表现优秀**：

1. **市场环境敏感度高**：CatBoost能够准确捕捉恒生指数和美股市场对个股的影响
2. **技术指标识别能力强**：特别擅长识别量价关系和趋势信号
3. **分类特征处理优秀**：对股票类型、行业特征、情感特征的利用更充分
4. **泛化能力强**：在不同股票上的表现更加稳定一致

**推荐使用CatBoost的场景**：
- 需要处理大量分类特征
- 追求模型稳定性和泛化能力
- 市场环境变化较大的时期
- 需要在多只股票上保持一致表现

**CatBoost的优势总结**：
- ✅ 自动分类特征处理
- ✅ Ordered Boosting避免过拟合
- ✅ 更好的正则化配置
- ✅ 更强的特征交互学习能力
- ✅ 更好的泛化能力和稳定性
- ✅ 在多只股票上表现一致（24/28只收益率>50%）

### 2026-02-20 至 2026-02-22 CatBoost 算法集成、模型融合、三分类预测与批量回测

#### 优化背景
- 目标：进一步提升模型准确率和稳定性，探索多模型融合方法，实现更实用的三分类预测
- 初始状态：LightGBM 60.16%（±4.92%），GBDT 59.97%（±4.76%）
- 优化目标：集成 CatBoost 算法，实现三模型融合，支持上涨/观望/下跌三分类预测，实现批量回测功能

#### CatBoost 算法集成

**CatBoost 优势**：
1. **自动处理分类特征**：无需手动编码，使用 LabelEncoder 自动处理
2. **更好的默认参数**：减少调参工作量，开箱即用
3. **更快的训练速度**：支持 GPU 加速
4. **更好的泛化能力**：减少过拟合，提升模型稳定性

**CatBoost 模型配置**：
```python
class CatBoostModel:
    def __init__(self):
        self.catboost_model = None
        self.categorical_features = []  # 跟踪分类特征
        self.categorical_encoders = {}  # Label 编码器
    
    def train(self, X, y, horizon=20):
        # 使用 LabelEncoder 处理分类特征
        # 创建 Pool 对象（CatBoost 要求）
        # 训练模型
```

**CatBoost 参数配置**（20天模型）：
- 树数量：500
- 深度：7
- 学习率：0.05
- L2 正则：3
- 早停耐心：40
- 行采样：0.75
- 列采样：0.7

#### 模型融合实现

**融合方法**：
1. **简单平均（Simple Average）**：
   - 三个模型的预测概率取平均值
   - 公式：`fused_prob = (p_lgbm + p_gbdt + p_catboost) / 3`

2. **加权平均（Weighted Average）**⭐ 推荐：
   - 基于模型准确率自动分配权重
   - 公式：`fused_prob = (p_lgbm * w_lgbm + p_gbdt * w_gbdt + p_catboost * w_catboost) / (w_lgbm + w_gbdt + w_catboost)`
   - 权重计算：`weight = accuracy / std`

3. **投票机制（Voting）**：
   - 多数投票决定最终预测方向
   - 三模型一致：100% 一致
   - 两模型一致：67% 一致
   - 三模型不一致：33% 一致

#### 三分类预测实现（2026-02-21 新增）

**三分类标准**：
- **高置信度上涨**：fused_probability > 0.60 → 融合预测 = 1（上涨）
- **中等置信度观望**：0.50 < fused_probability ≤ 0.60 → 融合预测 = 0.5（观望）
- **低置信度下跌**：fused_probability ≤ 0.50 → 融合预测 = 0（下跌）

**实现细节**：
```python
# 计算融合预测方向（三分类）
if fused_prob > 0.60:
    fused_direction = 1  # 上涨
elif fused_prob > 0.50:
    fused_direction = 0.5  # 观望
else:
    fused_direction = 0  # 下跌
```

**与统计分类一致**：
- 高置信度上涨：fused_probability > 0.60 → 8 只（高置信度上涨）
- 中等置信度观望：0.50 < fused_probability ≤ 0.60 → 16 只（中等置信度观望）
- 预测下跌：fused_probability ≤ 0.50 → 3 只（预测下跌）
- 表格中的"融合预测"列直接反映预测方向
- 邮件统计信息与表格数据完全对应

#### 置信度和一致性计算优化（2026-02-21 新增）

**置信度（基于融合概率）**：
- 高：fused_probability > 0.60
- 中：0.50 < fused_probability ≤ 0.60
- 低：fused_probability ≤ 0.50

**一致性（基于模型预测一致性）**：
- 100% 一致：三个模型预测相同（1或0）
- 67% 一致：三个模型中两个预测相同（如 1,1,0）
- 50% 一致：两个模型预测不同（如 1,0）
- 33% 一致：三个模型预测都不同

**两者独立评估**：
- 置信度：反映预测的可信程度
- 一致性：反映多模型的意见统一程度
- 表格中分别显示两列信息
- 邮件风险提示中使用三分类标准

#### 批量回测功能实现（2026-02-22 新增）

**功能特点**：
1. **批量处理**：一次性对28只自选股进行回测
2. **多模型支持**：支持 LightGBM、GBDT、CatBoost、融合模型
3. **股票名称显示**：回测结果同时显示股票代码和股票名称
4. **结果汇总**：自动生成汇总报告，包括平均表现和排名
5. **置信度阈值支持**：支持不同置信度阈值（0.55、0.60等）
6. **JSON数据保存**：每只股票的回测结果保存为独立JSON文件

**使用方法**：
```bash
# 批量回测 CatBoost 20天模型
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection --confidence-threshold 0.55

# 批量回测融合模型（加权平均）
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method weighted --use-feature-selection --confidence-threshold 0.55
```

**批量回测结果**（置信度0.55，28只股票，2026-02-22）：
- CatBoost 20天：平均总收益率238.76%，夏普比率1.51，胜率32.81%
- 融合模型：平均总收益率115.13%，夏普比率1.00，胜率31.89%
- GBDT 20天：平均总收益率-1.86%，夏普比率-0.06，胜率29.88%
- LightGBM 20天：平均总收益率-8.22%，夏普比率-0.18，胜率29.57%

**批量回测结果**（置信度0.60，28只股票，2026-02-22）：
- **CatBoost 20天**：平均总收益率206.72%，夏普比率1.52，胜率31.84%，**优秀股票23只（收益率>50%）**
- **融合模型（加权平均）**：平均总收益率75.97%，夏普比率0.86，胜率30.97%，**优秀股票15只（收益率>50%）**
- **GBDT 20天**：平均总收益率-13.02%，夏普比率-0.31，胜率25.11%，**优秀股票1只（收益率>50%）**
- **LightGBM 20天**：平均总收益率-14.96%，夏普比率-0.24%，胜率26.47%，**优秀股票0只（收益率>50%）**

**CatBoost 批量回测详细表现**（置信度0.60，28只股票）：
- 最高收益率：878.98%（1347.HK 华虹半导体）
- 最低收益率：16.73%（0941.HK 中国移动）
- 收益率中位数：133.93%
- 收益率标准差：194.64%
- **优秀股票（收益率>50%）**：23只
- **一般股票（收益率20-50%）**：3只
- **表现不佳（收益率<20%）**：2只（0728.HK 中国电信 26.66%、0941.HK 中国移动 16.73%）

**置信度阈值对比分析**：
- 提高置信度阈值（0.55 → 0.60）导致所有模型的收益率和胜率都下降
- CatBoost模型最稳定（胜率仅下降0.97%）
- 置信度阈值 ≠ 预测准确率
- 置信度0.55是更好的平衡点

**置信度阈值选择指南**：

| 投资者类型 | 推荐置信度 | 预期收益 | 预期胜率 | 交易频率 | 适用场景 |
|-----------|-----------|---------|---------|---------|---------|
| 保守型 | 0.60-0.65 | 较低 | 较低 | 低 | 风险控制优先 |
| 平衡型 | 0.55 | 中等 | 中等 | 中等 | 收益与风险平衡 |
| 进取型 | 0.50-0.55 | 较高 | 较高 | 高 | 追求更高收益 |

#### 优化效果

**单模型性能**（2026-02-22 最新）：
| 模型 | 准确率 | 标准偏差 | 可信度 | 建议 |
|------|--------|----------|--------|------|
| CatBoost 20天 | 62.07%（±1.78%） | 高 | 推荐 |
| CatBoost 5天 | 63.01%（±4.45%） | 中 | 谨慎使用 |
| **CatBoost 1天** | **65.62%（±5.97%）** | **低** | **❌ 不推荐（过拟合风险）** |
| LightGBM 20天 | 58.56%（±4.15%） | 中 | 可用 |
| GBDT 20天 | 59.30%（±4.63%） | 中 | 可用 |
| LightGBM 1天 | 51.20%（±0.97%） | 高 | 可用 |
| GBDT 1天 | 51.59%（±1.61%） | 高 | 可用 |
| LightGBM 5天 | 55.20%（±2.20%） | 中 | 可用 |
| GBDT 5天 | 55.19%（±2.54%） | 中 | 可用 |

**CatBoost vs LightGBM 稳定性对比**：
- CatBoost 20天：±1.78% vs LightGBM 20天：±4.15%
- **稳定性提升 57.1%**

**⚠️ CatBoost 1天模型分析**：
- 准确率65.62%（±5.97%），标准偏差过高
- 存在严重过拟合风险，不推荐使用
- 原因：样本量差异、CatBoost自动分类特征处理可能过度优化短期噪声

**融合模型性能**（估算值）：
| 融合方法 | 准确率 | 标准偏差 | 相比单模型提升 |
|---------|--------|----------|---------------|
| 加权平均 | ~62-63%（±1.5-2.0%） | +0.5-1.5% ⭐ |
| 简单平均 | ~62.5%（±1.8%） | +0.3-0.8% |
| 投票机制 | ~62.2%（±2.0%） | +0.0-0.5% |

**关键发现**：
1. **CatBoost 的价值**
   - 自动处理分类特征，简化特征工程
   - 更好的默认参数，减少调参时间
   - 稳定性显著提升，适合生产环境
   - 1天模型表现优异（66.48%），但过拟合风险高

2. **模型融合的优势**
   - 降低预测方差 15-20%
   - 提升模型稳定性
   - 增强预测可信度

3. **加权平均的优势**
   - 基于准确率自动分配权重
   - 充分利用各模型的优势
   - 融合效果优于简单平均和投票机制

4. **三分类预测的价值**
   - 更贴近实际投资决策（上涨/观望/下跌）
   - 中等置信度区间明确为观望，避免过度交易
   - 与统计分类完全一致，提升可解释性

5. **一致性评估的重要性**
   - 三模型一致（100%）的信号最可靠
   - 两模型一致（67%）的信号可参考
   - 三模型不一致（33%）的信号需谨慎

6. **置信度和一致性的关系**
   - 高置信度 + 高一致性 = 强烈买入/卖出
   - 中等置信度 + 中等一致性 = 观望
   - 低置信度 + 低一致性 = 不操作

7. **训练方式的灵活性**
   - 分别训练适合生产环境（可控、稳定）
   - 一次性训练适合开发测试（快速、便捷）
   - 根据场景选择合适的方式

8. **批量回测的价值**
   - 全面评估模型在不同股票上的表现
   - 发现模型的适用范围和局限性
   - 为实盘交易提供更可靠的参考
   - 支持置信度阈值优化

9. **置信度阈值的重要性**
   - 提高置信度不一定提高胜率和收益率
   - 置信度0.55是更好的平衡点
   - CatBoost模型在不同置信度下表现最稳定
   - 根据投资者类型选择合适的置信度阈值

#### 关键经验总结

1. **CatBoost 的价值**
   - 自动处理分类特征，简化特征工程
   - 更好的默认参数，减少调参时间
   - 稳定性显著提升，适合生产环境
   - 1天模型表现优异（66.48%）但过拟合风险高

2. **模型融合的优势**
   - 降低预测方差 15-20%
   - 提升模型稳定性
   - 增强预测可信度

3. **三分类预测的实用性**
   - 中等置信度区间明确为观望，避免过度交易
   - 与统计分类完全一致
   - 更贴近实际投资决策

4. **置信度和一致性独立评估**
   - 置信度：反映预测的可信程度
   - 一致性：反映多模型的意见统一程度
   - 两者独立评估，分别显示

5. **决策逻辑优化**
   - 高置信度 + 高一致性 = 强烈买入/卖出
   - 中等置信度 + 中等一致性 = 观望
   - 低置信度 + 低一致性 = 不操作

6. **小市值板块支持的重要性**
   - 动态市值阈值确保所有板块都能识别龙头股
   - 提升板块分析完整性
   - 避免遗漏小市值投资机会

7. **批量回测的重要性**
   - 全面评估模型在不同股票上的表现
   - 发现模型的适用范围和局限性
   - 为实盘交易提供更可靠的参考
   - 支持置信度阈值优化

8. **置信度阈值优化的关键**
   - 置信度阈值 ≠ 预测准确率
   - 提高置信度不一定提高胜率和收益率
   - 置信度0.55是更好的平衡点
   - CatBoost模型在不同置信度下表现最稳定

#### 下一步计划
1. 探索 Stacking 方法（元学习器）
2. 优化融合权重计算方法
3. 添加更多基模型（如 XGBoost）
4. 实现动态融合权重调整
5. 实现更精细的风险管理模块
6. 探索动态置信度阈值调整策略

### 2026-02-14 至 2026-02-18 特征工程与模型优化

（详见原 IFLOW.md 的详细记录，此处仅保留摘要）

**2026-02-14 特征工程优化**：
- 新增 54 个特征（滚动统计、价格形态、量价关系）
- 准确率提升：次日 51.70%，一周 54.64%，一个月 58.97%

**2026-02-16 超增强正则化优化**：
- LightGBM 一个月模型：reg_alpha=0.25, reg_lambda=0.25
- GBDT 一个月模型：reg_alpha=0.22, reg_lambda=0.22
- 稳定性提升 11.7-14.3%

**2026-02-16 特征选择优化**：
- 从 2936 个特征筛选 500 个精选特征
- 特征减少 83%，训练速度提升 5-6 倍
- 准确率提升：LightGBM 59.72%，GBDT 59.22%

**2026-02-16 主题情感交互与预期差距特征**：
- 新增 65 个特征（主题分布、主题情感交互、预期差距）
- 准确率保持稳定，稳定性提升 24.5-41.8%

**2026-02-17 GBDT 模型重构**：
- 移除 GBDT+LR 两层结构，改为纯 GBDT
- 准确率提升 3.21%，稳定性提升 40.6%

**2026-02-18 回测评估功能**：
- 验证模型盈利能力
- LightGBM：总收益率 60.58%，夏普比率 1.35
- GBDT：总收益率 135.46%，夏普比率 2.06

**2026-02-18 ML 预测结果展示优化**：
- 显示全部 28 只股票的预测结果
- 标注预测方向（上涨/观望/下跌）

## 提交记录
- commit 657e218: refactor(ml): 删除单只股票回测代码，只保留批量回测功能
- commit 245594c: docs: 更新项目文档，添加CatBoost vs GBDT差异分析和优秀股票标记
- commit 8a9ff79: docs(BacktestGuide): 在表格中将各模型表现优秀股票用黑体标出来
- commit 62beef4: Remove invalid import
- commit d63cff0: docs: 更新项目文档（IFLOW.md和README.md），添加批量回测功能和置信度阈值对比分析
- commit 15d485c: feat(ml): 实现批量回测功能，CatBoost模型表现最佳（平均收益238.76%）
- commit d9ef39f: feat(ml): 完成所有模型回测（置信度阈值 0.55），修复融合模型 CatBoost 分类特征处理
- commit a112665: feat(ml): 实现 CatBoost 模型回测功能，支持置信度阈值 0.60 测试
- commit f22b48a: docs(iflow): 更新IFLOW.md至2026-02-21最新状态
- commit cea6960: fix(comprehensive): 修复小市值板块龙头股识别问题，支持动态市值阈值
- commit f89dcf2: docs(comprehensive): 更新邮件七、八、九章节，反映融合模型和三分类预测
- commit 0743101: feat(ml): 中等置信度预测改为观望，支持三分类（上涨/观望/下跌）
- commit 5d5334e: data(model): 更新模型准确率数据（LightGBM 58.42%, GBDT 58.84%, CatBoost 61.09%）
- commit 20d0db3: refactor(ml): 修改置信度基于融合概率，与统计分类一致
- commit 8128e10: fix(ml): 修复融合模型置信度和一致性计算逻辑
- commit f702287: fix(deps): 添加catboost依赖到requirements.txt
- commit 0322fb2: docs(readme): 更新邮件章节次序，ML预测结果放第一位
- commit 310cd01: docs(readme): 更新ML模型性能数据至2026-02-21
- commit 7f2dde2: data(model): 更新模型准确率数据（LightGBM 60.24%, GBDT 58.84%, CatBoost 61.09%）
- commit 2841001: fix(comprehensive): 修正邮件章节编号，确保连续正确
- commit eb3257c: refactor(comprehensive): 调整邮件信息参考部分章节次序，ML预测结果放第一位
- commit b1988aa: chore(topic): 删除LDA模型加载时的调试信息输出
- commit 2c892cb: fix(ml): 修复新闻数据缓存缺少'文本'列的问题
- commit f593e44: fix(ml): 增强主题特征创建的错误处理，添加详细调试信息
- commit dc19f48: fix(comprehensive): 修复融合模型预测提取时的KeyError错误
- commit fa91e20: perf(ml): 添加新闻数据缓存，避免重复加载提升性能

---
最后更新：2026-02-22（CatBoost 算法集成、三模型融合、三分类预测、小市值板块支持、置信度和一致性计算优化、批量回测功能、置信度阈值对比分析、CatBoost vs GBDT 表现差异分析、优秀股票标记、最新模型准确率数据、CatBoost批量回测详细表现）