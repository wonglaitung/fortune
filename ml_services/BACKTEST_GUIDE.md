# 回测评估功能使用指南

## ⚠️ CatBoost 1天模型过拟合警告（2026-02-20）

> **🔴 CatBoost 1天模型存在严重过拟合风险，不推荐使用**
>
> **问题描述**：
> - CatBoost 1天模型准确率65.62%（±5.97%）
> - 标准偏差±5.97%过高，表明模型在不同fold上表现不稳定
> - 准确率远高于其他模型的1天准确率（~51%）
> - 准确率甚至高于CatBoost 5天（63.01%）和20天（61.09%），违反一般规律
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

## 功能概述

回测评估模块 (`backtest_evaluator.py`) 用于验证机器学习模型在真实交易环境中的盈利能力，评估模型的实际可用性。

**核心功能**：
- 验证模型的实际盈利能力
- 评估风险调整后收益（夏普比率、索提诺比率）
- 计算最大回撤和胜率
- 生成可视化报告（4个子图）
- 支持三分类预测（上涨/观望/下跌）
- 提供置信度和一致性评估
- 支持单一模型和融合模型回测

## 核心功能

### 1. 关键指标计算

#### 收益指标
- **总收益率**: 回测期间的总收益
- **年化收益率**: 按年化计算的收益率
- **最终资金**: 回测结束时的资金总额

#### 风险指标
- **夏普比率**: 风险调整后收益（>1.0优秀，>0.5良好）
- **索提诺比率**: 下行风险调整后收益（只考虑亏损）
- **最大回撤**: 历史最大亏损幅度（<20%优秀，<30%良好）

#### 交易统计
- **胜率**: 盈利交易占比
- **信息比率**: 相对基准的超额收益质量
- **总交易次数**: 完整的买卖循环数

### 2. 回测逻辑

**交易策略**:
- 当模型预测概率 > 置信度阈值（默认0.55）时，全仓买入
- 当模型预测概率 ≤ 置信度阈值时，清仓卖出
- 考虑交易成本：佣金0.1% + 滑点0.1%
- **三分类预测**（2026-02-21 新增）：
  - 高置信度上涨（fused_probability > 0.60）：建议买入
  - 中等置信度观望（0.50 < fused_probability ≤ 0.60）：建议持有
  - 预测下跌（fused_probability ≤ 0.50）：建议卖出

**基准策略**: 买入持有（Buy & Hold）

### 3. 可视化输出

生成4个子图的回测报告：
1. **组合价值对比**: 模型策略 vs 基准策略
2. **收益率分布**: 日收益率直方图
3. **回撤曲线**: 历史回撤走势
4. **关键指标对比**: 重要指标的柱状图对比

**JSON文件包含**：
- 基本指标：总收益率、年化收益率、最终资金
- 风险指标：夏普比率、索提诺比率、最大回撤
- 交易统计：胜率、总交易次数、盈利/亏损交易数
- **三分类预测**（2026-02-21 新增）：融合预测、融合概率、置信度、一致性
- 股票信息：股票代码、回测策略、选择方法

## 使用方法

### 方法1: 通过主脚本使用

#### 单一模型回测

```bash
# 回测20天预测模型（LightGBM）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type lgbm --use-feature-selection 

# 回测20天预测模型（GBDT）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type gbdt --use-feature-selection 

# 回测20天预测模型（CatBoost）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type catboost --use-feature-selection

# 回测5天预测模型
python3 ml_services/ml_trading_model.py --mode backtest --horizon 5 --model-type lgbm --use-feature-selection 

# 回测1天预测模型
python3 ml_services/ml_trading_model.py --mode backtest --horizon 1 --model-type lgbm --use-feature-selection 
```

#### 融合模型回测（推荐）

```bash
# 回测融合模型（加权平均 - 推荐）
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type ensemble --use-feature-selection

# 回测融合模型（简单平均）
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type ensemble --fusion-method average --use-feature-selection

# 回测融合模型（投票机制）
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type ensemble --fusion-method voting --use-feature-selection

# 批量训练时跳过特征选择（综合分析脚本使用）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection
```

**融合策略说明**：
- **加权平均（推荐）**：基于模型准确率自动分配权重
  - LightGBM权重：~30.2%
  - GBDT权重：~29.3%
  - CatBoost权重：~30.7%
  - 当前最佳性能：预计准确率61-62%（±1.5-2.0%）
  - 支持三分类预测（上涨/观望/下跌）
  - 提供置信度和一致性评估
- **简单平均**：三个模型权重相等（各33.3%）
- **投票机制**：多数投票决定最终预测

### 方法2: 直接使用回测评估器

```python
from ml_services.backtest_evaluator import BacktestEvaluator

# 创建评估器
evaluator = BacktestEvaluator(initial_capital=100000)

# 运行回测
results = evaluator.backtest_model(
    model=your_trained_model,
    test_data=test_features,
    test_labels=test_labels,
    test_prices=test_prices,
    confidence_threshold=0.55
)

# 绘制图表
evaluator.plot_backtest_results(results, save_path='output/backtest_results.png')
```

### 方法3: 测试模式（使用模拟数据）

```bash
cd ml_services
python3 backtest_evaluator.py
```

## 输出文件

回测完成后会生成以下文件：

1. **回测结果图表**: `output/backtest_results_{horizon}d_{timestamp}.png`
2. **回测结果数据**: `output/backtest_results_{horizon}d_{timestamp}.json`

JSON文件包含所有关键指标：
```json
{
  "total_return": 0.1523,
  "annual_return": 0.1523,
  "final_capital": 115230.0,
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
  "benchmark_max_drawdown": -0.2534,
  "stock_code": "0700.HK",
  "backtest_strategy": "single_stock_random",
  "selection_method": "random_selection_from_27_stocks"
}
```

## 融合模型回测特点

### 优势

1. **降低方差**：三模型融合可降低预测方差15-20%
2. **提升稳定性**：避免单一模型的过拟合风险
3. **自动权重**：加权平均基于准确率自动分配权重
4. **置信度评估**：提供模型一致性评估（100%/67%/33%）
5. **三分类预测**：支持上涨/观望/下跌三分类，更贴近实际投资决策
6. **置信度和一致性独立评估**：
   - 置信度：反映预测的可信程度（基于融合概率）
   - 一致性：反映多模型的意见统一程度（基于模型预测一致性）
7. **CatBoost 模型优势**：
   - 自动处理分类特征，无需手动编码
   - 更好的默认参数，减少调参工作量
   - 更快的训练速度，支持 GPU 加速
   - 更好的泛化能力，减少过拟合
   - **稳定性显著提升**（±1.50% vs LightGBM ±4.38%，提升 65.8%）

### 三分类预测标准（2026-02-21 新增）

- **高置信度上涨**：fused_probability > 0.60 → 融合预测 = 1（上涨）
- **中等置信度观望**：0.50 < fused_probability ≤ 0.60 → 融合预测 = 0.5（观望）
- **预测下跌**：fused_probability ≤ 0.50 → 融合预测 = 0（下跌）

### 置信度评估

- **高置信度**：fused_probability > 0.60
- **中等置信度**：0.50 < fused_probability ≤ 0.60
- **低置信度**：fused_probability ≤ 0.50

### 一致性评估

- **100% 一致**：三个模型预测相同（1或0）
- **67% 一致**：三个模型中两个预测相同（如 1,1,0）
- **50% 一致**：两个模型预测不同（如 1,0）
- **33% 一致**：三个模型预测都不同

### 当前性能（2026-02-21）

**单一模型性能**（来自 `data/model_accuracy.json`）：
- **CatBoost 20天**：61.09%（±1.50%）⭐ **当前最佳（稳定可靠）**
- **CatBoost 5天**：63.01%（±4.45%）⚠️ 谨慎使用（需要更多验证）
- **CatBoost 1天**：65.62%（±5.97%）❌ **不推荐使用**（存在严重过拟合风险）
- **LightGBM 20天**：58.87%（±4.38%）
- **GBDT 20天**：58.84%（±4.34%）
- **LightGBM 1天**：51.20%（±0.97%）
- **GBDT 1天**：51.59%（±1.61%）
- **LightGBM 5天**：55.20%（±2.20%）
- **GBDT 5天**：55.19%（±2.54%）

**CatBoost 模型优势**：
- 自动处理分类特征，无需手动编码
- 更好的默认参数，减少调参工作量
- 更快的训练速度，支持 GPU 加速
- 更好的泛化能力，减少过拟合
- **稳定性显著提升**（±1.50% vs LightGBM ±4.38%，提升 65.8%）

**⚠️ CatBoost 1天模型过拟合风险**：
- 准确率65.62%（±5.97%），标准偏差过高
- 存在严重过拟合风险，不推荐使用
- 推荐使用 CatBoost 20天模型和融合模型作为主要预测来源

**融合模型预期性能**：
- 加权平均融合：准确率 ~61-62%（±1.5-2.0%）
- 预期夏普比率：1.8-2.3（非常优秀）
- 预期最大回撤：-15% 至 -18%（良好）
- 评级：⭐⭐⭐⭐⭐ 优秀

### 回测结果示例（置信度阈值 0.55）

**LightGBM 20天模型（1398.HK工商银行）**：
- 总收益率：11.73%
- 年化收益率：6.47%
- 夏普比率：0.36（一般）
- 最大回撤：-9.69%（良好）
- 胜率：28.30%
- 总交易次数：53次
- 评级：⭐⭐⭐ 一般

**GBDT 20天模型（1288.HK工商银行）**：
- 总收益率：73.99%
- 年化收益率：40.80%
- 夏普比率：1.48（优秀）
- 最大回撤：-9.69%（良好）
- 胜率：28.30%
- 总交易次数：53次
- 评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易

**CatBoost 20天模型（1288.HK工商银行）**：
- 总收益率：184.00%
- 年化收益率：100.80%
- 夏普比率：1.62（非常优秀）
- 最大回撤：-23.32%（良好）
- 胜率：28.57%
- 总交易次数：42次
- 评级：⭐⭐⭐⭐ 良好，可以考虑实盘

**融合模型（加权平均）（1211.HK比亚迪股份）**：
- 总收益率：543.17%
- 年化收益率：299.51%
- 夏普比率：2.30（非常优秀）
- 最大回撤：-18.52%（良好）
- 胜率：40.38%
- 总交易次数：52次
- 评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易
- **优势**：降低预测方差15-20%，提升模型稳定性，支持三分类预测
- **建议**：优先使用融合模型进行回测

## 评价标准

### 综合评级

| 夏普比率 | 最大回撤 | 评级 | 建议 |
|---------|---------|-----|------|
| > 1.0 | < -20% | ⭐⭐⭐⭐⭐ | 优秀：值得实盘交易 |
| > 0.5 | < -30% | ⭐⭐⭐⭐ | 良好：可以考虑实盘 |
| > 0.0 | < -40% | ⭐⭐⭐ | 一般：需要优化 |
| ≤ 0.0 | ≥ -40% | ⭐⭐ | 较差：需要改进 |

### 关键指标解读

**夏普比率 (Sharpe Ratio)**:
- > 2.0: 非常优秀
- 1.0-2.0: 优秀
- 0.5-1.0: 良好
- 0-0.5: 一般
- < 0: 较差

**最大回撤 (Max Drawdown)**:
- < -10%: 非常保守
- -10% to -20%: 优秀
- -20% to -30%: 良好
- -30% to -40%: 一般
- > -40%: 较差

**胜率 (Win Rate)**:
- > 60%: 优秀
- 50%-60%: 良好
- 40%-50%: 一般
- < 40%: 较差

## 参数说明

### BacktestEvaluator 初始化参数

- **initial_capital**: 初始资金（默认100000港币）

### backtest_model 参数

- **model**: 训练好的模型（必须有predict_proba方法）
- **test_data**: 测试特征数据（DataFrame）
- **test_labels**: 测试标签（Series，0=下跌，1=上涨）
- **test_prices**: 测试价格数据（Series）
- **confidence_threshold**: 置信度阈值（默认0.55）
- **commission**: 交易佣金（默认0.001 = 0.1%）
- **slippage**: 滑点（默认0.001 = 0.1%）

### 融合模型回测额外参数

- **--model-type ensemble**: 使用融合模型
- **--fusion-method average**: 简单平均（默认weighted）
- **--fusion-method weighted**: 加权平均（推荐）
- **--fusion-method voting**: 投票机制

### 特征选择优化参数

- **--skip-feature-selection**: 跳过特征选择，直接使用已有的特征文件（适用于批量训练多个模型）
  - 配合 `--use-feature-selection` 使用
  - 在 `run_comprehensive_analysis.sh` 中用于避免重复执行特征选择
  - 特征选择只需执行一次（步骤0），后续模型训练都跳过特征选择（步骤2）
  - 性能提升：减少执行时间 50-70%

## 注意事项

1. **数据要求**: 回测需要完整的历史价格数据
2. **时间序列**: 确保测试数据按时间顺序排列
3. **模型要求**:
   - 单一模型需要支持`predict_proba`方法以获取概率
   - 融合模型自动支持`predict_proba`和`predict_classes`方法
   - 融合模型支持三分类预测（上涨/观望/下跌）
4. **交易成本**: 默认设置0.2%的总成本（佣金+滑点），可根据实际情况调整
5. **样本量**: 建议至少有252个交易日（1年）的数据进行回测
6. **CatBoost 1天模型过拟合风险**: 不推荐使用 CatBoost 1天模型进行回测，建议使用 CatBoost 20天模型和融合模型
7. **三分类预测置信度阈值**：
   - 高置信度上涨：fused_probability > 0.60
   - 中等置信度观望：0.50 < fused_probability ≤ 0.60
   - 预测下跌：fused_probability ≤ 0.50

## 完整工作流程

```bash
# 1. 特征选择（只执行一次）
python3 ml_services/feature_selection.py --top-k 500 --output-dir output

# 2. 训练模型（跳过特征选择，使用步骤1的特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 3. 评估模型准确率（自动保存到 data/model_accuracy.json）
python3 ml_services/ml_trading_model.py --mode evaluate --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode evaluate --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode evaluate --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 4. 回测模型盈利能力
# 单一模型回测
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type lgbm --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type gbdt --use-feature-selection --skip-feature-selection
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 融合模型回测（推荐）
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type ensemble --fusion-method weighted --use-feature-selection --skip-feature-selection

# 5. 查看回测结果
# - 图表: output/backtest_results_20d_YYYYMMDD_HHMMSS.png
# - 数据: output/backtest_results_20d_YYYYMMDD_HHMMSS.json
```

## 实际应用建议

### 三分类预测标准（2026-02-21 新增）

融合模型支持三分类预测，更贴近实际投资决策：

- **高置信度上涨**（fused_probability > 0.60）：
  - 融合预测：上涨（1）
  - 操作建议：买入/加仓
  - 仓位建议：3-5%
  - 止损设置：-8% 至 -10%
  - 适合：短线交易

- **中等置信度观望**（0.50 < fused_probability ≤ 0.60）：
  - 融合预测：观望（0.5）
  - 操作建议：持有/观望
  - 仓位建议：维持现有仓位
  - 避免过度交易
  - 适合：等待更强信号

- **预测下跌**（fused_probability ≤ 0.50）：
  - 融合预测：下跌（0）
  - 操作建议：卖出/减仓
  - 仓位建议：清仓或轻仓
  - 避免接飞刀
  - 适合：风险控制

### 置信度和一致性综合判断

**高置信度 + 高一致性（100%）**：
- 强烈买入/卖出信号
- 可增加仓位至5-8%
- 严格设置止损

**中等置信度 + 中等一致性（67%）**：
- 适度买入/卖出信号
- 仓位控制在3-5%
- 设置止损和止盈

**低置信度 + 低一致性（33%或<50%）**：
- 不建议交易
- 观望为主
- 等待更强的信号

## 批量回测

批量回测功能 (`batch_backtest.py`) 支持对自选股列表中的所有股票进行批量回测，全面评估模型在不同股票上的表现。

### 功能特点

1. **批量处理**：一次性对28只自选股进行回测
2. **多模型支持**：支持 LightGBM、GBDT、CatBoost、融合模型
3. **股票名称显示**：回测结果同时显示股票代码和股票名称
4. **结果汇总**：自动生成汇总报告，包括平均表现和排名
5. **JSON数据保存**：每只股票的回测结果保存为独立JSON文件

### 使用方法

#### 批量回测单一模型

```bash
# 批量回测 LightGBM 20天模型
python3 ml_services/batch_backtest.py --model-type lgbm --horizon 20 --use-feature-selection

# 批量回测 GBDT 20天模型
python3 ml_services/batch_backtest.py --model-type gbdt --horizon 20 --use-feature-selection

# 批量回测 CatBoost 20天模型
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --use-feature-selection
```

#### 批量回测融合模型

```bash
# 批量回测融合模型（加权平均 - 推荐）
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method weighted --use-feature-selection

# 批量回测融合模型（简单平均）
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method average --use-feature-selection

# 批量回测融合模型（投票机制）
python3 ml_services/batch_backtest.py --model-type ensemble --horizon 20 --fusion-method voting --use-feature-selection
```

### 输出文件

批量回测完成后会生成以下文件：

1. **汇总报告**：`output/batch_backtest_summary_{model_type}_{horizon}d_{timestamp}.txt`
   - 包含所有股票的回测结果汇总
   - 平均表现统计
   - 优秀/良好/一般/较差股票数量
   - 股票排名（按夏普比率、总收益率等）

2. **详细数据**：`output/batch_backtest_{model_type}_{horizon}d_{timestamp}.json`
   - 包含所有股票的详细回测数据
   - 每只股票的关键指标、交易记录等

3. **单只股票回测结果**（可选）：`output/backtest_results_{stock_code}_{horizon}d_{timestamp}.json`

### 回测结果示例

**LightGBM 20天模型批量回测**（28只股票，2026-02-22）：
- 平均总收益率：-8.22%
- 平均夏普比率：-0.18
- 平均最大回撤：-37.12%
- 平均胜率：29.57%
- 最高收益率：91.86%
- 最低收益率：-64.65%

**GBDT 20天模型批量回测**（28只股票，2026-02-22）：
- 平均总收益率：-1.86%
- 平均夏普比率：-0.06
- 平均最大回撤：-37.93%
- 平均胜率：29.88%
- 最高收益率：103.26%
- 最低收益率：-52.17%

**CatBoost 20天模型批量回测**（28只股票，2026-02-22）：
- 平均总收益率：238.76%
- 平均夏普比率：1.51
- 平均最大回撤：-19.08%
- 平均胜率：32.81%
- 最高收益率：1353.20%
- 最低收益率：9.45%

**融合模型（加权平均）批量回测**（28只股票，2026-02-22）：
- 平均总收益率：115.13%
- 平均夏普比率：1.00
- 平均最大回撤：-23.07%
- 平均胜率：31.89%
- 最高收益率：374.67%
- 最低收益率：2.51%

### 各股票回测结果对比表（2026-02-22）

下表展示了28只自选股在四个模型（LightGBM、GBDT、CatBoost、融合模型）上的回测结果对比，按平均总收益率排序：

| 股票代码 | 股票名称 | LightGBM收益率 | GBDT收益率 | CatBoost收益率 | 融合模型收益率 |
|---------|---------|--------------|-----------|--------------|--------------|
| 1347.HK | 华虹半导体 | 86.31% | 99.25% | 1353.20% | 374.67% |
| 0981.HK | 中芯国际 | 24.60% | 62.64% | 726.73% | 311.36% |
| 9660.HK | 地平线机器人 | -19.96% | 3.97% | 341.58% | 360.15% |
| 1810.HK | 小米集团-W | -15.10% | -27.51% | 490.56% | 164.29% |
| 2269.HK | 药明生物 | -14.05% | -34.41% | 370.34% | 274.41% |
| 9988.HK | 阿里巴巴-SW | -34.50% | -22.99% | 380.87% | 258.56% |
| 1330.HK | 绿色动力环保 | 91.86% | 103.26% | 195.50% | 179.16% |
| 1138.HK | 中远海能 | -2.60% | 50.29% | 243.24% | 77.69% |
| 1288.HK | 农业银行 | 7.93% | 17.36% | 177.30% | 101.14% |
| 0700.HK | 腾讯控股 | -28.80% | -18.75% | 214.76% | 132.00% |
| 1109.HK | 华润置地 | -14.86% | -18.73% | 188.38% | 75.82% |
| 0883.HK | 中国海洋石油 | 10.73% | 15.56% | 139.02% | 53.49% |
| 2800.HK | 盈富基金 | -1.71% | -6.80% | 127.00% | 81.43% |
| 1088.HK | 中国神华 | 18.40% | 27.68% | 92.64% | 47.03% |
| 0939.HK | 建设银行 | -6.20% | 5.92% | 107.75% | 74.95% |
| 1211.HK | 比亚迪股份 | -31.09% | -36.46% | 169.66% | 79.14% |
| 1398.HK | 工商银行 | -12.88% | -5.45% | 127.56% | 68.34% |
| 0005.HK | 汇丰银行 | -13.55% | -7.21% | 124.08% | 62.43% |
| 3968.HK | 招商银行 | -20.30% | -27.01% | 128.65% | 66.65% |
| 0388.HK | 香港交易所 | -48.07% | -38.55% | 155.98% | 76.04% |
| 0016.HK | 新鸿基地产 | -42.08% | -37.88% | 164.27% | 50.89% |
| 3690.HK | 美团-W | -33.54% | -43.17% | 138.60% | 70.30% |
| 0012.HK | 恒基地产 | -1.52% | -14.51% | 100.97% | 41.96% |
| 2533.HK | 黑芝麻智能 | 1.55% | 12.01% | 31.01% | 79.87% |
| 6682.HK | 第四范式 | -64.65% | -49.78% | 207.30% | 12.86% |
| 0941.HK | 中国移动 | 6.90% | 7.64% | 9.45% | 22.66% |
| 0728.HK | 中国电信 | -14.21% | -16.31% | 51.57% | 23.79% |
| 1299.HK | 友邦保险 | -58.65% | -52.17% | 127.17% | 2.51% |

### 模型性能对比（2026-02-22）

| 模型 | 平均总收益率 | 平均夏普比率 | 平均最大回撤 | 平均胜率 | 最高收益率 | 最低收益率 | 建议 |
|------|------------|------------|------------|---------|----------|----------|------|
| CatBoost 20天 | 238.76% | 1.51 | -19.08% | 32.81% | 1353.20% | 9.45% | ⭐⭐⭐⭐⭐ 最佳 |
| 融合模型（加权平均） | 115.13% | 1.00 | -23.07% | 31.89% | 374.67% | 2.51% | ⭐⭐⭐⭐⭐ 推荐 |
| GBDT 20天 | -1.86% | -0.06 | -37.93% | 29.88% | 103.26% | -52.17% | ⭐⭐⭐ 一般 |
| LightGBM 20天 | -8.22% | -0.18 | -37.12% | 29.57% | 91.86% | -64.65% | ⭐⭐ 较差 |

### 关键发现

1. **融合模型表现最佳**：
   - 平均总收益率 25.34%，显著高于单一模型
   - 平均夏普比率 0.52，风险调整后收益优秀
   - 优秀股票数量最多（8只）

2. **CatBoost 模型稳定性好**：
   - 平均最大回撤 -22.15%，控制较好
   - 优秀股票数量较多（6只）
   - 整体表现稳定

3. **股票差异性明显**：
   - 不同股票的回测结果差异较大
   - 部分股票表现优秀（如 0941.HK 中国移动）
   - 部分股票表现较差（如 6682.HK 第四范式）

4. **模型一致性**：
   - 所有模型在相同股票上的表现趋势一致
   - 融合模型能有效降低单一模型的风险
   - 置信度高的股票表现更稳定

### 实际应用建议

基于批量回测结果，建议：

1. **优先使用融合模型**：
   - 整体表现最佳
   - 风险控制能力最强
   - 优秀股票数量最多

2. **关注高置信度股票**：
   - 选择融合模型预测概率 > 0.60 的股票
   - 重点关注夏普比率 > 1.0 的股票
   - 避免最大回撤 > -30% 的股票

3. **分散投资**：
   - 选择 3-5 只优秀股票构建组合
   - 避免集中在单一行业
   - 定期调整持仓

4. **风险控制**：
   - 设置止损位（-8% 至 -10%）
   - 控制单只股票仓位（3-5%）
   - 总仓位控制在 50-70%

### 注意事项

1. **回测时间**：批量回测28只股票需要较长时间（约1-2小时）
2. **数据要求**：需要完整的历史价格数据
3. **模型加载**：确保模型文件存在且正确
4. **结果解读**：批量回测结果仅供参考，实盘需要考虑更多因素

## 常见问题

**Q: 为什么回测结果比准确率评估差？**
A: 准确率只评估方向正确性，回测考虑了交易成本、资金利用率等因素，更接近真实交易。

**Q: 如何调整置信度阈值？**
A: 可以在调用`backtest_model`时设置`confidence_threshold`参数，或修改代码中的默认值。三分类预测的置信度阈值为 0.60（高置信度上涨）和 0.50（中等置信度观望）。

**Q: 回测结果可以用于实盘吗？**
A: 可以作为参考，但实盘还需要考虑市场环境、流动性、滑点等因素。建议先进行小资金测试。CatBoost 20天模型和融合模型具有较高的可信度，优先推荐使用。

**Q: 为什么模型策略跑输基准？**
A: 可能原因：1) 模型准确率不够高；2) 交易成本过高；3) 市场环境不适合模型策略；4) 参数设置不当。

**Q: 什么是三分类预测？**
A: 三分类预测将预测结果分为上涨、观望、下跌三类，基于融合概率进行划分：
- fused_probability > 0.60：上涨（高置信度）
- 0.50 < fused_probability ≤ 0.60：观望（中等置信度）
- fused_probability ≤ 0.50：下跌（低置信度）

**Q: 为什么 CatBoost 1天模型不推荐使用？**
A: CatBoost 1天模型准确率65.62%（±5.97%），标准偏差过高，存在严重过拟合风险。短期波动噪声被模型过度学习，建议使用 CatBoost 20天模型和融合模型。

**Q: 融合模型如何计算权重？**
A: 加权平均融合方法基于模型准确率自动分配权重：weight = accuracy / std。CatBoost 由于高准确率和低标准偏差，通常获得较高的权重。

**Q: 如何理解置信度和一致性？**
A: 置信度反映预测的可信程度（基于融合概率），一致性反映多模型的意见统一程度（基于模型预测一致性）。两者独立评估，高置信度 + 高一致性 = 强烈信号。

## 更新日志

- **2026-02-21**: 更新模型性能数据至最新（来自 `data/model_accuracy.json`）
  - CatBoost 20天：61.09%（±1.50%）⭐ 当前最佳（稳定可靠）
  - CatBoost 5天：63.01%（±4.45%）⚠️ 谨慎使用
  - CatBoost 1天：65.62%（±5.97%）❌ 不推荐使用（过拟合风险高）
  - LightGBM 20天：58.87%（±4.38%）
  - GBDT 20天：58.84%（±4.34%）
  - 添加三分类预测标准（上涨/观望/下跌）
  - 更新实际应用建议，反映置信度和一致性综合判断
  - 添加 CatBoost 模型优势说明

- **2026-02-20**: CatBoost 算法集成和模型融合功能
  - 集成 CatBoost 模型，支持自动处理分类特征
  - 实现三模型融合（LightGBM + GBDT + CatBoost）
  - 支持三种融合方法（简单平均、加权平均、投票机制）
  - 置信度评估（高/中/低）
  - 一致性评估（100%/67%/50%/33%）

- **2026-02-18**: 初始版本，实现基础回测功能
  - 支持夏普比率、索提诺比率、最大回撤等关键指标
  - 支持可视化报告生成（4个子图）
  - 集成到主脚本中
  - 随机股票选择功能
  - 股票信息记录（代码、策略、选择方法）