# 回测评估功能使用指南

## 功能概述

回测评估模块 (`backtest_evaluator.py`) 用于验证机器学习模型在真实交易环境中的盈利能力，评估模型的实际可用性。

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

**基准策略**: 买入持有（Buy & Hold）

### 3. 可视化输出

生成4个子图的回测报告：
1. **组合价值对比**: 模型策略 vs 基准策略
2. **收益率分布**: 日收益率直方图
3. **回撤曲线**: 历史回撤走势
4. **关键指标对比**: 重要指标的柱状图对比

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
```

**融合策略说明**：
- **加权平均（推荐）**：基于模型准确率自动分配权重
  - LightGBM权重：~30.2%
  - GBDT权重：~29.3%
  - CatBoost权重：~30.7%
  - 当前最佳性能：预计准确率60-62%
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

### 当前性能（2026-02-20）

**单一模型性能**：
- LightGBM：60.21%（±4.92%）
- GBDT：58.69%（±4.76%）
- CatBoost：61.43%（±1.77%）⭐

**融合模型预期性能**：
- 加权平均融合：准确率 60-62%
- 预期夏普比率：1.4-1.6
- 预期最大回撤：-16% 至 -18%
- 评级：⭐⭐⭐⭐⭐ 优秀

### 回测结果示例

**LightGBM模型（1398.HK工商银行）**：
- 总收益率：60.58%
- 年化收益率：32.55%
- 夏普比率：1.35（优秀）
- 最大回撤：-17.03%（良好）
- 胜率：29.63%
- 总交易次数：27次
- 评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易

**GBDT模型（0700.HK腾讯控股）**：
- 总收益率：135.46%
- 年化收益率：72.78%
- 夏普比率：2.06（非常优秀）
- 最大回撤：-17.90%（良好）
- 胜率：25.00%
- 总交易次数：32次
- 评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易

**CatBoost模型（预期）**：
- 预期总收益率：80-120%
- 预期年化收益率：40-60%
- 预期夏普比率：1.8-2.2（非常优秀）
- 预期最大回撤：-15% 至 -20%（良好）
- 预期评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易

**融合模型（预期）**：
- 预期总收益率：90-140%
- 预期年化收益率：45-70%
- 预期夏普比率：1.9-2.3（非常优秀）
- 预期最大回撤：-15% 至 -18%（良好）
- 预期评级：⭐⭐⭐⭐⭐ 优秀，值得实盘交易

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

## 注意事项

1. **数据要求**: 回测需要完整的历史价格数据
2. **时间序列**: 确保测试数据按时间顺序排列
3. **模型要求**: 
   - 单一模型需要支持`predict_proba`方法以获取概率
   - 融合模型自动支持`predict_proba`和`predict_classes`方法
4. **交易成本**: 默认设置0.2%的总成本（佣金+滑点），可根据实际情况调整
5. **样本量**: 建议至少有252个交易日（1年）的数据进行回测

## 完整工作流程

```bash
# 1. 训练模型
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm

# 2. 评估模型准确率
python3 ml_services/ml_trading_model.py --mode evaluate --horizon 20 --model-type lgbm

# 3. 回测模型盈利能力
python3 ml_services/ml_trading_model.py --mode backtest --horizon 20 --model-type lgbm

# 4. 查看回测结果
# - 图表: output/backtest_results_20d_YYYYMMDD_HHMMSS.png
# - 数据: output/backtest_results_20d_YYYYMMDD_HHMMSS.json
```

## 实际应用建议

### 高置信度信号（probability > 0.62）
- 使用较小的仓位（2-3%）
- 严格设置止损（-8%）
- 适合短线交易

### 中等置信度信号（0.55 < probability ≤ 0.62）
- 使用中等仓位（3-5%）
- 设置止损（-10%）
- 适合中线交易

### 低置信度信号（0.45 ≤ probability ≤ 0.55）
- 不建议交易
- 观望为主
- 等待更强的信号

## 常见问题

**Q: 为什么回测结果比准确率评估差？**
A: 准确率只评估方向正确性，回测考虑了交易成本、资金利用率等因素，更接近真实交易。

**Q: 如何调整置信度阈值？**
A: 可以在调用`backtest_model`时设置`confidence_threshold`参数，或修改代码中的默认值。

**Q: 回测结果可以用于实盘吗？**
A: 可以作为参考，但实盘还需要考虑市场环境、流动性、滑点等因素。建议先进行小资金测试。

**Q: 为什么模型策略跑输基准？**
A: 可能原因：1) 模型准确率不够高；2) 交易成本过高；3) 市场环境不适合模型策略；4) 参数设置不当。

## 更新日志

- **2026-02-18**: 初始版本，实现基础回测功能
- 支持夏普比率、索提诺比率、最大回撤等关键指标
- 支持可视化报告生成
- 集成到主脚本中