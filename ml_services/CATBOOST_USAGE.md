# CatBoost 模型使用指南

## 概述

CatBoost 是 Yandex 开发的梯度提升库，已成功集成到项目中。CatBoost 具有以下优势：

1. **自动处理分类特征**：无需手动编码，减少预处理工作量
2. **更好的默认参数**：减少调参工作量，开箱即用
3. **更快的训练速度**：支持 GPU 加速
4. **更好的泛化能力**：减少过拟合，提升模型稳定性

## 快速开始

### 1. 训练 CatBoost 模型

```bash
# 训练 1 天预测模型
python3 ml_services/ml_trading_model.py --mode train --model-type catboost --horizon 1

# 训练 5 天预测模型
python3 ml_services/ml_trading_model.py --mode train --model-type catboost --horizon 5

# 训练 20 天预测模型（使用特征选择）
python3 ml_services/ml_trading_model.py --mode train --model-type catboost --horizon 20 --use-feature-selection
```

### 2. 预测股票涨跌

```bash
# 使用最新数据预测
python3 ml_services/ml_trading_model.py --mode predict --model-type catboost --horizon 20

# 基于指定日期预测
python3 ml_services/ml_trading_model.py --mode predict --model-type catboost --horizon 20 --predict-date 2026-02-20
```

### 3. 模型回测

```bash
# 回测 20 天预测模型
python3 ml_services/ml_trading_model.py --mode backtest --model-type catboost --horizon 20 --use-feature-selection
```

## CatBoost 模型参数

### 默认参数配置

| 预测周期 | 树数量 | 深度 | 学习率 | L2正则 | 早停耐心 | 行采样 | 列采样 |
|---------|--------|------|--------|--------|----------|--------|--------|
| 1天     | 500    | 7    | 0.05   | 3      | 40       | 0.75   | 0.7    |
| 5天     | 500    | 6    | 0.05   | 3      | 50       | 0.7    | 0.6    |
| 20天    | 400    | 5    | 0.04   | 5      | 60       | 0.6    | 0.6    |

### 参数说明

- **n_estimators**：树的数量（迭代次数）
- **depth**：树的深度（控制模型复杂度）
- **learning_rate**：学习率（控制每棵树的贡献）
- **l2_leaf_reg**：L2正则化系数（防止过拟合）
- **early_stopping_rounds**：早停耐心（防止过拟合）
- **subsample**：行采样比例（随机梯度下降）
- **colsample_bylevel**：列采样比例（特征随机性）

## 与其他模型对比

### CatBoost vs LightGBM vs GBDT

| 特性 | CatBoost | LightGBM | GBDT |
|------|----------|----------|------|
| 分类特征处理 | 自动 | 需手动编码 | 需手动编码 |
| 默认参数 | 优秀 | 良好 | 需调优 |
| 训练速度 | 快 | 最快 | 慢 |
| 过拟合控制 | 优秀 | 良好 | 一般 |
| GPU 支持 | 是 | 是 | 否 |
| 可解释性 | 好 | 好 | 一般 |

## 模型输出文件

### 训练输出

- `data/ml_trading_model_catboost_1d.pkl` - 1天模型文件
- `data/ml_trading_model_catboost_5d.pkl` - 5天模型文件
- `data/ml_trading_model_catboost_20d.pkl` - 20天模型文件
- `output/catboost_feature_importance.csv` - 特征重要性文件

### 预测输出

- `data/ml_trading_model_catboost_predictions_1d.csv` - 1天预测结果
- `data/ml_trading_model_catboost_predictions_5d.csv` - 5天预测结果
- `data/ml_trading_model_catboost_predictions_20d.csv` - 20天预测结果

### 回测输出

- `output/backtest_results_20d_YYYYMMDD_HHMMSS.png` - 回测结果图表
- `output/backtest_results_20d_YYYYMMDD_HHMMSS.json` - 回测结果数据

## 准确率管理

CatBoost 模型的准确率会自动保存到 `data/model_accuracy.json` 文件中，格式如下：

```json
{
  "catboost_1d": {
    "model_type": "catboost",
    "horizon": 1,
    "accuracy": 0.5234,
    "std": 0.0245,
    "timestamp": "2026-02-20 15:30:00"
  },
  "catboost_5d": {
    "model_type": "catboost",
    "horizon": 5,
    "accuracy": 0.5567,
    "std": 0.0289,
    "timestamp": "2026-02-20 15:35:00"
  },
  "catboost_20d": {
    "model_type": "catboost",
    "horizon": 20,
    "accuracy": 0.6012,
    "std": 0.0456,
    "timestamp": "2026-02-20 15:40:00"
  }
}
```

## 使用示例

### Python 代码示例

```python
from ml_services.ml_trading_model import CatBoostModel
from config import WATCHLIST

# 创建模型实例
model = CatBoostModel()

# 训练模型
feature_importance = model.train(
    codes=WATCHLIST,
    horizon=20,
    use_feature_selection=True
)

# 保存模型
model.save_model('data/my_catboost_model.pkl')

# 加载模型
model.load_model('data/my_catboost_model.pkl')

# 预测单只股票
result = model.predict('0700.HK')
print(f"预测结果: {result}")
```

## 最佳实践

1. **使用特征选择**：推荐使用 `--use-feature-selection` 参数，只使用500个精选特征
2. **先训练20天模型**：20天模型准确率最高，适合中长期投资
3. **定期重新训练**：建议每周重新训练模型，保持模型时效性
4. **关注过拟合**：注意训练/验证差距，如果差距过大需要增强正则化
5. **对比多种模型**：同时训练 LightGBM、GBDT、CatBoost，选择表现最好的

## 注意事项

1. **首次训练时间较长**：CatBoost 首次训练可能需要10-30分钟
2. **内存占用**：确保有足够的内存（建议 > 8GB）
3. **数据质量**：确保股票数据完整，缺失数据会影响模型性能
4. **特征一致性**：训练和预测时使用相同的特征列
5. **版本兼容**：CatBoost 版本建议 >= 1.0.0

## 故障排除

### 问题1：导入失败

```
ImportError: No module named 'catboost'
```

**解决方案**：
```bash
pip install catboost
```

### 问题2：训练失败

```
ValueError: 没有可用的数据
```

**解决方案**：
- 检查网络连接
- 确认股票代码正确
- 检查数据源是否可用

### 问题3：预测失败

```
ValueError: 模型未训练，请先调用train()方法
```

**解决方案**：
- 确保先训练模型
- 检查模型文件是否存在
- 确认模型路径正确

## 性能基准

基于2026年2月的测试数据：

| 预测周期 | LightGBM | GBDT | CatBoost |
|---------|----------|------|----------|
| 1天     | 51.88%   | 52.00% | 待测试 |
| 5天     | 54.64%   | 53.75% | 待测试 |
| 20天    | 59.72%   | 59.22% | 待测试 |

*注：CatBoost 的准确率将在首次训练后更新*

## 参考资料

- [CatBoost 官方文档](https://catboost.ai/docs/)
- [CatBoost GitHub](https://github.com/catboost/catboost)
- [CatBoost 论文](https://arxiv.org/abs/1706.09516)

## 更新日志

- **2026-02-20**：CatBoost 模型首次集成到项目
- 支持训练、预测、回测功能
- 支持特征选择
- 自动保存准确率到 JSON 文件
- 完整的命令行参数支持