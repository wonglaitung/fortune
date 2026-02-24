# 特征选择方法汇总

## 快速对比

| 预测周期 | 统计方法准确率 | 模型重要性法 | 速度提升 | 推荐方案 |
|---------|--------------|------------|---------|---------|
| **1天** | 51.59% | 51.94% (Top 50) | +28.5% | Top 50 |
| **5天** | 56.47% | 56.47% (稳定高100) | +28.7% | 稳定高100 |
| **20天** | 61.87% | 61.87% (Top 100) | +20.1% | Top 100 |

## 最佳模型对比（全部模型）

| 模型 | 预测周期 | 准确率 | 标准差 | 推荐度 |
|------|---------|--------|--------|--------|
| **LightGBM** | 20天 | 61.87% | 3.71% | ⭐⭐⭐⭐⭐ |
| **CatBoost** | 20天 | 61.36% | 1.98% | ⭐⭐⭐⭐⭐ |
| **CatBoost** | 5天 | 63.01% | 4.45% | ⭐⭐⭐⭐ |
| **LightGBM** | 5天 | 56.47% | 2.27% | ⭐⭐⭐⭐ |
| **GBDT** | 20天 | 60.04% | 5.28% | ⭐⭐⭐ |
| **CatBoost** | 1天 | 65.62% | 5.97% | ⚠️ 过拟合 |
| **LightGBM** | 1天 | 51.59% | 2.25% | ⭐⭐⭐ |

## 方法对比结论

| 维度 | 统计方法 | 模型重要性法 | 胜出者 |
|------|---------|------------|--------|
| 准确率 | 56.64% (平均) | 56.76% (平均) | 模型重要性法 |
| 速度 | N/A | +20-32% | 模型重要性法 |
| 稳定性 | 未考虑 | 考虑CV指标 | 模型重要性法 |
| 实用性 | 需验证 | 直接可用 | 模型重要性法 |

## 最终推荐

**推荐使用：模型重要性法**

### 具体配置
```bash
# 1天预测 - 使用 Top 50 特征
python scripts/train_with_feature_selection.py \
  --mode train --model_type lightgbm --horizon 1 \
  --feature_set top_50

# 5天预测 - 使用稳定高100 特征
python scripts/train_with_feature_selection.py \
  --mode train --model_type lightgbm --horizon 5 \
  --feature_set stable_high_100

# 20天预测 - 使用 Top 100 特征
python scripts/train_with_feature_selection.py \
  --mode train --model_type lightgbm --horizon 20 \
  --feature_set top_100
```

### 优化效果
- ✅ 训练速度提升 20-32%
- ✅ 准确率保持（56-62%）
- ✅ 模型更稳定（考虑CV指标）
- ✅ 特征数量减少 97-99%

## 详细报告

完整的对比分析请查看：`docs/feature_selection_methods_comparison.md`
