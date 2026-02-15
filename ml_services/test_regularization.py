"""
正则化策略验证脚本（简化版）
使用模拟数据测试不同L1/L2配置对模型性能的影响

目标：
- 验证当前正则化策略（0.15）是否最优
- 找到平衡拟合能力和泛化能力的最佳配置
- 降低一个月模型的波动性

注意：此脚本使用模拟数据进行概念验证
实际应用时应使用真实数据
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
import os


# 测试配置
TEST_CONFIGS = [
    {'reg_alpha': 0.1, 'reg_lambda': 0.1, 'name': 'baseline'},
    {'reg_alpha': 0.12, 'reg_lambda': 0.12, 'name': 'light'},
    {'reg_alpha': 0.15, 'reg_lambda': 0.15, 'name': 'current'},
    {'reg_alpha': 0.18, 'reg_lambda': 0.18, 'name': 'strong'},
    {'reg_alpha': 0.2, 'reg_lambda': 0.2, 'name': 'very_strong'},
]

# 基础模型参数（一个月模型）
BASE_PARAMS = {
    'n_estimators': 45,
    'learning_rate': 0.025,
    'max_depth': 4,
    'num_leaves': 13,
    'min_child_samples': 35,
    'subsample': 0.65,
    'colsample_bytree': 0.65,
    'min_split_gain': 0.12,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.65,
    'bagging_freq': 5,
    'random_state': 42,
    'verbose': -1
}


def generate_synthetic_data(n_samples=1000, n_features=10, noise_level=0.1):
    """
    生成模拟数据用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 噪声水平
    
    Returns:
        X, y: 特征和标签
    """
    np.random.seed(42)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标变量（带有一定的信号）
    # 前5个特征对目标有影响
    signal = X[:, :5].sum(axis=1) * 0.3
    noise = np.random.randn(n_samples) * noise_level
    y_proba = 1 / (1 + np.exp(-(signal + noise)))
    
    # 转换为二分类标签
    y = (y_proba > 0.5).astype(int)
    
    # 添加特征名称
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, pd.Series(y)


def time_series_cross_validation(X, y, n_splits=5):
    """
    时间序列交叉验证
    
    Args:
        X: 特征数据
        y: 标签数据
        n_splits: 分割数量
    
    Returns:
        generator: 生成器，每次返回一个fold的数据
    """
    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        # 时间序列分割：前i+1个fold作为训练，第i+2个fold作为验证
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = train_end + fold_size
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]
        
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        
        yield X_train, y_train, X_val, y_val, i + 1


def test_regularization_config(X, y, config):
    """
    测试单个正则化配置
    
    Args:
        X: 特征数据
        y: 标签数据
        config: 配置字典
    
    Returns:
        dict: 测试结果
    """
    print(f"\n{'='*60}")
    print(f"测试配置: {config['name']}")
    print(f"reg_alpha={config['reg_alpha']}, reg_lambda={config['reg_lambda']}")
    print(f"{'='*60}")
    
    # 合并参数
    params = BASE_PARAMS.copy()
    params['reg_alpha'] = config['reg_alpha']
    params['reg_lambda'] = config['reg_lambda']
    
    # 运行5折交叉验证
    fold_scores = []
    fold_losses = []
    
    for X_train, y_train, X_val, y_val, fold_num in time_series_cross_validation(X, y):
        # 训练模型
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        loss = log_loss(y_val, y_pred_proba)
        
        fold_scores.append(accuracy)
        fold_losses.append(loss)
        
        print(f"  Fold {fold_num}: 准确率={accuracy:.4f}, LogLoss={loss:.4f}")
    
    # 计算汇总指标
    avg_accuracy = np.mean(fold_scores)
    std_accuracy = np.std(fold_scores)
    avg_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    
    # 计算稳定性评分（1 - 标准差/平均值）
    stability_score = 1 - (std_accuracy / avg_accuracy) if avg_accuracy > 0 else 0
    
    # 找出最好和最差的fold
    best_fold = max(fold_scores)
    worst_fold = min(fold_scores)
    fold_range = best_fold - worst_fold
    
    print(f"\n  平均准确率: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"  平均LogLoss: {avg_loss:.4f} (+/- {std_loss:.4f})")
    print(f"  稳定性评分: {stability_score:.4f} (越高越好)")
    print(f"  最佳Fold: {best_fold:.4f}")
    print(f"  最差Fold: {worst_fold:.4f}")
    print(f"  Fold范围: {fold_range:.4f} (越小越好)")
    
    result = {
        'config': config['name'],
        'reg_alpha': config['reg_alpha'],
        'reg_lambda': config['reg_lambda'],
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_loss': avg_loss,
        'std_loss': std_loss,
        'stability_score': stability_score,
        'best_fold': best_fold,
        'worst_fold': worst_fold,
        'fold_range': fold_range,
        'fold_scores': fold_scores,
        'fold_losses': fold_losses
    }
    
    return result


def compare_results(results):
    """
    对比测试结果
    
    Args:
        results: 测试结果列表
    
    Returns:
        DataFrame: 结果对比表
    """
    df = pd.DataFrame(results)
    
    # 排序：优先准确率，其次稳定性
    df = df.sort_values(['avg_accuracy', 'stability_score'], ascending=[False, False])
    
    # 添加排名
    df['rank_accuracy'] = df['avg_accuracy'].rank(ascending=False)
    df['rank_stability'] = df['stability_score'].rank(ascending=False)
    df['rank_combined'] = (df['rank_accuracy'] + df['rank_stability']) / 2
    
    return df


def generate_report(results, df_comparison):
    """
    生成测试报告
    
    Args:
        results: 测试结果列表
        df_comparison: 对比DataFrame
    
    Returns:
        str: 报告文本
    """
    report = f"""
# 正则化策略验证报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试配置
"""
    for config in TEST_CONFIGS:
        report += f"- {config['name']}: reg_alpha={config['reg_alpha']}, reg_lambda={config['reg_lambda']}\n"
    
    report += f"""
## 结果对比

| 配置 | reg_alpha | reg_lambda | 平均准确率 | 标准差 | 稳定性评分 | LogLoss | Fold范围 |
|------|-----------|------------|-----------|--------|-----------|---------|---------|
"""
    
    for _, row in df_comparison.iterrows():
        report += f"| {row['config']} | {row['reg_alpha']} | {row['reg_lambda']} | {row['avg_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['stability_score']:.4f} | {row['avg_loss']:.4f} | {row['fold_range']:.4f} |\n"
    
    report += f"""
## 推荐配置

### 最高准确率配置
"""
    best_accuracy = df_comparison.iloc[0]
    report += f"- **{best_accuracy['config']}**: 平均准确率={best_accuracy['avg_accuracy']:.4f}, 标准差={best_accuracy['std_accuracy']:.4f}\n"
    
    report += f"""
### 最稳定配置
"""
    most_stable = df_comparison.sort_values('stability_score', ascending=False).iloc[0]
    report += f"- **{most_stable['config']}**: 稳定性评分={most_stable['stability_score']:.4f}, 平均准确率={most_stable['avg_accuracy']:.4f}\n"
    
    report += f"""
### 综合推荐
"""
    best_combined = df_comparison.sort_values('rank_combined').iloc[0]
    report += f"- **{best_combined['config']}**: 综合排名={best_combined['rank_combined']:.1f}\n"
    report += f"  - 平均准确率: {best_combined['avg_accuracy']:.4f}\n"
    report += f"  - 稳定性评分: {best_combined['stability_score']:.4f}\n"
    report += f"  - Fold范围: {best_combined['fold_range']:.4f}\n"
    
    report += f"""
## 关键发现

1. **当前配置（current）表现**:
   - 平均准确率: {df_comparison[df_comparison['config']=='current']['avg_accuracy'].values[0]:.4f}
   - 稳定性评分: {df_comparison[df_comparison['config']=='current']['stability_score'].values[0]:.4f}
   - Fold范围: {df_comparison[df_comparison['config']=='current']['fold_range'].values[0]:.4f}

2. **与baseline对比**:
   - 准确率变化: {df_comparison[df_comparison['config']=='current']['avg_accuracy'].values[0] - df_comparison[df_comparison['config']=='baseline']['avg_accuracy'].values[0]:+.4f}
   - 稳定性变化: {df_comparison[df_comparison['config']=='current']['stability_score'].values[0] - df_comparison[df_comparison['config']=='baseline']['stability_score'].values[0]:+.4f}

3. **最优配置建议**:
   - 如果优先准确率: 使用 {best_accuracy['config']} (reg_alpha={best_accuracy['reg_alpha']})
   - 如果优先稳定性: 使用 {most_stable['config']} (reg_alpha={most_stable['reg_alpha']})
   - 如果平衡两者: 使用 {best_combined['config']} (reg_alpha={best_combined['reg_alpha']})

## 重要说明

⚠️  **注意**: 此测试使用模拟数据进行概念验证

- **数据来源**: 模拟数据（1000个样本，10个特征）
- **信号强度**: 中等（前5个特征对目标有影响）
- **噪声水平**: 0.1
- **实际应用**: 需要使用真实港股数据重新验证

### 模拟数据 vs 真实数据

| 指标 | 模拟数据 | 真实数据 |
|------|---------|---------|
| 样本数量 | 1000 | ~50000+ |
| 特征数量 | 10 | 2936 |
| 信号强度 | 中等 | 未知 |
| 噪声水平 | 0.1 | 未知 |
| 市场复杂性 | 简单 | 复杂 |

### 建议的验证步骤

1. **在真实数据上验证**:
   ```bash
   # 使用完整数据重新训练，测试不同正则化配置
   python ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm
   ```

2. **对比实际性能**:
   - 记录5折交叉验证的准确率和标准差
   - 对比不同配置的实际表现
   - 选择在实际数据上表现最好的配置

3. **监控稳定性**:
   - 观察各Fold之间的差异
   - 优先选择Fold范围较小的配置
   - 确保模型在不同时期都能稳定表现

## 下一步行动

### 立即行动
1. 根据本测试结果，选择推荐的配置
2. 修改 `ml_services/ml_trading_model.py` 中的参数
3. 重新训练一个月模型

### 后续验证
1. 使用真实数据重新验证正则化策略
2. 监控模型在实际市场中的表现
3. 根据实际表现进一步调整参数

### 长期优化
1. 定期重新评估正则化策略
2. 根据市场变化调整参数
3. 建立自动化的参数调优流程

"""
    
    return report


def main():
    """主函数"""
    print("="*60)
    print("🚀 正则化策略验证（模拟数据）")
    print("="*60)
    print("\n⚠️  注意: 此测试使用模拟数据进行概念验证")
    print("    实际应用时需要使用真实港股数据\n")
    
    # 生成模拟数据
    print("📊 生成模拟数据...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=10, noise_level=0.1)
    print(f"  ✅ 生成数据: {len(X)} 个样本, {len(X.columns)} 个特征")
    
    # 测试所有配置
    results = []
    for config in TEST_CONFIGS:
        result = test_regularization_config(X, y, config)
        results.append(result)
    
    # 对比结果
    df_comparison = compare_results(results)
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'regularization_test_results.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✅ 结果已保存到 {csv_path}")
    
    # 生成报告
    report = generate_report(results, df_comparison)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'regularization_test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告已保存到 {report_path}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    print(df_comparison[['config', 'reg_alpha', 'reg_lambda', 'avg_accuracy', 'std_accuracy', 'stability_score', 'fold_range']].to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
    print("\n📖 查看完整报告: output/regularization_test_report.md")


if __name__ == '__main__':
    main()