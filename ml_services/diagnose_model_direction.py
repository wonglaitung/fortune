#!/usr/bin/env python3
"""
诊断模型学习方向问题

验证假设：
1. 训练集和测试集的特征-标签关系是否一致？
2. 模型是否学到了相反的方向？
3. 哪些特征方向正确，哪些错误？
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'ml_services'))

from ml_trading_model import CatBoostModel
from config import WATCHLIST


def analyze_feature_label_relationship(df, feature_cols, label_col='Label', split_date=None, date_col='Date'):
    """分析特征与标签的关系"""
    results = []

    if split_date:
        train_df = df[df[date_col] < split_date].copy()
        test_df = df[df[date_col] >= split_date].copy()

        for col in feature_cols:
            if col not in df.columns:
                continue

            # 训练集分析
            train_valid = train_df[[col, label_col]].dropna()
            if len(train_valid) < 10:
                continue
            train_corr = train_valid[col].corr(train_valid[label_col])
            train_label1_mean = train_df[train_df[label_col] == 1][col].mean()
            train_label0_mean = train_df[train_df[label_col] == 0][col].mean()

            # 测试集分析
            test_valid = test_df[[col, label_col]].dropna()
            if len(test_valid) < 10:
                continue
            test_corr = test_valid[col].corr(test_valid[label_col])
            test_label1_mean = test_df[test_df[label_col] == 1][col].mean()
            test_label0_mean = test_df[test_df[label_col] == 0][col].mean()

            results.append({
                'feature': col,
                'train_corr': train_corr,
                'test_corr': test_corr,
                'train_label1_mean': train_label1_mean,
                'train_label0_mean': train_label0_mean,
                'test_label1_mean': test_label1_mean,
                'test_label0_mean': test_label0_mean,
                'train_diff': train_label1_mean - train_label0_mean,
                'test_diff': test_label1_mean - test_label0_mean,
                'direction_change': np.sign(train_corr) != np.sign(test_corr) if train_corr != 0 and test_corr != 0 else False
            })

    return pd.DataFrame(results)


def analyze_prediction_direction(y_true, y_prob, y_pred):
    """分析预测方向是否正确"""
    from sklearn.metrics import accuracy_score, roc_auc_score

    # 正常预测
    acc_normal = accuracy_score(y_true, y_pred)
    auc_normal = roc_auc_score(y_true, y_prob)

    # 反转预测
    y_pred_reversed = 1 - y_pred
    y_prob_reversed = 1 - y_prob
    acc_reversed = accuracy_score(y_true, y_pred_reversed)
    auc_reversed = roc_auc_score(y_true, y_prob_reversed)

    print("\n" + "="*60)
    print("预测方向分析")
    print("="*60)
    print(f"正常预测: 准确率={acc_normal:.4f}, AUC={auc_normal:.4f}")
    print(f"反转预测: 准确率={acc_reversed:.4f}, AUC={auc_reversed:.4f}")

    if acc_reversed > acc_normal + 0.02:  # 显著差异阈值
        print("\n⚠️ 警告: 反转预测准确率显著更高！模型可能学到了错误的方向")
        print(f"   准确率差异: +{(acc_reversed - acc_normal)*100:.2f}%")
    elif acc_reversed > acc_normal:
        print("\n⚠️ 注意: 反转预测准确率略高，需进一步验证")
        print(f"   准确率差异: +{(acc_reversed - acc_normal)*100:.2f}%")
    else:
        print("\n✅ 正常预测准确率更高，模型方向正确")

    return {
        'acc_normal': acc_normal,
        'acc_reversed': acc_reversed,
        'auc_normal': auc_normal,
        'auc_reversed': auc_reversed,
        'should_reverse': acc_reversed > acc_normal + 0.02
    }


def check_regime_shift(df, date_col='Date', label_col='Label'):
    """检查市场环境是否发生变化"""
    print("\n" + "="*60)
    print("市场环境变化分析（按年份）")
    print("="*60)

    df['Year'] = pd.to_datetime(df[date_col]).dt.year

    yearly_stats = []
    for year in sorted(df['Year'].unique()):
        year_df = df[df['Year'] == year]
        if len(year_df) < 10:
            continue

        label1_ratio = year_df[label_col].mean()

        # 计算标签收益差异
        if 'Actual_Return' in year_df.columns:
            label1_return = year_df[year_df[label_col] == 1]['Actual_Return'].mean()
            label0_return = year_df[year_df[label_col] == 0]['Actual_Return'].mean()
            return_diff = label1_return - label0_return
        else:
            label1_return = 0
            label0_return = 0
            return_diff = 0

        yearly_stats.append({
            'year': year,
            'samples': len(year_df),
            'label1_ratio': label1_ratio,
            'label1_return': label1_return,
            'label0_return': label0_return,
            'return_diff': return_diff
        })

    yearly_df = pd.DataFrame(yearly_stats)
    print(yearly_df.to_string(index=False))

    return yearly_df


def analyze_top_features_importance(catboost_model, feature_cols, X_train, y_train, X_test, y_test):
    """分析特征重要性及其在训练/测试集上的表现"""

    # 获取特征重要性
    importance = catboost_model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("Top 20 特征重要性及其在训练/测试集上的相关性")
    print("="*60)

    top_features = feature_importance.head(20)['feature'].tolist()

    results = []
    for feat in top_features:
        if feat not in feature_cols:
            continue
        idx = feature_cols.index(feat)

        # 训练集相关性
        train_data = np.column_stack([X_train[:, idx], y_train])
        train_data = train_data[~np.isnan(train_data[:, 0])]
        if len(train_data) > 10:
            train_corr = np.corrcoef(train_data[:, 0], train_data[:, 1])[0, 1]
        else:
            train_corr = 0

        # 测试集相关性
        test_data = np.column_stack([X_test[:, idx], y_test])
        test_data = test_data[~np.isnan(test_data[:, 0])]
        if len(test_data) > 10:
            test_corr = np.corrcoef(test_data[:, 0], test_data[:, 1])[0, 1]
        else:
            test_corr = 0

        results.append({
            'feature': feat,
            'importance': importance[idx],
            'train_corr': train_corr,
            'test_corr': test_corr,
            'direction_change': np.sign(train_corr) != np.sign(test_corr) if not np.isnan(train_corr) and not np.isnan(test_corr) and train_corr != 0 and test_corr != 0 else False
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 统计方向变化
    direction_changes = results_df['direction_change'].sum()
    print(f"\n方向变化的特征数量: {direction_changes}/{len(results_df)}")

    return results_df


def main():
    print("="*60)
    print("模型学习方向诊断")
    print("="*60)

    horizon = 20

    # 初始化模型
    model = CatBoostModel()

    # 加载股票列表
    stocks = list(WATCHLIST.keys())[:15]  # 使用前15只股票进行测试

    print(f"\n使用股票: {stocks}")

    # 准备数据
    print("\n准备数据...")
    df = model.prepare_data(
        codes=stocks,
        horizon=horizon,
        use_feature_cache=False  # 不使用缓存，确保数据最新
    )

    if df is None or df.empty:
        print("没有有效数据")
        return

    print(f"\n合并后数据: {len(df)} 条记录")

    # 检查日期列名
    print(f"数据列: {df.columns.tolist()[:10]}...")
    print(f"索引类型: {type(df.index)}")
    print(f"索引名称: {df.index.name}")

    # 检查是否有日期列
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'date' in df.columns:
        date_col = 'date'
    elif df.index.name in ['Date', 'date', None] and hasattr(df.index, 'to_series'):
        # 使用索引作为日期
        df = df.reset_index()
        date_col = df.columns[0]
        print(f"使用索引作为日期列: {date_col}")
    else:
        # 尝试从索引获取日期
        df = df.reset_index()
        date_col = df.columns[0]

    print(f"日期列: {date_col}")
    print(f"日期列样本: {df[date_col].head(3).tolist()}")

    # 时间分割
    dates = sorted(df[date_col].unique())
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    print(f"训练集截止日期: {split_date}")
    print(f"训练集: {len(df[df[date_col] < split_date])} 条")
    print(f"测试集: {len(df[df[date_col] >= split_date])} 条")

    # 获取特征列
    feature_cols = model.get_feature_columns(df)
    print(f"特征数量: {len(feature_cols)}")

    # 分析关键特征
    key_features = ['Volatility_60d', 'Trend_Slope_60d', 'MA250_Slope', 'PE', 'sentiment_ma14',
                    'RSI_14', 'MACD', 'BB_Position', 'Volume_Ratio_20d', 'US_10Y_Yield',
                    'HSI_Regime_Duration', 'HSI_Regime_Prob_1']

    # 过滤存在的特征
    existing_features = [f for f in key_features if f in feature_cols]
    print(f"\n关键特征（存在的）: {existing_features}")

    print("\n" + "="*60)
    print("关键特征与标签的关系（训练集 vs 测试集）")
    print("="*60)

    relationship_df = analyze_feature_label_relationship(
        df, existing_features, 'Label', split_date, date_col
    )

    if not relationship_df.empty:
        print("\n训练集 vs 测试集 特征-标签关系:")
        print(relationship_df.to_string(index=False))

        # 检查方向变化
        direction_changes = relationship_df[relationship_df['direction_change']]
        if len(direction_changes) > 0:
            print(f"\n⚠️ 发现 {len(direction_changes)} 个特征在训练集和测试集之间方向发生变化:")
            print(direction_changes[['feature', 'train_corr', 'test_corr']].to_string(index=False))

    # 检查市场环境变化
    check_regime_shift(df, date_col)

    # 训练模型并分析预测方向
    print("\n" + "="*60)
    print("训练模型并分析预测方向")
    print("="*60)

    train_df = df[df[date_col] < split_date].copy()
    test_df = df[df[date_col] >= split_date].copy()

    # 只保留数值型特征
    numeric_feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    print(f"数值型特征数量: {len(numeric_feature_cols)}")

    # 处理 NaN
    train_df[numeric_feature_cols] = train_df[numeric_feature_cols].fillna(train_df[numeric_feature_cols].median())
    test_df[numeric_feature_cols] = test_df[numeric_feature_cols].fillna(train_df[numeric_feature_cols].median())

    # 准备训练数据
    X_train = train_df[numeric_feature_cols].values
    y_train = train_df['Label'].values
    X_test = test_df[numeric_feature_cols].values
    y_test = test_df['Label'].values

    # 训练模型
    from catboost import CatBoostClassifier, Pool

    catboost_model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        depth=7,
        learning_rate=0.03,
        n_estimators=600,
        l2_leaf_reg=2,
        subsample=0.75,
        colsample_bylevel=0.75,
        random_seed=2020,
        verbose=False,
        early_stopping_rounds=80,
        allow_writing_files=False
    )

    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_test, label=y_test)

    catboost_model.fit(train_pool, eval_set=val_pool, verbose=False)

    # 预测
    y_prob = catboost_model.predict_proba(X_test)[:, 1]
    y_pred = catboost_model.predict(X_test)

    # 分析预测方向
    direction_result = analyze_prediction_direction(y_test, y_prob, y_pred)

    # 分析特征重要性
    importance_df = analyze_top_features_importance(
        catboost_model, numeric_feature_cols, X_train, y_train, X_test, y_test
    )

    # 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)

    if direction_result['should_reverse']:
        print("\n🔴 问题确认: 模型学到了与特征-标签关系相反的方向")
        print("\n可能原因:")
        print("1. 训练集和测试集的市场环境不同（Regime Shift）")
        print("2. 某些高权重特征的方向在训练集和测试集之间发生了变化")
        print("3. 类别不平衡导致模型偏向某一类")

        print("\n建议解决方案:")
        print("1. 缩短训练窗口，让模型学习近期的市场逻辑")
        print("2. 使用单调约束（monotone_constraints）强制特征方向")
        print("3. 增加样本权重，让近期样本权重更高")
        print("4. 不要简单反转预测概率 - 这会破坏正确学习的特征")
    else:
        print("\n✅ 模型方向正确，无需反转预测")
        print("\n当前模型表现:")
        print(f"  准确率: {direction_result['acc_normal']:.4f}")
        print(f"  AUC: {direction_result['auc_normal']:.4f}")

    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    main()