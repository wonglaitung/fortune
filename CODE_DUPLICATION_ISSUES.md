# 模型类代码重复问题分析

## 问题概述

检查发现 `ml_trading_model.py` 中存在大量代码重复和逻辑混入问题。

## 已修复的问题

### ✅ LightGBMModel 中的 GBDT 逻辑（已修复）

**位置**: `LightGBMModel.train()` 方法，第 1815-1823 行

**问题描述**:
- `LightGBMModel` 中包含了 GBDT 的特征选择逻辑
- 由于 `LightGBMModel.model_type` 始终是 `'lgbm'`，这部分代码永远不会执行
- 违反了单一职责原则

**修复**:
```python
# 修复前（错误）：
elif use_feature_selection and self.model_type == 'gbdt':
    print("\n🎯 应用特征选择（GBDT）...")
    ...

# 修复后（正确）：
# 已删除这段代码，因为这是 GBDT 的逻辑
```

## 待修复的问题

### ❌ 代码重复：`load_selected_features` 方法

**重复次数**: 3 次
**位置**:
- `LightGBMModel.load_selected_features()` - 第 1565 行
- `GBDTModel.load_selected_features()` - 第 2164 行
- `CatBoostModel.load_selected_features()` - 第 2782 行

**代码量**: 约 67 行

**问题描述**:
- 三个类中的 `load_selected_features` 方法完全相同
- 造成代码冗余，维护困难
- 修改时需要同时修改三个地方

**建议解决方案**: 创建一个基类或混入类（Mixin）

---

### ❌ 代码重复：`prepare_data` 方法

**重复次数**: 3 次
**位置**:
- `LightGBMModel.prepare_data()` - 第 1616 行
- `GBDTModel.prepare_data()` - 第 2215 行
- `CatBoostModel.prepare_data()` - 第 2833 行

**代码量**: 约 140+ 行

**问题描述**:
- 三个类中的 `prepare_data` 方法逻辑高度相似
- 只有部分细节不同（例如并行下载 vs 串行下载）
- 大量重复的特征计算代码

**建议解决方案**:
1. 提取公共逻辑到基类
2. 将不同的行为通过模板方法模式处理

---

### ❌ 代码重复：`get_feature_columns` 方法

**重复次数**: 2 次
**位置**:
- `LightGBMModel.get_feature_columns()` - 第 1758 行
- `GBDTModel.get_feature_columns()` - 第 2337 行

**代码量**: 约 15 行

**问题描述**:
- `LightGBMModel` 和 `GBDTModel` 中的 `get_feature_columns` 方法完全相同
- 用于排除 20 个中间计算列（如 `Open`, `High`, `Low`, `MA5`, `RSI` 等）
- **`CatBoostModel` 没有这个方法**（见下文）

**建议解决方案**: 移动到基类中

---

### ⚠️ 一致性问题：特征列排除方式不同

**问题描述**:
三个模型类在特征列排除上存在**不一致**：

| 模型 | `get_feature_columns` | 排除的列数 | 排除方式 |
|------|---------------------|----------|---------|
| **LightGBMModel** | ✅ 有 | 20+ | 调用 `get_feature_columns(df)` 方法 |
| **GBDTModel** | ✅ 有 | 20+ | 调用 `get_feature_columns(df)` 方法 |
| **CatBoostModel** | ❌ 没有 | 3 | 内联在 `train()` 方法中 |

**具体差异**:

1. **LightGBMModel / GBDTModel** 排除：
   ```python
   exclude_columns = ['Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                     'Future_Return', 'Label', 'Prev_Close',
                     'Vol_MA20', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
                     'BB_upper', 'BB_lower', 'BB_middle',
                     'Low_Min', 'High_Max', '+DM', '-DM', '+DI', '-DI',
                     'TP', 'MF_Multiplier', 'MF_Volume']
   ```

2. **CatBoostModel** 只排除：
   ```python
   exclude_columns = ['Code', 'Label', 'Future_Return']
   ```

**潜在风险**:
- ⚠️ `CatBoostModel` 可能使用了中间计算列作为特征（如 `Open`, `High`, `Low`, `MA5`, `RSI` 等）
- ⚠️ 如果这些列在特征文件中，`CatBoostModel` 会使用它们，而其他两个模型不会
- ⚠️ 可能导致 `CatBoostModel` 的特征输入与其他模型不一致

**建议解决方案**:
1. 统一三个模型类的特征列排除方式
2. 将 `get_feature_columns` 方法移动到基类
3. 确保所有模型类使用相同的特征集

---

## 建议的架构改进方案

### 方案 1：创建基类 `BaseTradingModel`

```python
class BaseTradingModel:
    """交易模型基类"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.processor = BaseModelProcessor()
        self.feature_columns = []
        self.horizon = 1
        self.model_type = None  # 子类需要设置
        self.categorical_encoders = {}

    def load_selected_features(self, filepath=None, current_feature_names=None):
        """加载选择的特征列表（公共方法）"""
        # 实现代码...

    def get_feature_columns(self, df):
        """获取特征列（公共方法）"""
        # 实现代码...

    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False):
        """准备数据（公共方法，可被子类重写）"""
        # 实现代码...

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        """训练模型（抽象方法，子类必须实现）"""
        raise NotImplementedError("子类必须实现 train() 方法")

    def predict(self, code, predict_date=None, horizon=None):
        """预测（抽象方法，子类必须实现）"""
        raise NotImplementedError("子类必须实现 predict() method")

    def save_model(self, filepath):
        """保存模型（抽象方法）"""
        raise NotImplementedError("子类必须实现 save_model() method")

    def load_model(self, filepath):
        """加载模型（抽象方法）"""
        raise NotImplementedError("子类必须实现 load_model() method")


class LightGBMModel(BaseTradingModel):
    """LightGBM 模型"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = 'lgbm'

    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        # LightGBM 特定的训练逻辑
        pass

    # predict, save_model, load_model 等方法...


class GBDTModel(BaseTradingModel):
    """GBDT 模型"""

    def __init__(self):
        super().__init__()
        self.gbdt_model = None
        self.model_type = 'gbdt'

    # 类似实现...


class CatBoostModel(BaseTradingModel):
    """CatBoost 模型"""

    def __init__(self):
        super().__init__()
        self.catboost_model = None
        self.model_type = 'catboost'

    # 类似实现...
```

**优点**:
- ✅ 消除代码重复
- ✅ 统一接口
- ✅ 易于维护和扩展
- ✅ 符合 DRY（Don't Repeat Yourself）原则
- ✅ 符合开闭原则（对扩展开放，对修改关闭）

**缺点**:
- ⚠️ 需要大量重构工作
- ⚠️ 可能引入新的 bug（需要充分测试）

---

### 方案 2：使用 Mixin 类

如果 `prepare_data` 的差异较大，可以使用 Mixin 模式：

```python
class FeatureSelectionMixin:
    """特征选择混入类"""
    def load_selected_features(self, filepath=None, current_feature_names=None):
        # 实现...

class DataPreparationMixin:
    """数据准备混入类"""
    def prepare_data(self, codes, start_date=None, end_date=None, horizon=1, for_backtest=False):
        # 实现...

class LightGBMModel(FeatureSelectionMixin, DataPreparationMixin):
    """LightGBM 模型"""
    # 只实现 LightGBM 特有的逻辑...
```

---

## 影响范围分析

### 当前受影响的文件
- ✅ `ml_services/ml_trading_model.py` (已修复部分)
- ⚠️ `ml_services/batch_backtest.py` (使用了重复的方法)
- ⚠️ `ml_services/backtest_evaluator.py` (使用了重复的方法)

### 向后兼容性
- ✅ 类名重命名已有别名支持
- ⚠️ 方法重构可能影响现有代码（如果直接访问了这些方法）

---

## 建议的实施步骤

1. **创建基类** `BaseTradingModel`
2. **提取公共方法**到基类
3. **修改子类**继承基类
4. **移除重复代码**
5. **运行完整测试**确保功能正常
6. **更新文档**说明新的架构

---

## 总结

- ✅ **已修复**: LightGBMModel 中的 GBDT 逻辑混入
- ❌ **待修复**: 大量代码重复（约 300+ 行重复代码）
- ⚠️ **建议**: 创建基类来消除重复，提高代码可维护性

---

**创建日期**: 2026-02-25
**状态**: 部分修复完成
