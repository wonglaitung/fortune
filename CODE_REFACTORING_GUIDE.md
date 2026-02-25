# 代码重复问题解决 - 修改指南

## 已完成的修改

### ✅ 1. 创建了 BaseTradingModel 基类
- **位置**: `ml_services/ml_trading_model.py` 第 1553 行
- **内容**: 包含公共方法
  - `load_selected_features()` - 加载选择的特征列表
  - `get_feature_columns()` - 获取特征列（排除中间计算列）

### ✅ 2. 统一了特征列排除方式
- `CatBoostModel` 现在有 `get_feature_columns()` 方法
- `CatBoostModel.train()` 现在使用 `get_feature_columns()` 方法
- 三个模型现在都排除相同的 28 个列

## 需要完成的修改

### 第一步：删除 LightGBMModel 中的重复方法

**当前状态**: `LightGBMModel` 已经继承了 `BaseTradingModel`，但仍有重复方法

**需要删除**:
1. 第 1642-1691 行：`load_selected_features()` 方法（50行）
2. 第 1834-1850 行：`get_feature_columns()` 方法（17行）

**修改方式**:
```python
# 删除这两段重复代码，因为父类 BaseTradingModel 已经提供了
```

### 第二步：修改 GBDTModel 继承基类

**当前状态**: `GBDTModel` 不继承任何基类

**需要修改**:
```python
# 修改前（第 2229 行）:
class GBDTModel:

# 修改后:
class GBDTModel(BaseTradingModel):
```

**需要删除**:
1. 第 2241-2290 行：`load_selected_features()` 方法（50行）
2. 第 2413-2429 行：`get_feature_columns()` 方法（17行）

### 第三步：修改 CatBoostModel 继承基类

**当前状态**: `CatBoostModel` 不继承任何基类

**需要修改**:
```python
# 修改前（第 2840 行）:
class CatBoostModel:

# 修改后:
class CatBoostModel(BaseTradingModel):
```

**需要删除**:
1. 第 2859-2908 行：`load_selected_features()` 方法（50行）
2. 第 2960-2976 行：`get_feature_columns()` 方法（17行）

## 预期效果

### 删除的代码量
- **LightGBMModel**: 67 行
- **GBDTModel**: 67 行
- **CatBoostModel**: 67 行
- **总计**: 201 行重复代码

### 改进后的架构
```
BaseTradingModel (基类)
├── load_selected_features()  # 公共方法
└── get_feature_columns()    # 公共方法

LightGBMModel(BaseTradingModel)  # 继承基类
├── __init__()
├── prepare_data()  # 特定实现（并行下载）
├── train()         # 特定实现
├── predict()       # 特定实现
└── save_model()

GBDTModel(BaseTradingModel)      # 继承基类
├── __init__()
├── prepare_data()  # 特定实现（串行下载）
├── train()         # 特定实现
├── predict()       # 特定实现
└── save_model()

CatBoostModel(BaseTradingModel)  # 继承基类
├── __init__()
├── prepare_data()  # 特定实现（串行下载，不同实现）
├── train()         # 特定实现
├── predict()       # 特定实现
├── save_model()
└── predict_proba()
```

## 验证步骤

### 1. 编译检查
```bash
python3 -m py_compile ml_services/ml_trading_model.py
```

### 2. 运行特征列统一测试
```python
from ml_services.ml_trading_model import LightGBMModel, GBDTModel, CatBoostModel
import pandas as pd

# 创建测试数据并验证三个模型的特征列一致
```

### 3. 测试模型实例化
```python
lgbm = LightGBMModel()
gbdt = GBDTModel()
catboost = CatBoostModel()

# 验证它们都有 get_feature_columns 方法
assert hasattr(lgbm, 'get_feature_columns')
assert hasattr(gbdt, 'get_feature_columns')
assert hasattr(catboost, 'get_feature_columns')

# 验证它们都有 load_selected_features 方法
assert hasattr(lgbm, 'load_selected_features')
assert hasattr(gbdt, 'load_selected_features')
assert hasattr(catboost, 'load_selected_features')
```

### 4. 测试继承关系
```python
from ml_services.ml_trading_model import BaseTradingModel

lgbm = LightGBMModel()
gbdt = GBDTModel()
catboost = CatBoostModel()

# 验证继承关系
assert isinstance(lgbm, BaseTradingModel)
assert isinstance(gbdt, BaseTradingModel)
assert isinstance(catboost, BaseTradingModel)
```

## 注意事项

### 1. prepare_data 方法
- 三个模型的 `prepare_data()` 方法仍然保持各自的实现
- `LightGBMModel` 使用并行下载
- `GBDTModel` 和 `CatBoostModel` 使用串行下载
- 这个方法的差异是设计上的，不需要统一

### 2. 向后兼容性
- 所有公共方法的签名保持不变
- 外部调用代码不需要修改
- 类方法的行为保持一致

### 3. 测试覆盖
- 需要确保所有模型的功能测试通过
- 特别是特征选择和特征列排除的逻辑
- 需要验证模型训练和预测功能正常

## 执行计划

### 阶段1：删除 LightGBMModel 重复方法
- [ ] 删除第 1642-1691 行（load_selected_features）
- [ ] 删除第 1834-1850 行（get_feature_columns）
- [ ] 编译检查
- [ ] 测试功能

### 阶段2：修改 GBDTModel
- [ ] 修改类定义继承 BaseTradingModel
- [ ] 删除第 2241-2290 行（load_selected_features）
- [ ] 删除第 2413-2429 行（get_feature_columns）
- [ ] 编译检查
- [ ] 测试功能

### 阶段3：修改 CatBoostModel
- [ ] 修改类定义继承 BaseTradingModel
- [ ] 删除第 2859-2908 行（load_selected_features）
- [ ] 删除第 2960-2976 行（get_feature_columns）
- [ ] 编译检查
- [ ] 测试功能

### 阶段4：完整测试
- [ ] 运行所有模型类测试
- [ ] 验证特征列统一
- [ ] 验证继承关系
- [ ] 更新文档

## 预期结果

完成所有修改后：
- ✅ 消除 201 行重复代码
- ✅ 三个模型类继承统一的基类
- ✅ 特征列排除方式完全一致
- ✅ 代码可维护性显著提升
- ✅ 符合 DRY（Don't Repeat Yourself）原则

---

**创建日期**: 2026-02-25
**状态**: 待执行
