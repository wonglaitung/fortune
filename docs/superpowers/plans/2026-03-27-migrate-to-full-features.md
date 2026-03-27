# 迁移到全量特征（892个）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目的默认特征策略从"500个精选特征"改为"使用全量特征（892个）"，保持向后兼容性，简化训练流程

**Architecture:** 采用渐进式迁移策略，分4个阶段逐步实施：阶段1更新核心训练脚本，阶段2更新训练相关脚本，阶段3更新自动化脚本，阶段4更新文档。保留 `--use-feature-selection` 参数但默认行为改为使用全量特征，显式指定时显示弃用警告。

**Tech Stack:** Python 3.10+, CatBoost, argparse, Git

---

## 文件结构

**阶段1 - 核心训练脚本**（1个文件）：
- `ml_services/ml_trading_model.py` - 修改3个模型类的train()方法，添加弃用警告

**阶段2 - 训练相关脚本**（6个文件）：
- `ml_services/walk_forward_validation.py` - 移除特征选择参数
- `ml_services/walk_forward_by_sector.py` - 移除特征选择参数
- `ml_services/train_sector_model.py` - 移除特征选择参数
- `ml_services/batch_backtest.py` - 移除特征选择参数
- `ml_services/backtest_20d_horizon.py` - 移除特征选择参数
- `ml_services/compare_three_models_20d.py` - 移除特征选择参数

**阶段3 - 自动化脚本**（2个文件）：
- `scripts/run_comprehensive_analysis.sh` - 移除特征选择步骤，简化训练命令
- `scripts/run_model_comparison.sh` - 简化训练命令

**阶段4 - 文档**（3个文件）：
- `AGENTS.md` - 更新特征选择方法章节、训练命令、特征工程章节
- `lessons.md` - 添加全量特征优于500特征的经验章节
- `progress.txt` - 记录迁移过程和验证结果

---

## 阶段1：更新核心训练脚本

### Task 1: 修改 LightGBMModel.train() 方法

**Files:**
- Modify: `ml_services/ml_trading_model.py` (约第2291行)

- [ ] **Step 1: 读取 LightGBMModel.train() 方法**

运行: `sed -n '2291,2350p' ml_services/ml_trading_model.py`
Expected: 显示train()方法的完整定义

- [ ] **Step 2: 修改参数默认值和文档**

找到 `use_feature_selection` 参数定义，将默认值改为 `False`，更新文档字符串

```python
def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
    """
    训练LightGBM模型（默认使用全量特征892个）
    
    Args:
        codes: 股票代码列表
        start_date: 训练开始日期
        end_date: 训练结束日期
        horizon: 预测周期（天数）
        use_feature_selection: 是否使用特征选择（已弃用，默认False使用全量特征）
    
    Returns:
        特征重要性
    """
```

- [ ] **Step 3: 添加弃用警告和类变量**

在 `LightGBMModel` 类开头添加类变量，在 `train()` 方法开始处添加警告逻辑

```python
class LightGBMModel:
    _deprecation_warning_shown = False  # 类变量，控制警告只显示一次
    
    def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
        # 检查是否需要显示弃用警告
        if use_feature_selection and not LightGBMModel._deprecation_warning_shown:
            print("⚠️  警告：特征选择功能已弃用，建议使用全量特征（892个）。Walk-forward验证显示全量特征性能更好。")
            LightGBMModel._deprecation_warning_shown = True
        
        # ... 其余代码 ...
```

- [ ] **Step 4: 更新日志输出**

修改日志输出，明确显示使用的特征数量

```python
# 应用特征选择（可选）
if use_feature_selection:
    print("\n🎯 应用特征选择（LightGBM）...（已弃用）")
    selected_features = self.load_selected_features(current_feature_names=self.feature_columns)
    if selected_features:
        self.feature_columns = [col for col in self.feature_columns if col in selected_features]
        print(f"✅ 特征数量: {len(self.feature_columns)}（特征选择 - 已弃用）")
else:
    print(f"\n✅ 特征数量: {len(self.feature_columns)}（全量特征）")
```

- [ ] **Step 5: 验证修改**

运行: `python3 -m py_compile ml_services/ml_trading_model.py`
Expected: 无语法错误

- [ ] **Step 6: 提交阶段1-LightGBM**

```bash
git add ml_services/ml_trading_model.py
git commit -m "feat(migration/phase1): 修改LightGBMModel.train()方法

- 修改use_feature_selection默认值为False
- 添加弃用警告（类变量控制只显示一次）
- 更新日志输出显示特征数量"
```

---

### Task 2: 修改 GBDTModel.train() 方法

**Files:**
- Modify: `ml_services/ml_trading_model.py` (约第2911行)

- [ ] **Step 1: 读取 GBDTModel.train() 方法**

运行: `sed -n '2911,2970p' ml_services/ml_trading_model.py`
Expected: 显示train()方法的完整定义

- [ ] **Step 2: 修改参数默认值和文档**

将 `use_feature_selection` 默认值改为 `False`，更新文档字符串（同Task 1）

- [ ] **Step 3: 添加弃用警告和类变量**

在 `GBDTModel` 类开头添加类变量，在 `train()` 方法开始处添加警告逻辑（同Task 1）

- [ ] **Step 4: 更新日志输出**

修改日志输出，明确显示使用的特征数量（同Task 1）

- [ ] **Step 5: 验证修改**

运行: `python3 -m py_compile ml_services/ml_trading_model.py`
Expected: 无语法错误

- [ ] **Step 6: 提交阶段1-GBDT**

```bash
git add ml_services/ml_trading_model.py
git commit -m "feat(migration/phase1): 修改GBDTModel.train()方法

- 修改use_feature_selection默认值为False
- 添加弃用警告（类变量控制只显示一次）
- 更新日志输出显示特征数量"
```

---

### Task 3: 修改 CatBoostModel.train() 方法

**Files:**
- Modify: `ml_services/ml_trading_model.py` (约第3599行)

- [ ] **Step 1: 读取 CatBoostModel.train() 方法**

运行: `sed -n '3599,3658p' ml_services/ml_trading_model.py`
Expected: 显示train()方法的完整定义

- [ ] **Step 2: 修改参数默认值和文档**

将 `use_feature_selection` 默认值改为 `False`，更新文档字符串（同Task 1）

- [ ] **Step 3: 添加弃用警告和类变量**

在 `CatBoostModel` 类开头添加类变量，在 `train()` 方法开始处添加警告逻辑（同Task 1）

- [ ] **Step 4: 更新日志输出**

修改日志输出，明确显示使用的特征数量（同Task 1）

- [ ] **Step 5: 验证修改**

运行: `python3 -m py_compile ml_services/ml_trading_model.py`
Expected: 无语法错误

- [ ] **Step 6: 测试默认行为（使用全量特征）**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost`
Expected: 日志显示"特征数量: 892（全量特征）"

- [ ] **Step 7: 测试弃用警告**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection`
Expected: 显示弃用警告"⚠️ 警告：特征选择功能已弃用..."

- [ ] **Step 8: 提交阶段1-CatBoost**

```bash
git add ml_services/ml_trading_model.py
git commit -m "feat(migration/phase1): 修改CatBoostModel.train()方法

- 修改use_feature_selection默认值为False
- 添加弃用警告（类变量控制只显示一次）
- 更新日志输出显示特征数量
- 验证默认行为和弃用警告正常工作"
```

---

## 阶段2：更新训练相关脚本

### Task 4: 修改 walk_forward_validation.py

**Files:**
- Modify: `ml_services/walk_forward_validation.py`

- [ ] **Step 1: 读取参数定义部分**

运行: `grep -n "add_argument.*feature" ml_services/walk_forward_validation.py`
Expected: 找到特征选择相关的参数定义

- [ ] **Step 2: 移除特征选择参数**

删除以下参数：
- `--use-feature-selection`
- `--skip-feature-selection`

删除示例：
```python
# 删除这两行
parser.add_argument('--use-feature-selection', action='store_true',
                   help='使用特征选择')
parser.add_argument('--skip-feature-selection', action='store_true',
                   help='跳过特征选择，直接使用已有的特征文件')
```

- [ ] **Step 3: 修改模型训练调用**

移除 `use_feature_selection` 参数传递：

```python
# 修改前
model.train(stock_codes, horizon=args.horizon, use_feature_selection=args.use_feature_selection)

# 修改后
model.train(stock_codes, horizon=args.horizon)
```

- [ ] **Step 4: 更新帮助文档**

确保 `--help` 输出不包含特征选择参数说明

- [ ] **Step 5: 验证修改**

运行: `python3 ml_services/walk_forward_validation.py --help | grep -i feature`
Expected: 不输出任何内容（已移除特征选择参数）

- [ ] **Step 6: 提交阶段2-walk_forward_validation**

```bash
git add ml_services/walk_forward_validation.py
git commit -m "feat(migration/phase2): 移除walk_forward_validation.py的特征选择参数

- 删除--use-feature-selection和--skip-feature-selection参数
- 移除相关逻辑和变量
- 更新帮助文档"
```

---

### Task 5: 修改 walk_forward_by_sector.py

**Files:**
- Modify: `ml_services/walk_forward_by_sector.py`

- [ ] **Step 1: 读取参数定义部分**

运行: `grep -n "add_argument.*feature" ml_services/walk_forward_by_sector.py`
Expected: 找到特征选择相关的参数定义

- [ ] **Step 2: 移除特征选择参数**

删除特征选择参数（同Task 4）

- [ ] **Step 3: 修改模型训练调用**

移除 `use_feature_selection` 参数传递（同Task 4）

- [ ] **Step 4: 验证修改**

运行: `python3 ml_services/walk_forward_by_sector.py --help | grep -i feature`
Expected: 不输出任何内容

- [ ] **Step 5: 提交阶段2-walk_forward_by_sector**

```bash
git add ml_services/walk_forward_by_sector.py
git commit -m "feat(migration/phase2): 移除walk_forward_by_sector.py的特征选择参数

- 删除--use-feature-selection和--skip-feature-selection参数
- 移除相关逻辑和变量"
```

---

### Task 6: 修改 train_sector_model.py

**Files:**
- Modify: `ml_services/train_sector_model.py`

- [ ] **Step 1: 读取参数定义部分**

运行: `grep -n "add_argument.*feature" ml_services/train_sector_model.py`
Expected: 找到特征选择相关的参数定义

- [ ] **Step 2: 移除特征选择参数**

删除特征选择参数（同Task 4）

- [ ] **Step 3: 修改模型训练调用**

移除 `use_feature_selection` 参数传递（同Task 4）

- [ ] **Step 4: 验证修改**

运行: `python3 ml_services/train_sector_model.py --help | grep -i feature`
Expected: 不输出任何内容

- [ ] **Step 5: 提交阶段2-train_sector_model**

```bash
git add ml_services/train_sector_model.py
git commit -m "feat(migration/phase2): 移除train_sector_model.py的特征选择参数

- 删除--use-feature-selection和--skip-feature-selection参数
- 移除相关逻辑和变量"
```

---

### Task 7: 修改 batch_backtest.py

**Files:**
- Modify: `ml_services/batch_backtest.py`

- [ ] **Step 1: 读取参数定义部分**

运行: `grep -n "add_argument.*feature" ml_services/batch_backtest.py`
Expected: 找到特征选择相关的参数定义

- [ ] **Step 2: 移除特征选择参数**

删除特征选择参数（同Task 4）

- [ ] **Step 3: 修改模型训练调用**

移除 `use_feature_selection` 参数传递（同Task 4）

- [ ] **Step 4: 验证修改**

运行: `python3 ml_services/batch_backtest.py --help | grep -i feature`
Expected: 不输出任何内容

- [ ] **Step 5: 提交阶段2-batch_backtest**

```bash
git add ml_services/batch_backtest.py
git commit -m "feat(migration/phase2): 移除batch_backtest.py的特征选择参数

- 删除--use-feature-selection和--skip-feature-selection参数
- 移除相关逻辑和变量"
```

---

### Task 8: 修改 backtest_20d_horizon.py

**Files:**
- Modify: `ml_services/backtest_20d_horizon.py`

- [ ] **Step 1: 读取参数定义部分**

运行: `grep -n "add_argument.*feature" ml_services/backtest_20d_horizon.py`
Expected: 找到特征选择相关的参数定义

- [ ] **Step 2: 移除特征选择参数**

删除特征选择参数（同Task 4）

- [ ] **Step 3: 修改模型训练调用**

移除 `use_feature_selection` 参数传递（同Task 4）

- [ ] **Step 4: 验证修改**

运行: `python3 ml_services/backtest_20d_horizon.py --help | grep -i feature`
Expected: 不输出任何内容

- [ ] **Step 5: 提交阶段2-backtest_20d_horizon**

```bash
git add ml_services/backtest_20d_horizon.py
git commit -m "feat(migration/phase2): 移除backtest_20d_horizon.py的特征选择参数

- 删除--use-feature-selection和--skip-feature-selection参数
- 移除相关逻辑和变量"
```

---

### Task 9: 修改 compare_three_models_20d.py

**说明**：该文件不使用argparse，没有命令行参数。
但train_model()函数签名（第51行）中有use_feature_selection参数，该参数在函数体内未被使用。
建议移除该参数或添加注释说明。

**Files:**
- Modify: `ml_services/compare_three_models_20d.py` (约第51行)

- [ ] **Step 1: 读取函数签名**

运行: `sed -n '51,60p' ml_services/compare_three_models_20d.py`
Expected: 显示train_model函数定义

- [ ] **Step 2: 移除未使用的参数**

将函数签名中的 `use_feature_selection=True` 参数移除：

```python
# 修改前
def train_model(model_type, horizon=20, use_feature_selection=True):

# 修改后
def train_model(model_type, horizon=20):
```

同时更新文档字符串，移除该参数的说明。

- [ ] **Step 3: 验证修改**

运行: `python3 -m py_compile ml_services/compare_three_models_20d.py`
Expected: 无语法错误

- [ ] **Step 4: 提交阶段2-compare_three_models_20d**

```bash
git add ml_services/compare_three_models_20d.py
git commit -m "feat(migration/phase2): 移除compare_three_models_20d.py中未使用的参数

- 移除train_model()函数中未使用的use_feature_selection参数
- 更新函数文档字符串"
```

---

## 阶段3：更新自动化脚本

### Task 10: 修改 run_comprehensive_analysis.sh

**Files:**
- Modify: `scripts/run_comprehensive_analysis.sh`

- [ ] **Step 1: 读取脚本内容**

运行: `cat scripts/run_comprehensive_analysis.sh`
Expected: 显示完整脚本内容

- [ ] **Step 2: 移除步骤0（特征选择步骤）**

删除以下部分：

```bash
# 删除整个步骤0
# 步骤0: 运行特征选择脚本，生成500个精选特征（使用F-test方法）
echo "=========================================="
echo "📊 步骤 0/5: 运行特征选择（使用statistical方法）"
echo "=========================================="
echo ""
python3 ml_services/feature_selection.py --method statistical --top-k 500 --horizon 20 --output-dir output
```

- [ ] **Step 3: 更新步骤1（训练命令）**

修改训练命令，移除特征选择参数：

```bash
# 修改前
echo "=========================================="
echo "📊 步骤 1/5: 训练 CatBoost 20天模型"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 修改后
echo "=========================================="
echo "📊 步骤 1/4: 训练 CatBoost 20天模型（全量特征892个）"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

- [ ] **Step 4: 重新编号步骤**

将后续步骤编号减1（步骤2→步骤1，步骤3→步骤2，等等）

- [ ] **Step 5: 更新输出文件列表**

移除特征选择结果相关的输出文件说明

- [ ] **Step 6: 验证修改**

运行: `bash -n scripts/run_comprehensive_analysis.sh`
Expected: 无语法错误

- [ ] **Step 7: 提交阶段3-run_comprehensive_analysis**

```bash
git add scripts/run_comprehensive_analysis.sh
git commit -m "feat(migration/phase3): 简化run_comprehensive_analysis.sh

- 移除步骤0（特征选择步骤）
- 简化训练命令（移除--use-feature-selection参数）
- 重新编号步骤"
```

---

### Task 11: 修改 run_model_comparison.sh

**Files:**
- Modify: `scripts/run_model_comparison.sh`

- [ ] **Step 1: 读取脚本内容**

运行: `cat scripts/run_model_comparison.sh`
Expected: 显示完整脚本内容

- [ ] **Step 2: 查找并移除特征选择相关步骤**

搜索并删除特征选择步骤（如果有）

- [ ] **Step 3: 简化训练命令**

修改所有训练命令，移除 `--use-feature-selection` 和 `--skip-feature-selection` 参数

- [ ] **Step 4: 验证修改**

运行: `bash -n scripts/run_model_comparison.sh`
Expected: 无语法错误

- [ ] **Step 5: 提交阶段3-run_model_comparison**

```bash
git add scripts/run_model_comparison.sh
git commit -m "feat(migration/phase3): 简化run_model_comparison.sh

- 移除特征选择相关步骤
- 简化训练命令（移除--use-feature-selection参数）"
```

---

## 阶段4：更新文档

### Task 12: 更新 AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 读取特征选择方法章节**

运行: `grep -n "特征选择方法" AGENTS.md | head -5`
Expected: 找到特征选择方法章节的位置

- [ ] **Step 2: 更新推荐方法**

修改推荐方法为"使用全量特征（892个）"：

```markdown
### 特征选择方法
- **推荐方法**：使用全量特征（892个）⭐
- **已弃用方法**：统计方法（500个精选特征）- 保留但不推荐使用
```

- [ ] **Step 3: 更新训练命令示例**

将所有训练命令更新为不包含特征选择参数：

```bash
# 更新前
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection

# 更新后
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

（需要在AGENTS.md中找到所有训练命令并更新）

- [ ] **Step 4: 更新特征工程章节**

修改特征数量说明：

```markdown
- **特征数量**：892个（全量特征）⭐ **业界标准**
```

- [ ] **Step 5: 添加全量特征 vs 500特征对比验证章节**

在适当位置添加新章节：

```markdown
### 全量特征 vs 500特征对比验证（2026-03-27）

**验证方法**：Walk-forward验证（业界标准）
- 验证对象：银行股板块（6只股票）
- 验证周期：2024-01-01 至 2025-12-31
- 验证参数：12个月训练窗口，1个月测试窗口，1个月滚动步长

**性能对比**：
| 指标 | 全量特征（892个） | 500特征 | 改进幅度 |
|------|------------------|---------|---------|
| 年化收益率 | 40.42% | 30.28% | **+10.14%** |
| 索提诺比率 | 1.9023 | 1.0400 | **+83%** |

**关键发现**：
- CatBoost的自动特征选择机制优于预选择
- 信息保留完整，避免特征选择导致的信息丢失
- 特征组合（520个交叉特征）可能包含重要非线性关系
```

- [ ] **Step 6: 标记特征选择为已弃用**

在相关章节添加"已弃用但保留"标记

- [ ] **Step 7: 验证修改**

运行: `python3 -m markdown AGENTS.md 2>&1 | head -20`
Expected: 无严重错误

- [ ] **Step 8: 提交阶段4-AGENTS**

```bash
git add AGENTS.md
git commit -m "docs(migration/phase4): 更新AGENTS.md特征选择方法章节

- 推荐方法改为全量特征（892个）
- 标记500特征为已弃用方法
- 更新所有训练命令示例
- 添加全量特征vs 500特征对比验证章节"
```

---

### Task 13: 更新 lessons.md

**Files:**
- Modify: `lessons.md`

- [ ] **Step 1: 读取lessons.md内容**

运行: `wc -l lessons.md`
Expected: 显示文件行数

- [ ] **Step 2: 添加全量特征优于500特征章节**

在文件末尾添加新章节：

```markdown
### 全量特征（892个）优于500特征（2026-03-27验证）

**验证背景**：
- 验证方法：Walk-forward验证（业界标准）
- 验证对象：银行股板块（6只股票）
- 验证周期：2024-01-01 至 2025-12-31
- 验证参数：12个月训练窗口，1个月测试窗口，1个月滚动步长

**验证结果**：
- 全量特征（892个）：年化收益率 40.42%，索提诺比率 1.9023
- 500特征：年化收益率 30.28%，索提诺比率 1.0400
- 改进幅度：年化收益率+10.14%，索提诺比率+83%

**关键发现**：
1. CatBoost的自动特征选择机制优于预选择
2. 信息保留完整，避免特征选择导致的信息丢失
3. 特征组合（520个交叉特征）可能包含重要非线性关系
4. Walk-forward验证是唯一可信的评估方法

**经验教训**：
- ✅ 全量特征策略优于特征选择策略（CatBoost场景）
- ✅ Walk-forward验证是评估真实预测能力的唯一方法
- ✅ 避免过度特征工程，让模型自己学习
- ⚠️ 特征数量应在合理范围内（890-895个），避免过大
```

- [ ] **Step 3: 标记旧经验为已过时**

找到"固定500特征是最优方案"相关章节，标记为"已过时，仅供参考"

- [ ] **Step 4: 添加总结章节**

在适当位置添加总结章节：

```markdown
### 全量特征 vs 500特征验证的重要影响（2026-03-27）

**验证方法**：
- Walk-forward验证（业界标准）
- 12个月训练窗口，1个月测试窗口，1个月滚动步长
- 6只银行股，2024-01-01至2025-12-31

**关键洞察**：
1. **CatBoost自动特征选择机制优于预选择**
   - CatBoost内置的Ordered Boosting和特征重要性评估更有效
   - 预选择可能丢失重要的特征组合信息

2. **信息保留的完整性至关重要**
   - 500特征策略丢失了约44%的特征信息（892→500）
   - 特别是520个交叉特征中可能包含重要的非线性关系

3. **性能提升显著**
   - 年化收益率+10.14%（40.42% vs 30.28%）
   - 索提诺比率+83%（1.9023 vs 1.0400）
   - 这是一个重大的性能提升

**业界实践启示**：
- Bloomberg、Two Sigma、Renaissance Technologies等顶级量化机构通常使用500-1000个特征
- CatBoost等现代GBDT框架的自动特征选择机制已经非常成熟
- 预选择在深度学习场景可能有用，但在GBDT场景下效果有限

**未来方向**：
- 保持全量特征策略
- 监控CatBoost的特征重要性，识别关键特征
- 探索动态特征选择策略（基于市场环境自适应）
```

- [ ] **Step 5: 验证修改**

运行: `python3 -m markdown lessons.md 2>&1 | head -20`
Expected: 无严重错误

- [ ] **Step 6: 提交阶段4-lessons**

```bash
git add lessons.md
git commit -m "docs(migration/phase4): 添加全量特征优于500特征的经验章节

- 添加详细的验证背景、结果、关键发现
- 标记旧经验为已过时
- 添加总结章节，记录验证方法和业界实践启示"
```

---

### Task 14: 更新 progress.txt

**Files:**
- Modify: `progress.txt`

- [ ] **Step 1: 读取progress.txt内容**

运行: `cat progress.txt | tail -30`
Expected: 显示文件末尾30行

- [ ] **Step 2: 添加迁移完成记录**

在"最近完成"部分添加：

```markdown
[2026-03-27] 完成从500特征迁移到全量特征（892个）
- 基于2026-03-27的Walk-forward验证结果
- 年化收益率提升：+10.14%（40.42% vs 30.28%）
- 索提诺比率提升：+83%（1.9023 vs 1.0400）
- 修改文件：12个（1核心脚本+6训练脚本+2自动化脚本+3文档）
- 采用4阶段渐进式迁移策略
- 保持向后兼容性，保留特征选择代码但标记为弃用
```

- [ ] **Step 3: 添加验证结果摘要**

在适当位置添加：

```markdown
全量特征 vs 500特征验证结果（2026-03-27）：
- 验证方法：Walk-forward验证（银行股板块，6只股票）
- 验证周期：2024-01-01 至 2025-12-31
- 关键发现：CatBoost的自动特征选择机制优于预选择
- 性能提升：年化收益率+10.14%，索提诺比率+83%
```

- [ ] **Step 4: 更新待处理事项（如有）**

删除或更新相关的待处理事项

- [ ] **Step 5: 提交阶段4-progress**

```bash
git add progress.txt
git commit -m "docs(migration/phase4): 记录迁移过程和验证结果

- 添加迁移完成记录
- 添加验证结果摘要
- 更新待处理事项"
```

---

## 最终验证

### Task 15: 集成测试

- [ ] **Step 1: 测试默认训练（使用全量特征）**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost`
Expected: 日志显示"特征数量: 892（全量特征）"
注意：此步骤需要5-10分钟

- [ ] **Step 2: 测试CatBoost弃用警告**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection`
Expected: 显示弃用警告"⚠️ 警告：特征选择功能已弃用..."
注意：此步骤需要5-10分钟

- [ ] **Step 3: 测试LightGBM弃用警告**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type lgbm --use-feature-selection`
Expected: 显示弃用警告"⚠️ 警告：特征选择功能已弃用..."
注意：此步骤需要5-10分钟

- [ ] **Step 4: 测试GBDT弃用警告**

运行: `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type gbdt --use-feature-selection`
Expected: 显示弃用警告"⚠️ 警告：特征选择功能已弃用..."
注意：此步骤需要5-10分钟

- [ ] **Step 5: 测试Walk-forward验证**

运行: `python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20 --test-window 1 --train-window 12 --folds 1`
Expected: 正常运行，不包含特征选择步骤
注意：此步骤需要30-60分钟

- [ ] **Step 6: 测试批量回测**

运行: `python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6 --stocks 0700.HK 0939.HK`
Expected: 正常运行，不包含特征选择步骤
注意：此步骤需要10-20分钟

- [ ] **Step 7: 验证所有文档更新**

运行: `grep -r "500特征\|500个特征" AGENTS.md lessons.md | grep -v "已弃用\|已过时\|2026-03-27"`
Expected: 不输出任何内容（所有500特征引用都已标记为已弃用或已过时）

- [ ] **Step 8: 提交最终验证**

```bash
git status
git diff --stat
git commit -m "test: 完成迁移后的集成测试

- 验证默认训练使用全量特征
- 验证所有模型的弃用警告正常显示（CatBoost/LightGBM/GBDT）
- 验证Walk-forward和批量回测正常运行
- 验证所有文档更新完成"
```

---

## 回滚计划

如果迁移后出现严重问题，按以下步骤回滚：

1. **恢复核心训练脚本**：
```bash
git revert <latest-commit-on-phase1>
```

2. **恢复训练相关脚本**：
```bash
git revert <latest-commit-on-phase2>
```

3. **恢复自动化脚本**：
```bash
git revert <latest-commit-on-phase3>
```

4. **恢复文档**：
```bash
git revert <latest-commit-on-phase4>
```

5. **重新运行特征选择脚本**：
```bash
python3 ml_services/feature_selection.py --method statistical --top-k 500 --horizon 20 --output-dir output
```

6. **重新训练模型**：
```bash
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection
```

---

## 成功标准

- ✅ 所有训练脚本正常运行
- ✅ 默认使用892个特征
- ✅ 弃用警告正常显示
- ✅ Walk-forward验证结果符合预期（年化收益率≥35%，索提诺比率≥1.5）
- ✅ 所有文档更新完成
- ✅ 向后兼容性保持（旧命令仍然可用）

---

## 预计时间

- 阶段1：30分钟
- 阶段2：1小时
- 阶段3：30分钟
- 阶段4：1小时
- 最终验证：30分钟
- **总计**：3.5小时

---

## 附录

### 相关文档

- 设计文档：`docs/superpowers/specs/2026-03-27-migrate-to-full-features-design.md`
- Walk-forward验证报告（全量特征）：`output/walk_forward_sector_bank_catboost_20d_20260327_142106.md`
- Walk-forward验证报告（500特征）：`output/walk_forward_sector_bank_catboost_20d_20260327_145815.md`
- 综合对比分析：`output/feature_comparison_final_20260327.md`

### 参考命令

```bash
# 验证特征数量
python3 -c "from ml_services.ml_trading_model import FeatureEngineer; fe = FeatureEngineer(); df = fe.get_sample_data('0005.HK'); features = fe.create_features(df, horizon=20); print(f'实际特征数量: {len(features.columns)}')"

# 测试训练
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 测试Walk-forward验证
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20

# 测试批量回测
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6

# 测试综合分析
./scripts/run_comprehensive_analysis.sh
```