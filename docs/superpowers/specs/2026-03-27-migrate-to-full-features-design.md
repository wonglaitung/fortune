# 从500特征迁移到全量特征（892个）设计文档

**文档版本**: 1.0
**创建日期**: 2026-03-27
**作者**: iFlow CLI
**状态**: 待审核

---

## 1. 概述

### 1.1 迁移背景

项目当前使用**500个精选特征**（F-test + 互信息混合方法）进行模型训练。基于2026-03-27的Walk-forward验证结果，发现**全量特征（892个）**在性能上显著优于500特征。

### 1.2 迁移目标

将项目的默认特征策略从"500个精选特征"改为"使用全量特征（8922个）"，同时：
- 保持向后兼容性
- 保留特征选择代码（标记为弃用）
- 简化训练流程
- 更新所有相关文档

### 1.3 验证结果摘要

**验证方法**：Walk-forward验证（业界标准）
- 验证对象：银行股板块（6只股票）
- 验证周期：2024-01-01 至 2025-12-31
- 验证参数：12个月训练窗口，1个月测试窗口，1个月滚动步长

**性能对比**：
| 指标 | 全量特征（892个） | 500特征 | 改进幅度 |
|------|------------------|---------|---------|
| 年化收益率 | 40.42% | 30.28% | **+10.14%** |
| 索提诺比率 | 1.9023 | 1.0400 | **+83%** |
| 夏普比率 | -0.0235 | -0.0501 | +53% |
| 平均收益率 | 3.21% | 2.40% | +34% |

**关键发现**：
- CatBoost的自动特征选择机制优于预选择
- 信息保留完整，避免特征选择导致的信息丢失
- 特征组合（520个交叉特征）可能包含重要非线性关系

---

## 2. 设计方案

### 2.1 迁移策略

采用**渐进式迁移**（方案1），分4个阶段逐步实施：

1. **阶段1**：更新核心训练脚本
2. **阶段2**：更新训练相关脚本
3. **阶段3**：更新自动化脚本
4. **阶段4**：更新文档

### 2.2 向后兼容性

- 保留 `--use-feature-selection` 参数
- 参数默认行为改为使用全量特征
- 显式指定参数时显示弃用警告
- 保留所有特征选择代码

### 2.3 不更新的内容

- 深度学习实验脚本（`lstm_experiment.py`、`transformer_experiment.py`）保持现状
- 特征选择相关代码保留但不推荐使用

---

## 3. 详细实施方案

### 3.1 阶段1：更新核心训练脚本

**文件**：`ml_services/ml_trading_model.py`

**修改内容**：

1. **修改 `train()` 方法的 `use_feature_selection` 参数默认值**
   - 统一默认值为 `False`（使用全量特征）
   - 更新参数文档说明

2. **添加弃用警告**
   - 当用户显式指定 `--use-feature-selection` 参数时
   - 显示警告：`⚠️  警告：特征选择功能已弃用，建议使用全量特征（892个）。Walk-forward验证显示全量特征性能更好。`
   - 警告只显示一次（使用类变量控制）

3. **更新日志输出**
   - 训练开始时显示：`特征数量: 892（全量特征）`
   - 如果使用特征选择：`特征数量: 500（特征选择 - 已弃用）`

4. **不删除任何代码**
   - 保留 `load_selected_features()` 方法
   - 保留特征选择逻辑
   - 保留 `feature_selection.py` 文件

**修改位置**：
- `LightGBMModel.train()`：约第2291行
- `GBDTModel.train()`：约第2911行
- `CatBoostModel.train()`：约第3599行

**验证方法**：
```bash
# 测试默认行为（应使用全量特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 测试弃用警告（应显示警告）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection
```

---

### 3.2 阶段2：更新训练相关脚本

**文件列表**（6个）：
1. `ml_services/walk_forward_validation.py`
2. `ml_services/walk_forward_by_sector.py`
3. `ml_services/train_sector_model.py`
4. `ml_services/batch_backtest.py`
5. `ml_services/backtest_20d_horizon.py`
6. `ml_services/compare_three_models_20d.py`

**修改内容**：

1. **移除 `--use-feature-selection` 和 `--skip-feature-selection` 参数**
2. **移除相关逻辑和变量**
3. **更新帮助文档**

**修改示例**（以 `train_sector_model.py` 为例）：

**移除前**：
```python
parser.add_argument('--use-feature-selection', action='store_true',
                   help='使用特征选择')
parser.add_argument('--skip-feature-selection', action='store_true',
                   help='跳过特征选择，直接使用已有的特征文件')

# 训练模型
feature_importance = model.train(
    stock_codes,
    horizon=args.horizon,
    use_feature_selection=args.use_feature_selection or args.skip_feature_selection
)
```

**移除后**：
```python
# 训练模型（默认使用全量特征）
feature_importance = model.train(
    stock_codes,
    horizon=args.horizon
)
```

**验证方法**：
```bash
# 测试所有脚本是否正常工作（不使用特征选择参数）
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20
python3 ml_services/train_sector_model.py --sector bank --horizon 20
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6
python3 ml_services/backtest_20d_horizon.py --start-date 2025-01-01 --end-date 2025-01-31 --horizon 20
```

---

### 3.3 阶段3：更新自动化脚本

**文件列表**（2个）：
1. `scripts/run_comprehensive_analysis.sh`
2. `scripts/run_model_comparison.sh`

**修改内容**：

1. **`run_comprehensive_analysis.sh`**：
   - 移除步骤0（特征选择步骤）
   - 更新步骤1（训练命令，移除 `--use-feature-selection --skip-feature-selection`）
   - 重新编号步骤（步骤0→步骤1，步骤1→步骤2，等等）
   - 更新输出文件列表（移除特征选择结果）

2. **`run_model_comparison.sh`**：
   - 移除特征选择相关步骤（如果有）
   - 简化训练命令

**修改前**：
```bash
# 步骤0: 运行特征选择脚本，生成500个精选特征（使用F-test方法）
echo "=========================================="
echo "📊 步骤 0/5: 迥行特征选择（使用statistical方法）"
echo "=========================================="
echo ""
python3 ml_services/feature_selection.py --method statistical --top-k 500 --horizon 20 --output-dir output

# 步骤1: 训练 CatBoost 20天模型
echo "=========================================="
echo "📊 步骤 1/5: 训练 CatBoost 20天模型"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection --skip-feature-selection
```

**修改后**：
```bash
# 步骤1: 训练 CatBoost 20天模型（使用全量特征）
echo "=========================================="
echo "📊 步骤 1/4: 训练 CatBoost 20天模型（全量特征892个）"
echo "=========================================="
echo ""
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

**验证方法**：
```bash
# 测试综合分析脚本
./scripts/run_comprehensive_analysis.sh

# 验证输出文件
ls -lt data/llm_recommendations_*.txt
ls -lt data/ml_trading_model_catboost_predictions_20d.csv
ls -lt data/comprehensive_recommendations_*.txt
```

---

### 3.4 阶段4：更新文档

**文件列表**（3个）：
1. `AGENTS.md`
2. `lessons.md`
3. `progress.txt`

**AGENTS.md 更新内容**：

1. **特征选择方法章节**：
   - 推荐方法改为："使用全量特征（892个）"
   - 已弃用方法："统计方法（500个精选特征）" - 保留但不推荐使用

2. **所有训练命令示例**：
   - 移除 `--use-feature-selection` 和 `--skip-feature-selection` 参数

3. **特征工程章节**：
   - 特征数量改为：892个（全量特征）

4. **添加新章节**：
   - 全量特征 vs 500特征对比验证
   - 包含详细的验证结果和对比数据

5. **标记特征选择为已弃用**：
   - 在多个章节添加"已弃用但保留"标记

**lessons.md 更新内容**：

1. **添加新章节**：
   - "全量特征（892个）优于500特征（2026-03-27验证）"
   - 包含详细的验证背景、结果、关键发现
   - 标记旧的"固定500特征是最优方案"为"已过时，仅供参考"

2. **添加总结章节**：
   - "全量特征vs 500特征验证的重要影响（2026-03-27）"
   - 记录验证方法、关键洞察、经验教训
   - 说明对项目的影响和业界实践的启示

**progress.txt 更新内容**：

1. **在"最近完成"部分添加**：
   - 迁移完成记录
   - 验证结果摘要
   - 关键发现

2. **更新待处理事项**（如有）

---

## 4. 修改文件列表

### 4.1 完整文件列表（12个文件）

**阶段1 - 核心训练脚本**（1个文件）：
- `ml_services/ml_trading_model.py`

**阶段2 - 训练相关脚本**（6个文件）：
- `ml_services/walk_forward_validation.py`
- `ml_services/walk_forward_by_sector.py`
- `ml_services/train_sector_model.py`
- `ml_services/batch_backtest.py`
- `ml_services/backtest_20d_horizon.py`
- `ml_services/cluster_three_models_20d.py`

**阶段3 - 自动化脚本**（2个文件）：
- `scripts/run_comprehensive_analysis.sh`
- `scripts/run_model_comparison.sh`

**阶段4 - 文档**（3个文件）：
- `AGENTS.md`
- `lessons.md`
- `progress.txt`

### 4.2 不修改的文件

- 深度学习实验脚本（`lstm_experiment.py`、`transformer_experiment.py`）
- 特征选择相关代码（保留但标记为弃用）
- 其他未列出的文件

---

## 5. 风险和缓解措施

### 5.1 预期风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 训练时间增加 | 训练时间可能增加10-20% | 无（性能提升值得时间成本） |
| 模型文件变大 | 模型文件大小增加约10% | 无（磁盘空间充足） |
| 现有脚本不兼容 | 用户可能仍在使用旧参数 | 保留参数，显示弃用警告，保持向后兼容 |
| 文档不一致 | 用户可能看到矛盾的说明 | 一次性更新所有文档，保持一致性 |

### 5.2 错误处理

1. **参数验证**
   - 当用户使用 `--use-feature-selection` 时显示弃用警告
   - 警告只显示一次，使用类变量控制

2. **向后兼容性保证**
   - 保留所有特征选择相关代码
   - 即使显式指定 `--use-feature-selection` 也能正常工作
   - 只是不推荐使用

3. **日志输出改进**
   - 明确显示使用的特征数量和策略
   - 帮助用户理解当前配置

---

## 6. 测试策略

### 6.1 单元测试

```bash
# 测试1：默认行为（应使用全量特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
# 预期输出：使用 892 个特征（全量特征）

# 测试2：弃用警告（应显示警告）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection
# 预期输出：⚠️ 警告：特征选择功能已弃用... + 使用 500 个特征
```

### 6.2 集成测试

```bash
# 测试完整综合分析流程
./scripts/run_comprehensive_analysis.sh

# 测试Walk-forward验证
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20

# 测试批量回测
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6
```

### 6.3 回归测试

- 确保所有现有功能正常工作
- 对比迁移前后的性能指标（确保没有退化）
- 验证GitHub Actions自动化流程正常

---

## 7. 成功标准

### 7.1 功能正常

- ✅ 所有训练脚本正常运行
- ✅ 默认使用892个特征
- ✅ 弃用警告正常显示

### 7.2 性能验证

- ✅ Walk-forward验证结果符合预期（年化收益率40.42%，索提诺比率1.9023）
- ✅ 无性能退化

### 7.3 文档完整

- ✅ AGENTS.md 更新完成
- ✅ lessons.md 添加新经验
- ✅ progress.txt 记录迁移过程

### 7.4 向后兼容

- ✅ 旧命令仍然可用（显示警告）
- ✅ 特征选择代码保留但标记为弃用

---

## 8. 实施时间表

### 8.1 预计总时间

**总计**：约4小时

### 8.2 时间分配

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 阶段1 | 更新核心训练脚本 | 30分钟 |
| 阶段2 | 更新训练相关脚本 | 1小时 |
| 阶段3 | 更新自动化脚本 | 30分钟 |
| 阶段4 | 更新文档 | 1小时 |
| 验证 | 每个阶段验证（15分钟×4） | 1小时 |
| 最终测试 | 集成测试和回归测试 | 30分钟 |
| **总计** | | **3.75小时** |

### 8.3 提交策略

- 每个阶段完成后提交一次
- 提交信息格式：`feat: 迁移到全量特征 - 阶段X/4 - [简短描述]`
- 最终提交：`feat: 完成从500特征迁移到全量特征`

---

## 9. 后续计划

### 9.1 短期（1-3个月）

1. 监控模型性能变化
2. 收集新的回测数据
3. 评估是否需要进一步优化

### 9.2 中期（3-6个月）

1. 考虑动态特征选择策略
2. 探索特征工程优化方向
3. 评估是否需要增加新特征

### 9.3 长期（6-12个月）

1. 定期重新验证特征策略
2. 探索新的特征组合
3. 评估模型性能趋势

---

## 10. 附录

### 10.1 相关文档

- Walk-forward验证报告（全量特征）：`output/walk_forward_sector_bank_catboost_20d_20260327_142106.md`
- Walk-forward验证报告（500特征）：`output/walk_forward_sector_bank_catboost_20d_20260327_145815.md`
- 综合对比分析：`output/feature_comparison_final_20260327.md`

### 10.2 参考命令

```bash
# 训练模型（使用全量特征）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 生成预测
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost

# 批量回测
python3 ml_services/batch_backtest.py --model-type catboost --horizon 20 --confidence-threshold 0.6

# Walk-forward验证
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20

# 综合分析
./scripts/run_comprehensive_analysis.sh
```

### 10.3 联系和支持

如有问题或疑问，请参考：
- `lessons.md`：经验教训和最佳实践
- `AGENTS.md`：完整项目文档
- `progress.txt`：项目进度跟踪

---

**文档结束**