# 模型更新验证报告（个股 20d：CatBoost → LightGBM）

## 更新类型
- 模型类型：**个股模型**（20d 信号）
- 更新内容：学习器 **CatBoost → LightGBM**（复用同一 PIT 管线，`learner='lightgbm'`；见 `docs/MODEL_IMPROVEMENT_PLAN.md` §5.18）
- 更新日期：2026-09-25

---

## 个股模型验证结果

### Walk-forward 测试（38 folds，同参数同折，2023-06 ~ 2026-07）

| 指标 | 更新前 CatBoost | 更新后 LightGBM | 评估 |
|------|----------------|-----------------|------|
| 合并准确率 | 51.6% | **52.3%**（p=0.036，better） | ✅ 提升 |
| 横截面 Rank IC | +0.0231 | **+0.0318**（+38%） | ✅ 提升 |
| 横截面 ICIR | 0.134 | **0.187** | ✅ 提升 |
| 超额 lift（信号胜率−基准） | +1.8pp | **+2.3pp** | ✅ 提升 |
| 数据泄漏检查 | 准确率 <65% ✅ | 准确率 52.3% <65% ✅ | 无泄漏 |

### 5.5 严谨合并评估（backtest_eval）

- 信号胜率 51.8% vs 无条件买入基准 49.5% → **超额 lift +2.3pp**（p=0.16，未显著但方向为正）
- 交易信号平均收益 2.1% vs 全样本 1.8%（+0.3pp）

### 5.6 月度护栏判定（monthly_guardrail，DECISIONS D2）

| 指标 | 值 | 门槛 | 判定 |
|------|-----|------|------|
| 20d 中性 TopK 净IR | 1.06 [−0.07,2.18] | ≥0.7 | ✅ |
| PBO | 0.47 | <0.5 | ✅ |
| DSR（最优 top5-neutral） | 0.981 | ≥0.95 | ✅ |
| **结论** | | | **🟢 通过（可升级）** |

### Fold 详细分析（38 折）

- 平均准确率：52.17%
- 平均收益/期：+2.15%
- 波动：Fold 11 收益 +13.96%（准确率仅 27.3%）、Fold 16 +16.32%；Fold 22 −6.89%、Fold 36 −7.55%
- 结论：收益与准确率不同步（fold 层面），符合"横截面 IC≠组合收益"；整体正收益、无异常泄漏折

### 因果链 / 三周期
- 按 PIT/embargo 全期口径，三周期模式≈随机（不构成独立交易信号），本轮未重跑（与学习器无关）

---

## 文档/代码更新清单

- [x] `ml_services/ml_trading_model.py`：`learner='lightgbm'` 分支（train/predict/save/load 适配）
- [x] `ml_services/walk_forward_validation.py`：`--loss-function` / `--learner`
- [x] `scripts/train_lightgbm_20d.py`：生产 LightGBM 20d 训练
- [x] `comprehensive_analysis.py`：20d 优先加载 `ml_trading_model_lightgbm_20d.pkl`
- [x] 删除废弃 `LightGBMModel`/`GBDTModel` 旧实现（~1290 行 → 兼容存根）
- [x] `AGENTS.md` / `docs/DEPLOYMENT.md` / `docs/MODEL_IMPROVEMENT_PLAN.md`（§5.18）/ `progress.txt`
- [x] 语法检查 `py_compile` 通过

---

## 下一步建议
- 20d 中性 TopK **按 LightGBM + top5-neutral 配置**运行；每月 `monthly_guardrail.py` 复核（PBO/DSR 监控）
- 仍受 2025 行情主导（36m 🟢 但 12m 全期 🟡）→ 跨期稳健性持续观察
- 其余周期（1d/5d）保持 CatBoost（LightGBM 仅用于 20d）

---
*报告生成：2026-09-25（模型验证技能 SOP）*