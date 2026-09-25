# LightGBM A/B：1d / 5d 管线级验证报告

- **日期**：2026-09-26
- **目的**：20d 已升级 LightGBM（§5.18）后，验证 1d/5d 是否也应换学习器
- **方法**：`walk_forward_validation.py --learner lightgbm`，与 CatBoost 基线**同折同参数**复跑

## 一、配置（与基线严格一致）

| 项 | 值 |
|----|----|
| 折数 | 38 folds（train 36m，test 1m，step 1m） |
| 验证日期 | 2020-06-01 至 2026-07-31（测试折 2023-06 ~ 2026-07） |
| 股票 | 59 只 |
| 特征选择 | **是**（每折 Top-500，防穿越） |
| 置信阈值 | 0.55 |
| 学习器 | `LGBMClassifier`（n_estimators=400），复用 CatBoost 管线 |

**数据源**：
- 1d：`output/20260923_105752_catboost_1d` vs `output/20260926_002743_catboost_1d`（日志确认 `learner=lightgbm`）
- 5d：`output/20260922_212530_catboost_5d` vs `output/20260926_005557_catboost_5d`

---

## 二、1d A/B

| 指标 | CatBoost | **LightGBM** | 判读 |
|------|----------|--------------|------|
| 合并准确率 | 50.8% | **51.8%** | 样本大故显著，但幅度仍小 |
| 超额 lift | +1.2pp (p=0.010) | **+1.3pp (p=0.008)** | 基本持平，略优 |
| 折内 avg_ic | 0.0082 | **0.0222** | **+171%** |
| 折内 avg_rank_ic | 0.0158 | **0.0341** | **+116%** |
| **净IR** | −2.04 | −1.52 | 两者都为负 |
| **PBO** | 0.64 | **0.31** | LightGBM 过拟合风险更低 |
| **DSR** | 0.000 | 0.000 | — |
| 累计净收益 | −83.7% | −76.4% | 两者皆巨亏 |
| 护栏判定（D2） | 🔴 停用 | 🔴 停用 | **IR≤0** |

**逐年 lift**：两者 2023 均为负（−1.2/−1.1pp），2024-2026 均为正，LightGBM 2026 更好（+2.9 vs +2.1pp）。

**结论**：LightGBM 在 1d **信号层显著更强**（RankIC 翻倍）、过拟合更低（PBO 0.64→0.31），
但组合层两者 **净IR 均 ≤0 → 双双 🔴 停用**。1d 本来即「不可交易」（多空毛收益 0.09%/期 < 成本 0.5%），
换学习器无法改变这一结论。

---

## 三、5d A/B

| 指标 | **CatBoost** | LightGBM | 判读 |
|------|--------------|----------|------|
| 合并准确率 | **51.8%** | 51.2% | CatBoost 更高 |
| 超额 lift | **+1.6pp** (p=0.076) | +0.6pp (p=0.49) | **CatBoost 更优且更显著** |
| 折内 avg_ic | **0.067** | 0.0555 | CatBoost 更高 |
| 折内 avg_rank_ic | **0.0862** | 0.0748 | CatBoost 更高 |
| **净IR** | **0.57** [−0.58,1.66] | 0.25 [−0.96,1.36] | **CatBoost 更高** |
| **PBO** | **0.36** | 0.50（临界） | **CatBoost 更低** |
| **DSR** | **0.813** | 0.637 | **CatBoost 更高** |
| 累计净收益 | **+42.2%** | +10.3% | **CatBoost 更高** |
| 护栏判定（D2） | 🟡 保留低配 | 🟡 保留低配 | 均未达升级线 |

**逐年 lift**：

| 年份 | CatBoost | LightGBM |
|------|----------|----------|
| 2023 | −2.0pp | +1.2pp |
| 2024 | −0.6pp | **−3.5pp** |
| 2025 | **+3.4pp** | +3.0pp |
| 2026 | **+0.8pp** | **−2.1pp** |
| lift 标准差 | 5.9pp | 6.9pp |

**结论**：5d 上 **CatBoost 全面占优**（lift、IC、净IR、PBO、DSR、累计收益），且 **2026 最新年度
CatBoost +0.8pp 而 LightGBM −2.1pp**、标准差更大（6.9 vs 5.9pp）→ LightGBM 在 5d 不仅均值更差，
稳定性也更差 → **不升级**。

---

## 四、汇总：学习器选择是"分周期"的

| 周期 | 学习器 | 结果 |
|------|--------|------|
| **20d** | ✅ **LightGBM**（已上线） | RankIC +38%、护栏 🟢（§5.18，2026-09-25） |
| **5d** | ✅ **维持 CatBoost** | lift +1.6 vs +0.6pp、净IR 0.57 vs 0.25、2026 为正 vs 为负 |
| **1d** | ⚪ 两者皆停用 | 净IR −2.04 / −1.52 → 🔴 停用；仅保留报告展示 |

> **教训**：学习器优劣**不是全局属性，而是随预测周期而变**——
> 同一条管线、同一份数据、同一组折，20d 上 LightGBM 赢、5d 上 CatBoost 赢、1d 上两者都不可用。
> 因此**不能"全部转 LightGBM"**，必须按周期分别 A/B。

## 五、产物

| 文件 | 内容 |
|------|------|
| `output/backtest_eval_lightgbm_1d.md` / `backtest_eval_1d.md` | 1d lift/方向技能（LGBM vs CatBoost） |
| `output/backtest_eval_lightgbm_5d.md` / `backtest_eval_5d.md` | 5d lift/方向技能（LGBM vs CatBoost） |
| `output/monthly_guardrail_lightgbm_{1d,5d}.md` | LightGBM 月度护栏 |
| `output/monthly_guardrail_catboost_{1d,5d}.md` | CatBoost 月度护栏 |
| `output/wf_lgbm_{1d,5d}.log` | 运行日志（本地，不入库） |

> 注：walk-forward 报告文件名仍标 `catboost_*`（沿用 `--model-type` 默认值），
> 实际学习器以日志 `🔁 学习器: lightgbm` 为准（待改进：报告应记录 learner 字段）。
