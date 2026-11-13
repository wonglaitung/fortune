# 预测周期扫描（A 阶段）预注册

> **创建**：2026-10-14 | **状态**：已预注册，未开跑前写死
> **来源**：B 阶段（信号-持有期敏感度）对个股/恒指均为 null（无更优持有期），
> 按用户指示仍在**限量网格**上执行 A，以排除「换周期重训本身会更强」的可能。
> 编号分配前先 `grep` 确认，正文不占用 D 编号。

## 1. 问题

**现行港股个股周期 = 1d / 5d / 20d（生产 20d）。**
A 阶段问：是否存在候选预测周期 h* ∈ {3, 10, 30}，
其**样本外交易价值**优于现行 20d 生产周期？

## 2. 控制变量（只改 horizon）

| 项 | 值 |
|---|---|
| 学习器 | CatBoost（全部周期统一，超参 20d 调优版：depth 8 / lr 0.06 / 400 树） |
| 数据窗口 | `--start-date 2020-06-01 --end-date 2026-07-31`（与 20d 基线同窗；测试期 2023-06→2026-07） |
| 特征口径 | `--use-feature-selection`（Top500 + F1/HIBOR FORCE 默认开，D22/D23/D24） |
| 时点口径 | `mode='production'`（T 收盘特征 + T+1 成交标签，D25） |
| 三冻结 | `US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot` `HK_MARKET_SNAPSHOT_DIR=data/hk_market_snapshot` `GATE_SOURCE_CSV=output/20261013_192444_catboost_20d/prediction_analysis.csv` |
| 确定性 env | `scripts/run_walk_forward.sh`（`PYTHONHASHSEED=0`、线程数固定） |
| 提交 | `--no-commit`（不受污染） |

## 3. 预注册判据（判定式，须**全部**满足才算 h* 胜出）

1. **A 闸**：`python3 scripts/presentation_gate.py --input <csv> --horizon h` **非零退出即阻断**；个股准确率 **≤65%**（>65% 视为泄漏）。
2. **PBO < 0.5**
3. **DSR ≥ 0.95**
4. **净IR 95%CI 下界 > 0**
5. **方向 lift > 0**（准确率 − 永远看涨基准）
6. **相对基线**：候选净IR 点估计 **>** 20d 基线净IR，且差异在 **Fold 聚类**下方向一致。
7. **多重比较**：3 候选对 α=0.05 做 Bonferroni（α_eff=0.0167）；**不取最好一轮**。
8. **双跑 md5 一致**（可复现性硬门槛；不一致即禁止引用）。

**失败判据**：无候选同时满足 1–6；或中标候选双跑 md5 不一致。
**升配判据**：**全部满足**才建议将生产 20d 切换/新增 h*。

## 4. 评估命令（每个周期 + 20d 基线）

```
python3 scripts/presentation_gate.py --input output/<dir>/prediction_analysis.csv --horizon H
python3 ml_services/backtest_eval.py --input output/<dir>/prediction_analysis.csv --horizon H
python3 ml_services/monthly_guardrail.py --horizon H --pred <csv>
python3 ml_services/portfolio_backtest.py --horizon H --topk 10 --pred <csv>
```

## 5. 产物

- `output/<ts>_catboost_{3,10,20,30}d/prediction_analysis.csv`（各 2 轮，比 md5）
- `output/horizon_scan_report.md`（汇总对比表 + 判定）
- 更新 `progress.txt`；结论若为否决，`lessons.md` 追加一条

## 6. 诚实性声明

- B 阶段已给**低先验**（信号对持有期不敏感）；A 的结果**不可**被解读为「周期可自由调优」。
- 若结果 null，须如实记录，**不得**再调 horizon（等同在噪声里捞信号）。
