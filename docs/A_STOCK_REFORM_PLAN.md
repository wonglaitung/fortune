# A 股对照港股改造方案（差距分析与分期执行）

> **版本**：v1.2 | **创建日期**：2026-09-28 | **状态**：执行中（P0/P1/P2/P3.1 主体完成，P3.2 与 P4.1 A/B 跑数中）
> **性质**：方案文档 + 执行跟踪（完成项打 ✅，执行后更新本文档）
> **基线**：港股 2026-09 系列改造（评估 D3 / 组合层 D2 / 概率校准 / 门槛 D8 / 展示同口径等）

---

## 一、背景

港股近期完成一轮系统性改造（评估口径、组合层护栏、概率校准、市场门槛、展示同口径、复权、
邮件口径白话化等）。A 股系统多数对应能力缺失或停留旧口径，本文档做全量差距分析并分期实施。

**总原则**：
1. **口径诚实优先于新功能**——先保证 A 股评估/展示不再用绝对准确率误导（D3），再谈升级。
2. **复用港股工具，不复制代码**——`backtest_eval`/`monthly_guardrail`/`portfolio_backtest`
   输入均为 `prediction_analysis.csv`（市场中立），仅需列名对齐 + 板块映射分支（`--market a`）。
3. **受 D1 约束**（个股横截面 alpha 停止投入）——A 股改造聚焦"评估基建/口径诚实"，
   不新增横截面 alpha 挖掘（行业中性 TopK 目标集等不做）。
4. **数据不足的项延后**——校准/监控依赖已评估历史（现仅 132 条且停回写），排 P3 并等待拍板。

---

## 二、差距矩阵（港股基线 → A 股现状）

| # | 港股已改造 | A 股现状 | 差距 | 分期 |
|---|---|---|---|---|
| 1 | 评估 D3：backtest_eval（lift/方向技能/IC CI/n_eff/逐月/校准分桶） | walk-forward 只报 accuracy/f1/夏普/IC，无基准扣除 | 🔴 | P1 |
| 2 | 组合层 D2：monthly_guardrail（净IR/PBO/DSR）+ portfolio_backtest（超额CI/逐年） | 完全没有 | 🔴 | P2 |
| 3 | 概率校准：DailyConfidence Isotonic + 置信度（12,484 条历史） | 无校准；`a_stock_prediction_history` 仅 132 条且已停回写 | 🔴 数据瓶颈 | P3 |
| 4 | 展示同口径：方向/模式按校准概率重判（lessons 三.19） | raw 自判自洽（暂无矛盾）→ 校准后必须连带 | 🟡 | 随 3 |
| 5 | 门槛分位 D8：GATE_QUANTILES P92/P90（walk-forward 分布 PIT） | 绝对 0.50/0.65/0.70（`a_stock_comprehensive_analysis.py:850`） | 🔴 | P3 |
| 6 | 邮件口径 D3 白话 | 「强买入阈值0.60略高于准确率」= 绝对准确率推阈值（D3 反例） | 🟡 | P0 |
| 7 | 学习器分周期 A/B（D10：20d=LGBM） | 纯 CatBoost，无 A/B | 🟡 成本高 | P4 |
| 8 | 复权 qfq 全链（监控 exit 修复） | 训练/walk-forward 均 qfq 自洽（`a_stock_walk_forward.py:477`），评估已停无 exit | 🟢 基本无 | — |
| 9 | 图表 D3（CI 误差须/显著性/护栏块） | 无性能图表（六边形强度图≠评估图表） | 🟡 依赖 1/2 | P4 |
| 10 | 性能监控报告（D3 诚实摘要） | 09-27 主动停用（history 不回写） | 🟡 需拍板 | P4 |
| 11 | 主表链接化（gu.qq.com/hkXXXXX） | 无（可用 sh600000/sz000001 格式） | 🟢 | P0 |
| 12 | 回测产物入库工具 | `scripts/commit_backtest_result.py` 明确排除 a_stock | 🟢 | P1 |
| 13 | 邮件术语白话化 | 未做 | 🟢 | P0 |
| 14 | 三周期模式净贡献+Bonferroni 验证 | `A_STOCK_TRANSMISSION_ACCURACY` 注释自认「参考港股，需根据A股验证结果更新」 | 🟡 | P0 诚实化 |
| 15 | D2 行业中性 TopK 目标集 🎯 | 无 | ⚪ **不做**（D1） | — |
| 16 | 仓位口径 C（0.60/0.55 档+门槛优先） | LLM 建议仓位（保守/适度/激进%）+ 强买0.60/买0.50 两套体系 | 🟡 | P4 对齐 |

**关键发现（成本利好）**：
- `backtest_eval._normalize_columns` **已兼容 A 股列命名**（code/predict_prob/fold）；
- 三工具输入均为 CSV（市场中立），A 股导出注释已写「与港股格式一致」但实际列名不同——
  **P0 列名对齐后 P1/P2 基本直吃**；
- A 股 walk-forward 内部已有 `market_layer`/`dynamic_threshold`/IC/夏普，只是没写进 CSV 与报告主口径。

---

## 三、关键约束

1. **A 股历史数据瓶颈**：`data/a_stock_prediction_history.json` 仅 132 条（每周期 44 条），
   2026-09-27 起停回写 → 校准器（MIN_SAMPLES=200/周期）拟合不了；性能监控无数据。
2. **双入口**：`a_stock_walk_forward.py`（根目录，AGENTS 指定入口）与
   `ml_services/a_stock_walk_forward.py` 并存——执行前确认唯一活入口，避免改错文件。
3. **旧 CSV 缺列**：已有 `output/20260722_182121_a_stock_catboost_20d/prediction_analysis.csv`
   无 `Date`/`Market_Layer`/`Dynamic_Threshold` → guardrail/portfolio 直跑会 KeyError，
   **P2 需用新导出格式重跑 walk-forward**（或只重跑 20d）。
4. **默认 glob 混入**：`monthly_guardrail.latest_pred` 的 `output/*_catboost_20d` 会匹配
   `*_a_stock_catboost_20d`（港股默认跑时可能误取 A 股 CSV）→ P2 顺手排除。
5. **D1 张力**：评估重建 ≠ 挖新 alpha；P4 学习器 A/B 需拍板后才做
   （**2026-09-28 已拍板：三周期全做**——用户明确覆盖 D1 的"停止投入"默认，见决策点 3）。

---

## 四、分期方案

### P0 快赢（口径/文案/展示，不依赖数据）

| # | 内容 | 文件 | 验证 |
|---|---|---|---|
| P0.1 | walk-forward CSV 列名对齐港股（`Stock_Code`/`Fold`/`Predict_Direction`）+ 补 `Date`/`Market_Layer`/`Dynamic_Threshold` 列；更新消费方 | `a_stock_walk_forward.py`、`a_stock_comprehensive_analysis.py:495` | py_compile + 列名断言测试 |
| P0.2 | 邮件删「绝对准确率推阈值」段 → D3 白话（准确率仅背景、买入依据=市场调整门槛+概率档） | `a_stock_comprehensive_analysis.py` L1707 区 | 文案检查 |
| P0.3 | 传导模式胜率诚实化：`A_STOCK_TRANSMISSION_ACCURACY` 展示处标注「未验证，参考港股」 | 同上 L94/L787/L3022 | 文案检查 |
| P0.4 | 主表股票代码链接化（`gu.qq.com/sz000001`/`sh600000`，5位码+沪深前缀） | `a_stock_comprehensive_analysis.py` 表格行 | URL 生成测试 |
| P0.5 | 邮件术语白话化（Walk-forward→历史回测验证 等，对齐港股 dd2718a7 口径） | 同上 | 文案检查 |

### P1 评估重建（核心价值）

| # | 内容 | 文件 | 验证 |
|---|---|---|---|
| P1.1 | `backtest_eval.py` 加 `--market a`：板块映射换 `A_STOCK_SECTOR_MAPPING` + A 股板块中文名 | `ml_services/backtest_eval.py` | `--market a` 跑通 |
| P1.2 | 跑 A 股 1d/5d/20d 三份 backtest_eval → lift/方向技能/IC 基线报告 | CLI | `output/backtest_eval_a_*` 产出 |
| P1.3 | 回测产物入库工具扩 a_stock（复用港股每次入库自动清旧机制） | `scripts/commit_backtest_result.py` | dry-run |

### P2 组合层（依赖 P0.1 列名 + 新格式 CSV）

| # | 内容 | 文件 | 验证 |
|---|---|---|---|
| P2.1 | `monthly_guardrail`/`portfolio_backtest` 加 `--market a`（板块映射）；`latest_pred` 默认 glob 排除 `*_a_stock_*` | 两工具 + `eval_overfit` | `--market a` 跑通 |
| P2.2 | 重跑 A 股 walk-forward 20d（新导出格式）→ 净IR/PBO/DSR + 超额 CI/逐年 | `a_stock_walk_forward.py` + 两工具 | D2 判定产出 |
| P2.3 | A 股 D2 判定记录（档位沿用港股：净IR≥0.7 且 PBO<0.5 且 DSR≥0.95） | `docs/DECISIONS.md`（追加） | 文档 |

### P3 校准+门槛（依赖决策点 1/2）

| # | 内容 | 前置 |
|---|---|---|
| P3.1 | DailyConfidence 参数化扩展 A 股（history_file/HORIZONS）+ **方向/模式同口径连带**（lessons 三.19） | 校准数据源拍板 |
| P3.2 | 门槛分位化（D8 同款，A 股 walk-forward 分布建快照） | P3.1 |
| P3.3 | 恢复/决定 A 股 history 回写 | 决策点 1 |

### P4 可选增强（全部需拍板）

学习器 A/B（D10）｜ 图表 D3｜ 性能监控恢复 ｜ 仓位口径 C 对齐 ｜ 入库扩展。

### 不做清单

- D2 行业中性 TopK 目标集 🎯（D1：横截面 alpha 停止投入）
- 恢复性能报告「市场分布节」（09-27 已简化）

---

## 五、决策点（**2026-09-28 全部已拍板**）

| # | 决策 | 拍板结果 | 影响 |
|---|---|---|---|
| 1 | A 股评估历史是否恢复回写？ | **维持停用** | P3.3 不做；监控链路不动 |
| 2 | 校准数据源 | **walk-forward OOF 立即拟合**（PIT，从 `prediction_analysis.csv`） | P3.1 解锁，不等 history；需约定滚动更新节奏 |
| 3 | P4 学习器 A/B | **三周期全做**（1d/5d/20d，LightGBM vs CatBoost） | P4.1 解锁（覆盖 D1"停止投入"默认）；需先给 A 股模型加 model_type 开关 |
| 4 | A 股 D2 护栏档位 | **沿用港股三门槛**（净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 → 🟢15-20%） | P2.3 判定口径确定 |

> 决策 2 与 1 的组合含义：校准概率完全来自回测 OOF 分布（无生产样本回流），
> 分布随新 walk-forward 重跑更新——P3.1 须写明"快照日期 + 重跑即再拟合"。

---

## 六、执行状态跟踪

- [x] P0.1 CSV 列名对齐 + Date/Market_Layer/Dynamic_Threshold（`build_prediction_analysis()` + 5 单测；消费方兼容新旧列名）
- [x] P0.2 邮件删绝对准确率推阈值（LLM prompt 改 D3 白话）
- [x] P0.3 传导模式胜率诚实化（`verified=False` + 表格/文本"未验证（参考港股）"标注）
- [x] P0.4 主表链接化（`_stock_chart_url`/`_code_link_html` → gu.qq.com sz/sh/bj）
- [x] P0.5 邮件术语白话化（邮件可见处已无 Walk-forward/Isotonic/分位等；仅 LLM prompt 内部保留）
- [x] P1.1 backtest_eval `--market a`（板块映射+中文名+前导零补齐；unknown 归零）
- [x] P1.2 A 股三周期 backtest_eval 基线报告 —— **20d 已按新格式 19 折重算**（`output/backtest_eval_a_stock_20d_20260928.md`：
      准确率 **58.4%** [55.3,61.5]、超额 lift **+7.1pp**（p=0.0033 显著）、逐年 lift 2025 +7.7pp / 2026 +5.5pp 双正、
      逐折 lift 正 18/19、板块层轨交IT +9.9pp/电子 +9.5pp 方向技能领先、能源 −1.9pp 唯一负、
      校准分桶单调（50-60%→53.0%，80%+→72.5%，可直接作 P3.1 OOF 校准素材）；
      **5d 已补**（`output/backtest_eval_a_stock_5d_20260928.md`：51.5%、lift +1.0pp p=0.497 不显著 → 5d 无边缘）；
      **1d 首跑因主力资金缓存过期被静默降级（lessons 三.23）已废弃、重跑中** —— 补跑后重出 `backtest_eval_a_stock_1d_20260928.md`）
- [x] P1.3 入库工具扩 a_stock —— **已具备**（`HK20D_RE` 排除 a_stock，A股目录只提交 CSV 不改 GATE_SNAPSHOT；
      `a_stock_walk_forward.py` 已集成 `--no-commit` 开关）
- [x] P2.1 guardrail/portfolio/eval_overfit `--market a` + `latest_pred` 默认 glob 排除 `*_a_stock_*`
      + `signal_lift` 缺 `Dynamic_Threshold` 回退 0.5（合成面板连通性验证通过）
- [x] P2.2 重跑 20d walk-forward + D2 判定 —— **完成**（`output/20260928_144833_a_stock_catboost_20d/`，
      19 折 19,578 行新格式；护栏 净IR **2.75** [1.41,4.63] ✅ / DSR **1.000** ✅ / **PBO 0.74** ❌ →
      **🟡 保留低配**；组合层超额IR **3.02** [1.96,4.80] 下界>0、逐年双正但仅 20 期 →
      `output/monthly_guardrail_20d_20260928_a.md`、`output/portfolio_20d_top10_a.md`；
      **5d 已补**：净IR 0.52/PBO 0.54/DSR 0.688 → 🟡、组合层超额 CI 跨 0、lift +1.0pp 不显著 → 5d 无边缘；
      **1d 首跑（lift −1.2pp p=0.04、净IR −2.39 → 🔴）特征集缺主力资金列（lessons 三.23），
       结论待补跑后重判，暂不写入 D11 定稿**）
- [x] P2.3 A 股 D2 判定记录入 DECISIONS —— **D11 + §四.附**（含复核命令与触发条件）
- [~] P3.1 OOF 校准扩展 A 股 —— **代码+20d 校准器已就绪**（`DailyConfidence(cal_prefix='a_stock_',
      oof_glob=...)` + `calibrate_probability()` 方向同口径 + 4 单测；`daily_confidence.py --market a --refit`
      拟合；20d/5d 校准器（`data/calibrators/a_stock_prob_cal_{20,5}.pkl`）已各从 19,578 条 OOF 拟出、
      快照元数据 `a_stock_cal_meta_{20,5}.json` 落盘；
      **1d 校准器已拟出但源 CSV 属降级首跑 → 1d 补跑完成后 `daily_confidence.py --market a` 再拟一次**；
      邮件 A 股概率已走校准链路）
- [x] P3.2 门槛分位化 D8 —— **完成**（`ml_services/a_stock_gates.py`：`A_GATE_QUANTILES{bear:0.92,weak:0.90}`、快照 bear **0.7497** / weak **0.7154**（源 `output/20260928_144833_a_stock_catboost_20d` 19,578 条 + `a_stock_prob_cal_20.pkl`，as_of 2026-07-31）、回退链 CSV→快照→绝对值 0.70/0.65；`a_stock_comprehensive_analysis.py`（`get_market_sentiment` 阈值、layer_names、LLM prompt、`market_adjust` 判定）与 `a_stock_email.py`（sentiment 动态阈值、prompt 文案）全部改分位；3 单测，全量 129 passed；运行时实测 bear 0.7497 / weak 0.7154 / normal 0.50）
- [ ] P3.3 ~~恢复 history 回写~~ → **不做**（决策点 1：维持停用）
- [~] P4.1 学习器 A/B 三周期 —— **开关已就绪**（`AStockTradingModel(learner=...)` +
      `--learner lightgbm`：LightGBM 走同口径 TSCV+样本权重、准确率键 `a_stock_lightgbm_{h}d` 分离、
      输出目录 `*_a_stock_lightgbm_{h}d` 不覆盖 catboost、guardrail A股 glob 覆盖双学习器；4 单测；
      **三周期 LightGBM 链已在 tmux `ablgbm` 串行跑（20d→5d→1d，日志 `/tmp/opencode/awf_lgbm.log`）**，
       同时 tmux `abcb1d` 并行补跑 CatBoost 1d（修 lessons 三.23 特征降级，日志 `/tmp/opencode/awf_cb1d_rerun.log`）；
       六份运行主力资金缓存全部命中、特征集对齐）
- [ ] P4.2 图表 D3｜ P4.3 仓位口径 C 对齐（决策点 4 已定口径）——待排期

> 执行完一项勾一项；完成后本文档随 `progress.txt` 一并更新。
