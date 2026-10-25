---
name: hypothesis-validation
description: >
  MUST USE when 需要验证一个市场/策略假设是否成立，或判断某个指标/结论是否可信。
  触发词：验证/证明/假设检验/回测/显著性/这策略行不行/这个能赚钱吗/有没有 alpha/
  这数可信吗/靠不靠谱/准不准；以及改动类命题：换池/换周期/加特征/择时/风格因子/板块轮动/
  成本/换手/仓位。本技能给出本项目（港股 / A股 量化）**验证假设的标准 SOP + 硬规则 +
  失败模式 + 命令附录**。凡是要把数字作为"结论"输出前，先跑本技能。
---

# 假设验证 SOP（本项目的"尺子"）

> **一句话**：把"任意市场假设 → 可信地信 / 否"固定成一条**自带反自欺机制**的流水线。
> **本技能的重点不是"会回测"，而是"在最容易被自己骗的地方都预埋了闸"。**
> **核心信念**：结论不可信，**十有八九不是算错，而是"口径选错"**
> （比较基准 / 采样粒度 / 显著性口径 / ⓠ 同一性）。所以 SOP 的重心是**堵口径**，不是堆指标。

## 何时用

- 用户问"X 策略行不行 / X 有没有 alpha / 这个数可信吗 / 换个周期(池、特征、成本)会更好吗"。
- **任何要作为"结论"输出的数字**：先过 ⓠ 同一性闸 + A 闸，才轮到其它闸。
- 也用于**提出**假设（第 0 步的"命题 + 预登记"），不只在"验证"时。

---

## SOP（7 步）

### 0. 命题 + 预登记（**看结果之前**）
- 把假设写成**可证伪**形式：`H0 / H1 + 数字或区间`（例：`H0: IC ≤ 0.0234`）。
- **先写死成功/失败判据**（数字/区间），**再看结果**。
- **先问三态**（lessons 三.75）：**算不了**（缺数据 / 无 PIT）？**说不准**（功效不足）？**先验已否**（已有强反证）？
  —— 任一命中 **就停，不跑**（省时且防止"跑到一个假阳"）。

### 1. ⓠ 同一性闸（先于一切）
被评估的**量** = 会用的**量**？四问任一"否"即止：
1. 社区集合（`community_ids`）严格相等？2. 列集合：生产 pkl ⊆ WF 可用列 **且** F1/HIBOR 两边都在？
3. 时点：`train()` / `predict()` 的 `use_shift`（`mode`）相等？4. 概率同源（`oof_glob` 校准 + 门槛 `GATE_SOURCE_CSV`）？
- 跑：`python3 -m pytest tests/test_pipeline_contract.py`（须 **0 xfailed**）。

### 2. 数据卫生 + 三冻结
- **无前视**：滚动特征 `.shift(1)`；标签 `Close.shift(-(h+lag))/Close.shift(-lag)-1`；外部行情（HSI/美股）走**对齐函数**，别裸 `merge`。
- **无幸存者**：池成员要 PIT（时点还原），别用"今天还在的成分"回填历史。
- **三冻结**：`HK_MARKET_SNAPSHOT_DIR` + `US_MARKET_SNAPSHOT_DIR` + `GATE_SOURCE_CSV` + **双跑 md5 一致**。
- ⚠️ **跨日/跨机不可逐行复现**（数据源日内更新）⇒ 只比**决策指标**，禁逐行 diff。

### 3. 口径（**最容易翻车的一步**）
- **能力**：看**部署口径**（行业内中性分、逐日 IC），**不看原始 prob**；看**超额**（减等权基准），**不看绝对 IR**。
- **显著**：必须**按折聚类**（行级 z 是**伪显著**）；重叠序列用 HAC / 块 bootstrap。
- **回撤/Calmar**：必须**逐日 mark-to-market**（非重叠期稀疏净值会**低估**回撤、**虚报**风控）。
- **降回撤类**：必须加**静态等仓位对照** → 区分"真择时" vs "**只是少投**"。

### 4. 三道闸
- **A 泄漏闸**：`presentation_gate.py`（非零退出即阻断）；结论须带**适用范围**。
- **B 过拟合闸**：**PBO**（对零假设带）+ **DSR**。
- **C 净IR闸**：净 IR 95%CI **下界>0** 且 **lift>0**。

### 5. 抗过拟合 + 稳健性（**不许取最好一轮**）
- 跨**相邻参数**符号是否一致？（`--sensitivity`；翻转 = 不稳健）
- **独立窗口**能复现？（换时间窗 = 真 OOS）
- **多重比较**：测了 N 格就按 N× 惩罚（max 统计量 / Bonferroni）。
- **vintage 稳定**？（换一天数据，符号是否翻转）

### 6. 功效声明
- 报 **MDE**（最小可探测效应）：**"不显著" ≠ "没有"**（窗口小 ⇒ 测不出小 edge）。

### 7. 结论 + 记录
- 判 `🟢 / 🟡 / 🔴` + **适用范围**（**WF 内部比较 ✓ / 外推生产 ✗**）。
- 落 `docs/DECISIONS.md`（新 D 号：预登记 + 判决）→ `lessons.md` → `docs/BASELINES.md`；`doc_gate` 绿。

---

## 硬规则（**判据式** —— 在合法数据上跑一遍会 FAIL 的，才算规则）

```
G1 未预登记就出结论            → FAIL
G2 行级 z 当显著               → FAIL（改折聚类）
G3 用绝对 IR 当能力            → FAIL（改超额，减等权）
G4 用非重叠期稀疏净值算回撤    → FAIL（改逐日 mark）
G5 降回撤不加静态等仓位对照    → FAIL
G6 报告"最好一格"当结论        → FAIL（须给全格 + 相邻参数 + 独立窗）
G7 结论无"适用范围"标注        → FAIL
G8 不报 MDE                    → FAIL
G9 ⓠ 未过（契约有 xfailed）    → FAIL（后面闸不必跑）
```

---

## 失败模式库（前车之鉴）

- **行级 z 伪显著**：行级 z=6.3 → 折聚类 t=1.40（**p=0.167**）。
- **稀疏净值低估回撤**：20d 非重叠 maxDD −9.9% → **逐日 −18.9%** ⇒ "✅ 会躲"是假象（D35 §十二）。
- **取最好一轮**：MA200 单点过三关 → 相邻 150/250 全垮（**D43**）；板块 (L20,K2) 单格（**D36**）；价值 E/P 独立窗负（**D42**）。
- **比较基准选错**：波动目标"降回撤"只是"少投"（**D32**）；刹车亦然（D35 §十二）。
- **beta ≠ alpha**：绝对 IR 好看，可能只是**含 beta**（三.68）。
- **三态前置**：缺 PIT 基本面 → **算不了**（D40/D41）；~39 期 → **MDE≈1.6** ⇒ **说不准**（三.75）。

---

## 模板（填空）

```
## 命题
H0：__（数字）      H1：__（且显著口径 = __）

## 三态前置
算不了? __   说不准?(MDE=__)   先验已否? __        # 任一命中 ⇒ 停

## 预登记判据（跑前写死）
__

## 运行
命令 = __      冻结 = __      双跑 md5 = __

## 结果
口径 = 部署口径 + 超额 | 折聚类 t = __ | maxDD(逐日) = __ | 静态对照 maxDD = __
| PBO = __ | DSR = __ | MDE = __ | 相邻参数符号 = __

## 判决
🟢 / 🟡 / 🔴   适用范围：WF 内部比较 ✓ / 外推生产 ✗

## 记录
DECISIONS D__ ；lessons __ ；BASELINES __
```

---

## 附录：命令（**工具改了以 `docs/REVIEW_GATES.md` 为准**）

```
# ⓠ 同一性闸（须 0 xfailed）
python3 -m pytest tests/test_pipeline_contract.py

# A 泄漏闸
python3 scripts/presentation_gate.py --input <csv> --horizon N

# B/C 护栏：PBO/DSR/净IR/超额/MDE/部署口径 IC
python3 ml_services/monthly_guardrail.py --horizon 20 --pred <csv> --output <md>

# 相邻参数敏感性（符号是否跨参一致）
python3 ml_services/monthly_guardrail.py --horizon 20 --pred <csv> --sensitivity

# 组合层超额 CI + 逐年
python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10 --pred <csv>

# Walk-forward（三冻结 + 双跑）
HK_MARKET_SNAPSHOT_DIR=data/hk_market_snapshot \
US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot \
GATE_SOURCE_CSV=output/<基线>/prediction_analysis.csv \
  scripts/run_walk_forward.sh --model-type lgbm --horizon 20 --use-feature-selection \
  --start-date 2020-06-01 --end-date 2026-07-31 --no-commit

# 时序/择时类策略评估（唯一实现：逐日 mark + 静态对照 + MDE）
python3 ml_services/hsi_risk_overlay.py --enh nasdaq --horizon 20
python3 scripts/classic_timing_backtest.py

# 收尾（断链/表格/过期/A闸）
python3 scripts/doc_gate.py
```

> 详规（9 项清单 / 硬约束 / 案例）：`docs/REVIEW_GATES.md`
> 数字（唯一真相源）：`docs/BASELINES.md` ｜ 教训：`lessons.md` ｜ 决策：`docs/DECISIONS.md`
> 工程守则：`AGENTS.md` ｜ AI 协作流程：`docs/AI_WORKFLOW.md`
