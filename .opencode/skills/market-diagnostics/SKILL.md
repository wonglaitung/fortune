---
name: market-diagnostics
description: >
  MUST USE when 需要回答“这个跑赢/跑输是真的吗、来自哪里”，或做指数 vs 个股分解、
  IC/流动性/Beta/行业归因、成本校准、事件研究、幸存者检查、端点/权重敏感性。
  触发词：归因/诊断/为什么跑赢/为什么亏/指数 vs 个股/幸存者/含息/成本/换手/事件研究/
  这个结论可信吗。本技能给出项目通用的**市场诊断 SOP + 硬规则 + 工具命令**；
  不预测收益，只把收益/指标拆到可复核的口径。
---

# 市场诊断与归因 SOP（本项目的“诊断尺子”）

> 目标：把「看着跑赢」拆成 **股息 / 端点 / 权重 / 幸存者 / 成本 / 真实选股** 六件事。
> 一句话：**先问口径，再看数字；先拆来源，再谈能力。**

## 何时用

- 指数 vs 个股/组合：为什么十年涨/跌、谁贡献的；
- 某策略「超额为正」：是 alpha、beta、风格，还是成本/基准口径；
- IC 很高但不赚钱：信息落在哪个分层、能不能交易；
- 事件后漂移：盈警/配售/供股/卖空；
- 任何要作为「结论」输出的收益/超额数字。

不适用：预测模型训练（走 `hypothesis-validation`）；个股基本面（走 `equity-analysis`）。

---

## 0. 先定口径（不做这步，后面全错）

| 问题 | 必须写死 |
|---|---|
| 基准 | 价格指数还是**含息**？等权还是市值加权？ |
| 宇宙 | 当前池还是 PIT 池？含不含退市？ |
| 收益 | 前复权？含股息？币种？ |
| 端点 | 起止日；是否报多端点？ |
| 成本 | 往返多少？flat 还是分层模型？ |
| 显著性 | 行级还是折/月聚类？ |

**数据缺口（本项目现状）**：无历史股数/权重、无退市股、无 2010–2023 PIT 成分、南向只有总量。
⇒ 缺这些时，只能做**描述性诊断**，不能声称「无偏回测」。

---

## 1. 指数 vs 个股：六步拆解

1. **股息**：价格指数 vs 含息基准（如 2800.HK）；十年股息差常达 30–40pp；
2. **端点**：至少报「全期 + 两个高点起算」；同一段历史换起点可翻符号；
3. **权重**：等权 vs 市值加权；少数巨头可主导指数；
4. **成分/风格**：老经济 vs 新经济、大小盘、价值/成长；
5. **幸存者**：当前池回填历史 ≠ 当时池；中位数比均值更诚实；
6. **成本/换手**：扣往返成本后再比较。

> 示例结论模板：「恒指价格 +5.0%，含息 +43.8%；310 当前池中位 +2.1%；
> 自选 30 +204% 但属事后挑选的赢家子集」——每句都要能复算。

## 2. 信息层归因（IC）

```bash
python3 scripts/ic_attribution.py --input output/<run>/prediction_analysis.csv --horizon 20 --write-subsets
```

- 逐日横截面 IC，按**流动性 / Beta / 行业**分层；
- 同时报「毛 IC」与「扣成本后净收益」；
- 若 IC 集中在低流动/低 Beta 分位，先问「能不能买到」。

## 3. 成本校准

```bash
python3 scripts/cost_calibration.py --pred output/<run>/prediction_analysis.csv
python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10 --cost-mode model --pred ...
```

- 往返 vs 单边；ADV/佣金/价差/冲击分层；
- 报「隐含往返成本」和「盈亏平衡成本」；
- 成本假设能决定符号（如蜡烛图毛≈0）。

## 4. 事件研究

```bash
python3 scripts/hkex_event_study.py --horizon 20
python3 scripts/sfc_signal.py --fwd 20 --lag 10
```

- 入场时点（公告次日）、持有期、基准（等权/行业/流动性匹配）；
- 按月/事件聚类，不报行级 t；
- 先区分「毛 CAR」与「扣成本后」。

## 5. 稳健性

```bash
python3 scripts/vintage_sensitivity.py --horizon 20 --topk 10
python3 ml_services/monthly_guardrail.py --horizon 20 --pred ... --sensitivity
```

- 子期（牛/熊/震荡）、多端点、跨快照、相邻参数；
- 禁止「取最好一格」；扩展网格后才通过的格子，必须未见段复验。

## 6. 功效与结论

- 报 **MDE**：月度策略常需 >1.6%/月才能测出；
- 「不显著」≠「没有」；「显著」也要看聚类口径；
- 结论必须带**适用范围**（本宇宙/本窗口/本成本）。

---

## 硬规则（判据式）

```
G1 价格指数 vs 前复权个股、不说明股息   → FAIL
G2 当前池回填历史、不标幸存者           → FAIL
G3 只报单一端点/单一窗口                → FAIL
G4 不定义基准（等权/市值/含息）          → FAIL
G5 行级 z/t 当显著                      → FAIL（改折/月聚类）
G6 说“没有 edge”却不报 MDE              → FAIL
G7 取网格最好一格当结论                 → FAIL
G8 把诊断结论当交易信号                 → FAIL
```

## 失败模式库

- **股息漏算**：价格指数十年 vs 含息差 30–40pp；
- **幸存者**：当前池中位数远高于真实全市场；
- **端点**：2018/2021 高点起算，恒指十年为负；
- **权重幻觉**：等权小盘跑赢 ≠ 指数选股差；
- **成本翻转**：毛≈0 的策略扣费后为负；
- **小样本/多重检验**：几十个检验里总有一个「显著」；
- **缓存/快照混用**：不同 vintage 的数字不可直接比。

## 输出模板

```
## 口径（基准/宇宙/端点/成本/聚类）
## 指数 vs 个股（股息/端点/权重/幸存者）
## 信息层（IC 分层）
## 组合层（净超额/成本/盈亏平衡）
## 事件（CAR/聚类）
## 稳健性（子期/端点/快照/相邻参数）
## 功效（MDE）与适用范围
## 数据缺口
```

## 命令速查

```bash
python3 scripts/ic_attribution.py --input <csv> --horizon 20
python3 scripts/cost_calibration.py --pred <csv>
python3 scripts/hkex_event_study.py --horizon 20
python3 scripts/sfc_signal.py --fwd 20 --lag 10
python3 scripts/vintage_sensitivity.py --horizon 20 --topk 10
python3 ml_services/monthly_guardrail.py --horizon 20 --pred <csv>
python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10 --pred <csv>
python3 scripts/presentation_gate.py --input <csv> --horizon 20
```
