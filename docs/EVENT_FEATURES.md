# 公司行动事件表（P0）

> 状态：**P0 完成；P1 未跑**。30 只样本事件数过低，按预登记不满足事件研究门槛。
> 生成：`python3 scripts/build_event_table.py`；输出 `data/events/events.csv`（gitignore，可重生成）。

## 1. 目标与边界

- 只做**公司行动**，不做宏观/行业事件。
- **机械调整**（股息、拆合股、供股除权）优先修正价格，不当作 alpha 特征。
- 事件表**高精度优先**：宁缺毋滥，`confidence=review` 的条目在 P1 人工复核。

## 2. 字段

| 字段 | 含义 |
|---|---|
| `code` | 5 位港股代码 |
| `name` | 自选池名称 |
| `event_type` | 见下表 |
| `announce_ts` | 公告时间（PIT 基准） |
| `confidence` | `high` / `review` |
| `direction` | `+` / `-` / `0`（先验方向，P1 验证） |
| `stage` | `announcement` / `completion` / `update` |
| `title` / `url` | 原文与链接 |
| `source` | `HKEXnews` |

## 3. 事件类型与规则

| 类型 | 标题特征 | 方向 | 置信 |
|---|---|---|---|
| `PROFIT_WARNING` | `PROFIT WARNING` / `盈警` | − | high |
| `POSITIVE_PROFIT_ALERT` | `POSITIVE PROFIT ALERT` / `盈喜` | + | high |
| `PLACEMENT_COMPANY` | `PLACING OF NEW SHARES` / `ISSUE OF NEW SHARES` / `SUBSCRIPTION OF NEW SHARES` | − | high |
| `PLACEMENT_SHAREHOLDER` | `PLACING OF (EXISTING) SHARES BY` | − | review |
| `RIGHTS_ISSUE` | `RIGHTS ISSUE` / `供股` | − | review |
| `OPEN_OFFER` | `OPEN OFFER` | − | high |
| `CB_ISSUANCE` | `ISSUE OF ... CONVERTIBLE BONDS/NOTES` | − | high |
| `CB_CONVERSION` | `CONVERSION OF CONVERTIBLE` / `ADJUSTMENT TO CONVERSION PRICE` | − | review |
| `PRIVATISATION` | `PRIVATI[ZS]ATION` | 0 | high |
| `SCHEME` | `SCHEME OF ARRANGEMENT` | 0 | high |
| `LOCKUP` | `LOCK-UP UNDERTAKING/EXTENSION` | + | review |

**排除**：`SHARE AWARD`、`SHARE OPTION`、`SCRIP`、`DIVIDEND`、`CBE`、第三方（`TRINKAUS`、`BANK OF COMMUNICATIONS`）。

## 4. PIT 规则

1. 特征只能用 `announce_ts` 之后的数据；统一 **T+1** 可用。
2. `completion` / `update` 不是新信息，P1 默认只用 `stage=announcement`。
3. 除权日/生效日与公告日分开存；P1 再加 `effective_ts`。
4. 事件表按 `(code, type, date, title)` 去重；同一事件的多份公告保留 `stage` 区分。

## 5. 30 只覆盖（2016–2026）

| 类型 | 全部 | 仅 announcement |
|---|---:|---:|
| 盈喜 | 30 | 30 |
| 盈警 | 9 | 9 |
| 大股东配售 | 16 | 14 |
| 公司配售 | 11 | 8 |
| 锁定 | 7 | 7 |
| CB 转股/调整 | 6 | 6 |
| 私有化 | 7 | 6 |
| CB 发行 | 6 | 3 |
| **合计** | **92** | **83** |

> 供股/公开招股/协议安排在 30 只中为 **0**；不是分类器漏，是样本本身没有。

## 6. P0 结论

- 事件表可用，分类器有单测；PIT 字段齐全。
- **30 只事件太稀疏**：单一类型最多 30 个，低于 P1 的 100 个门槛；
  按预登记，**30 只不足以做事件研究**，更不能据此判“有/没有”。
- 价格端：快照为腾讯前复权（qfq），股息/拆合股已调整；30 只无供股事件，
  暂无需 TERP 修正。总回报口径用 `dca_eval.py --total-return`。

## 7. 下一步

- **P1 扩到 310 只**（标题级提取成本低）：预计事件数 ×8–10，单一类型可过 100。
- 先跑 `PROFIT_WARNING + POSITIVE_PROFIT_ALERT`（数量最多、方向明确），
  再跑 `PLACEMENT_COMPANY`、`PRIVATISATION`、`LOCKUP`。
- 明细级（折价、稀释比例、TERP）仍在 30 只上做，避免一次性 310 份 PDF 解析。

## 8. P1 结果（310 只，2016–2026）

- 事件表：**2,246 条**，首次公告 **2,050 条**；事件-周期样本 **8,114**。
- 事件研究：公告日 **T+1 入场**，超额 = 个股 − HSI；t 按月份聚类。

| 事件 | 周期 | N | 均值 | 中位 | 月聚类 t | 子窗同号 |
|---|---:|---:|---:|---:|---:|---|
| 盈警 | 1d | 717 | −0.34% | −0.39% | −1.64 | 是 |
| 盈警 | 5d | 715 | −1.20% | −1.11% | −1.43 | 是 |
| 盈喜 | 60d | 657 | +3.79% | −1.32% | +2.18 | **否**（+6.1% → −1.9%） |
| 公司配售 | 60d | 133 | −3.18% | −3.62% | −0.47 | **否**（+0.6% → −9.5%） |
| 供股 | 60d | 105 | +4.00% | +3.06% | +1.33 | **否**（+5.3% → −2.0%） |
| CB 发行 | 60d | 133 | +3.49% | +1.39% | +1.43 | **否**（+9.8% → −4.4%） |
| 协议安排 | 60d | 20 | −10.18% | −9.32% | −5.55 | 是，但 N=20 |
| 大股东配售 | 1d | 24 | −1.41% | −1.67% | −2.51 | 样本太小 |

**P1 判决**：🔴 没有事件同时满足「N≥100 + |t|≥2 + 子窗同号」。
- 盈警方向稳定（负），但 t 仅 −1.4~−1.6；5d MDE≈1.7%，功效不足。
- 盈喜 60d 均值正、中位负，且子窗翻转 ⇒ 离群值驱动，不可用。
- 配售/供股/CB 的 60d 均值均因子窗翻转，不稳。
- 协议安排 t 很强，但 N=20 且属特殊交易，不能当通用信号。
- 大股东配售/锁定 N<30，不做结论。

**P2 不启动**：没有稳定事件信号，增量 IC 无前提。若未来重做，需：
1. 事件日期精确到公告时点（盘中/盘后）；
2. 控制规模、折价、市场状态；
3. 用 310 之外的独立窗口；
4. 先补退市股，消除幸存者。

## 9. 自动更新（2026-10-31 起）

| 频率 | Workflow | 更新 |
|---|---|---|
| 每天 16:30 | `comprehensive-analysis.yml` | 追加 `data_freshness.md`（只读检查） |
| 每周六 08:00 | `data-refresh.yml` | 公告 2016–2026 → 事件表 → 提交 |
| 每月 1 日 18:00 | `monthly-retrain.yml` | 恒指 PIT + 新鲜度 + 重训 → 模型 |

- 入库：`data/events/events_310.csv` + `manifest_310.json`（约 600 KB）。
- 不入库：公告 JSON/PDF、`fund_cache`、快照、CCASS 原始数据。
- 本地同步：`bash scripts/sync_data.sh`；需要 CCASS 私有仓加 `--ccass`。

## 10. P2a：事件规模抽取（30 只）

工具：`scripts/extract_event_magnitude.py`（PDF → 规模字段）。
输出：`data/events/event_magnitudes.csv`（24 条：高置信 8、低置信 16、followup 7）。

| 事件 | 关键规模 |
|---|---|
| 中芯国际 2017 配售 | 稀释 3.07%，折价 4.91%，募资/市值 3.0% |
| 黑芝麻 2025 配售 | 稀释 8.29%，折价 11.79%，募资/市值 8.0% |
| 中国平安 2024 CB | 转股稀释 3.56%，本金 HK$273 亿 |
| 汇丰 2025 私有化 | 要约价 HK$155，溢价 30.3%（后续更新 33.1%） |
| 小米 2019 锁定 | 365 天 |

**低置信/缺口**：药明生物配售只抽到折价（6.7–8.1%）与规模（3–4% 市值），稀释比例缺失；
阿里配售、部分 CB/锁定公告未披露关键字段；同一事件有多份公告（`is_followup=1`）。

**规则**：P2b 只用 `magnitude_conf=high` 的规模；低置信只保留事件 dummy/decay，缺失显式标记。

## 11. P2b：事件特征初筛

模块：`ml_services/feature_engineering/event_features.py`；检验：`scripts/event_feature_ic.py`。
特征：事件衰减（20d 半衰、90d 截止）+ 高置信规模（配售稀释/折价、CB、要约溢价、锁定）。

**30 只**：仅 15 只有事件、88 个月；除私有化（7 个月，t +2.53）外无 |t|≥2。
**310 只**：242 只、88 个月、19,381 行：

| 特征 | 有效月 | 均值 IC | NW t | MDE | 判断 |
|---|---:|---:|---:|---:|---|
| 盈警衰减 | 88 | −0.0266 | −2.39 | 0.0223 | 方向稳定，未过多重比较 |
| 大股东配售衰减 | 30 | +0.0347 | +2.11 | 0.0329 | 边缘，样本少 |
| 公司配售衰减 | 68 | −0.0031 | −0.32 | 0.0192 | 无 |
| 盈喜衰减 | 88 | +0.0022 | +0.23 | 0.0193 | 无 |
| CB 发行/转股 | 71–74 | +0.002~+0.004 | <0.6 | — | 无 |
| 配售规模（高置信） | 5 | — | — | — | 样本不足 |

**结论**：事件特征没有稳健的独立信号；`盈警` 是唯一方向稳定但未过多重比较的。
P2c 只做「最小事件 block + WF 消融」，不承诺提升；不过则降级为风险标记。
