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
