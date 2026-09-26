# docs/ 文档索引

> **本目录回答"该看哪份文档"。**
> 项目总入口是根目录 [../AGENTS.md](../AGENTS.md)（规则 + 常用命令）；
> 本文件是 docs/ 内 18 份文档的**导航地图**，不重复任何文档的内容。

---

## 一、按场景导航（最常用）

| 我想做… | 看这一份 |
|---|---|
| **决定要不要用 / 加仓 / 停用某个信号** | [DECISIONS.md](DECISIONS.md) → [DEPLOYMENT.md](DEPLOYMENT.md) |
| **从零建一个量化系统** | [QUANT_SYSTEM_METHODOLOGY.md](QUANT_SYSTEM_METHODOLOGY.md) |
| **验证新模型 / 新特征 / 新周期** | [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) ＋ 技能 `.opencode/command/model_validation.md` |
| **加一个新特征** | [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)（8 步验证清单） |
| **搞懂 1/5/20 天三周期的结论** | [THREE_HORIZON_ANALYSIS.md](THREE_HORIZON_ANALYSIS.md) |
| **知道最近做过哪些实验、结论是什么** | [MODEL_IMPROVEMENT_PLAN.md](MODEL_IMPROVEMENT_PLAN.md) §5.1–5.21 |
| **排查我踩过的坑** | [../lessons.md](../lessons.md) |
| **A 股系统的完整设计** | [A_STOCK_DESIGN.md](A_STOCK_DESIGN.md) |
| **每日跑分析、看输出** | [../AGENTS.md](../AGENTS.md) → [../README.md](../README.md) |
| **新电脑配 SSH** | [SSH_SETUP.md](SSH_SETUP.md) |

---

## 二、完整目录

### A. 核心活文档（做判断前必读，随每轮验证更新）

| 文档 | 回答什么 | 更新频率 |
|---|---|---|
| [DECISIONS.md](DECISIONS.md) | **做了什么决定、为什么**（D1–D10） | 每轮结论后 |
| [QUANT_SYSTEM_METHODOLOGY.md](QUANT_SYSTEM_METHODOLOGY.md) | **系统该怎么建**（五层 / 七原则 / 七阶段 / 三道闸门） | 低频（方法级，长期有效） |
| [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) | **怎么验**（指标定义 / 月度护栏 / 最新验证结果） | 每轮 walk-forward |
| [DEPLOYMENT.md](DEPLOYMENT.md) | **实盘怎么用**（底仓 + 战术 + 风控 + 仓位） | 闸门判定变化时 |
| [MODEL_IMPROVEMENT_PLAN.md](MODEL_IMPROVEMENT_PLAN.md) | **做过哪些实验、结论**（§5.1–5.21 实验日志） | 每个实验后 |
| [THREE_HORIZON_ANALYSIS.md](THREE_HORIZON_ANALYSIS.md) | **三周期信号的真实价值**（八模式 / 传导） | 每轮复测 |
| [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) | **怎么设计与验证特征** | 特征管线变更时 |

### B. 设计与规范

| 文档 | 回答什么 |
|---|---|
| [A_STOCK_DESIGN.md](A_STOCK_DESIGN.md) | A 股系统完整设计（53 只池 / 样本权重 / 涨跌停处理） |
| [programmer_skill.md](programmer_skill.md) | 开发规范（需求分析 / 整体设计 / 公共代码提取 / 改完即测） |

### C. 交易与操作指引

| 文档 | 回答什么 | 时效性 |
|---|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | 仓位架构与每日/每月复核流程 | 活文档 |
| [BANK_AND_ETF_TRADING_GUIDE.md](BANK_AND_ETF_TRADING_GUIDE.md) | 银行股与盈富基金买卖指引 | 规则类 |
| [SECTOR_ROTATION_TRADING_RULES.md](SECTOR_ROTATION_TRADING_RULES.md) | 板块轮动交易法则 | 规则类 |
| [ANOMALY_DETECTION_GUIDE.md](ANOMALY_DETECTION_GUIDE.md) | 异常检测的用法与限制 | 规则类 |
| [CLASSIC_TRADING_THEORIES.md](CLASSIC_TRADING_THEORIES.md) | 股票买卖经典理论（参考） | 静态参考 |

### D. 分析报告（**快照，带日期，不追新**）

> ⚠️ 以下为 **2026-04 前后的横截面分析**，反映当时数据；结论**未经 PIT/embargo 复验**，
> 不得直接当交易信号。若要更新，需按 [QUANT_SYSTEM_METHODOLOGY.md](QUANT_SYSTEM_METHODOLOGY.md) 的
> 时点与口径要求重跑。

| 文档 | 快照日期 | 内容 |
|---|---|---|
| [SECTOR_ROTATION_ANALYSIS.md](SECTOR_ROTATION_ANALYSIS.md) | 2026-04-17 | 10 板块轮动规律（736 交易日） |
| [STOCK_NETWORK_ANALYSIS.md](STOCK_NETWORK_ANALYSIS.md) | 2026-04-30 | 46 股 / 15 板块网络（MST+PMFG+…） |
| [STOCK_CORRELATION_ANALYSIS.md](STOCK_CORRELATION_ANALYSIS.md) | 2026-04-28 | 股票关联与因果关系 |
| [FEATURE_IMPORTANCE_ANALYSIS.md](FEATURE_IMPORTANCE_ANALYSIS.md) | 2026-04（局部更新） | 多周期特征重要性 Top10 |

### E. 环境与工程

| 文档 | 回答什么 |
|---|---|
| [SSH_SETUP.md](SSH_SETUP.md) | 新电脑配置 GitHub SSH 认证 |

---

## 三、本目录之外的关键文档

| 文档 | 定位 |
|---|---|
| [../AGENTS.md](../AGENTS.md) | **总入口**：规则、常用命令、最新验证数值、核心警告 |
| [../README.md](../README.md) | 项目介绍与架构 |
| [../lessons.md](../lessons.md) | **踩过的坑**（编号 + 版本日志，事件级，会累积） |
| [../progress.txt](../progress.txt) | 逐日进展流水（过程记录） |
| `.opencode/command/model_validation.md` | 模型验证 S.O.P.（walk-forward → 5.5 → 5.6 全流程） |

---

## 四、三类文档的分工（避免读错）

| 你想知道 | 读 | 不要读 |
|---|---|---|
| **做了什么决定** | `DECISIONS.md` | （结论会过期，别去旧报告里翻） |
| **系统该怎么建** | `QUANT_SYSTEM_METHODOLOGY.md` | （方法长期有效，别把项目结论当方法） |
| **踩过什么坑** | `lessons.md` | （别重复犯同一个错） |
| **怎么验** | `VALIDATION_GUIDE.md` | （别拿绝对准确率当指标） |

---

## 五、维护规则

1. **活文档**（A 组）：每次 walk-forward / 决策变更后同步刷新，数值须能从入库 CSV 复现。
2. **快照文档**（D 组）：只在重跑时整体替换，标题保留快照日期；**不逐条修补**。
3. **新增文档时**：同时更新本索引；若属"长期有效"，在头部标注定位（参考 `QUANT_SYSTEM_METHODOLOGY.md` 文档分层）。
4. **口径变更时**：按原则 5（立宪 + 清洗 + 审计）全库检查旧数字，本索引同步加废弃标注。
