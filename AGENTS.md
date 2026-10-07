# 金融资产智能分析系统 - opencode 规则

> 本文件是 opencode 的规则入口，已通过 `opencode.json` 的 `instructions` 字段自动加载。
> **📏 只放「每次会话都要遵守的指令」**（业界基准 **<500 行** —— Cursor 官方上限；超出即说明参考资料该外迁）。参考资料按需查阅、**不随会话加载**：
>
> | 要看什么 | 去哪 |
> |---|---|
> | ⭐ **全部 Walk-forward 实测数字** | [docs/BASELINES.md](docs/BASELINES.md) — **唯一真相源**，其他文档一律指回此处 |
> | 三道闸 8 项清单 / 两条硬约束 / 实验方法论正文 | [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md) — 含全部案例与失职复盘 |
> | 数据流 / 特征架构 / 环境变量 / 自动化调度 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
> | 经验教训（因果链与历史观测值的**唯一出处**） | [lessons.md](lessons.md) |
> | 文档索引（"该看哪份"导航） | [docs/README.md](docs/README.md) |
> | 编程规范 | [docs/programmer_skill.md](docs/programmer_skill.md) |
> | 进度 / 项目介绍 | [progress.txt](progress.txt) / [README.md](README.md) |

---

## 📋 项目概览

**双市场支持**：
- 🇭🇰 **港股** - 恒生指数三周期预测、个股预测、异常检测（32只自选股；2026-10-06 剔除 2 只 ETF 后）
- 🇨🇳 **A股** - 三周期预测、综合买卖建议、板块分析（53只股票池）

**核心理念**：人机混合智能 - 融合大模型推理能力与机器学习预测精度

---

## ⚡ 常用命令

### 测试与验证

```bash
# 语法检查（每次修改后必须执行）
python3 -m py_compile <文件路径>

# 运行所有测试
python3 -m pytest tests/ -v

# 运行单个测试
python3 -m pytest tests/test_anomaly_integrator.py -v
```

### 核心功能命令

#### 港股系统

| 任务 | 命令 | 运行时机 |
|------|------|---------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` | 收市后 |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` | ⚠️ 收市后（16:00 HKT） |
| **个股详细分析** | `python3 comprehensive_analysis.py --stocks 2318.HK` | 收市后 |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` | 收市后推荐 |
| **个股Walk-forward验证** | `scripts/run_walk_forward.sh --model-type catboost --horizon 20 --use-feature-selection`（确定性 env 固化，双跑验收/复现必用；直接 python3 等价但不固化 `PYTHONHASHSEED`/线程数） |
| **降噪对照实验** | `... run_walk_forward.sh ... --top-k 30`（每折特征数；降噪路径已实证排除，见 lessons 三.37） |
| **⭐ 可复现跑（推荐）** | `python3 scripts/pin_macro_snapshot.py` 先冻结快照，再 `US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot GATE_SOURCE_CSV=output/<基线>/prediction_analysis.csv scripts/run_walk_forward.sh ...`——**双冻结，缺一不可**（lessons 三.29/三.30）。验收=同配置连跑两次 CSV md5 相同 |
| **跨快照稳健性** | `python3 scripts/vintage_sensitivity.py --horizon 20 --topk 10` —— 扫描全部港股 20d 快照输出 [min,max] 区间（**禁止取最好一轮**） |
| **相对 alpha 专用校验** | `python3 scripts/rel_alpha_check.py --input <csv> --horizon 20` —— 重算 Relative_Return，**含 Fold 聚类显著性**（行级 z 是伪显著） | 成功后自动入库（见下方 Git 规范；`--no-commit` 关闭） |
| **恒指Walk-forward验证** | `python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20` | - |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection` | - |
| **生产 LightGBM 20d** | `python3 scripts/train_lightgbm_20d.py` | 20d 信号默认学习器（A/B 胜出，见 §5.18）；**1d/5d 维持 CatBoost**（§5.20/D10） |
| **模型预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost --use-feature-selection` | - |
| **特征选择** | `python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20` | - |
| **超参数调优** | `python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30` | - |
| **股票网络分析** | `python3 ml_services/stock_network_analysis.py --skip-pmfg` | - |
| **回测评估（lift/方向技能）** | `python3 ml_services/backtest_eval.py --input output/<dir>/prediction_analysis.csv --horizon 20` | ⭐ 评估必用 |
| **月度护栏（净IR/PBO/DSR）** | `python3 ml_services/monthly_guardrail.py --horizon 20` | 每次 Walk-forward 后必跑，判定见 `docs/DECISIONS.md` D2 |
| **组合层复核（超额CI/逐年）** | `python3 ml_services/portfolio_backtest.py --horizon 20 --topk 10 --pred output/<dir>/prediction_analysis.csv` | 每次 20d Walk-forward 后另跑（技能阶段 5.7）；超额IR CI 下界>0 且逐年多数为正 |
| **性能监控** | `python3 ml_services/performance_monitor.py --mode all --no-email` | - |
| **风险回报率分析** | `python3 ml_services/risk_reward_analyzer.py --stocks watchlist --style moderate` | - |

#### A股系统

| 任务 | 命令 | 运行时机 |
|------|------|---------|
| **A股综合分析（完整流程）** | `./scripts/run_a_stock_analysis.sh` | ⚠️ 收市后（15:15 CST） |
| **A股模型训练** | `python3 a_stock_ml_model.py --mode train --horizon 20` | - |
| **A股模型预测** | `python3 a_stock_ml_model.py --mode predict --horizon 20 --core-only` | - |
| **A股大模型建议** | `python3 a_stock_email.py --force --no-email` | - |
| **A股综合分析** | `python3 a_stock_comprehensive_analysis.py --llm-file data/a_stock_llm_*.txt --use-cached-predictions` | - |
| **A股Walk-forward验证** | `bash scripts/run_a_stock_walk_forward.sh --horizon 20`（固化确定性 env；直接 python3 等价但不固化线程数/哈希种子） | 成功后自动入库 CSV（`--no-commit` 关闭） |

### 缓存管理

```bash
# 清除港股特征缓存（新增特征后必须执行）
rm -rf data/feature_cache/*.pkl

# 清除港股原始数据缓存
rm -rf data/stock_cache/*.pkl

# 清除A股特征缓存
rm -rf data/a_stock_feature_cache/*.pkl

# 清除A股原始数据缓存
rm -rf data/a_stock_cache/*.pkl
```

### 安装与配置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp set_key.sh.sample set_key.sh
# 编辑 set_key.sh，填写邮箱和API密钥
source set_key.sh

# 3. 验证安装
python hsi_email.py --no-email
```

> 环境变量与依赖清单见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §环境配置

---

## ⚠️ 核心警告

> **一行一条，只写「规则 + 判定式 + 指针」。**
> 因果链 / 实验过程 / 历史观测值 → [lessons.md](lessons.md)；实测数字 → [docs/BASELINES.md](docs/BASELINES.md)；
> 三道闸与硬约束正文 → [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)。

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward 准确率 **>65%（个股）/ >80%（恒指）** 即泄漏信号（个股正常区间 50–60%）。**三条硬检查**：①`Predict_Prob` 饱和成 0/1；②预测 UP 组 `Relative_Return`（= Actual_Return − HSI未来收益，**CSV 无此列须重建**）**最大值 ≤0 = 符号退化**（合法数据实测 **+0.42**、泄漏版 **−1e-05**，勿把方向抄反）；③分组符号一致性。**可执行：`python3 scripts/presentation_gate.py --input <csv> --horizon N`（非零退出即阻断）**。出处与抄反事故见 lessons 三.28 / 三.47 |
| **判读顺序不可颠倒** | ①绝对值异常(准确率/IC/概率饱和) → ②PBO<0.5 → ③DSR≥0.95 → ④净IR CI 不跨0+lift>0 → ⑤才谈仓位。**PBO/DSR 检测不出泄漏**（实测泄漏版 PBO 0.43、DSR 1.000 全过，实际是假信号），见 lessons 三.28 |
| **静默 NaN 特征** | 外部行情(HSI/美股)为午夜时间戳、个股为收盘 16:00，裸 `merge(left_index,…)` 会整列 NaN 且**不报错**；一律走 `_align_to_index`；修对齐后必须清特征缓存再验证（lessons 三.27） |
| **取数窗口硬编码** | `period_days=1460` 曾写死在个股取数处，使 2016–2021 永远取不到（宏观修了个股漏修）；改口径须 grep 全部调用点，一律用 `period_days_needed`（lessons 三.26） |
| **验证产物完整性** | fold 异常被吞会静默产出残缺 CSV（17/38 折仍打印"验证完成"）；跑完必核对**折数/行数**（lessons 三.20，已加完整性闸门缺折即非零退出） |
| **Walk-forward 复现口径** | **同机同代码双跑现已 bit 级复现**（2026-09-28 修复互信息无种子/缓存非原子写/模型非确定参数，lessons 三.25）——不复现即 bug，用 `WF_X_DETAIL=1` 列指纹定位（`scripts/run_walk_forward.sh` 固化 env）；**跨日/跨机**对比只看 lift/护栏等决策指标，禁逐行 diff CSV（三.22）；TopK 组合层指标（39→71 期）单轮判定不可信 |
| **静态快照穿越** | 网络/情感/主题/基本面等「最新值广播到全部历史行」= 未来穿越，回测须用 PIT 时点还原。**修 PIT 必须同步修 `community_ids` 集合来源**——`create_market_network_interaction_features` 遍历**传入列表**而非 `df.unique()`，PIT 集合比静态多一个时，多出的社区会**静默**丢失全部 `net_constraint_*` 交叉特征。A股污染量化与四次归因实验见 D13 |
| **绝对准确率/胜率陷阱** | 趋势板块基准本就高；评估看 **lift（胜率−基准）** 与 **方向技能（准确率−永远看涨）**，不看绝对值 |
| **降噪救不回信号** | Top500→Top30 单变量对照（同窗口/折数/学习器/快照/门槛基准）：净IR 0.42→0.64 但 **CI 仍跨 0、PBO 越过门槛、DSR 不过** → 三闸门全不过。**路径关闭基于纪律**——勿再调 Top50/100（等于在噪声里捞信号，违反预注册纪律）。见 lessons 三.37 |
| **个股横截面 alpha 已穷尽** | 完整管线 DSR 0.87 不过、扩池不增信号、简单模型 IC 转负；2026-09-30 相对标签（剥离 Beta）受控复验仍未 rescue → **停止投入**，见 D1 |
| **恒指方向信号不优于持有** | 信号 Long/Flat 1/5/20d IR（−0.68/0.13/0.11）均低于买入持有（0.38/0.39/0.49） |
| **异常大跌抄底是真信号但强依赖行情** | z≤−3 持 5d 净IR 1.55 显著；但逐年依赖 2025，**仅限大盘上行期（恒指>MA200）战术使用**；短期强势过滤有害 |
| **港股个股信号多为"2025 现象"** | 多策略复核收益集中于 2025 反弹年；跨年稳健性普遍不足，须逐年看 |
| **IC 计算** | IC 必须用实际收益率，不能用二元标签；收益率计算必须与训练一致 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **CatBoost 1天模型** | 噪音大，仅供参考 |
| **学习器不可全局替换** | LightGBM 并非全面更优：20d LGBM 赢、**5d CatBoost 更优**（lift/IR/PBO/DSR 六项全占优、2026 转负）、1d 两者净IR≤0 双停 → 换学习器必须按周期分别 A/B（§5.20/D10） |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **恒指 vs 个股** | 恒指 20d 59.1%（n_eff≈35 不显著）vs 个股 20d 51.8%（不显著）；**恒指三周期模式样本量小、个股模式样本量大但效应 ≤±2.7pp，均不可交易** |
| **高置信度风险** | 高置信度预测错误时损失可达 -73%，必须设置止损 |
| **网络社区特征一致性** | 训练时保存 `model.community_ids`，预测时使用相同社区 ID 列表 |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测预处理必须一致 |
| **默认值设计** | 默认值必须与有效值范围分离，使用 -1 表示"未知"，基本面特征用 NaN |
| **训练时 NaN** | 不要用 `df.dropna()` 删除所有 NaN，只删除标签和关键列 |
| **绝对值特征** | 跨股票训练时，绝对价格/成交量特征必须标准化或排除 |
| **市场情绪数据源** | 必须使用所有股票收益率计算上涨比例，与 walk-forward 验证一致 |
| **双模式预测** | 收市后预测使用 `mode='production'`（当日数据），Walk-forward 使用 `mode='backtest'`（T-1 数据） |
| **yfinance 盘中数据** | yfinance 日线数据在盘中可能不准确，推荐使用腾讯财经接口获取实时报价 |
| **复权口径一致性** | 训练/预测/评估统一用**腾讯前复权（qfq）**（港股 `get_hk_stock_data_tencent`、A股 `get_a_stock_data`）；性能监控 exit 价必须 qfq 同源，禁用 yfinance 未复权价当 exit（跨除息日 actual_return 失真，实测虚高 62%，见 lessons 三.17） |
| **校准概率与方向同口径** | Isotonic 校准后 `direction`/`pattern` 必须按**校准概率 ≥0.5** 重判（`apply_to_results` 已同步 + `rebuild_three_horizon_patterns` 重算模式），否则邮件出现「↑ 0.49」自相矛盾，见 lessons 三.19 |
| **A股数据源 640 行硬上限** | ⚠️ 腾讯 A股接口**无论 `period_days` 给多少恒返回 640 行**（实测 640→641 / 1500→640 / 3000→640），约 2.5 年上限。加 12 个月训练窗后**可用测试期不足 2 年**（实测 2025-01→2026-07，2025 占 63.6%）→ **低 PBO/高 DSR 不可信**，是 A股不予升配的硬限制（D13）。AKShare 备源为备选 |
| **A股涨跌停差异** | 主板10%涨跌停，创业板20%涨跌停，混合训练时需标签标准化 |
| **A股股票代码前导零** | 保存CSV时必须用字符串格式 `zfill(6)`，否则前导零丢失（002655→2655） |
| **A股样本权重** | 核心股权重3.0倍，扩展股1.0倍，训练时需传入 `sample_weight` |
| **行级显著性=伪显著** | Walk-forward 的行级 z 值忽略折间+横截面相关（同折共享模型、同日 58 股共享同一 HSI 未来收益），有效独立观测≈**折数**而非行数。实测行级 z=6.3 → Fold 聚类 t=1.40 **p=0.167**。任何命中率/准确率/lift 的显著性必须**按 Fold 聚类后检验**（lessons 三.29） |
| **GATE 同源三依赖** | 门槛/校准器/快照/基准CSV 必须同源：①宏观 `US_MARKET_SNAPSHOT_DIR` ②分位基准 `GATE_SOURCE_CSV` ③Isotonic 校准器须随模型重训**用 OOF 重拟合**（`oof_glob`，别用生产历史——新模型无历史可拟合）。**`bear==weak` 通常是概率被压平（无 edge），不是分位算错**（lessons 三.30） |
| ~~跨快照数值极差~~ | ❌ **已撤销（2026-10-05）**：曾归因「跨快照漂移」，实为**旧特征缓存内容差异**——清缓存后连跑两次 md5 一致、同配置极差 **0.00**。**教训**：指标分叉时先做「清缓存重跑」判别（最便宜），再谈机制解释；见 lessons 三.42 |
| ~~港股数据末日敏感性~~ | ❌ **已撤销（2026-10-05）**：`WALKFORWARD_DATA_END` 解决的是**不存在的问题**（数据源当日本就只到 07-30），反引入实截断。**教训**：指标变化先问「基线是否也有此差异」——基线末日同为 07-30 则末日不是变量 |
| **跨日回测不可复现** | `us_market_data` 缓存**仅当天有效**且是宏观特征唯一来源 → 同代码跨天跑结果不同（实测 abs20d 净IR 0.34/0.42、PBO 0.41/0.64）。三.25 的 bit 级复现**仅限同一天**；**多跑几次总能撞到三门槛全过**，禁止取最好的一次当结论 |
| **冻结必须双向落盘** | 只写成功数据、**失败不写哨兵** = 没冻结（每次重试，成败纯看网络）。**只做一半的冻结比不冻结更危险**——制造「已冻结」的错觉。两个易错点：①`load_frozen` 须**数据优先于哨兵**；②冻结模式下抓取失败写 `.failed` 且**不再重试网络**。见 lessons「38. 冻结机制必须双向落盘」 |
| **市场门槛优先分位、退化回退绝对值** | 分位法隐含前提「模型有 edge」；**模型无 edge 时校准把概率压回基准率 → 分位锚在 0.5 → 门控静默失效**（实测 bear 通过率 17.2%→98.2%）。已加退化保护（`GATE_MIN_UNIQUE=30`/`GATE_MIN_SPREAD=0.05`），退化即回退 bear 0.70/weak 0.65。**`bear==weak` 是分布退化信号，不是分位算错**。分位口径：熊市/弱震荡用 `GATE_QUANTILES` P92/P90（PIT），数据源须为 walk-forward 回测分布（`prediction_history` 右尾过窄会失效）；见 lessons 三.30、D8 |

---

## 🤖 机器学习模型

> ## ⭐ 基线与可信度数字已全部外迁 → **[docs/BASELINES.md](docs/BASELINES.md)**
> 恒指三周期 / 三周期模式 / 个股完整模型 / 86-fold / 新池 10-06 当前有效基线，**只写那一份**。
> 本节只保留**不随重跑变化**的模型设计（配置、输出文件、过滤器、判据）。

**新增利率特征**（2026-05-23）：
- 多期限美债收益率：US_2Y_Yield, US_10Y_Yield, US_30Y_Yield
- 中国国债收益率：CN_10Y_Yield
- 期限利差：US_2Y_10Y_Spread（收益率曲线斜率）, US_10Y_30Y_Spread
- 中美利差：CN_US_10Y_Spread（资金流向驱动）, CN_US_Spread_Change_5d, CN_US_Spread_Z_Score
- 通过网络交叉特征区分个股（如 `net_centrality_CN_US_10Y_Spread`）

**特征选择**：使用 Top 500 特征，特征减少 55.8%，性能优于全量特征

**Walk-forward 输出文件**（保存到 `output/YYYYMMDD_HHMMSS_catboost_20d/`）：
- `fold_metrics_detail.json` - 每个 Fold 的指标 + **Top 100 特征重要性**
- `prediction_analysis.csv` - 所有预测详情（用于 Fold 盈亏比分析）
- `validation_summary.json` - 总体验证结果

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨 |
| 特征数量 | ~1450 → 500 | 推荐使用 Top 500 特征选择 |
| 随机种子 | 42（固定） | 确保可重现性 |

**20天模型参数**（超参数优化后）：

| 参数 | 值 |
|------|-----|
| n_estimators | 400 |
| depth | 8 |
| learning_rate | 0.06 |
| l2_leaf_reg | 2 |
| subsample | 0.75 |
| colsample_bylevel | 0.8 |

### 新特征上线验证清单

详见 [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)，8个验证步骤：

1. 泄漏检查 - 所有特征使用 `shift(1)`
2. **绝对值特征标准化** - 跨股票训练必须标准化
3. **市场级特征交叉** - 对所有股票同值的特征必须交叉
4. **特征单调性** - 交叉特征保持逻辑单调性
5. Walk-forward 验证 - 准确率达标
6. SHAP 排名 - 进入 top 30
7. Pearson 相关性 - 与现有特征 < 0.8
8. 随机种子稳定性 - 波动 < 2%

### 市场情绪过滤器

**核心原理**：市场上涨比例有强自相关性（lag=1 自相关系数 0.929），滞后1天数据能有效识别极端市场环境。

**阈值分层**：

| 层级 | 上涨比例 | 动态阈值 | 操作 |
|------|---------|---------|------|
| extreme_bear | <20% | 1.0 | 暂停交易 |
| bear | 20-30% | 0.70 | 高置信 |
| weak | 30-40% | 0.65 | 谨慎 |
| normal | >40% | 0.50 | 标准 |

**验证效果**：准确率 62.0% → 70.7%（+8.7%），总收益 +63.44

**代码**：`ml_services/market_regime.py` - MarketSentimentFilter 类

**使用要点**：
- 数据源：使用所有股票收益率计算上涨比例，与 walk-forward 验证一致
- 无前瞻性偏差：严格使用滞后1天数据（`lookback_days=1`）
- 生产集成：`comprehensive_analysis.py` 中已集成

### 市场状态稳定性检测

**核心原理**：HMM 市场状态持续时间（Regime_Duration）反映状态稳定性，短持续时间意味着频繁转换，预测可靠性下降。

**状态稳定性判断**：

| Regime_Duration | 稳定性 | 建议 |
|-----------------|--------|------|
| < 5 天 | ⚠️ 不稳定 | 降低仓位 |
| 5-15 天 | 🟡 中等 | 正常交易 |
| > 15 天 | ✅ 稳定 | 趋势明确 |

**代码**：`data_services/regime_detector.py` - RegimeDetector 类

**生产集成**：
- `comprehensive_analysis.py` 的 `get_current_market_state()` 函数
- 邮件报告中展示市场状态持续时间

---

## 🔧 开发规范

### 代码修改原则

1. **修改完即测试**：每次修改后立即执行 `python3 -m py_compile <文件>`
2. **避免硬编码路径**：使用 `os.path.dirname(os.path.abspath(__file__))` 获取脚本目录
3. **HTTP API 超时处理**：调用 API 时必须设置超时时间——**含依赖链内部请求**（akshare 等库函数常无 timeout，会静默挂死整条流水线，见 lessons 三.21）
4. **语言规范**：对话和注释使用简体中文，变量名/函数名使用英文

### 🚨 对抗性审核：三道闸（强制，不可跳过）

> **8 项审核清单、两条硬约束的失职复盘、实验方法论全文与案例 → [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)**
> 此处只留**触发时机 + 硬规则**，防止「改了 AGENTS 却没改规程」的双写漂移。

三个触发时机（**其中两个与「有没有改动」无关**）：

| 闸 | 触发时机 | 判据 |
|---|---|---|
| **A 呈现闸** | 任何指标要作为**结论**输出前 | 绝对值异常 → `python3 scripts/presentation_gate.py --input <csv> --horizon N`，**非零退出即阻断** |
| **B 提交闸** | 任何代码/配置/文档改动后、commit 前 | [8 项清单](docs/REVIEW_GATES.md)逐项过，任一不过即重审 |
| **C 升级闸** | 任何 🟢/升配/加仓/改阈值 决策前 | 双冻结 + 双跑 md5 一致 + 禁止取最好一轮 |

**B 闸首条**：改动完成后，**像攻击对手一样攻击自己的改动**，确认无问题才能提交。

**两条硬约束（规则版）**：

1. **因果结论必须先验证再写** —— 凡「因为 X 所以 Y」必附**聚类检验或复算证据**；
   **行级统计不得用于因果推断**（实测行级 z=6.3 → Fold 聚类 t=1.40, **p=0.167** 不显著）。
2. **得出 🟢/🟡/🔴 或任何阈值/仓位变更后，强制**：

   ```bash
   grep -nE "仓位|放大|可升至|≤[0-9]+%|净IR [0-9]" docs/DEPLOYMENT.md README.md
   ```

   逐条**读**命中行，确认**没有与当前判定相反的无条件现状建议**。
   「🟢 升级 → 才可放大至 15–20%」这类**条件规则是对的，勿误删**。

**实验方法论三条（跑实验前）**：

1. **先写「本次改动会连带改变哪些东西」** —— 答不出即尚未理解该改动，不该开跑。
2. **预注册判据 > 让 AI 复述需求** —— 看到结果**之前**，把成功/失败写成数字或区间。
3. **验证前先校验「数据非空」，再校验语义** —— **空数据不能证伪任何命题**。

**判读顺序不可颠倒**：绝对值异常 → PBO<0.5 → DSR≥0.95 → 净IR CI 不跨0 + lift>0 → 才谈仓位。

---

### 数据泄漏防护

高风险特征必须使用 `.shift(1)` 避免使用当日数据：
- 所有 `.rolling()` 计算的特征
- `future_return` 必须使用 `.shift(-N)` 计算未来收益

```python
# ❌ 错误：使用当日数据
future_return = returns.rolling(5).sum()

# ✅ 正确：使用未来数据
future_return = returns.rolling(5).sum().shift(-5)
```

### CatBoost 分类特征处理

训练和预测时必须一致处理分类特征 NaN：
```python
# 训练时
df[col] = df[col].fillna('unknown').astype(str)
encoder = LabelEncoder()
df[col] = encoder.fit_transform(df[col])

# 预测时
test_df[col] = test_df[col].fillna('unknown').astype(str)
test_df[col] = test_df[col].apply(
    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
)
```

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
  - **例外**：回测 `prediction_analysis.csv` 由 `scripts/commit_backtest_result.py` 自动入库
    （分位门槛数据源，CI/本地须同源；每次入库自动 `git rm --cached` 旧的港股 20d CSV
    只留最新，防仓库膨胀；快照维护见 `ml_services/market_regime.py` 注释）
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`

---

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**对抗性审核有三个触发时机（见 [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)）**：
- **指标呈现前**：任何数字要作为结论汇报时，先查绝对值异常（可能一个字都没改）
- **改动提交前**：任何代码/配置/文档改动后，攻击自己的改动再 commit
- **升级决策前**：任何 🟢/升配/加仓/改阈值前，双冻结 + 双跑 md5 一致

2026-10-02 实证两次：①**未改任何代码**，仅因汇报 rel20d 结果触发 A 闸，
抓到 100% 准确率泄漏（PBO 0.43/DSR 1.000 全过却是假信号）；
②文档同步触发 B 闸，发现 4 处自相矛盾，其中 `DEPLOYMENT.md` 仍写着
已被证伪的「🟢 可放大至 15%」——若照此实盘会**加倍错误仓位**。

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

**模型更新后**：运行 Walk-forward 验证确认性能，使用 `/model_validation` 命令执行标准验证流程

**特征修改后**：清除缓存 `rm -rf data/feature_cache/*.pkl`

---

## 🔗 快速链接

- **⭐ Walk-forward 基线（唯一真相源）**：[docs/BASELINES.md](docs/BASELINES.md)
- **⭐ 三道闸 / 硬约束 / 实验方法论正文**：[docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)
- **架构 / 特征架构 / 环境变量 / 调度**：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **特征工程**：[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) - 完整指南（含案例分析）
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **验证方法**：[docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md)
- **模型改进计划**：[docs/MODEL_IMPROVEMENT_PLAN.md](docs/MODEL_IMPROVEMENT_PLAN.md) - 业界基准驱动 ⭐
- **决策备忘**：[docs/DECISIONS.md](docs/DECISIONS.md) - 个股 alpha 停止投入等关键决策 ⭐
- **建设方法论**：[docs/QUANT_SYSTEM_METHODOLOGY.md](docs/QUANT_SYSTEM_METHODOLOGY.md) - 五层结构 / 七原则 / 七阶段 / 三道闸 ⭐
- **量化交易误解**：[docs/量化交易误解-正式文档.md](docs/量化交易误解-正式文档.md) - 四道陷阱方法论注脚 ⭐
- **实盘部署**：[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - 底仓+战术+风控与月度护栏 ⭐
- **A股设计**：[docs/A_STOCK_DESIGN.md](docs/A_STOCK_DESIGN.md) - A股系统完整设计文档
- **SSH认证**：[docs/SSH_SETUP.md](docs/SSH_SETUP.md) - 新电脑配置 GitHub SSH 认证指南