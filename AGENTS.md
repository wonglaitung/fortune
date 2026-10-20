# 金融资产智能分析系统 - opencode 规则

> 本文件是 opencode 的规则入口，已通过 `opencode.json` 的 `instructions` 字段自动加载。
> **📏 只放「每次会话都要遵守的指令」**（业界基准 **<500 行** —— Cursor 官方上限；超出即说明参考资料该外迁）。参考资料按需查阅、**不随会话加载**：
>
> | 要看什么 | 去哪 |
> |---|---|
> | ⭐ **全部 Walk-forward 实测数字** | [docs/BASELINES.md](docs/BASELINES.md) — **唯一真相源**，其他文档一律指回此处 |
> | ⓪ 同一性闸 + 三道闸 9 项清单 / 两条硬约束 / 实验方法论正文 | [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md) — 含全部案例与失职复盘 |
> | 非专家怎么跟 AI 协作（命题前置 / 判读顺序 / 纠偏 / 收尾检查表） | [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md) |
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

## 🧩 三大功能（2026-10-16）

> 项目**现在**的定位——**两个市场预测系统 + 一个假设验证框架**。注意性质不同：

| # | 功能 | 是什么 | **诚实定性** |
|---|---|---|---|
| ① | 🇭🇰 **港股预测系统** | 恒指三周期 + 个股预测 + 异常检测 | **能跑的系统；alpha 已验证 = null**（三层/交叉/风格/关联，见三层总账） |
| ② | 🇨🇳 **A股预测系统** | 三周期 + 综合买卖建议 + 板块分析 | **能跑的系统；alpha null + 数据硬顶 640 行 ⇒ 不予升配**（D13） |
| ③ | 🧾 **假设验证框架**（"尺子" + 技能） | `REVIEW_GATES` + 三道闸 + `strategy_eval` + `hypothesis-validation` 技能 | **项目最可信的资产**——真正"经得起检验"的部分 |

- **外围辅助**（非核心预测）：crypto / 黄金监控、性能监控、新闻抓取、SFC/南向取数。
- ⚠️ **一句话**：**①② 的价值在工程 / 展示 / 流程 / 风控，不在"能赚钱"；③ 才是"科学"那部分。**
- 用尺量任何策略前，先走技能 `.opencode/skills/hypothesis-validation/SKILL.md`（G1–G9）。

---

## 🧾 三层总账（alpha 结论 · 唯一入口，2026-10-16）

> 一句话：**"大市 / 板块 / 个股差异"三层，及其交叉 / 风格 / 关联传播，我们全都单独测过，挖 alpha 全部失败。**
> 细节见 `docs/DECISIONS.md`（D1 口径备忘旁的总账）与各 D 号。

| 层 | 是什么 | 模型 / 实验 | 结论 |
|---|---|---|---|
| ① **大市** | 恒指方向（涨/跌） | ML 择时（`hsi_*`）+ 经典规则（**D43**） | ❌ ML 不优于持有；简单规则也**仅 MA200 单点过、相邻参数即垮（不稳健）** |
| ② **板块** | 行业轮动（哪个板块强） | 板块动量轮动（D36 / D37） | ❌ 主判据不过 + 独立窗口不复现 |
| ③ **个股差异** | 个股跑赢同类（剥①②后的残差） | 个股横截面模型 | ❌ 全周期全 null（D1 / D33 / D34 / D35） |
| ④ **风格** | 价值 / 质量 / 动量 / 低波 | 价格型（D39）+ PIT 近似基本面（D41 / D42） | ❌ 价格型 null；**E/P 最接近（IR 0.64）但独立窗为负、滞后加大即弱 ⇒ 钉死**；规模无股数史 ⇒ 算不了 |
| ⑤ **关联 / 题材** | 题材内挑股（分区）+ 沿关联边传播 | **D51**（题材分区）+ **D52**（E1 A-H lead-lag） | ❌ 题材内 **N_eff≈1 ⇒ 结构性测不出**；E1 边可交易方向 **null** |

⇒ **①×② 交叉（D38）也 ❌**（分状态切换"周期/防御"，0/4，且不如"死拿周期组"）。
⇒ **"理念 / 形态"编码类也 ❌**：技术形态 → **不立项（D44）**；大师 / 短线理念→特征 → **不立项（D46，适用范围见 D47）**。
⇒ **唯一"有点东西"的"降回撤"最后也证明只是"少投"**（D35 §十二）。
⇒ **V4 换标签（rank 横截面回归）也不救**：**D59**——个股标签 `1/5/20 二分类 → 5/10/20 rank 分位回归`，C 闸（净IR/PBO/DSR/超额CI）**三周期全不过**，仍是 alpha null；**该改写是口径/架构变更，不是 alpha 提升**。
⚠️ **口径 = "未发现（不显著）" ≠ "已证明没有"**（功效有限，D45 / D49）；**勿把"系统挂了大市/板块组件（regime 门控、HSI 模型、板块展示）"误读为"在分析大市/板块的 alpha"**——那些是风控 / 择时 / 展示层。

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
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` | ⚠️ 收市后（**16:30 HKT**；cron 已改 `30 8` UTC，D25） |
| **个股详细分析** | `python3 comprehensive_analysis.py --stocks 2318.HK` | 收市后 |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` | 收市后推荐 |
| **个股Walk-forward验证** | `scripts/run_walk_forward.sh --model-type catboost --horizon 20 --use-feature-selection`（确定性 env 固化，双跑验收/复现必用；直接 python3 等价但不固化 `PYTHONHASHSEED`/线程数。**D23 起默认 `F1_FEATURES=1 F1_FORCE=1`（例外观察采用；D20 ⑦ 期为 =0）**；复现 2026-10-11 前一切基线须显式 `F1_FEATURES=0 F1_FORCE=0`，复现 F1-on 旧基线（212920/234941/B2）同 =1 但另需 `HIBOR_FEATURES=0`） | - |
| **SFC 卖空周报周更** | `python3 scripts/fetch_sfc_short.py`（增量抓 SFC 官方每周 CSV 追加 `data/sfc_short/`；`--dry` 预览 / `--full` 回填）；已挂进 `run_comprehensive_analysis.sh` 前置步骤 | 每周（SFC 周五发布后） |
| **降噪对照实验** | `... run_walk_forward.sh ... --top-k 30`（每折特征数；降噪路径已实证排除，见 lessons 三.37） | - |
| **⭐ 可复现跑（推荐）** | `python3 scripts/pin_macro_snapshot.py` 先冻结宏观快照，`python3 scripts/pin_hk_snapshot.py` 先冻结个股行情快照，再 `US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot HK_MARKET_SNAPSHOT_DIR=data/hk_market_snapshot GATE_SOURCE_CSV=output/<基线>/prediction_analysis.csv scripts/run_walk_forward.sh ...`——**三冻结，缺一不可**（lessons 三.29/三.30/三.50）。验收=同配置连跑两次 CSV md5 相同。注意：runner **默认 F1=0**（D20 ⑦ 生产口径；复现 F1-on 旧基线须 `F1_FEATURES=1 F1_FORCE=1`）；`data/sfc_short/` 为周更文件不属三冻结——PIT t-7 + WF 窗口止于新周之前 ⇒ 追加不改窗口内特征，跨周复现仍须同版本 CSV；WF `--start-date` 是**全程起点**（测试期起点=start+训练窗），复现测试期 2023-06→2026-07 须 `--start-date 2020-06-01`（lessons 三.52） | - |
| **跨快照稳健性** | `python3 scripts/vintage_sensitivity.py --horizon 20 --topk 10` —— 扫描全部港股 20d 快照输出 [min,max] 区间（**禁止取最好一轮**） | - |
| **相对 alpha 专用校验** | `python3 scripts/rel_alpha_check.py --input <csv> --horizon 20` —— 重算 Relative_Return，**含 Fold 聚类显著性**（行级 z 是伪显著） | 本地校验工具（不改库；入库由 `commit_backtest_result.py` 负责） |
| **恒指Walk-forward验证** | `python3 ml_services/hsi_walk_forward.py --train-window 12 --horizon 20` | - |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection` | - |
| **生产 CatBoost 20d** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection`（F1/HIBOR 默认开，D23/D22） | **20d 信号默认学习器（D20 ⑦ 四格多数判：C1 夺魁，2026-10-10 切换）**；5d/10d 仍 CatBoost（D10；1d 已停用 D59）；LGBM pkl 降级为回退留档；⚠️ F1 与 HIBOR **均例外默认开且均配 FORCE**（D23/D22/D24，各自 env 独立回退）；FORCE 防慢变量被统计选择静默丢弃（着装检查 K4：判据=不得静默丢弃，验证=训练日志 FORCE 纳入行 + gain 榜含该列，**不是**「统计选择 Top500 保留」） |
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
> ⓪ 同一性闸 / 三道闸与硬约束正文 → [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)。

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward 准确率 **>65%（个股）/ >80%（恒指）** 即泄漏信号（个股正常区间 50–60%）。**三条硬检查**：①`Predict_Prob` 饱和成 0/1；②预测 UP 组 `Relative_Return`（= Actual_Return − HSI未来收益，**CSV 无此列须重建**）**最大值 ≤0 = 符号退化**（泄漏版 **−1e-05** 贴零是稳定信号；合法单轮值 1d旧 **+0.42** / 20d旧 **+2.02** / 20dD25新 **+1.58**——**分周期分口径，仅供参照不是阈值**；勿把方向抄反）；③分组符号一致性。**可执行：`python3 scripts/presentation_gate.py --input <csv> --horizon N`（非零退出即阻断）**。出处与抄反事故见 lessons 三.28 / 三.47 |
| **判读顺序不可颠倒** | **⓪ 同一性**（WF 的函数 = 生产的函数？训练时点 = 推理时点？被评估的概率 = 生产下单用的概率？）→ ①绝对值异常(准确率/IC/概率饱和) → ②PBO<0.5 → ③DSR≥0.95 → ④净IR CI 不跨0+lift>0 → ⑤才谈仓位。**⓪ 不过则后面四闸不必跑**（①~④ 问「这个数可不可信」，⓪ 问「这个数说的是不是我要的东西」）。**PBO/DSR 检测不出泄漏**（实测泄漏版 PBO 0.43、DSR 1.000 全过，实际是假信号），见 lessons 三.28 |
| **WF ≠ 生产（同一性缺口）** | ⚠️ **已于 D25/D30 逐条修复（⓪ 四问全通）**，本行保留为问题清单存档：①时点差 1 天（D25）②`net_comm_*` 编码（D30）③特征列集合（O4 清单+契约测试）④1d/5d pkl 同代（例行重训）⑤Isotonic 同源（D26 probability/confidence 拆分）。**原 2026-10-12 实测 5 条独立不一致**（数字与证据出处见 lessons 三.60；**本行不写可变计数，计数一律查 BASELINES.md**）：①**训练/推理时点差 1 天**——`train()` 不传 mode → `use_shift=1`(T-1)，`predict()` 默认 `mode='production'` → `use_shift=0`(T)，WF 全程 T-1（`ml_trading_model.py:4048/4466/4822`）；②**`net_comm_*` 编码**——WF 每股动态提取（缺列=NaN），生产传静态∪PIT（缺列=0），CatBoost 原生区分二者；③特征列集合（生产 pkl = 静态名单 + FORCE 追加 vs WF 每折 Top-500+FORCE，静态名单含 **0 个 HIBOR/0 个 F1**）；④1d/5d pkl 仍是 09-30 旧模型；⑤**Isotonic 只在生产侧**且港股未传 `oof_glob` ⇒ A 闸跑在未校准概率、生产阈值跑在校准后概率。**②（编码实质不同）单独即阻断外推；①（时点错配）单独即令外推不成立**——方向与幅度均未测，不得假设为同向（见 lessons 三.60）。**根因＝口径挂在调用点的默认参数上，方法本身不保证任何东西**（`mode`/`community_ids`/`selected_features` 三入参 × 三路径 = 三种组合）。**⓪ 尚无自动脚本**（靠人工核对），但**契约测试已建**（`tests/test_pipeline_contract.py` 6 条 strict-xfail 锁定上述 5 条差异）|
| **静默 NaN 特征** | 外部行情(HSI/美股)为午夜时间戳、个股为收盘 16:00，裸 `merge(left_index,…)` 会整列 NaN 且**不报错**；一律走 `_align_to_index`；修对齐后必须清特征缓存再验证（lessons 三.27） |
| **取数窗口硬编码** | `period_days=1460` 曾写死在个股取数处，使 2016–2021 永远取不到（宏观修了个股漏修）；改口径须 grep 全部调用点，一律用 `period_days_needed`（lessons 三.26） |
| **验证产物完整性** | fold 异常被吞会静默产出残缺 CSV（17/38 折仍打印"验证完成"）；跑完必核对**折数/行数**（lessons 三.20，已加完整性闸门缺折即非零退出） |
| **Walk-forward 复现口径** | **同机同代码双跑现已 bit 级复现**（2026-09-28 修复互信息无种子/缓存非原子写/模型非确定参数，lessons 三.25）——不复现即 bug，用 `WF_X_DETAIL=1` 列指纹定位（`scripts/run_walk_forward.sh` 固化 env）；**跨日/跨机**对比只看 lift/护栏等决策指标，禁逐行 diff CSV（三.22）；TopK 组合层指标（39→71 期）单轮判定不可信 |
| **静态快照穿越** | 网络/情感/主题/基本面等「最新值广播到全部历史行」= 未来穿越，回测须用 PIT 时点还原。**修 PIT 必须同步修 `community_ids` 集合来源**——`create_market_network_interaction_features` 遍历**传入列表**而非 `df.unique()`，PIT 集合比静态多一个时，多出的社区会**静默**丢失全部 `net_constraint_*` 交叉特征。A股污染量化与四次归因实验见 D13 |
| **绝对准确率/胜率陷阱** | 趋势板块基准本就高；评估看 **lift（胜率−基准）** 与 **方向技能（准确率−永远看涨）**，不看绝对值 |
| **降噪救不回信号** | Top500→Top30 单变量对照（同窗口/折数/学习器/快照/门槛基准）：净IR 0.42→0.64 但 **CI 仍跨 0、PBO 越过门槛、DSR 不过** → 三闸门全不过。**路径关闭基于纪律**——勿再调 Top50/100（等于在噪声里捞信号，违反预注册纪律）。见 lessons 三.37 |
| **个股横截面 alpha 已穷尽** | 完整管线 DSR 0.87 不过、扩池不增信号、简单模型 IC 转负；2026-09-30 相对标签（剥离 Beta）受控复验仍未 rescue → **停止投入**，见 D1 |
| **F1 例外观察采用（D23；原 D20 ⑦ 关闭）** | 默认值**只在 `ml_services/run_manifest.py::FLAG_DEFAULTS` 定义一次**（O5/D25），两个 runner 只透传不写 `${X:-1}`；生产/回测默认 `F1_FEATURES=1 F1_FORCE=1`（FORCE 必配，实测 0/7 会被 Top500 静默丢弃）；**复现 2026-10-11 前一切基线须显式 =0**；杀条件 K1 净IR 或 CI 下界≤0 / K2 DSR<0.95 / K3 A 闸非零 / K4 着装 0/7 即回退，R1 两跑归因；罪名维持②「进训练差异不可分辨（非有害）」，B 检验 p=0.18/0.52 如实保留；SFC/基本面管线保留数据不进生产；复现 F1-on 旧基线（212920/234941/B2）须显式 `F1_FEATURES=1 F1_FORCE=1`。⚠️ WF `--start-date`=**全程起点**（含 36 月训练窗），复现 D20 测试期 2023-06→2026-07 须 `--start-date 2020-06-01`，传 2023-06 只得 2 折废跑（lessons 三.52） |
| **HIBOR 特征默认已开（D22 例外采用）+ HIBOR_FORCE 强制纳入（D24）** | 默认同 O5 只在 `FLAG_DEFAULTS` 定义；生产/回测默认 `HIBOR_FEATURES=1 HIBOR_FORCE=1`（默认跟随）；**必须配 FORCE**——生产读静态名单（10-10 生成）含 **0 个 HIBOR 列**，统计法按样本内方差系统性淘汰慢变量（族审计：F1 0/7、HIBOR 0/500、网络族富集 1.30x），不强制则「算了不用」空转（2026-10-12 实测纳入 60 列）；回退 `HIBOR_FORCE=0`；**复现 2026-10-11 前一切旧基线（152157/234941/212920/B2/F1-on）须显式 `HIBOR_FEATURES=0 HIBOR_FORCE=0`**；杀条件 K1 净IR 或 CI 下界≤0 / K2 DSR<0.95 / K3 A 闸非零，触发即改回默认 0（预注册见 DECISIONS D22）；判定 HIBOR-null 不因采用而改写 |
| **恒指方向信号不优于持有** | 信号 Long/Flat 5/10/20d IR（0.24/0.07/0.20）均低于买入持有（0.50/0.47/0.49） |
| **异常大跌抄底是真信号但强依赖行情** | z≤−3 持 5d 净IR 1.55 显著；但逐年依赖 2025，**仅限大盘上行期（恒指>MA200）战术使用**；短期强势过滤有害 |
| **港股个股信号多为"2025 现象"** | 多策略复核收益集中于 2025 反弹年；跨年稳健性普遍不足，须逐年看 |
| **IC 计算** | IC 必须用实际收益率，不能用二元标签；收益率计算必须与训练一致 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| ~~CatBoost 1天模型~~ | ❌ **已停用（D59）**：周期改 5/10/20，1d 不再生产 |
| **学习器不可全局替换** | LightGBM 并非全面更优：20d LGBM 赢、**5d CatBoost 更优**（lift/IR/PBO/DSR 六项全占优、2026 转负）、1d 两者净IR≤0 双停 → 换学习器必须按周期分别 A/B（D10） |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **恒指 vs 个股** | 恒指 5/10/20d 57.3%/58.5%/52.4%（折聚类后均不显著）vs 个股 20d 51.8%（不显著）；**恒指三周期八模式、个股三周期模式折聚类后均不显著（效应 ≤±2.7pp），均不可交易** |
| **高置信度风险** | 高置信度预测错误时损失可达 -73%，必须设置止损 |
| **网络社区特征一致性** | 训练时保存 `model.community_ids`，预测时使用相同社区 ID 列表 |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测预处理必须一致 |
| **默认值设计** | 默认值必须与有效值范围分离，使用 -1 表示"未知"，基本面特征用 NaN |
| **训练时 NaN** | 不要用 `df.dropna()` 删除所有 NaN，只删除标签和关键列 |
| **绝对值特征** | 跨股票训练时，绝对价格/成交量特征必须标准化或排除 |
| **市场情绪数据源** | 必须使用所有股票收益率计算上涨比例，与 walk-forward 验证一致 |
| **时点口径（D25 已执行）** | **全链路统一 `mode='production'`（T 收盘数据）**：WF 3 处 + 生产 train 1 处均**显式**传 mode（不留默认）。决策时点 16:30 HKT → **T+1 成交** → 标签 = `Close.shift(-(h+lag))/Close.shift(-lag)-1`（T+1→T+1+h），`LABEL_ENTRY_LAG_DAYS` **全项目唯一定义**。港股收盘竞价 16:00–16:10、**16:10 后当天不再成交** ⇒ 按 T 收盘买入物理不可行。⚠️ **2026-10-13 起所有旧基线（`output/*_catboost_20d/` 15 轮）口径已作废**，双跑重验中，见 DECISIONS D25 / lessons 三.60 |
| **yfinance 盘中数据** | yfinance 日线数据在盘中可能不准确，推荐使用腾讯财经接口获取实时报价 |
| **复权口径一致性** | 训练/预测/评估统一用**腾讯前复权（qfq）**（港股 `get_hk_stock_data_tencent`、A股 `get_a_stock_data`）；性能监控 exit 价必须 qfq 同源，禁用 yfinance 未复权价当 exit（跨除息日 actual_return 失真，实测虚高 62%，见 lessons 三.17） |
| ~~校准概率与方向同口径~~ | ❌ **已撤销（D26）**：旧规则把 `probability`（P(上涨)）覆写为 P(方向判对) 并按其 ≥0.5 重判方向 → 20d 校准器在 p<0.741 为平台 0.4933，32 只邮件全显示「上涨概率 0.49」、三重确认 0.55 门槛结构性不可达、原始 0.74 强涨股被翻成「↓」。**现行**：`probability`/`direction` 一律用模型原始输出（与 A 闸/WF 同源，阈值 ≥0.5），`confidence` 单独承载 P(判对)；某周期 confidence 横截面极差 <0.02 ⇒ 撤销并告警（退化守卫）。见 DECISIONS D26 / lessons 三.61 |
| **A股数据源 640 行硬上限** | ⚠️ 腾讯 A股接口**无论 `period_days` 给多少恒返回 640 行**（实测 640→641 / 1500→640 / 3000→640），约 2.5 年上限。加 12 个月训练窗后**可用测试期不足 2 年**（实测 2025-01→2026-07，2025 占 63.6%）→ **低 PBO/高 DSR 不可信**，是 A股不予升配的硬限制（D13）。AKShare 备源为备选 |
| **A股涨跌停差异** | 主板10%涨跌停，创业板20%涨跌停，混合训练时需标签标准化 |
| **A股股票代码前导零** | 保存CSV时必须用字符串格式 `zfill(6)`，否则前导零丢失（002655→2655） |
| **A股样本权重** | 核心股权重3.0倍，扩展股1.0倍，训练时需传入 `sample_weight` |
| **行级显著性=伪显著** | Walk-forward 的行级 z 值忽略折间+横截面相关（同折共享模型、同日 58 股共享同一 HSI 未来收益），有效独立观测≈**折数**而非行数。实测行级 z=6.3 → Fold 聚类 t=1.40 **p=0.167**。任何命中率/准确率/lift 的显著性必须**按 Fold 聚类后检验**（lessons 三.29） |
| **GATE 同源三依赖** | 门槛/校准器/快照/基准CSV 必须同源：①宏观 `US_MARKET_SNAPSHOT_DIR` ②分位基准 `GATE_SOURCE_CSV` ③Isotonic 校准器须随模型重训**用 OOF 重拟合**（`oof_glob`，别用生产历史——新模型无历史可拟合）。**`bear==weak` 通常是概率被压平（无 edge），不是分位算错**（lessons 三.30） |
| ~~跨快照数值极差~~ | ❌ **已撤销（2026-10-05）**：曾归因「跨快照漂移」，实为**旧特征缓存内容差异**——清缓存后连跑两次 md5 一致、同配置极差 **0.00**。**教训**：指标分叉时先做「清缓存重跑」判别（最便宜），再谈机制解释；见 lessons 三.42 |
| ~~港股数据末日敏感性~~ | ❌ **已撤销（2026-10-05）**：`WALKFORWARD_DATA_END` 解决的是**不存在的问题**（数据源当日本就只到 07-30），反引入实截断。**教训**：指标变化先问「基线是否也有此差异」——基线末日同为 07-30 则末日不是变量 |
| **跨日回测不可复现** | `us_market_data` 缓存**仅当天有效**且是宏观特征唯一来源 → 同代码跨天跑结果不同（实测 abs20d 净IR 0.34/0.42、PBO 0.41/0.64）。三.25 的 bit 级复现**仅限同一天**；**多跑几次总能撞到三门槛全过**，禁止取最好的一次当结论。⚠️ **个股同日版更隐蔽**：港股 OHLC 实时取数、特征缓存键含 `last_date` → 数据源在**同一天内的数小时窗口**更新（如 10-06 行情午后到）即令末日漂移、缓存键全变、全量重算、共同期 40% 特征列值变 → 双跑 md5 必不一致。宏观有 `pin_macro_snapshot.py`，**个股原有 `pin_hk_snapshot.py`（2026-10-08 补）**，缺则三冻结不全（lessons 三.50） |
| **冻结必须双向落盘** | 只写成功数据、**失败不写哨兵** = 没冻结（每次重试，成败纯看网络）。**只做一半的冻结比不冻结更危险**——制造「已冻结」的错觉。两个易错点：①`load_frozen` 须**数据优先于哨兵**；②冻结模式下抓取失败写 `.failed` 且**不再重试网络**。见 lessons「38. 冻结机制必须双向落盘」 |
| **市场门槛优先分位、退化回退绝对值** | 分位法隐含前提「模型有 edge」；**模型无 edge 时校准把概率压回基准率 → 分位锚在 0.5 → 门控静默失效**（实测 bear 通过率 17.2%→98.2%）。已加退化保护（`GATE_MIN_UNIQUE=30`/`GATE_MIN_SPREAD=0.05`），退化即回退 bear 0.70/weak 0.65。**`bear==weak` 是分布退化信号，不是分位算错**。分位口径：熊市/弱震荡用 `GATE_QUANTILES` P92/P90（PIT），数据源须为 walk-forward 回测分布（`prediction_history` 右尾过窄会失效）。**⚠️ 2026-10-12 实测：真实分布上守卫恒触发——港股校准后唯一值 13<30、P92−P90 0.0103<0.05 ⇒ 分位门槛自 10-02 起从未生效，港股与 A 股现行生效门槛都是绝对 0.70/0.65**；以「分位」措辞叙述的历史结论须按绝对阈值理解。见 lessons 三.30、三.61、D8 |

---

## 🗣 回答规范（每次答题适用）

1. **措辞**：短句、主动语态、一个概念一个名字（ASD-STE100 八成严格度）。
2. **因果**：凡「因为 X 所以 Y」必须附复算或聚类检验，否则标注 **仅判断**。
3. **判定式**：规则写成可判定形式——在合法数据上跑一遍会 FAIL 的，才算规则。
4. **先图后文**：系统架构、数据流、状态流转、判读顺序，先给图再给文字。**图必须是 ASCII 文本图**（代码块内画框线/箭头，CLI 直接可读）；**仅对话输出禁用 ```mermaid 代码块**——CLI 不渲染 Mermaid，只显示源码等于没图。**文档（README / docs/*.md）不受此限，可继续用 mermaid**。
5. **只留可 grep 的**：数值与判定写进 Markdown；**HTML / SVG / 视频不入库**。
6. **含 `|` 的 shell 命令放代码块、不进表格**（表格内转义会改变命令语义）。
7. **流程与检查表** → [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md)

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

**特征选择**：使用 Top 500 特征，特征减少 55.8%，性能优于全量特征；**20d 生产现为 Top500（F1-off，D20 ⑦）**；F1-on 复验时才 `F1_FORCE=1` 并入 7 特征（507）

**Walk-forward 输出文件**（保存到 `output/YYYYMMDD_HHMMSS_catboost_20d/`）：
- `fold_metrics_detail.json` - 每个 Fold 的指标 + **Top 100 特征重要性**
- `prediction_analysis.csv` - 所有预测详情（用于 Fold 盈亏比分析）
- `validation_summary.json` - 总体验证结果

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨 |
| 特征数量 | ~1450 → 500（F1-off 生产口径；F1-on 复验时 507） | 推荐使用 Top 500 特征选择 |
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

详见 [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)，9个验证步骤：

1. 泄漏检查 - 所有特征使用 `shift(1)`
2. **绝对值特征标准化** - 跨股票训练必须标准化
3. **市场级特征交叉** - 对所有股票同值的特征必须交叉
4. **特征单调性** - 交叉特征保持逻辑单调性
5. Walk-forward 验证 - 准确率达标
6. SHAP 排名 - 进入 top 30
7. Pearson 相关性 - 与现有特征 < 0.8
8. 随机种子稳定性 - 波动 < 2%
9. **着装检查** - 首跑后确认进所选 Top500 名单；0/N=静默丢弃=假上线（慢变量须 F1_FORCE 强制纳入，D20 ①前车/D23 K4）

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

### 🚨 对抗性审核：⓪ 同一性闸 + 三道闸（强制，不可跳过）

> **9 项审核清单、两条硬约束的失职复盘、实验方法论全文与案例 → [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)**
> 此处只留**触发时机 + 硬规则**，防止「改了 AGENTS 却没改规程」的双写漂移。

四个触发时机（**其中三个与「有没有改动」无关**）：

| 闸 | 触发时机 | 判据 |
|---|---|---|
| **⓪ 同一性闸** | 任何指标要作为**结论**输出前（先于 A 闸） | 四问，任一「否」即止：① **社区集合**严格相等（`community_ids`）② **列集合**：生产 pkl ⊆ WF 可用列集 **且** F1/HIBOR 列两边都在 **且** 选择规则同源（**不要要求两边集合逐字相等**——WF 每折重选，恒不相等）③ **时点**：`train()` 与 `predict()` 的 `use_shift` 相等；④ 概率同源（`oof_glob` 校准 + 门槛基准 CSV）。⚠️ **部分可执行**：`python3 scripts/parity_check.py`（产物/静态/env 层，**Q3/Q4 运行时相等仍须人工**）；契约测试兜底：`python3 -m pytest tests/test_pipeline_contract.py` |
| **A 呈现闸** | 任何指标要作为**结论**输出前 | 绝对值异常 → `python3 scripts/presentation_gate.py --input <csv> --horizon N`，**非零退出即阻断**；结论须标注「适用范围：WF 内部比较 ✓ / 外推生产 ✗」。✅ **D45 起每次输出自动带该标注**（原无工具强制）；**结论必须条件化**：任何「不立项/无 edge/升配」等结论须带**适用范围**，`doc_gate.py L6` 对 DECISIONS 结论段强制（既有冻结/新增强制）——起源 D46→D47 |
| **B 提交闸** | 任何代码/配置/文档改动后、commit 前 | [9 项清单](docs/REVIEW_GATES.md)逐项过，任一不过即重审（第 9 项=同一性） |
| **C 升级闸** | 任何 🟢/升配/加仓/改阈值 决策前 | 三冻结 + 双跑 md5 一致 **∧ ⓪ 通过** + 禁止取最好一轮。⚠️ md5 一致**只证同一路径可复现、不证两路径相同**——2026-10-12 复现的是一个没人部署的函数 |

> **尺子边界（D45，用本框架量任何策略时须同报三项）**：① **适用范围**（本宇宙 / 本管线；非通用）② **功效/MDE**（`monthly_guardrail.min_detectable_ir`；"不显著 ≠ 没有"）③ **相邻参数一致性**（`monthly_guardrail --sensitivity`；符号翻转 = 不稳健）。
>
> **任何"结论"输出前，先走技能 `hypothesis-validation`**（`.opencode/skills/hypothesis-validation/SKILL.md`）：7 步 SOP（命题预登记 → ⓪同一性 → 数据卫生 → 口径 → 三道闸 → 稳健性 → 功效MDE → 记录）+ 硬规则 **G1–G9** + 失败模式库。触发词："验证 / 行不行 / 有没有 alpha / 可不可信 / 换池·换周期·加特征·择时"。

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

**判读顺序不可颠倒**：**⓪ 同一性**（WF=生产？时点对齐？概率同源？）→ 绝对值异常 → PBO<0.5 → DSR≥0.95 → 净IR CI 不跨0 + lift>0 → 才谈仓位。**⓪ 不过则后面不必跑**。

**「同源」是通用原则，不只针对概率分位**（7 项：特征列集合 / 社区集合 / 时点口径 / 数据窗口 / 快照 / 校准器来源 / 门槛基准）；**每个 flag 的默认值只许一处**（当前违反 4 处，lessons 三.60 O5）。

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

- **数据文件不新增入库**（可执行判据在 `scripts/doc_gate.py` S1，pre-commit 强制）：
  - 新增 `.pkl` / `.csv` → 拦；**更新既有跟踪文件不受限**
    （`data/a_stock_models/*` 等由 CI 定期更新，属既有实践）
  - 永不入库：`data/model_accuracy.json`（运行时状态）、`data/us_market_cache/`、`*.bak`
  - **例外**：回测 `prediction_analysis.csv`（仅 `output/*_catboost_20d/` 路径）由
    `scripts/commit_backtest_result.py` 自动入库
    （分位门槛数据源，CI/本地须同源；每次入库自动 `git rm --cached` 旧的港股 20d CSV
    只留最新，防仓库膨胀；快照维护见 `ml_services/market_regime.py` 注释）
  - **例外**：SFC 卖空周报 `data/sfc_short/sfc_short_long.csv` + `manifest.json`
    （模型输入特征，D20 生产纳入；`fetch_sfc_short.py` 周更、工作流 commit 写回，
    类比上条例外；教训见 lessons 三.51——被 gitignore 的生产依赖 = 生产端不存在）
  - **例外**：A-H 溢价数据集 `data/ah_premium/ah_*.csv` + `fx_hkdcny.csv` + `manifest.json`
    （D21 M1/M1b 复现资产；`ml_services/fetch_ah_premium.py` 可全量回拉，但 A 股长窗历史
    仅免费源可得，不入库=换机不可复现，同 lessons 三.51）
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`
- **B 闸自动执行点**：`.githooks/pre-commit` 跑 `scripts/doc_gate.py --staged`，
  不过即拒绝提交。一次性安装（**本地配置，新 clone 需重跑**）：

  ```
  git config core.hooksPath .githooks && chmod +x .githooks/pre-commit
  ```

  覆盖断链 / 表格 / 转义管道 / 过期残留 / A 闸 / 禁入文件 / py 语法；
  **不覆盖全量 pytest**（约 5 分钟，仍是 B 闸人工步骤）。

---

## 🔢 决策编号分配（2026-10-14 新增，防再次撞车）

**为什么加这条**：2026-10-13 晚有第二个会话并行写入，**「社区集合显式化」与「probability 口径拆分」
同时被编成 D26**，导致同一仓库里两个 D26 各指一事（BASELINES/A_STOCK_REFORM_PLAN/lessons 三.63
一度用「D26」指前者，AGENTS/DECISIONS 用「D26」指后者）。

**规则**：
1. **编号在 `docs/DECISIONS.md` 里唯一**；写入前先 `grep -c "^## D<下一号>" docs/DECISIONS.md` 确认未被占用
2. **两个会话同时在写时**：由**后启动的一方**顺延取号，不得复用任何已出现过的编号
3. **历史条目不重写编号**（progress.txt / lessons 是编年日志）——改号时在**文末**加一条编号映射说明
4. 改号必须同步：`grep -rn "<旧号>" docs/ AGENTS.md lessons.md`，逐条判断指向后改，不得全局替换

> 已发生的改号：社区集合显式化 **D26 → D30**（编号撞车）。
> 说「社区集合 / `net_comm_*` 编码」= **D30**；说「probability 与 confidence 口径」= **D26**。

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**对抗性审核有三个触发时机（见 [docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)）**：
- **指标呈现前**：任何数字要作为结论汇报时，先查绝对值异常（可能一个字都没改）
- **改动提交前**：任何代码/配置/文档改动后，攻击自己的改动再 commit
- **升级决策前**：任何 🟢/升配/加仓/改阈值前，三冻结 + 双跑 md5 一致

2026-10-02 实证两次：①**未改任何代码**，仅因汇报 rel20d 结果触发 A 闸，
抓到 100% 准确率泄漏（PBO 0.43/DSR 1.000 全过却是假信号）；
②文档同步触发 B 闸，发现 4 处自相矛盾，其中 `DEPLOYMENT.md` 仍写着
已被证伪的「🟢 可放大至 15%」——若照此实盘会**加倍错误仓位**。

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

**模型更新后**：运行 Walk-forward 验证确认性能，使用 `/model_validation` 命令执行标准验证流程

**特征修改后**：清除缓存 `rm -rf data/feature_cache/*.pkl`

**改动收尾时**：跑 `python3 scripts/doc_gate.py`（断链 / 表格完整性 / 转义管道 /
过期残留 / A 闸可运行 / **结论条件化 L6**），非零退出即阻断；含 `|` 的 shell 命令放代码块不进表格；
HTML / SVG / 视频不入库。完整检查表见 [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md) §8

---

## 🔗 快速链接

- **⭐ Walk-forward 基线（唯一真相源）**：[docs/BASELINES.md](docs/BASELINES.md)
- **⭐ ⓪ 同一性闸 / 三道闸 / 硬约束 / 实验方法论正文**：[docs/REVIEW_GATES.md](docs/REVIEW_GATES.md)
- **架构 / 特征架构 / 环境变量 / 调度**：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **特征工程**：[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) - 完整指南（含案例分析）
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **验证方法**：[docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md)
- **模型改进计划**：[docs/MODEL_IMPROVEMENT_PLAN.md](docs/MODEL_IMPROVEMENT_PLAN.md) - 三层提升假设与预注册判据 ⭐
- **决策备忘**：[docs/DECISIONS.md](docs/DECISIONS.md) - 个股 alpha 停止投入等关键决策 ⭐
- **建设方法论**：[docs/QUANT_SYSTEM_METHODOLOGY.md](docs/QUANT_SYSTEM_METHODOLOGY.md) - 五层结构 / 七原则 / 七阶段 / 三道闸 ⭐
- **量化交易误解**：[docs/量化交易误解-正式文档.md](docs/量化交易误解-正式文档.md) - 四道陷阱方法论注脚 ⭐
- **实盘部署**：[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - 底仓+战术+风控与月度护栏 ⭐
- **A股设计**：[docs/A_STOCK_DESIGN.md](docs/A_STOCK_DESIGN.md) - A股系统完整设计文档
- **SSH认证**：[docs/SSH_SETUP.md](docs/SSH_SETUP.md) - 新电脑配置 GitHub SSH 认证指南