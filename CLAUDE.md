# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 港股智能分析系统 - 快速参考

> **📚 详细文档**：特征工程、验证方法等完整指南请查看 [docs/](docs/) 目录
> **⚠️ 经验教训**：关键警告和最佳实践请参阅 [lessons.md](lessons.md)
> **🔧 编程规范**：开发流程、系统设计决策请遵守 [docs/programmer_skill.md](docs/programmer_skill.md)

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

| 任务 | 命令 |
|------|------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` 或 `python3 comprehensive_analysis.py` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **个股Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection` |
| **模型预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost --use-feature-selection` |
| **特征选择** | `python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20` |
| **超参数调优** | `python3 ml_services/hyperparameter_tuner.py --horizon 20 --n-iter 30` |
| **股票网络分析** | `python3 ml_services/stock_network_analysis.py --skip-pmfg` |
| **性能监控** | `python3 ml_services/performance_monitor.py --mode all --no-email` |

### 缓存管理

```bash
# 清除特征缓存（新增特征后必须执行）
rm -rf data/feature_cache/*.pkl

# 清除原始数据缓存
rm -rf data/stock_cache/*.pkl
```

---

## ⚠️ 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward准确率 >65%（个股）或 >80%（恒指）通常是数据泄漏信号 |
| **IC 计算** | IC 必须用实际收益率，不能用二元标签；收益率计算必须与训练一致 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **CatBoost 1天模型** | 噪音大，仅供参考 |
| **深度学习模型** | LSTM/Transformer F1≈0，**不推荐** |
| **加密货币策略** | 股票异常策略**不适用于**加密货币 |
| **恒指 vs 个股** | 恒指准确率显著高于个股（81% vs 54%），个股预测需谨慎 |
| **高置信度风险** | 高置信度预测错误时损失可达 -73%，必须设置止损 |
| **网络社区特征一致性** | 训练时保存 `model.community_ids`，预测时使用相同社区 ID 列表 |
| **分类特征 NaN** | CatBoost 预测时必须处理分类特征 NaN，训练和预测预处理必须一致 |
| **默认值设计** | 默认值必须与有效值范围分离，使用 -1 表示"未知"，基本面特征用 NaN |
| **训练时 NaN** | 不要用 `df.dropna()` 删除所有 NaN，只删除标签和关键列 |

---

## 📐 数据流架构

```
外部数据源 → data_services/ → 分析层 → ml_services/ → 输出
    ↓              ↓              ↓            ↓          ↓
腾讯财经      技术指标计算    异常检测     CatBoost    邮件报告
yfinance     基本面数据      综合分析     Walk-forward  JSON文件
AKShare      南向资金        主力追踪     性能监控
```

**关键依赖关系**：
- `comprehensive_analysis.py` 整合：大模型建议 + CatBoost预测 + 异常检测 + 板块分析
- `hsi_prediction.py` 调用 `ml_services/hsi_ml_model.py` 进行CatBoost预测
- `detect_stock_anomalies.py` 使用 `anomaly_detector/` 模块的双层检测（Z-Score + Isolation Forest）
- `config.py` 定义股票板块映射 `STOCK_SECTOR_MAPPING` 和自选股列表 `WATCHLIST`

**特征模块**（动态构建，自动同步）：
- `data_services/calendar_features.py` - 日历效应（22个特征）
- `data_services/volatility_model.py` - GARCH 波动率（4个特征）
- `data_services/regime_detector.py` - HMM 市场状态检测（10个特征）
- `ml_services/stock_network_analysis.py` - 股票网络分析（社区ID、中心性等）

**数据存储**：
- `data/hsi_models/` - 恒指CatBoost模型（.cbm）和特征配置（.json）
- `data/stock_cache/` - 原始数据缓存（7天有效期）
- `data/feature_cache/` - 特征缓存（7天有效期，170x加速）
- `output/` - 分析报告和回测结果

---

## 🤖 机器学习模型

### 模型可信度（Walk-forward 验证）

**恒指增强模型**（33特征）：

| 周期 | 准确率 | 推荐度 |
|------|--------|--------|
| **20天** | **81.24%** | ⭐⭐⭐⭐⭐ 推荐 |
| 5天 | 60.26% | ⭐⭐⭐ 趋势确认 |
| 1天 | 50.11% | ⚠️ 噪音大 |

**个股完整模型**（2026-05-09 验证，12 folds，57只股票，Top 300特征）：

| 指标 | 数值 | 评估 |
|------|------|------|
| 平均准确率 | 52.40% | ✅ 正常范围 |
| 平均夏普比率 | 4.85 | ✅ 优秀 |
| 平均 IC | 0.2096 | ✅ 有效 |
| 平均 Rank IC | 0.2196 | ✅ 有效 |
| 平均最大回撤 | -0.94% | ✅ 优秀 |

### CatBoost 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **预测阈值** | 0.5 | 概率 > 0.5 预测上涨 |
| 特征数量 | 1191 → 730 → 300 | 完整特征 → 去冗余 → Top 300 最优 |
| 随机种子 | 42（固定） | 确保可重现性 |

**20天模型参数**：

| 参数 | 值 |
|------|-----|
| n_estimators | 400 |
| depth | 7 |
| learning_rate | 0.04 |
| l2_leaf_reg | 2 |
| subsample | 0.75 |
| colsample_bylevel | 0.6 |

---

## ⚙️ 环境配置

### 必填环境变量

| 变量名 | 说明 |
|--------|------|
| `SMTP_SERVER` | SMTP 服务器地址 |
| `EMAIL_SENDER` | 发件人邮箱 |
| `EMAIL_PASSWORD` | 邮箱应用密码 |
| `RECIPIENT_EMAIL` | 收件人邮箱列表 |
| `QWEN_API_KEY` | 通义千问 API 密钥 |

### 主要依赖

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm` `hmmlearn` `arch` `networkx`

---

## 🔧 开发规范

### 代码修改原则

1. **修改完即测试**：每次修改后立即执行 `python3 -m py_compile <文件>`
2. **避免硬编码路径**：使用 `os.path.dirname(os.path.abspath(__file__))` 获取脚本目录
3. **HTTP API 超时处理**：调用 API 时必须设置超时时间
4. **语言规范**：对话和注释使用简体中文，变量名/函数名使用英文

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
- GitHub Actions：排程控制在 cron，不在代码中重复判断
- 推送冲突：使用 `git pull --rebase`

---

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

**模型更新后**：运行 Walk-forward 验证确认性能

---

## 🔗 快速链接

- **经验教训**：[lessons.md](lessons.md) - 关键警告和最佳实践
- **进度跟踪**：[progress.txt](progress.txt) - 项目当前进展
- **详细文档**：[docs/](docs/) - 特征工程、验证方法等
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **特征重要性分析**：[docs/FEATURE_IMPORTANCE_ANALYSIS.md](docs/FEATURE_IMPORTANCE_ANALYSIS.md)
- **股票网络分析**：[docs/STOCK_NETWORK_ANALYSIS.md](docs/STOCK_NETWORK_ANALYSIS.md)
