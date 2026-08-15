# 金融资产智能分析系统 - opencode 规则

> 本文件是 opencode 的规则入口。**完整项目规则见 [CLAUDE.md](CLAUDE.md)**，已通过 `opencode.json` 的 `instructions` 字段自动加载，无需重复维护。

---

## 📋 项目概览

**双市场支持**：
- 🇭🇰 **港股** - 恒生指数三周期预测、个股预测、异常检测（31只自选股）
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
```

### 港股系统

| 任务 | 命令 |
|------|------|
| **恒生指数预测** | `python3 hsi_prediction.py --no-email` |
| **综合分析** | `./scripts/run_comprehensive_analysis.sh` |
| **港股异常检测** | `python3 detect_stock_anomalies.py --mode standalone --mode-type deep` |
| **个股Walk-forward验证** | `python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20` |
| **模型训练** | `python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection` |
| **模型预测** | `python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost --use-feature-selection` |
| **特征选择** | `python3 ml_services/feature_selection.py --method statistical --top-k 300 --horizon 20` |

### A股系统

| 任务 | 命令 |
|------|------|
| **A股综合分析** | `./scripts/run_a_stock_analysis.sh` |
| **A股模型训练** | `python3 a_stock_ml_model.py --mode train --horizon 20` |
| **A股模型预测** | `python3 a_stock_ml_model.py --mode predict --horizon 20 --core-only` |

### 缓存管理

```bash
rm -rf data/feature_cache/*.pkl        # 清除港股特征缓存
rm -rf data/stock_cache/*.pkl          # 清除港股原始数据缓存
rm -rf data/a_stock_feature_cache/*.pkl # 清除A股特征缓存
rm -rf data/a_stock_cache/*.pkl        # 清除A股原始数据缓存
```

---

## ⚠️ 核心警告

| 警告 | 说明 |
|------|------|
| **数据泄漏** | Walk-forward准确率 >65%（个股）或 >80%（恒指）通常是数据泄漏信号 |
| **预测阈值** | 方向判断用 **0.5**，不是 0.65 |
| **恒指 vs 个股** | 恒指准确率显著高于个股（81% vs 54%），个股预测需谨慎 |
| **CatBoost 分类特征 NaN** | 训练和预测预处理必须一致，分类特征 NaN 用 'unknown' 填充 |
| **双模式预测** | 收市后预测使用 `mode='production'`，Walk-forward 使用 `mode='backtest'` |
| **A股涨跌停差异** | 主板10%，创业板20%，混合训练时需标签标准化 |
| **A股股票代码前导零** | 保存CSV时必须 `zfill(6)` |
| **A股样本权重** | 核心股权重3.0倍，扩展股1.0倍 |

---

## 📐 数据流架构

```
外部数据源（腾讯财经/AKShare）→ data_services/（数据处理）→ 分析层（异常检测/综合分析）→ ml_services/（机器学习）→ 输出（邮件/JSON/微信）
```

**关键依赖关系**：
- `comprehensive_analysis.py` 整合：大模型建议 + CatBoost预测 + 异常检测 + 板块分析
- `hsi_prediction.py` 调用 `ml_services/hsi_ml_model.py` 进行CatBoost预测
- `detect_stock_anomalies.py` 使用 `anomaly_detector/` 模块的双层检测（Z-Score + Isolation Forest）
- `config.py` 定义股票板块映射和自选股列表（31只）

**特征模块**（动态构建，自动同步）：
- `data_services/calendar_features.py` - 日历效应（22个特征）
- `data_services/volatility_model.py` - GARCH 波动率（4个特征）
- `data_services/regime_detector.py` - HMM 市场状态检测（10个特征）
- `ml_services/stock_network_analysis.py` - 股票网络分析

**消息服务模块**：
- `message_services/email_sender.py` - 统一邮件发送
- `message_services/wechat_work_bot.py` - 企业微信机器人
- `message_services/wxpusher_bot.py` - WxPusher 推送
- `message_services/notifier.py` - 统一通知接口

---

## 🏗️ 特征架构（单一真相源）

**核心原则**：特征处理逻辑只在 `ml_trading_model.py` 中维护，其他模块通过导入或方法调用复用。

- `BaseTradingModel.get_feature_columns()` - 排除绝对值特征，返回有效特征列表
- `BaseTradingModel.prepare_features_for_selection()` - 特征选择专用方法
- `BaseTradingModel.prepare_data()` - 完整特征准备
- `FeatureEngineer` - 技术指标、交叉特征、单调性、NaN处理

**新增特征时**：只需修改 `ml_trading_model.py`，`feature_selection.py` 自动同步。

---

## 🤖 机器学习模型

**恒指增强模型**（2026-05-18 验证，33特征）：20天准确率 **81.22%**，5天 **65.86%**，1天 51.49%

**个股完整模型**（2026-05-24 验证，12 folds，57只股票）：平均准确率 58.77%，夏普 6.45

### CatBoost 配置

| 参数 | 值 |
|------|-----|
| 预测阈值 | 0.5 |
| 随机种子 | 42（固定） |
| n_estimators | 400 |
| depth | 8 |
| learning_rate | 0.06 |

---

## 🔧 开发规范

### 代码修改原则

1. **修改完即测试**：每次修改后立即执行 `python3 -m py_compile <文件>`
2. **避免硬编码路径**：使用 `os.path.dirname(os.path.abspath(__file__))`
3. **HTTP API 超时处理**：调用 API 时必须设置超时时间
4. **语言规范**：对话和注释使用简体中文，变量名/函数名使用英文

### 数据泄漏防护

高风险特征必须使用 `.shift(1)` 避免使用当日数据：
- 所有 `.rolling()` 计算的特征
- `future_return` 必须使用 `.shift(-N)` 计算未来收益

### Git 提交规范

- 文件上传：只提交 `.md` 格式，不提交 `.json`/`.csv`
- 推送冲突：使用 `git pull --rebase`

---

## 📝 会话工作流

**会话开始时**：读取 `progress.txt` 了解项目进展，审查 `lessons.md` 检查错误

**功能更新后**：更新 `progress.txt` 记录进展，如有新学习心得更新 `lessons.md`

**模型更新后**：运行 Walk-forward 验证确认性能（使用 `/model_validation` 命令）

**特征修改后**：清除缓存 `rm -rf data/feature_cache/*.pkl`

---

## 🔗 快速链接

- **完整规则**：[CLAUDE.md](CLAUDE.md) - 项目主文档（自动加载）
- **经验教训**：[lessons.md](lessons.md)
- **进度跟踪**：[progress.txt](progress.txt)
- **特征工程**：[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)
- **三周期分析**：[docs/THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md)
- **验证方法**：[docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md)
- **A股设计**：[docs/A_STOCK_DESIGN.md](docs/A_STOCK_DESIGN.md)
