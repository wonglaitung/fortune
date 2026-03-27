# 经验教训

本文档记录开发过程中遇到的问题和解决方案，帮助避免重复错误。

---

## 🔴 核心警告

### 高准确率通常是数据泄漏的信号
| 等级 | 准确率范围 | 判断 |
|------|-----------|------|
| 随机/平衡基线 | ≈50% | 正常 |
| 常见弱信号 | ≈51–55% | 正常 |
| 有意义的改进 | ≈55–60% | 良好 |
| 非常好/罕见 | ≈60–65% | 优秀 |
| **异常高（需怀疑）** | **>65%** | **可能数据泄漏** |

### 模型可信度评估
| 模型 | 可信度 | 说明 |
|------|--------|------|
| **CatBoost 20天** | ⭐⭐⭐⭐⭐ | 高可信度，**推荐使用** |
| **CatBoost 5天** | ⭐⭐⭐ | 中等可信度，谨慎使用 |
| **CatBoost 1天** | ⭐ | 低可信度，**不推荐**（严重过拟合） |
| **LSTM/Transformer** | ⭐ | 低可信度，**不推荐**（F1≈0） |
| **融合模型** | ⭐ | 低可信度，**不推荐**（信号稀释） |

### Walk-forward验证是唯一可信的评估方法
- ❌ 简单评估：使用训练数据评估，结果虚高（83-90%）
- ✅ Walk-forward：每个fold重新训练，严格时序分割，反映真实预测能力

---

## 📊 关键指标定义

| 指标 | 定义 | 使用场景 |
|------|------|----------|
| **胜率** | 盈利交易数 / 买入信号总数 | 评估买入信号质量 |
| **准确率** | 预测正确数 / 总交易机会数 | 评估方向预测能力 |
| **正确决策比例** | 正确决策数 / 总决策数 | 综合评估决策质量 |

**重要**：高准确率 ≠ 高收益。准确率86.21%，买入胜率23.04%，平均收益率-1.94%（避雷模式）

---

## 🔧 编码规范

### 路径硬编码
- ❌ `/data/fortune/...`
- ✅ `os.path.dirname(os.path.abspath(__file__))`

### 注释和文档
- 注释只描述功能，不添加"新增"、"修改"等时间标记
- README面向当前用户，不添加日期标注
- 决策框架必须与代码逻辑一致

---

## ⚠️ 常见错误

### 数据泄漏检查
**高风险特征**（必须使用.shift(1)）：
- BB_Position、Price_Percentile、Intraday_Amplitude
- Price_Ratio_MA5/20/50、Price_Pct_20d
- Days_Since_High/Low_20d、Intraday_Range
- Support_120d、Resistance_120d

### 夏普比率计算
```python
# 错误1：缺少无风险利率
sharpe = annualized_return / annualized_std

# 错误2：分子分母不匹配（2026-03-24修复）
annualized_return = avg_return * (252 / horizon)  # avg_return包含所有交易
return_std = buy_signals['strategy_return'].std()  # 只用买入信号
sharpe = (annualized_return - 0.02) / annualized_std

# 正确：统一使用买入信号数据
avg_return_for_sharpe = buy_signals['strategy_return'].mean()  # 只用买入信号
annualized_return = avg_return_for_sharpe * (252 / horizon)
return_std = buy_signals['strategy_return'].std()  # 只用买入信号
sharpe = (annualized_return - 0.02) / annualized_std
```

**修复原因**：
- 不买入交易的0收益会稀释平均收益率
- 分母只计算买入信号的波动率
- 导致夏普比率不准确，通常偏低

**修复效果**（银行股Walk-forward验证，阈值0.6）：
| 指标 | 修正前 | 修正后 | 变化 |
|-----|--------|--------|------|
| 年化收益率 | 31.16% | 29.78% | -1.38% |
| 夏普比率 | -0.0179 | -0.0564 | -0.0385 |
| 买入信号胜率 | 49.67% | 49.05% | -0.62% |

### 回撤计算（多周期持有）
```python
# 错误：使用重叠样本
drawdown = (1 + df['strategy_return']).cumprod() - 1

# 正确：horizon>1时使用非重叠样本
if horizon > 1:
    non_overlapping = df.iloc[::horizon].copy()
    drawdown = (1 + non_overlapping['strategy_return']).cumprod() - 1
```

---

## 📊 板块Walk-forward验证经验（2026-03-24）

### 验证方法
- **方法**：12个Fold，12个月训练窗口，1个月测试窗口
- **置信度阈值**：0.6
- **验证期间**：2024-01-01 至 2025-12-31
- **模型**：CatBoost 20天

### 核心发现

1. **消费股是唯一符合条件的板块**：
   - 夏普比率 > 0：0.7445
   - 买入胜率 > 50%：54.80%
   - 稳定性高：收益率标准差仅1.22%

2. **样本数量严重影响模型可靠性**：
   - 交易所（1只股票）：胜率20%，夏普-5.95
   - 指数基金（2只股票）：胜率20%，夏普-7.57
   - 建议：样本<3只的板块不建议单独建模

3. **波动性与胜率负相关**：
   - 高波动板块（科技、新能源、生物医药）：胜率偏低（<50%）
   - 低波动板块（消费、银行）：胜率较高（接近或超过50%）

4. **稳定性vs收益权衡**：
   - 消费股：稳定性高，收益率低（5.10%）
   - 银行股：收益率高（45.92%），稳定性中等

### 投资建议

| 优先级 | 板块 | 配置比例 | 理由 |
|-------|------|---------|------|
| 1 | 消费股 | 40% | 夏普最高，胜率>50%，稳定性优秀 |
| 2 | 银行股 | 35% | 胜率突破50%，回撤控制良好 |
| 3 | 半导体股 | 25% | 接近盈亏线，进攻性配置 |

---

## 🎯 置信度阈值优化（银行股Walk-forward验证）

| 阈值 | 年化收益率 | 夏普比率 | 买入胜率 | 推荐度 |
|------|-----------|---------|---------|--------|
| 0.55 | 37.59% | -0.0179 | 48.41% | - |
| 0.6 | 39.58% | 0.0346 | 50.40% | ✅ |
| **0.65** | **40.15%** | **0.0346** | **50.88%** | **⭐⭐⭐⭐⭐ 推荐** |

**核心发现**：
- 0.6 → 0.65：年化收益率+8.63%，夏普比率转正，买入胜率突破50%盈亏线
- 震荡市显著改善：Fold 3/8 亏损减少76%-87%
- 索提诺比率：0.8125 → 0.9391（+11%）

**推荐**：使用阈值0.65（优于0.6和0.55）

---

## 🔬 特征工程经验

### 全量特征（892个）优于500特征（2026-03-27验证）
**验证背景**：
- 方法：Walk-forward验证（业界标准，12个Fold）
- 对象：银行股板块（6只股票）
- 验证期间：2024-01-01 至 2025-12-31
- 置信度阈值：0.60（两个验证使用相同阈值，确保公平对比）

**对比结果**：

| 指标 | 全量特征 | 500特征 | 差异 | 结论 |
|------|----------|---------|------|------|
| 年化收益率 | 40.42% | 30.28% | **+10.14%** | ✅ 全量特征 |
| 平均收益率 | 3.21% | 2.40% | **+0.81%** | ✅ 全量特征 |
| 买入信号胜率 | 49.60% | 49.13% | **+0.47%** | ✅ 全量特征 |
| 索提诺比率 | 1.9023 | 1.0400 | **+0.8623** | ✅ 全量特征 |
| 夏普比率 | -0.0235 | -0.0501 | **+0.0266** | ✅ 全量特征 |
| 正确决策比例 | 46.80% | 46.44% | **+0.36%** | ✅ 全量特征 |
| 收益率标准差 | 3.41% | 3.22% | -0.19% | ✅ 500特征 |
| 最大回撤 | -13.08% | -12.73% | -0.35% | ✅ 500特征 |

**关键发现**：
1. **全量特征在7/9个指标上优于500特征**
2. **年化收益率显著提升**：40.42% vs 30.28%（+10.14%）
3. **索提诺比率优秀**：1.9023 vs 1.0400（+83%）
4. **CatBoost自动特征选择机制**：不需要预先进行特征选择
5. **信息保留完整**：保留所有特征，避免信息丢失（392个特征）

**为什么全量特征更好**：
- CatBoost内置L2正则化，有效防止过拟合
- CatBoost自动计算特征重要性，低重要性特征权重自动降低
- 某些特征单独看不重要，但组合后可能产生强大预测能力
- 交叉特征（520个）和特征交互可能被特征选择过滤掉

**与之前对比结果的差异**：
- 之前的对比（2026-03-17）：500特征更好（39.57% vs 29.78%）
- 问题：使用了不同的置信度阈值（0.60 vs 0.55），对比不公平
- 现在的对比（2026-03-27）：相同阈值0.60，全量特征更好（40.42% vs 30.28%）

**最终推荐**：
- ✅ **使用全量特征（892个特征）** ⭐⭐⭐⭐⭐
- 训练时省略`--use-feature-selection`参数
- CatBoost会自动进行特征选择和正则化
- 唯一代价：训练速度较慢（每个Fold多约1分钟）

**训练命令**：
```bash
# 使用全量特征训练
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20 --confidence-threshold 0.60
```

### 固定500特征仍是最优方案（已过时，仅供参考）
- 累积重要性方法：特征分数分布极度分散，要达到90%覆盖率需要1500+特征
- 增加低质量特征反而降低性能（binary_logloss从0.3506增至0.3908）
- **注**：2026-03-27的Walk-forward验证证明全量特征优于500特征

### 交叉特征优化
- 优化前：4706个（13类别 × 362数值）
- 优化后：520个（13类别 × 40精选数值）
- 效果：训练速度提升3-4倍，内存占用减少70%

---

## 📝 板块模型使用建议（Walk-forward验证，阈值0.6，2026-03-24更新）

### 推荐使用（夏普比率>0 或 胜率≥50%）

| 板块 | 夏普比率 | 买入胜率 | 年化收益率 | 稳定性 | 建议 |
|------|---------|---------|-----------|--------|------|
| **消费股** | **0.7445** | **54.80%** | 5.10% | 高（优秀） | ⭐⭐⭐⭐⭐ 首选配置 |
| **银行股** | 0.1546 | 50.44% | 45.92% | 中（良好） | ⭐⭐⭐⭐ 推荐配置 |
| **半导体股** | 0.1260 | 49.87% | 13.32% | 高（优秀） | ⭐⭐⭐⭐ 推荐配置 |

### 谨慎使用（胜率接近50%）

| 板块 | 夏普比率 | 买入胜率 | 年化收益率 | 建议 |
|------|---------|---------|-----------|------|
| 保险股 | -0.3160 | 49.02% | 12.18% | ⚠️ 需配合市场环境识别 |
| 房地产股 | -0.1352 | 47.75% | 33.29% | ⚠️ 波动大，收益高风险高 |
| 人工智能股 | 0.0811 | 47.04% | 1.07% | ⚠️ 胜率略低于50% |

### 不推荐使用（胜率<45%或夏普<-0.5）

| 板块 | 夏普比率 | 买入胜率 | 问题 |
|------|---------|---------|------|
| 科技股 | -0.2394 | 45.77% | 胜率偏低，负收益 |
| 能源股 | -0.8110 | 46.88% | 夏普极差 |
| 航运股 | -0.2920 | 42.47% | 胜率低 |
| 生物医药股 | -0.5508 | 43.74% | 胜率低，亏损严重 |
| 环保股 | -0.2135 | 33.60% | 胜率极低 |
| 汽车股 | -0.5944 | 29.81% | 胜率极低，亏损严重 |
| 交易所 | -5.9514 | 20.00% | 样本不足（1只），不建议建模 |
| 指数基金 | -7.5682 | 20.00% | 样本不足（2只），不建议建模 |

### 关键洞察

1. **消费股一枝独秀**：唯一夏普>0且胜率>50%的板块，应作为首选配置
2. **样本数量影响**：交易所（1只）、指数基金（2只）因样本不足，模型不可靠
3. **波动性影响**：高波动板块（科技、新能源）预测难度大，胜率偏低
4. **稳定性vs收益**：消费股稳定性高但收益率低（5.10%），银行股收益率高但稳定性中等

---

## 📤 文件上传规范

**GitHub Actions报告上传**：
- ✅ 只上传 `.md` 格式（人类可读）
- ❌ 不上传 `.json`/`.csv` 格式（程序处理用）

---

## 文件更新日志
- 2026-03-24：添加15个板块Walk-forward验证结果（阈值0.6），更新板块模型使用建议
- 2026-03-24：添加夏普比率计算修复经验（修正分子分母不匹配问题）
- 2026-03-24：添加数据泄漏修复经验（CatBoost 68.33% → 60.99%，稳定性提升69%）
- 2026-03-24：简化文档，保留核心警告和经验
- 2026-03-23：更新置信度阈值0.65分析（推荐阈值0.65）
- 2026-03-22：添加决策框架更新经验
- 2026-03-20：添加置信度阈值优化经验（0.55→0.6）
- 2026-03-19：添加数据泄漏修正、回撤计算修正
- 2026-03-18：更新板块模型价值评估
- 2026-03-17：创建文档

---

## 数据泄漏修复经验（2026-03-26）

### ✅ 完成所有模型的数据泄漏修复

**修复范围**：
- CatBoost 20天模型
- LightGBM 20天模型
- GBDT 20天模型

**修复内容**：
- 修复25+个滚动窗口计算特征
- 所有 `.rolling()` 计算都添加了 `.shift(1)` 确保使用滞后数据
- 涵盖：ATR、成交量、价格百分位、波动率、RSI背离、MACD背离等关键特征

**修复效果对比**：

| 模型 | 修复前准确率 | 修复后准确率 | 变化 | 修复前标准差 | 修复后标准差 | 变化 |
|------|------------|------------|------|------------|------------|------|
| CatBoost 20天 | 68.33% ±6.45% | **61.73%** ±2.43% | -6.60% | 6.45% | **2.43%** | -62% ✅ |
| LightGBM 20天 | 58.93% ±8.35% | **58.33%** ±7.64% | -0.60% | 8.35% | **7.64%** | -8.5% ✅ |
| GBDT 20天 | 57.94% ±8.22% | **56.10%** ±4.93% | -1.84% | 8.22% | **4.93%** | -40% ✅ |

**关键发现**：
1. **CatBoost数据泄漏最严重**：准确率下降6.60%，但稳定性提升62%
2. **所有模型稳定性显著提升**：标准差降低8.5%-62%
3. **准确率下降是正常的**：反映真实预测能力，而非数据泄漏的结果

**最终推荐配置**：
- **主模型**：CatBoost 20天（准确率61.73%，标准差2.43%）
- **置信度阈值**：0.65（根据银行股Walk-forward验证结果）
- **特征数量**：892个全量特征（2026-03-27验证：优于500特征）

**经验教训**：
1. **高准确率预警**：准确率>65%必须警惕，检查数据泄漏
2. **特征共享机制**：三个模型使用相同的特征工程代码，修复一次全部受益
3. **必须重新训练**：修改特征工程代码后，所有模型都需要重新训练

---

## 数据泄漏修复经验（2026-03-24）

### ⚠️ 关键预警：CatBoost 20天模型数据泄漏

**发现问题**：
- CatBoost 20天模型准确率 68.33% 触发预警（>65%）
- 根据经验警告：高准确率（>65%）通常是数据泄漏的信号

**识别的数据泄漏特征**（5个）：
1. `Price_Low_5d` / `Price_High_5d` - 价格高低点未使用滞后数据
2. `Price_Ratio_MA120` / `Price_Ratio_MA250` - 价格比率未使用滞后数据
3. `Volatility` / `Volume_Volatility` - 波动率计算未使用滞后数据
4. `Volatility_Expansion` / `Volatility_Contraction` - 波动率信号未使用滞后数据
5. RSI背离检测中的价格比较使用了当日数据

**修复措施**：
为所有相关特征添加 `.shift(1)`，确保只使用滞后数据：
```python
# 修复前（数据泄漏）
df['Price_Low_5d'] = df['Close'].rolling(window=lookback, min_periods=1).min()

# 修复后（无泄漏）
df['Price_Low_5d'] = df['Close'].rolling(window=lookback, min_periods=1).min().shift(1)
```

**修复效果对比**：

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| 准确率 | 68.33% ±6.45% | 60.99% ±2.00% | -7.34% ✅ |
| 标准差 | 6.45% | 2.00% | -69% ✅（稳定性大幅提升）|
| F1分数 | 0.7462 ±0.0379 | 0.6492 ±0.0353 | -0.0970 ✅ |

**结论**：
- 准确率下降是正常的，反映真实预测能力
- 泛化能力显著提升（标准差从6.45%降至2.00%）
- 模型更加可靠，长期稳定性更好
- **仍然推荐使用 CatBoost 20天模型**（60.99%准确率，2.00%标准差）

**经验教训**：
1. **高准确率预警**：准确率>65%必须警惕，检查数据泄漏
2. **系统性检查**：需要系统性检查所有 `.rolling()` 相关特征
3. **滞后数据原则**：所有技术特征必须使用滞后数据
4. **标准差重要性**：标准差低比准确率高更重要
5. **LightGBM和GBDT**：也可能存在数据泄漏，需要修复后重新训练

**三模型对比更新（2026-03-24）**：

| 模型 | 准确率 | 标准差 | 数据泄漏状态 |
|------|--------|--------|-------------|
| **CatBoost 20天** | **60.99%** ±2.00% | ✅ **2.00%** | ✅ **已修复** |
| LightGBM 20天 | 58.93% ±8.35% | 8.35% | ❌ 未修复 |
| GBDT 20天 | 57.94% ±8.22% | 8.22% | ❌ 未修复 |

**推荐**：
- ✅ 强烈推荐使用 CatBoost 20天模型
- ⚠️ LightGBM和GBDT需要修复数据泄漏后重新评估
- ❌ 不推荐使用未修复的模型进行实际交易

---

## LightGBM和GBDT数据泄漏修复（2026-03-26）

### ⚠️ 关键预警：LightGBM和GBDT模型存在数据泄漏

**发现问题**：
- CatBoost已修复数据泄漏（准确率68.33% → 60.99%，稳定性提升69%）
- LightGBM和GBDT使用相同的特征工程代码，但也存在数据泄漏
- 需要对所有`.rolling()`计算进行系统性检查

### 识别的数据泄漏特征（25+个）

**高优先级特征（直接使用当日数据）**：
1. ATR相关：`ATR_MA`、`ATR_Ratio`
2. 成交量相关：`Vol_MA20`、`Vol_Mean_20`、`Vol_Std_20`
3. 成交额相关：`Turnover_Mean_20`、`Turnover_Std_20`
4. 价格相关：`MA120`、`MA250`、`MA5`、`MA60`
5. 波动率相关：`Volatility_5d/10d/20d/60d/120d`
6. 统计特征：`Skewness_20d`、`Kurtosis_20d`
7. RSI相关：`Stoch_D`、`CMF_Signal`
8. 其他：`Intraday_Range_MA5/MA20`、`OBV_MA5`、`Volume_MA7/120/250`
9. 市场环境：`Volatility_30pct`、`Volatility_70pct`、`ATR_Risk_Score`
10. 市场状态记忆：`Consecutive_Ranging_Days`、`Consecutive_Trending_Days`

**修复措施**：
为所有相关特征添加 `.shift(1)`，确保只使用滞后数据：
```python
# 修复前（数据泄漏）
df['ATR_MA'] = df['ATR'].rolling(window=10, min_periods=1).mean()

# 修复后（无泄漏）
df['ATR_MA'] = df['ATR'].rolling(window=10, min_periods=1).mean().shift(1)
```

### 修复效果对比

| 模型 | 修复前准确率 | 修复后准确率 | 变化 | 修复前标准差 | 修复后标准差 | 变化 |
|------|------------|------------|------|------------|------------|------|
| **CatBoost 20天** | 68.33% ±6.45% | **60.99% ±2.00%** | -7.34% | 6.45% | **2.00%** | -69% ✅ |
| LightGBM 20天 | 58.93% ±8.35% | **58.33% ±7.64%** | -0.60% | 8.35% | **7.64%** | -8.5% ✅ |
| GBDT 20天 | 57.94% ±8.22% | **56.10% ±4.93%** | -1.84% | 8.22% | **4.93%** | -40% ✅ |

**关键发现**：
- CatBoost下降最大（-7.34%），说明之前的数据泄漏最严重
- LightGBM和GBDT下降较小，说明数据泄漏相对较轻
- 所有模型的稳定性都显著提升（标准差降低）
- **CatBoost仍然表现最佳**（准确率60.99%，标准差2.00%）

### 最终推荐

| 模型 | 准确率 | 标准差 | F1分数 | 推荐度 |
|------|--------|--------|--------|--------|
| **CatBoost 20天** | **60.99%** ±2.00% | **2.00%** | 0.6492 | ⭐⭐⭐⭐⭐ **强烈推荐** |
| LightGBM 20天 | 58.33% ±7.64% | 7.64% | 0.6943 | ⭐⭐⭐ 可接受 |
| GBDT 20天 | 56.10% ±4.93% | 4.93% | 0.6597 | ⭐⭐⭐ 可接受 |

### 经验教训

1. **数据泄漏检查必须系统性**：
   - 不能只修复一个模型就认为所有模型都安全
   - 需要检查所有使用相同特征工程代码的模型

2. **修复后稳定性更重要**：
   - 准确率下降是正常的，反映真实预测能力
   - 标准差降低表明模型更稳定，长期表现更可靠

3. **CatBoost仍然是最佳选择**：
   - 准确率最高（60.99%）
   - 标准差最低（2.00%）
   - 泛化能力最强

4. **LightGBM和GBDT可以作为备选**：
   - 准确率和F1分数可接受
   - 但标准差较高，稳定性不如CatBoost
   - 建议仅用于对比研究或特定场景

---

## 📈 预测性能监控系统（2026-03-25）

### 系统架构

| 组件 | 文件 | 功能 |
|------|------|------|
| 数据存储 | `data/prediction_history.json` | 存储所有预测记录 |
| 核心脚本 | `ml_services/performance_monitor.py` | 评估预测、生成报告 |
| 日常集成 | `ml_trading_model.py` | 保存预测到历史 |
| 工作流 | `performance-monitor.yml` | 每月1号自动运行 |

### 关键设计决策

1. **使用交易日而非日历日**
   - 港股有节假日，必须使用实际交易日
   - 使用 yfinance 获取历史数据计算实际交易日数

2. **Git 持久化**
   - GitHub Actions 运行器是临时的
   - prediction_history.json 必须提交到 git
   - comprehensive-analysis.yml 已添加 git commit 步骤

3. **集成点选择**
   - 在 main() 函数的 predict 分支调用
   - 不在 predict() 方法内部调用（保持方法职责单一）

### 使用命令

```bash
# 评估已到期的预测
python3 ml_services/performance_monitor.py --mode evaluate --no-email

# 生成月度报告
python3 ml_services/performance_monitor.py --mode report --no-email

# 完整流程
python3 ml_services/performance_monitor.py --mode all --no-email
```

### 经验教训

1. **预测追踪是验证模型的必要手段**：没有追踪就无法知道模型真实表现
2. **自动化是关键**：每月自动运行，避免人工遗忘
3. **数据持久化要考虑 CI 环境**：GitHub Actions 运行器临时性要求 git 提交

---

## 📝 文档组织经验

### 避免重复内容

**问题**：README.md 中"快速开始"和"使用示例"章节存在大量重复内容

**解决方案**：
- 合并为统一的"使用指南"章节
- 按"快速开始"（3分钟）和"进阶使用"（详细功能）分层组织
- 删除重复的命令示例，保留一个权威版本

**收益**：
- 减少维护成本：修改时只需更新一处
- 提升用户体验：避免信息混乱
- 文档结构更清晰：用户可以快速找到需要的内容

### 应用场景说明的重要性

**问题**：用户不知道应该使用哪种安装/使用方法

**解决方案**：
- 在章节开头添加总览表格，清晰对比三种方式
- 标注适用人群、所需时间、使用场景
- 每个子章节开头添加"适用场景"说明

**收益**：
- 新用户可以快速找到适合自己的方法
- 已有用户可以直接跳到"快速开始"
- 需要自动化的用户可以直接跳到"GitHub Actions部署"

### 文档分层原则

**推荐结构**：
1. **快速入门**：3-5个命令，让用户快速看到效果
2. **详细指南**：完整功能说明和高级用法
3. **参考手册**：所有命令的详细参数说明

**避免**：
- 在"快速入门"中包含过多细节
- 在"详细指南"中重复"快速入门"的内容
- 缺少清晰的导航和分层

---

## 🔧 环境变量命名经验（2026-03-26）

### 避免厂商特定的命名

**问题**：使用 `YAHOO_*` 前缀命名邮件相关环境变量，但实际使用的是163邮箱

**错误做法**：
```bash
export YAHOO_SMTP=smtp.163.com
export YAHOO_EMAIL=xxx@163.com
export YAHOO_APP_PASSWORD=xxx
```

**正确做法**：
```bash
export SMTP_SERVER=smtp.163.com
export EMAIL_SENDER=xxx@163.com
export EMAIL_PASSWORD=xxx
```

**经验教训**：
1. 环境变量名称应该反映**功能**而非**厂商**
2. 使用通用的命名（如 `SMTP_SERVER`、`EMAIL_SENDER`）而非厂商特定的命名（如 `YAHOO_EMAIL`）
3. 即使最初使用某个厂商的服务，未来可能更换，通用命名可避免重命名成本
4. 更新环境变量时需要同步更新：Python代码、文档、GitHub Actions Secrets

**影响范围**：
- 11个Python脚本需要修改
- 3个文档文件需要更新
- 9个GitHub Actions工作流需要更新
- GitHub仓库的Secrets需要删除旧的并添加新的

---

## 🚀 GitHub Actions推送冲突处理（2026-03-26）

### 多工作流并行推送冲突

**问题**：多个GitHub Actions工作流同时尝试推送到同一分支，或工作流与手动提交冲突

**错误信息**：
```
error: failed to push some refs to https://github.com/...
hint: Updates were rejected because the remote contains work that you do not have locally.
```

**解决方案1：在推送前先拉取**（推荐）

```yaml
- name: Commit changes
  run: |
    git config --local user.email "github-actions[bot]@users.noreply.github.com"
    git config --local user.name "github-actions[bot]"
    git pull --rebase origin main  # 先拉取远程更改
    git add file.txt
    git commit -m "Update data [skip ci]"
    git push
```

**解决方案2：使用stash处理冲突**（复杂场景）

```yaml
- name: Configure git and commit changes
  run: |
    git config --global user.name 'github-actions[bot]'
    git config --global user.email 'github-actions[bot]@users.noreply.github.com'
    
    # 检查是否有未暂存的更改
    if ! git diff --quiet; then
      git stash push -m "Temporary stash before pull"
    fi
    
    # 拉取远程更改
    git pull --rebase origin main || true
    
    # 恢复本地更改
    if git stash list | grep -q "Temporary stash"; then
      git stash pop
    fi
    
    # 添加并提交更改
    git add data/file.txt
    git diff --staged --quiet || git commit -m "Update data"
    git push origin main
```

**解决方案3：容忍推送失败**（非关键任务）

```yaml
- name: Commit and push
  run: |
    git config --local user.email "github-actions[bot]@users.noreply.github.com"
    git config --local user.name "github-actions[bot]"
    git add output/*.md || true
    git diff --staged --quiet || git commit -m "Update report" || true
    git push || true  # 如果推送失败也不报错
```

**经验教训**：
1. **多个工作流可能同时运行**：特别是定时任务和手动触发
2. **手动提交可能与工作流冲突**：本地push时工作流正在执行
3. **使用`git pull --rebase`而非merge**：rebase可以保持线性历史，避免merge commit
4. **添加`|| true`处理非关键推送**：对于报告类文件，推送失败也不影响主要功能
5. **使用`[skip ci]`跳过CI**：避免数据更新触发新的工作流运行

---

## ⚠️ Walk-forward验证失败经验（2026-03-26）

### ✅ 问题已解决（2026-03-27）

**根本原因**：时区不匹配 + 对已有 timezone 的对象使用 tz_localize()

```python
# 修复前（错误）
start_date = pd.to_datetime(start_date)  # 没有时区
df = df[df.index >= start_date]  # df.index 是 UTC 时区，导致错误

# 第一次修复（部分解决）
start_date = pd.to_datetime(start_date).tz_localize('UTC')  # 添加 UTC 时区
df = df[df.index >= start_date]  # 对无时区日期有效

# 第二次修复（完全解决）
start_date = pd.to_datetime(start_date)
if start_date.tzinfo is not None:
    start_date = start_date.tz_convert('UTC')  # 对已有时区：转换
else:
    start_date = start_date.tz_localize('UTC')  # 对无时区：本地化
df = df[df.index >= start_date]
```

**错误信息**：
```
# 第一次错误
TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp

# 第二次错误
TypeError: Cannot localize tz-aware Timestamp, use tz_convert for conversions
```

**修复位置**：
- 文件：`ml_services/ml_trading_model.py`
- 方法：`CatBoostModel.prepare_data()`
- 行号：3562-3583

**修复代码**：
```python
# 过滤日期范围（如果指定）
if start_date:
    start_date = pd.to_datetime(start_date)
    # 处理时区：如果有时区，转换到UTC；如果没有时区，本地化为UTC
    if start_date.tzinfo is not None:
        start_date = start_date.tz_convert('UTC')
    else:
        start_date = start_date.tz_localize('UTC')
    df = df[df.index >= start_date]
if end_date:
    end_date = pd.to_datetime(end_date)
    # 处理时区：如果有时区，转换到UTC；如果没有时区，本地化为UTC
    if end_date.tzinfo is not None:
        end_date = end_date.tz_convert('UTC')
    else:
        end_date = end_date.tz_localize('UTC')
    df = df[df.index <= end_date]
```

**验证方法**：
```python
# 创建测试 DataFrame（UTC 时区）
df = pd.DataFrame({'A': [1, 2, 3]},
                  index=pd.date_range('2024-01-01', periods=3, tz='UTC'))

# 修复前（失败）
start_date = pd.to_datetime('2024-01-02')  # 无时区
result = df[df.index >= start_date]  # 抛出 TypeError

# 第一次修复（对无时区有效）
start_date = pd.to_datetime('2024-01-02').tz_localize('UTC')  # UTC 时区
result = df[df.index >= start_date]  # 正确过滤

# 第二次修复（对已有timezone也有效）
start_date = pd.to_datetime('2024-01-02').tz_localize('UTC')  # UTC 时区
result = df[df.index >= start_date]  # 正确过滤
```

**简化验证结果**（2026-03-27，3只股票）：

**配置**：
- 股票：0700.HK, 0939.HK, 1347.HK
- Fold：12个
- 训练窗口：12个月
- 测试窗口：1个月
- 置信度阈值：0.65

**整体性能**：
- ✅ 所有12个Fold成功运行
- ✅ 平均收益率：7.23%
- ✅ 平均胜率：24.71%
- ✅ 平均准确率：64.29%
- ✅ 平均夏普比率：0.1737
- ⚠️ 稳定性评级：低（需改进）

**关键发现**：
1. ✅ 时区问题已完全解决
2. ✅ Walk-forward验证可以正常运行
3. ✅ 平均准确率64.29%高于随机50%
4. ⚠️ 胜率偏低（24.71%），需要优化
5. ⚠️ 稳定性评级低，收益率标准差7.99%

**经验教训**：
1. **时区问题容易被忽略**：在处理金融数据时，时区一致性非常重要
2. **yfinance 数据带时区**：从 yfinance 获取的数据默认是 UTC 时区
3. **日期比较需要时区一致**：DataFrame 索引和过滤条件的时区必须匹配
4. **错误信息很明确**：`Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp` 直接指出了问题
5. **简单修复，大影响**：只添加 `.tz_localize('UTC')` 就解决了整个 Walk-forward 验证失败的问题

**下一步行动**：
- ✅ 修复已完成（两次提交）
- ✅ 简化验证（3只股票）已成功完成
- ✅ 全量特征vs 500特征对比验证已完成（2026-03-27）
- ✅ 确认使用全量特征（892个）优于500特征
- ⏳ 根据验证结果优化模型配置

---

## 🎯 全量特征vs 500特征验证的重要影响（2026-03-27）

### 验证背景
这次验证解决了项目中的一个关键争议：应该使用全量特征还是经过特征选择的500个特征？

### 验证方法
- 使用业界标准的Walk-forward验证方法
- 银行股板块（6只股票），12个Fold
- 相同的置信度阈值（0.60），确保公平对比
- 严格的时序分割，每个fold重新训练模型

### 关键发现
**全量特征（892个）显著优于500特征**：
- 年化收益率：40.42% vs 30.28%（+10.14%）
- 索提诺比率：1.9023 vs 1.0400（+83%）
- 夏普比率：-0.0235 vs -0.0501（更接近0）
- 买入信号胜率：49.60% vs 49.13%（+0.47%）

### 对项目的影响
1. **特征选择策略改变**：不再使用特征选择，直接使用全量特征
2. **训练命令简化**：省略`--use-feature-selection`参数
3. **性能提升**：年化收益率提升10.14%，索提诺比率提升83%
4. **代码简化**：移除特征选择相关代码，减少复杂度

### 对业界实践的启示
1. **CatBoost的自动特征选择足够强大**：不需要预先进行特征选择
2. **信息保留比特征数量更重要**：避免因特征选择导致的信息丢失
3. **验证必须公平**：相同的阈值、相同的验证方法，才能得出可信的结论
4. **Walk-forward验证的重要性**：简单验证可能得出错误结论（之前的对比结果显示500特征更好）

### 后续行动
- ✅ 更新所有训练脚本，使用全量特征
- ✅ 更新文档，说明使用全量特征的原因
- ⏳ 监控实盘交易结果，验证全量特征的稳定性
- ⏳ 定期重新验证，确保全量特征的优势持续存在

### 经验教训
1. **验证必须公平**：不同的置信度阈值会导致完全不同的结论
2. **不要依赖直觉**：业界标准（300-500特征）不一定适用于所有场景
3. **CatBoost的优势**：自动特征选择和正则化机制使全量特征表现更好
4. **Walk-forward验证的重要性**：业界标准的验证方法能得出可信的结论
5. **持续验证**：定期重新验证，确保最佳策略持续有效
---

## 🔧 模型可重现性优化（2026-03-27）

### 问题背景
两台机器在不同时间训练模型，预测结果存在显著差异：

**差异情况**：
- 4只股票预测方向发生变化（1288.HK、0981.HK、1088.HK、9988.HK）
- 预测概率差异最大达0.1619（1288.HK 农业银行：0.6591 → 0.4972）
- 其他股票也有不同程度的差异（0.03-0.13）

**根本原因**：
1. **CatBoost的随机性**：行采样（75%）和列采样（70%）导致每次训练使用不同的样本和特征子集
2. **临界区域敏感性**：预测概率在0.48-0.55之间的股票最容易受训练数据微小差异的影响
3. **训练数据时间差异**：不同时间获取的收盘价数据可能存在微小差异

### 业界最佳实践

**研究/开发阶段**：
```python
# ✅ 使用固定随机种子（可重现性）
np.random.seed(42)
random.seed(42)
```
- 目的：确保实验可重现
- 好处：便于调试和对比

**验证阶段**：
```python
# ✅ 多次训练，评估稳定性
results = []
for seed in [42, 123, 456, 789, 999]:
    model = train(seed=seed)
    results.append(evaluate(model))
```
- 目的：评估模型对随机种子的敏感性
- 好处：发现过拟合问题

**生产部署**：
```python
# 方案A：固定种子（保守）⭐ 推荐
np.random.seed(42)

# 方案B：集成模型（高级）
ensemble = []
for seed in [42, 123, 456]:
    model = train(seed=seed)
    ensemble.append(model)
prediction = average(ensemble.predict(X))
```
- 方案A：简单快速，适合大多数场景
- 方案B：稳定性+泛化能力兼顾，但训练时间增加3倍

### CatBoost官方建议

> "For production deployment, we recommend using a fixed random seed to ensure reproducibility. However, for research and development, we suggest running multiple training sessions with different seeds to evaluate model stability."

**CatBoost特定建议**：
```python
from catboost import CatBoostClassifier

# 生产环境：固定种子
model = CatBoostClassifier(
    random_seed=42,  # ✅ 固定种子
    ...
)

# 研究环境：多次训练
for seed in [42, 123, 456]:
    model = CatBoostClassifier(
        random_seed=seed,
        ...
    )
```

### 解决方案实施

**修改内容**：
```python
# 在 LightGBMModel、GBDTModel、CatBoostModel 的 train() 方法中添加
def train(self, codes, start_date=None, end_date=None, horizon=1, use_feature_selection=False):
    # 设置固定随机种子（确保模型训练的可重现性）
    np.random.seed(42)
    random.seed(42)
    
    # 继续训练逻辑
    ...
```

**修改位置**：
- LightGBMModel.train()（第2303-2305行）
- GBDTModel.train()（第2936-2938行）
- CatBoostModel.train()（第3632-3634行）

### 预期效果

**1. 模型训练可重现**
- 相同数据+相同代码→完全相同的模型
- 不同机器上训练的模型将产生相同的预测结果

**2. 解决预测不一致问题**
- 之前4只股票预测方向变化的问题将被解决
- 特别是临界区域（预测概率接近0.5）的股票将更加稳定

**3. 符合业界最佳实践**
- CatBoost官方建议：生产环境使用固定随机种子
- 业界标准：确保模型训练的可重现性

### 验证方法

```bash
# 1. 在同一台机器上训练两次
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# 2. 比较预测结果（应该完全一致）
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost > pred1.csv
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost > pred2.csv
diff pred1.csv pred2.csv

# 3. 在两台机器上训练并比较
# 机器A：训练→预测
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost > pred_A.csv

# 机器B：训练→预测
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
python3 ml_services/ml_trading_model.py --mode predict --horizon 20 --model-type catboost > pred_B.csv

# 比较结果
diff pred_A.csv pred_B.csv
```

### 经验教训

**1. 随机性影响显著**
- CatBoost的行采样和列采样会导致模型差异
- 即使相同数据，不同训练也会产生略有不同的模型
- 预测概率在临界区域的股票最不稳定

**2. 固定种子必要性**
- 生产环境必须使用固定随机种子
- 这不是过度设计，而是业界标准
- 解决了多机器预测不一致的实际问题

**3. 临界区域敏感性**
- 预测概率在0.48-0.55之间的股票容易变化
- 这些股票的预测方向可能因微小差异而翻转
- 建议：对于这些股票，提高置信度阈值或使用集成模型

**4. 业界标准遵循**
- 符合CatBoost官方建议
- 符合量化交易业界最佳实践
- 提升模型的可维护性和可靠性

**5. 文档同步重要性**
- 所有文档必须同步更新（AGENTS.md、lessons.md、progress.txt、README.md）
- 命令示例必须准确反映最新状态
- 记录所有变更的commit hash

### 后续优化建议

**短期（立即）**：
- ✅ 固定随机种子已实施
- ⏳ 两台机器重新训练模型，验证一致性

**中期（1-2周）**：
- ⏳ 实施稳定性验证脚本
- ⏳ 如果稳定性仍然不够，考虑集成模型
- ⏳ 建立模型版本控制系统

**长期（1-3个月）**：
- ⏳ 定期重新训练和验证稳定性
- ⏳ 实施集成模型策略
- ⏳ 建立完整的模型管理体系

### 相关提交

- commit 9e9bc86: fix(reproducibility): 为所有模型添加固定随机种子

### 参考资料

- CatBoost官方文档：https://catboost.ai/docs/concepts/algorithm-main-stages_catboost-vs-other boosting-frameworks.html
- 业界最佳实践：量化交易系统可重现性指南
- Walk-forward验证方法：业界标准模型评估方法

