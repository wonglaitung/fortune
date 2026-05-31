---
description: 新增特征合规性验证流程 - 动态读取 FEATURE_ENGINEERING.md 验证清单
allowed-tools: read_file, write_file, edit_file, bash, grep, glob
---

## Context

你是港股智能分析系统的特征工程专家。当新增特征时，执行标准化的验证流程。

## 关键原则

**⚠️ 重要：验证标准以 `docs/FEATURE_ENGINEERING.md` 为准！**

本命令是执行框架，具体验证标准请动态读取：
- **8步验证清单**：`docs/FEATURE_ENGINEERING.md` 第 519-529 行
- **数据泄漏阈值**：`docs/FEATURE_ENGINEERING.md` 第 439-444 行
- **相关性阈值**：`docs/FEATURE_ENGINEERING.md` 第 447-465 行
- **默认值原则**：`docs/FEATURE_ENGINEERING.md` 第 486-500 行

---

## 验证流程

### 步骤 0：读取最新验证标准

**必须首先执行**：读取 FEATURE_ENGINEERING.md 获取最新验证清单。

```bash
# 读取 8 步验证清单
grep -A 15 "新特征上线验证清单" docs/FEATURE_ENGINEERING.md

# 读取数据泄漏阈值
grep -A 6 "数据泄漏阈值" docs/FEATURE_ENGINEERING.md

# 读取相关性标准
grep -A 5 "阈值：新增特征与现有特征相关性" docs/FEATURE_ENGINEERING.md
```

---

### 步骤 1：确认特征信息

请用户提供：
1. **新特征名称**：如 `Hybrid_Conditional_Vol`
2. **特征类型**：时间序列/市场级/网络/基本面/交叉
3. **实现文件**：如 `ml_services/hybrid_volatility_model.py`

---

### 步骤 2：执行 8 步验证

根据 FEATURE_ENGINEERING.md 的清单执行验证：

#### 2.1 数据泄漏检查

```bash
# 搜索新特征实现代码中的 .rolling() 和 .shift()
grep -n "New_Feature" <实现文件>
grep -n ".rolling(" <实现文件> | head -20
grep -n ".shift(" <实现文件> | head -20
```

**判断标准**（来自 FEATURE_ENGINEERING.md）：
- 所有 `.rolling()` 后必须有 `.shift(1)`
- 标签必须使用 `shift(-horizon)`

#### 2.2 绝对值特征标准化

```bash
# 检查是否在排除列表中
grep -A 50 "ABSOLUTE_PRICE_FEATURES = \[" ml_services/ml_trading_model.py
```

**判断标准**：
- 如是绝对价格/成交量：添加到排除列表或创建标准化替代
- 如是比率/百分比：无需标准化

#### 2.3 市场级特征交叉

```python
# 验证特征是否对所有股票同值
if df.groupby('Date')['New_Feature'].nunique().max() == 1:
    print("市场级特征，必须与股票特征交叉")
```

#### 2.4 特征单调性

检查交叉方式是否与特征单调性匹配（正向×正向→乘法，负向×负向→风险放大等）

#### 2.5 相关性分析

```python
# 计算相关性
import yfinance as yf
from ml_services.ml_trading_model import BaseTradingModel

hsi = yf.Ticker('^HSI')
df = hsi.history(period='2y')
model = BaseTradingModel()
df = model.prepare_data(df, mode='backtest')

new_feature = 'New_Feature'
correlations = df.corr()[new_feature].abs().sort_values(ascending=False)
print(f"最高相关: {correlations.index[1]} = {correlations.iloc[1]:.4f}")
print(f"结果: {'✅ 通过' if correlations.iloc[1] < 0.8 else '❌ 失败'}")
```

**阈值**：< 0.8（来自 FEATURE_ENGINEERING.md）

#### 2.6 特征统计

```python
# 计算基本统计
print(f"均值: {df[new_feature].mean():.4f}")
print(f"NaN比例: {df[new_feature].isna().sum() / len(df) * 100:.1f}%")
print(f"零值比例: {(df[new_feature] == 0).sum() / len(df) * 100:.1f}%")
```

**标准**：NaN < 30%，零值 < 50%

#### 2.7 默认值检查

检查默认值是否与有效值范围分离

#### 2.8 缓存清理（必须执行）

```bash
rm -rf data/feature_cache/*.pkl
```

---

### 步骤 3：语法检查

```bash
python3 -m py_compile <修改的文件>
```

---

### 步骤 4：输出验证报告

```markdown
# 新增特征合规性验证报告

## 特征信息
- 特征名称：[特征名]
- 特征类型：[类型]
- 实现文件：[文件]

## 验证结果（8步清单）

| 步骤 | 验证项 | 结果 | 详情 |
|------|--------|------|------|
| 1 | 数据泄漏 | ✅/❌ | |
| 2 | 绝对值标准化 | ✅/不适用/❌ | |
| 3 | 市场级特征交叉 | ✅/不适用/❌ | |
| 4 | 特征单调性 | ✅/❌ | |
| 5 | 相关性检查 | ✅/❌ | 最高相关: X.XX |
| 6 | 特征统计 | ✅/❌ | |
| 7 | 默认值设计 | ✅/❌ | |
| 8 | 缓存清理 | ✅/❌ | |

**验证通过项：X/8**
```

---

## 执行检查清单

验证前必须：
- [ ] 已读取 `docs/FEATURE_ENGINEERING.md` 最新验证标准

验证后必须：
- [ ] 已清除特征缓存 `rm -rf data/feature_cache/*.pkl`
- [ ] 已执行语法检查

---

## 快速参考

| 文件 | 内容 |
|------|------|
| `docs/FEATURE_ENGINEERING.md` | **验证标准（唯一真相源）** |
| `ml_services/ml_trading_model.py` | 特征工程实现 |
| `lessons.md` | 经验教训 |

**记住**：
1. 验证标准以 FEATURE_ENGINEERING.md 为准
2. 新增特征后必须清除缓存
3. 本命令是执行框架，不重复维护验证标准
