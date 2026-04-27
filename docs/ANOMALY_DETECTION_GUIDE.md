# 港股异常检测指南

> **最后更新**：2026-04-27

---

## 异常检测概览

### 目的

识别港股市场中的异常价格和成交量模式，为交易决策提供风险信号和机会识别。

### 应用场景

1. **风险预警**：识别极端市场波动，提前预警风险
2. **机会识别**：识别异常低点或高点，捕捉交易机会
3. **市场监控**：持续监控市场状态，及时发现异常

---

## 双层检测架构

### 架构设计

```
Layer 1: Z-Score 实时检测（每小时）
  ├─ 检测范围：价格异常、成交量异常、价格波动异常
  ├─ 检测方法：统计方法（Z-Score > 3）
  └─ 输出：快速异常列表（high/medium/low）

Layer 2: Isolation Forest 深度分析（凌晨2点）
  ├─ 检测范围：复杂异常模式、非线性行为
  ├─ 检测方法：无监督机器学习（Isolation Forest）
  └─ 输出：深度异常分析报告
```

### 检测方法对比

| 方法 | 类型 | 执行频率 | 检测范围 | 适用场景 |
|------|------|---------|---------|---------|
| **Z-Score检测器** | 统计方法 | 每小时 | 价格、成交量、波动异常 | 实时监控 |
| **Isolation Forest** | 机器学习 | 每天凌晨2点 | 复杂异常模式 | 深度挖掘 |

---

## 验证策略

| 异常类型 | 策略 | 5日收益 | 胜率 | 应用场景 |
|---------|------|---------|------|----------|
| **价格异常 + 当日下跌** | 抄底 | +4.12% | **72%** | 超跌反弹机会 |
| 价格异常 + 当日上涨 | 观望 | +1.96% | 54% | 追涨风险 |
| IF high 异常 | 减仓 | -3.04% | 43% | 多维异常预警 |

---

## 使用方法

```bash
# 快速检测（Z-Score）
python3 detect_stock_anomalies.py --mode standalone --mode-type quick

# 深度检测（Isolation Forest）
python3 detect_stock_anomalies.py --mode standalone --mode-type deep
```

---

## 模块结构

```
anomaly_detector/
├── __init__.py                    # 模块入口
├── zscore_detector.py             # Z-Score检测器
├── isolation_forest_detector.py   # Isolation Forest检测器
├── feature_extractor.py           # 特征提取器
├── anomaly_integrator.py          # 异常整合器
└── cache.py                       # 异常缓存
```

---

## 核心警告

⚠️ **股票策略不适用于加密货币市场**

港股和加密货币市场特性不同，本系统针对港股优化，加密货币需要专门的策略。

---

## 相关文件

- **主脚本**：`detect_stock_anomalies.py`
- **模块目录**：`anomaly_detector/`
- **缓存文件**：`data/anomaly_cache.json`
