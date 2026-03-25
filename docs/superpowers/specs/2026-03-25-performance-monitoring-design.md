# 预测性能监控系统设计文档

**日期**: 2026-03-25
**状态**: 待审核

---

## 1. 概述

### 1.1 目标
建立一个预测性能监控系统，追踪模型预测与实际结果的对比，识别表现不佳的股票和板块。

### 1.2 核心需求
- **目标**: 追踪实际收益 vs 预期收益，识别表现不佳的股票/板块
- **时间周期**: 中期（20天持有期对齐）
- **通知方式**: 每月独立邮件报告
- **频率**: 每月1号
- **指标**: 胜率 + 收益率 + 风险调整指标（全部类别）

### 1.3 设计方案
采用**预测追踪系统**方案：存储每次预测，20天后获取实际价格进行对比。

---

## 2. 架构设计

### 2.1 数据流

```
每日预测流程:
comprehensive-analysis.yml (GitHub Actions)
    │
    ├── ml_trading_model.py predict
    │       │
    │       ├── 生成预测 → predictions_20d.csv
    │       └── 保存历史 → prediction_history.json
    │
    └── git commit & push prediction_history.json

每月监控流程:
performance-monitor.yml (GitHub Actions)
    │
    ├── git pull (获取最新 prediction_history.json)
    │
    ├── performance_monitor.py
    │       │
    │       ├── 读取 prediction_history.json
    │       ├── 获取实际价格 (yfinance)
    │       ├── 计算指标
    │       └── 生成报告并发送邮件
```

### 2.2 组件清单

| 组件 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 预测存储 | `data/prediction_history.json` | 新建 | 存储历史预测，提交到 git |
| 监控脚本 | `ml_services/performance_monitor.py` | 新建 | 主监控脚本 |
| 预测保存逻辑 | `ml_services/ml_trading_model.py` | 修改 | 添加 `save_prediction_to_history()` |
| 综合分析工作流 | `.github/workflows/comprehensive-analysis.yml` | 修改 | 添加 git 提交步骤 |
| 月度监控工作流 | `.github/workflows/performance-monitor.yml` | 新建 | 每月1号自动运行 |

---

## 3. 数据结构

### 3.1 prediction_history.json

```json
{
  "predictions": [
    {
      "prediction_id": "20260325_0700_HK",
      "timestamp": "2026-03-25T09:00:00",
      "stock_code": "0700.HK",
      "stock_name": "腾讯控股",
      "sector": "tech",
      "horizon": 20,
      "predicted_direction": "up",
      "prediction_probability": 0.65,
      "confidence_level": "high",
      "entry_price": 380.50,
      "model_type": "catboost",
      "model_accuracy": 0.60,
      "outcome": null,
      "actual_return": null,
      "actual_direction": null,
      "evaluated_at": null
    }
  ]
}
```

### 3.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| prediction_id | string | 唯一标识：日期_股票代码 |
| timestamp | string | 预测时间 (ISO 8601) |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| sector | string | 板块代码 |
| horizon | int | 预测周期（天） |
| predicted_direction | string | 预测方向：up/down |
| prediction_probability | float | 预测概率 (0-1) |
| confidence_level | string | 置信度等级：high/medium/low |
| entry_price | float | 预测时价格 |
| model_type | string | 模型类型：catboost |
| model_accuracy | float | 模型历史准确率 |
| outcome | string/null | 结果：correct/incorrect/null |
| actual_return | float/null | 实际收益率 |
| actual_direction | string/null | 实际方向：up/down |
| evaluated_at | string/null | 评估时间 |

### 3.3 数据管理
- 每只股票每天只保留一条预测（同日重跑会覆盖）
- 预估数据量：28股票 × 250交易日 × 2年 = ~14K条
- 文件大小预估：< 5MB

---

## 4. 月度报告格式

### 4.1 报告结构

```
邮件主题: 📊 月度预测性能监控报告 - 2026年3月

════════════════════════════════════════════════════════════════
                         执行摘要
════════════════════════════════════════════════════════════════

| 指标 | 本月 | 上月 | 变化 |
|------|------|------|------|
| 预测准确率 | 58.2% | 60.1% | -1.9% |
| 买入胜率 | 52.3% | 50.8% | +1.5% |
| 平均收益 | 3.2% | 2.8% | +0.4% |
| 夏普比率 | 1.24 | 1.15 | +0.09 |

⚠️ 注意：预测准确率连续两个月下降

════════════════════════════════════════════════════════════════
                       板块表现排名
════════════════════════════════════════════════════════════════

| 排名 | 板块 | 准确率 | 胜率 | 收益率 | 夏普 |
|------|------|--------|------|--------|------|
| 1 | 消费股 | 62.1% | 58.2% | 4.5% | 1.54 |
| 2 | 银行股 | 60.3% | 54.1% | 3.8% | 1.42 |
| 3 | 半导体股 | 55.8% | 48.9% | 2.1% | 0.87 |
| ... | ... | ... | ... | ... | ... |

表现最差：科技股 (准确率 45.2%, 胜率 38.1%)

════════════════════════════════════════════════════════════════
                   个股表现 TOP/BOTTOM 10
════════════════════════════════════════════════════════════════

TOP 10 表现最佳:
| 股票 | 预测准确 | 买入胜率 | 平均收益 |
|------|---------|---------|---------|
| 汇丰银行 | 72.3% | 68.0% | 5.2% |
| ... | ... | ... | ... |

BOTTOM 10 需关注:
| 股票 | 预测准确 | 买入胜率 | 平均收益 |
|------|---------|---------|---------|
| 地平线机器人 | 38.2% | 25.0% | -4.8% |
| ... | ... | ... | ... |

════════════════════════════════════════════════════════════════
                     预测 vs 实际对比
════════════════════════════════════════════════════════════════

高置信度预测 (概率>60%):
  - 预测准确: 58.2%
  - 预测方向: 上涨 128次, 下跌 22次
  - 实际上涨: 78次, 实际下跌: 49次

中等置信度预测 (概率50-60%):
  - 预测准确: 51.1%
  - 预测方向: 上涨 163次, 下跌 44次
  - 实际上涨: 88次, 实际下跌: 119次

════════════════════════════════════════════════════════════════
                       趋势分析
════════════════════════════════════════════════════════════════

近6个月准确率趋势:
  10月: 61.2%  11月: 59.8%  12月: 58.5%
   1月: 60.1%   2月: 59.3%   3月: 58.2%  ↓

⚠️ 警告: 准确率呈下降趋势，建议检查模型是否需要重新训练

════════════════════════════════════════════════════════════════
                       风险指标
════════════════════════════════════════════════════════════════

| 指标 | 本月 | 上月 | 状态 |
|------|------|------|------|
| 最大回撤 | -8.2% | -6.5% | ⚠️ 扩大 |
| 收益波动 | 12.3% | 11.8% | 正常 |
| 盈亏比 | 1.45 | 1.38 | ✅ 改善 |

═══════════════════════════════════════════════════════════════
报告范围: 2026-03-01 至 2026-03-31
预测样本: 620个 (28只股票 × 22交易日)
已评估: 580个 (预测后满20天)
```

### 4.2 指标计算

**胜率类指标:**
- 预测准确率 = 正确预测数 / 总预测数
- 买入胜率 = 盈利交易数 / 买入信号数
- 正确决策比例 = 正确决策数 / 总决策数

**收益类指标:**
- 平均收益率 = Σ(actual_return) / N
- 累计收益率 = Π(1 + actual_return) - 1

**风险调整指标:**
- 夏普比率 = (年化收益率 - 无风险利率) / 年化标准差
- 最大回撤 = min(cumulative_return - peak)
- 盈亏比 = 平均盈利 / 平均亏损

---

## 5. 实现细节

### 5.1 ml_trading_model.py 修改

在 `CatBoostModel.predict()` 方法末尾添加：

```python
def save_prediction_to_history(self, stock_code, stock_name, sector, 
                                predicted_direction, probability, entry_price):
    """保存预测到历史记录"""
    history_file = os.path.join(self.data_dir, 'prediction_history.json')
    
    # 读取现有历史
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {'predictions': []}
    
    # 构建预测记录
    prediction = {
        'prediction_id': f"{datetime.now().strftime('%Y%m%d')}_{stock_code}",
        'timestamp': datetime.now().isoformat(),
        'stock_code': stock_code,
        'stock_name': stock_name,
        'sector': sector,
        'horizon': self.horizon,
        'predicted_direction': predicted_direction,
        'prediction_probability': probability,
        'confidence_level': 'high' if probability > 0.60 else ('low' if probability <= 0.50 else 'medium'),
        'entry_price': entry_price,
        'model_type': 'catboost',
        'model_accuracy': self.accuracy,
        'outcome': None,
        'actual_return': None,
        'actual_direction': None,
        'evaluated_at': None
    }
    
    # 更新或添加预测（同日同股票覆盖）
    existing_ids = [p['prediction_id'] for p in history['predictions']]
    if prediction['prediction_id'] in existing_ids:
        idx = existing_ids.index(prediction['prediction_id'])
        history['predictions'][idx] = prediction
    else:
        history['predictions'].append(prediction)
    
    # 保存
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
```

### 5.2 performance_monitor.py 主要函数

```python
def evaluate_predictions(history_file, horizon=20):
    """评估已满持有期的预测"""
    # 读取历史
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    today = datetime.now()
    for pred in history['predictions']:
        if pred['outcome'] is not None:
            continue
        
        pred_date = datetime.fromisoformat(pred['timestamp'])
        days_passed = (today - pred_date).days
        
        if days_passed >= horizon:
            # 获取实际价格
            actual_price = fetch_price(pred['stock_code'], today)
            pred['actual_return'] = (actual_price - pred['entry_price']) / pred['entry_price']
            pred['actual_direction'] = 'up' if pred['actual_return'] > 0 else 'down'
            pred['outcome'] = 'correct' if pred['predicted_direction'] == pred['actual_direction'] else 'incorrect'
            pred['evaluated_at'] = today.isoformat()
    
    # 保存更新后的历史
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    return history

def calculate_metrics(history, start_date, end_date):
    """计算指定时间范围内的指标"""
    # 筛选时间范围内的已评估预测
    predictions = [p for p in history['predictions'] 
                   if p['evaluated_at'] and 
                   start_date <= p['evaluated_at'] <= end_date]
    
    # 按股票/板块聚合计算指标
    ...

def generate_report(metrics, prev_metrics):
    """生成邮件报告"""
    ...

def send_email(report):
    """发送邮件"""
    ...
```

### 5.3 GitHub Actions 工作流修改

**comprehensive-analysis.yml 添加提交步骤:**

```yaml
- name: Run comprehensive analysis script
  env:
    ...
  run: |
    bash scripts/run_comprehensive_analysis.sh

- name: Commit prediction history
  run: |
    git config --local user.email "github-actions[bot]@users.noreply.github.com"
    git config --local user.name "github-actions[bot]"
    git add data/prediction_history.json
    git diff --quiet && git diff --staged --quiet || git commit -m "Update prediction history [skip ci]"
    git push
```

**新建 performance-monitor.yml:**

```yaml
name: 月度预测性能监控

on:
  schedule:
    - cron: '0 0 1 * *'  # 每月1号 UTC 00:00 (香港时间 08:00)
  workflow_dispatch:

jobs:
  monitor:
    runs-on: ubuntu-latest
    env:
      TZ: Asia/Hong_Kong
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run performance monitor
        env:
          YAHOO_EMAIL: ${{ secrets.YAHOO_EMAIL }}
          YAHOO_APP_PASSWORD: ${{ secrets.YAHOO_APP_PASSWORD }}
          RECIPIENT_EMAIL: mall_cn@hotmail.com, wonglaitung@gmail.com
          YAHOO_SMTP: ${{ secrets.YAHOO_SMTP }}
        run: |
          python3 ml_services/performance_monitor.py
```

---

## 6. 错误处理

### 6.1 价格获取失败
- 重试机制：最多3次，间隔5秒
- 失败后跳过该预测，下次运行时重试
- 记录失败日志

### 6.2 数据文件损坏
- 使用 JSON schema 验证
- 损坏时备份并重新初始化
- 发送告警邮件

### 6.3 邮件发送失败
- 重试机制：最多3次
- 记录失败日志到 `logs/performance_monitor_error.log`

---

## 7. 测试计划

### 7.1 单元测试
- `test_save_prediction_to_history()` - 预测保存逻辑
- `test_evaluate_predictions()` - 预测评估逻辑
- `test_calculate_metrics()` - 指标计算逻辑

### 7.2 集成测试
- 模拟完整流程：预测 → 保存 → 评估 → 报告
- 使用历史数据验证指标计算正确性

### 7.3 手动测试
```bash
# 本地运行监控（不发送邮件）
python3 ml_services/performance_monitor.py --no-email

# 指定日期范围测试
python3 ml_services/performance_monitor.py --start-date 2026-02-01 --end-date 2026-02-28 --no-email
```

---

## 8. 后续优化

### 8.1 短期（可选）
- 添加周度简报（仅摘要，无详细报告）
- 支持多模型对比（当前仅 CatBoost）

### 8.2 中期（可选）
- Web 仪表板可视化
- 模型漂移自动告警
- 自动触发模型重训练

### 8.3 长期（可选）
- 实盘账户对接
- 实际交易记录追踪
- 策略参数自动优化