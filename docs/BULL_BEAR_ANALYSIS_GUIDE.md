# 牛熊市分析自动化配置

## 目录结构

```
output/
├── bull_bear_analysis/              # 牛熊市分析报告专用目录
│   ├── 2024/                       # 按年份分类
│   │   ├── 2024-Q1.md             # 季度报告
│   │   ├── 2024-Q2.md
│   │   ├── 2024-Q3.md
│   │   └── 2024-Q4.md
│   └── 2025/
│       ├── 2025-Q1.md
│       ├── 2025-Q2.md
│       ├── 2025-Q3.md
│       └── 2025-Q4.md
└── bull_bear_analysis_latest.md    # 最新报告（自动更新）
```

## 使用方法

### 方式1：使用Shell脚本（推荐）⭐

**完整自动化流程**：回测 → 牛熊市分析 → 生成报告

```bash
# 使用默认参数（2024-01-01 至 2025-12-31）
./run_bull_bear_analysis.sh

# 自定义日期范围
./run_bull_bear_analysis.sh 2024-01-01 2025-12-31

# 自定义日期范围和输出格式
./run_bull_bear_analysis.sh 2024-01-01 2025-12-31 md
```

**执行流程**：
1. **步骤1**：运行20天持有期回测（CatBoost 20天模型）
2. **步骤2**：查找最新生成的回测交易记录文件
3. **步骤3**：使用新生成的交易记录进行牛熊市分析
4. **步骤4**：生成分析报告（CSV、JSON、Markdown）

**优势**：
- ✅ 完全自动化，无需手动运行回测
- ✅ 始终使用最新的回测数据
- ✅ 确保数据一致性

### 方式2：直接运行Python脚本（仅分析）

**仅进行牛熊市分析**（需要已有回测文件）

```bash
# 使用默认参数（2024-01-01 至 2025-12-31）
python3 ml_services/analyze_bull_bear_market_auto.py

# 自定义日期范围
python3 ml_services/analyze_bull_bear_market_auto.py --start-date 2024-01-01 --end-date 2025-12-31

# 指定输出格式（csv, json, md, all）
python3 ml_services/analyze_bull_bear_market_auto.py --output-format md

# 指定输出目录
python3 ml_services/analyze_bull_bear_market_auto.py --output-dir output/bull_bear_analysis

# 完整示例
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date 2024-01-01 \
    --end-date 2025-12-31 \
    --output-dir output/bull_bear_analysis \
    --output-format all
```

### 方式3：手动指定交易记录文件

```bash
# 自动查找最新的回测文件（默认）
python3 ml_services/analyze_bull_bear_market_auto.py

# 手动指定交易记录文件
python3 ml_services/analyze_bull_bear_market_auto.py \
    --trades-file output/backtest_20d_trades_20260307_002039.csv
```

### 方式4：仅运行回测（不进行牛熊市分析）

```bash
# 运行20天持有期回测
python3 ml_services/backtest_20d_horizon.py \
    --start-date 2024-01-01 \
    --end-date 2025-12-31 \
    --horizon 20 \
    --model-type catboost \
    --confidence-threshold 0.55 \
    --use-feature-selection \
    --skip-feature-selection
```

## 定时任务配置

### 方式1：Cron定时任务（Linux/Mac）

编辑crontab：
```bash
crontab -e
```

添加以下配置：

```cron
# 每月1号凌晨2点运行一次（分析上一季度数据）
0 2 1 * * cd /data/fortune && ./run_bull_bear_analysis.sh >> logs/bull_bear_analysis.log 2>&1

# 每周日凌晨3点运行一次（分析最近3个月数据）
0 3 * * 0 cd /data/fortune && python3 ml_services/analyze_bull_bear_market_auto.py --start-date $(date -d '3 months ago' +%Y-%m-%d) --end-date $(date +%Y-%m-%d) >> logs/bull_bear_analysis.log 2>&1
```

### 方式2：GitHub Actions自动化

创建 `.github/workflows/bull-bear-analysis.yml`：

```yaml
name: 牛熊市分析报告

on:
  schedule:
    # 每月1号凌晨2点运行（香港时间）
    - cron: '0 18 * 1 *'  # UTC时间18:00 = 香港时间02:00
  workflow_dispatch:  # 允许手动触发

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
      - name: 检出代码
        uses: actions/checkout@v3
      
      - name: 设置Python环境
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: 安装依赖
        run: |
          pip install pandas numpy yfinance
      
      - name: 运行牛熊市分析
        env:
          START_DATE: ${{ github.event.schedule && format('{0}-01-01', github.event.schedule.year) || '2024-01-01' }}
          END_DATE: ${{ format('{0}-12-31', github.event.schedule.year) || '2025-12-31' }}
        run: |
          cd /data/fortune
          python3 ml_services/analyze_bull_bear_market_auto.py \
            --start-date "$START_DATE" \
            --end-date "$END_DATE" \
            --output-format all
      
      - name: 上传分析报告
        uses: actions/upload-artifact@v3
        with:
          name: bull-bear-analysis-report
          path: output/bull_bear_analysis_*.md
          retention-days: 90
```

### 方式3：systemd定时任务（Linux）

创建 `/etc/systemd/system/bull-bear-analysis.service`：

```ini
[Unit]
Description=牛熊市分析服务
After=network.target

[Service]
Type=oneshot
User=marcowong
WorkingDirectory=/data/fortune
ExecStart=/data/fortune/run_bull_bear_analysis.sh
StandardOutput=append:/var/log/bull-bear-analysis.log
StandardError=append:/var/log/bull-bear-analysis-error.log
```

创建 `/etc/systemd/system/bull-bear-analysis.timer`：

```ini
[Unit]
Description=牛熊市分析定时任务

[Timer]
# 每月1号凌晨2点运行
OnCalendar=*-*-01 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

启用定时任务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable bull-bear-analysis.timer
sudo systemctl start bull-bear-analysis.timer
```

## 输出文件说明

### CSV文件
- **文件名**: `bull_bear_analysis_YYYYMMDD_HHMMSS.csv`
- **内容**: 每只股票在不同市场环境下的详细数据
- **用途**: 数据分析、Excel处理

### JSON文件
- **文件名**: `bull_bear_analysis_YYYYMMDD_HHMMSS.json`
- **内容**: 结构化数据，包含元数据和市场环境信息
- **用途**: API集成、程序化处理

### Markdown文件
- **文件名**: `bull_bear_analysis_YYYYMMDD_HHMMSS.md`
- **内容**: 人类可读的分析报告
- **用途**: 文档、报告、GitHub展示

## 查看最新报告

### 命令行查看
```bash
# 查看最新的Markdown报告
cat output/bull_bear_analysis_$(ls -t output/bull_bear_analysis_*.md | head -1)

# 使用less分页查看
less output/bull_bear_analysis_$(ls -t output/bull_bear_analysis_*.md | head -1)
```

### 浏览器查看（需要本地HTTP服务器）
```bash
# 启动简单的HTTP服务器
cd output
python3 -m http.server 8080

# 然后在浏览器中访问
# http://localhost:8080/bull_bear_analysis_latest.md
```

## 数据维护建议

### 定期清理旧文件
```bash
# 删除90天前的分析报告
find output -name "bull_bear_analysis_*" -mtime +90 -delete

# 或移动到归档目录
mkdir -p archive/bull_bear_analysis
find output -name "bull_bear_analysis_*" -mtime +90 -exec mv {} archive/bull_bear_analysis/ \;
```

### 保留最新报告
```bash
# 创建符号链接指向最新报告
cd output
ln -sf $(ls -t bull_bear_analysis_*.md | head -1) bull_bear_analysis_latest.md
```

## 常见问题

### Q1: 如何分析特定季度的数据？
```bash
# 2024年第一季度
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date 2024-01-01 \
    --end-date 2024-03-31

# 2024年第三季度
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date 2024-07-01 \
    --end-date 2024-09-30
```

### Q2: 如何分析最近一年的数据？
```bash
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date $(date -d '1 year ago' +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d)
```

### Q3: 如何只生成Markdown报告？
```bash
python3 ml_services/analyze_bull_bear_market_auto.py --output-format md
```

### Q4: 如何自定义输出目录？
```bash
python3 ml_services/analyze_bull_bear_market_auto.py \
    --output-dir output/bull_bear_analysis/2025/Q1
```

## 集成到现有工作流

### 方式1：集成到综合分析脚本
修改 `run_comprehensive_analysis.sh`：
```bash
#!/bin/bash

# ... 现有代码 ...

# 添加牛熊市分析
echo "📊 运行牛熊市分析..."
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date $(date -d '3 months ago' +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d) \
    --output-format md

# ... 继续执行其他分析 ...
```

### 方式2：集成到月度报告生成
创建新脚本 `run_monthly_report.sh`：
```bash
#!/bin/bash

# 生成月度报告
python3 ml_services/backtest_monthly_analysis.py

# 生成牛熊市分析
python3 ml_services/analyze_bull_bear_market_auto.py \
    --start-date $(date -d '1 year ago' +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d) \
    --output-format all
```