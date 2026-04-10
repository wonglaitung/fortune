---
name: anomaly-detector
description: "Detect anomalies in time series data using Z-Score and Isolation Forest methods. Supports multiple time intervals (minute, hour, day, week) and works with CSV/Excel data files."
license: MIT
---

# Anomaly Detector

时间序列数据异常检测工具，支持 Z-Score 和 Isolation Forest 两种检测方法。

## 检查操作系统

在使用本技能前，请确认您的操作系统类型：

### Windows
- 打开 CMD 或 PowerShell
- 查看提示符是否为 `C:\>` 或 `PS C:\>`
- **使用方法**：查看下方"Windows 用户"部分

### Linux/macOS
- 打开终端
- Linux 提示符通常包含 `$` 或用户名@主机名
- macOS 提示符通常是 `MacBook-Username:~ username$`
- **使用方法**：查看下方"Linux/macOS 用户"部分

### 快速检测命令

**Windows CMD：**
```cmd
ver
```

**Linux/macOS：**
```bash
uname -a
```

## 何时使用此技能

当您需要以下任一场景时使用此技能：

- **数据异常检测**：检测时间序列数据中的异常值
- **金融监控**：检测股票价格、成交量异常
- **系统监控**：检测服务器指标异常（CPU、内存、流量）
- **IoT 数据分析**：检测传感器数据异常
- **业务指标监控**：检测销售额、用户活跃度等异常波动

## 检测方法

**默认行为**：同时使用两种方法检测，提供更全面的异常分析。

### 1. Z-Score 检测（实时监控）

基于移动窗口的统计方法，适合实时监控场景。

**特点**：
- 快速响应，计算简单
- 基于历史均值和标准差
- 适合单维度指标检测

**参数**：
- `--window-size`：滚动窗口大小（自动设置或手动指定）
- `--threshold`：Z-Score 阈值（默认 3.0）

### 2. Isolation Forest 检测（深度分析）

基于机器学习的多维异常检测方法。

**特点**：
- 多维特征检测
- 无监督学习
- 适合复杂数据模式

**参数**：
- `--contamination`：异常比例（自动设置或手动指定）

### 3. 同时检测（默认）

同时运行 Z-Score 和 Isolation Forest 两种方法，提供更全面的异常检测覆盖。

**优势**：
- Z-Score 擅长检测突发性异常
- Isolation Forest 擅长检测复杂模式异常
- 两种方法互补，减少漏检

### 4. 自动参数设置

根据 `--time-interval` 自动设置最佳检测参数，无需手动调参：

| 时间间隔 | 窗口大小 | 阈值 | 异常比例 | 说明 |
|----------|----------|------|----------|------|
| `minute` | 60 | 3.0 | 0.02 | 分钟级：60分钟窗口 |
| `hour` | 24 | 3.0 | 0.03 | 小时级：24小时窗口 |
| `day` | 30 | 3.0 | 0.03 | 日级：30天窗口（默认） |
| `week` | 12 | 3.0 | 0.05 | 周级：12周窗口 |

**使用示例**：
```bash
# 自动设置参数（推荐）
python3 detect_anomaly.py data.csv --column price --time-interval hour

# 手动覆盖参数
python3 detect_anomaly.py data.csv --column price --time-interval hour --window-size 48
```

## 配置使用

技能脚本位于技能目录中，使用时请替换 `\full\path\to` 为实际的安装路径。

**脚本路径：**

- **Linux/macOS：** `/full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py`
- **Windows：** `\full\path\to\skills\anomaly-detector\detect_anomaly.bat`

**使用示例：**

```bash
# Linux/macOS
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column value

# Windows
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column value
```

**注意**：`\full\path\to` 为示例路径，请替换为实际的安装根路径。

## 安装依赖

技能需要以下 Python 包：

```bash
# Linux/macOS（使用 python3）
pip3 install -r /full/path/to/skills/anomaly-detector/requirements.txt

# Windows CMD（使用 python）
pip install -r \full\path\to\skills\anomaly-detector\requirements.txt

# 直接安装
pip install pandas numpy scikit-learn openpyxl
```

**依赖版本要求**：
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- openpyxl >= 3.0.0（Excel 支持）

## 核心能力

- **Z-Score 检测**：基于移动窗口的实时异常检测
- **Isolation Forest 检测**：多维特征深度分析
- **多时间间隔支持**：minute、hour、day、week
- **多数据格式支持**：CSV、Excel（.xlsx）
- **自动特征提取**：RSI、MACD、波动率等技术指标
- **严重程度分级**：high、medium、low

## 当前限制

- Isolation Forest 需要至少 100 个数据点
- Z-Score 窗口大小需要小于数据点数量
- 自动特征提取仅支持包含 OHLCV 列的数据

## 使用流程

### Windows 用户

使用批处理脚本直接运行，无需手动调用 Python 命令。

**批处理脚本位置：**
```
\full\path\to\skills\anomaly-detector\detect_anomaly.bat
```

**使用示例：**

```cmd
# 同时使用两种方法检测（默认）
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column price

# 指定时间戳列名（当自动检测失败时）
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column price --timestamp-column 日期

# 仅使用 Z-Score 检测
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column price --method zscore

# 仅使用 Isolation Forest 检测
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column price --method isolation-forest
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --method isolation-forest

# 指定窗口大小和阈值
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.csv --column price --window-size 60 --threshold 2.5

# 处理 Excel 文件
\full\path\to\skills\anomaly-detector\detect_anomaly.bat data.xlsx --column value --sheet Sheet1

# 查看完整帮助信息
\full\path\to\skills\anomaly-detector\detect_anomaly.bat --help
```

### Linux/macOS 用户

直接使用 Python 命令运行脚本。

```bash
# 自动检测（根据时间间隔自动设置最佳参数，推荐）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --time-interval hour

# 分钟级数据检测
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --time-interval minute

# 日级数据检测（默认）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price

# 指定时间戳列名（当自动检测失败时）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --timestamp-column 日期

# 多维特征提取（使用所有数值列）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column sales --multi-column --method isolation-forest

# 检测全部数据（不限制回溯天数）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --lookback-days 0

# 仅使用 Z-Score 检测
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --method zscore

# 手动指定参数（覆盖自动设置）
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.csv --column price --window-size 60 --threshold 2.5

# 处理 Excel 文件
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py data.xlsx --column value --sheet Sheet1

# 查看完整帮助信息
python3 /full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py --help
```

### 命令行参数

| 参数 | 简写 | 必需 | 说明 |
|------|------|------|------|
| `input_file` | - | ✅ | 输入数据文件路径（CSV 或 Excel） |
| `--column` | `-c` | ✅ | 要检测的列名 |
| `--method` | `-m` | ❌ | 检测方法：`both`（默认）、`zscore`、`isolation-forest` |
| `--window-size` | `-w` | ❌ | Z-Score 窗口大小（默认根据时间间隔自动设置） |
| `--threshold` | `-t` | ❌ | Z-Score 阈值（默认 3.0） |
| `--contamination` | - | ❌ | Isolation Forest 异常比例（默认根据时间间隔自动设置） |
| `--time-interval` | - | ❌ | 时间间隔：minute、hour、day（默认）、week |
| `--timestamp-column` | - | ❌ | 时间戳列名（可选，不指定则自动检测） |
| `--multi-column` | - | ❌ | 使用所有数值列进行多维特征提取（仅 Isolation Forest 有效） |
| `--lookback-days` | - | ❌ | Isolation Forest 回溯天数（默认 30，设为 0 检测全部数据） |
| `--sheet` | `-s` | ❌ | Excel 工作表名称（默认第一个工作表） |
| `--output` | `-o` | ❌ | 输出文件路径（默认打印到控制台） |
| `--help` | `-h` | ❌ | 显示帮助信息 |

## 数据格式要求

### CSV 文件格式

```csv
timestamp,price,volume
2024-01-01,100.0,1000
2024-01-02,102.5,1200
2024-01-03,98.0,950
...
```

### Excel 文件格式

- 支持 .xlsx 格式
- 第一行应为列名
- 必须包含时间戳列和数值列

### 时间戳要求

- 支持格式：`YYYY-MM-DD`、`YYYY-MM-DD HH:MM:SS`、ISO 8601
- 自动检测列名：`timestamp`、`date`、`datetime`、`time`（不区分大小写）
- 如果列名不在自动检测列表中，请使用 `--timestamp-column` 参数手动指定

## 输出格式

**默认输出**：同时显示 Z-Score 和 Isolation Forest 两种方法的检测结果。

### Z-Score 检测输出

```
=== 异常检测结果 ===
检测方法: Z-Score
检测指标: price
窗口大小: 30
阈值: 3.0

发现 3 个异常:

[1] 2024-01-15 10:00:00
    类型: price
    严重程度: high
    Z-Score: 4.52
    当前值: 150.00
    均值: 100.25
    标准差: 11.02

[2] 2024-01-20 14:30:00
    类型: price
    严重程度: medium
    Z-Score: 3.21
    当前值: 135.50
    均值: 101.02
    标准差: 10.76
...
```

### Isolation Forest 检测输出

```
=== 异常检测结果 ===
检测方法: Isolation Forest
异常比例: 0.03

发现 5 个异常:

[1] 2024-01-15 10:00:00
    类型: isolation_forest
    严重程度: high
    异常分数: -0.15
    特征数: 10

[2] 2024-01-20 14:30:00
    类型: isolation_forest
    严重程度: medium
    异常分数: -0.08
    特征数: 10
...
```

## 最佳实践

### 1. 推荐使用自动参数设置（默认）

只需指定 `--time-interval`，参数自动优化：

```bash
# 小时级数据 - 自动设置 24 小时窗口
python3 detect_anomaly.py data.csv --column price --time-interval hour

# 分钟级数据 - 自动设置 60 分钟窗口
python3 detect_anomaly.py data.csv --column price --time-interval minute

# 日级数据（默认）- 自动设置 30 天窗口
python3 detect_anomaly.py data.csv --column price
```

### 2. 使用默认的双方法检测

两种方法互补：
- Z-Score 检测突发性异常（如价格暴跌）
- Isolation Forest 检测复杂模式异常

### 3. 手动调整参数（覆盖自动设置）

```bash
# 更敏感的检测（降低阈值）
python3 detect_anomaly.py data.csv --column price --threshold 2.0

# 更稳定的检测（提高阈值）
python3 detect_anomaly.py data.csv --column price --threshold 4.0

# 自定义窗口大小
python3 detect_anomaly.py data.csv --column price --time-interval hour --window-size 48
```

### 4. 单独使用某种方法

```bash
# 仅 Z-Score（适合实时监控、小数据集）
python3 detect_anomaly.py data.csv --column price --method zscore

# 仅 Isolation Forest（适合大数据集、复杂模式）
python3 detect_anomaly.py data.csv --column price --method isolation-forest
```

### 5. 导出结果

```bash
# 导出到 JSON 文件
python3 detect_anomaly.py data.csv --column price --output anomalies.json

# 导出到 CSV 文件
python3 detect_anomaly.py data.csv --column price --output anomalies.csv
```

## 常见问题

**Q: 提示"数据点不足"？**

A: Z-Score 需要至少 `window_size` 个数据点，Isolation Forest 建议至少 100 个数据点。

**Q: 如何处理缺失值？**

A: 脚本会自动跳过缺失值，但大量缺失可能影响检测效果。建议先清理数据。

**Q: Z-Score 和 Isolation Forest 结果不一致？**

A: 两种方法原理不同。Z-Score 基于统计分布，Isolation Forest 基于数据隔离。建议结合使用。

**Q: 如何选择合适的 contamination 参数？**

A: 根据历史经验设置。如果不确定，可从 0.01-0.05 开始尝试。

## 技术依赖

技能依赖的 Python 包已列在 `requirements.txt` 中：

- **pandas** >= 1.3.0: 数据处理
- **numpy** >= 1.20.0: 数值计算
- **scikit-learn** >= 1.0.0: Isolation Forest 模型
- **openpyxl** >= 3.0.0: Excel 文件支持

**依赖文件位置**：
```
/full/path/to/skills/anomaly-detector/requirements.txt
```

## 脚本位置

**Python 脚本：**
```
/full/path/to/skills/anomaly-detector/scripts/detect_anomaly.py
```

**Windows 批处理脚本：**
```
\full\path\to\skills\anomaly-detector\detect_anomaly.bat
```

## 相关资源

- [Z-Score 异常检测原理](https://en.wikipedia.org/wiki/Standard_score)
- [Isolation Forest 论文](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [iFlow CLI 技能系统文档](https://platform.iflow.cn/cli/examples/skill)
