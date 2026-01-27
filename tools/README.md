# 量化项目搜索工具

自动搜索和分析 GitHub 上热门的量化交易项目。

## 功能特性

- 🔍 自动搜索 GitHub 上的热门量化项目
- 📊 分析项目特点、技术栈和特性
- 📝 生成详细的分析报告（Markdown 格式）
- 💡 提供借鉴建议和集成方案
- 🔄 支持自定义搜索参数

## 安装依赖

```bash
pip install requests pandas
```

## 快速开始

### 基础使用

```python
from tools.quant_project_searcher import QuantProjectSearcher

# 创建搜索器
searcher = QuantProjectSearcher()

# 执行搜索
result = searcher.run_search(
    query="quantitative trading",
    min_stars=1000,
    limit=10,
    save=True
)
```

### 命令行使用

```bash
# 运行搜索
python tools/quant_project_searcher.py

# 运行示例
python tools/example_usage.py
```

## API 文档

### QuantProjectSearcher 类

#### `search_projects(query, language, min_stars, limit)`

搜索 GitHub 上的量化项目。

**参数:**
- `query` (str): 搜索关键词
- `language` (str): 编程语言
- `min_stars` (int): 最小 stars 数量
- `limit` (int): 返回结果数量限制

**返回:**
- `List[Dict]`: 项目列表

#### `analyze_projects(projects)`

分析项目特点。

**参数:**
- `projects` (List[Dict]): 项目列表

**返回:**
- `Dict`: 分析结果

#### `generate_report(projects, analysis)`

生成 Markdown 格式的分析报告。

**参数:**
- `projects` (List[Dict]): 项目列表
- `analysis` (Dict): 分析结果

**返回:**
- `str`: Markdown 格式的报告

#### `save_report(report, filename)`

保存报告到文件。

**参数:**
- `report` (str): 报告内容
- `filename` (str): 文件名

**返回:**
- `str`: 文件路径

#### `run_search(query, min_stars, limit, save, filename)`

执行完整的搜索和分析流程。

**参数:**
- `query` (str): 搜索关键词
- `min_stars` (int): 最小 stars 数量
- `limit` (int): 返回结果数量限制
- `save` (bool): 是否保存报告
- `filename` (str): 报告文件名

**返回:**
- `Dict`: 包含项目列表、分析结果和报告的字典

## 使用示例

### 示例 1: 基础搜索

```python
from tools.quant_project_searcher import QuantProjectSearcher

searcher = QuantProjectSearcher()
result = searcher.run_search(
    query="quantitative trading",
    min_stars=1000,
    limit=5,
    save=True
)

print(f"找到 {len(result['projects'])} 个项目")
```

### 示例 2: 自定义搜索

```python
searcher = QuantProjectSearcher()

# 搜索 Python 回测框架
projects = searcher.search_projects(
    query="backtesting",
    language="Python",
    min_stars=5000,
    limit=3
)

# 分析项目
analysis = searcher.analyze_projects(projects)

# 生成报告
report = searcher.generate_report(projects, analysis)

# 保存报告
searcher.save_report(report, "custom_report.md")
```

### 示例 3: 编程式使用

```python
searcher = QuantProjectSearcher()

# 获取项目
projects = searcher.search_projects(
    query="machine learning trading",
    min_stars=1000,
    limit=10
)

# 提取项目信息
for project in projects:
    print(f"{project['full_name']}: {project['stargazers_count']} stars")
    print(f"  语言: {project['language']}")
    print(f"  描述: {project['description']}")
```

## 输出报告

生成的报告包含以下内容：

1. **搜索结果概览**
   - 总项目数
   - 总 Stars
   - 编程语言分布

2. **热门项目列表**
   - 项目名称和链接
   - Stars 数量
   - 编程语言
   - 项目描述
   - 关键词标签

3. **特性分析**
   - 常见特性
   - Top 项目排名

4. **借鉴建议**
   - 值得学习的设计
   - 推荐集成的功能
   - 应用到当前项目的建议

5. **参考资源**
   - 相关项目链接
   - 文档链接

## 文件结构

```
tools/
├── __init__.py                  # 包初始化文件
├── quant_project_searcher.py    # 主模块
├── example_usage.py             # 使用示例
└── README.md                    # 说明文档

output/
├── quant_projects_search_report.md  # 搜索报告
└── backtesting_projects.md          # 回测项目报告
```

## 预定义的热门项目

工具内置了以下热门量化项目：

1. microsoft/qlib - AI 量化投资平台
2. wilsonfreitas/awesome-quant - 量化金融资源大全
3. mementum/backtrader - Python 回测框架
4. stefan-jansen/machine-learning-for-trading - 机器学习交易教程
5. QuantConnect/Lean - 算法交易引擎
6. AI4Finance-Foundation/FinRL - 强化学习交易库
7. polakowo/vectorbt - 向量化回测框架
8. kernc/backtesting.py - 简洁回测框架
9. edtechre/pybroker - 机器学习策略框架
10. TA-Lib/ta-lib-python - 技术分析库

## 注意事项

1. **GitHub API 限制**: 工具使用预定义的项目列表，避免频繁调用 GitHub API
2. **网络依赖**: 需要稳定的网络连接
3. **报告保存**: 默认保存到 `output/` 目录
4. **编码格式**: 报告使用 UTF-8 编码

## 扩展功能

### 添加新的项目

在 `quant_project_searcher.py` 的 `self.known_projects` 列表中添加：

```python
self.known_projects = [
    # ... 现有项目
    {
        "owner": "your-username",
        "repo": "your-repo",
        "keywords": ["keyword1", "keyword2"]
    }
]
```

### 自定义报告格式

修改 `generate_report` 方法来自定义报告格式。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License