#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 量化项目搜索和分析工具
自动搜索、分析和报告热门的量化交易项目
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class QuantProjectSearcher:
    """量化项目搜索和分析类"""

    def __init__(self):
        self.github_api_base = "https://api.github.com"
        self.search_results = []
        
        # 预定义的热门量化项目
        self.known_projects = [
            {
                "owner": "stefan-jansen",
                "repo": "machine-learning-for-trading",
                "keywords": ["machine learning", "trading", "algorithms"]
            },
            {
                "owner": "QuantConnect",
                "repo": "Lean",
                "keywords": ["backtesting", "algorithmic trading", "C#", "Python"]
            },
            {
                "owner": "mementum",
                "repo": "backtrader",
                "keywords": ["backtesting", "trading", "Python"]
            },
            {
                "owner": "wilsonfreitas",
                "repo": "awesome-quant",
                "keywords": ["quantitative", "finance", "resources"]
            },
            {
                "owner": "AI4Finance-Foundation",
                "repo": "FinRL",
                "keywords": ["reinforcement learning", "trading", "deep learning"]
            },
            {
                "owner": "polakowo",
                "repo": "vectorbt",
                "keywords": ["vectorized", "backtesting", "NumPy"]
            },
            {
                "owner": "microsoft",
                "repo": "qlib",
                "keywords": ["AI", "quantitative", "investment"]
            },
            {
                "owner": "kernc",
                "repo": "backtesting.py",
                "keywords": ["backtesting", "Python", "simple"]
            },
            {
                "owner": "edtechre",
                "repo": "pybroker",
                "keywords": ["machine learning", "strategy", "algorithmic trading"]
            },
            {
                "owner": "TA-Lib",
                "repo": "ta-lib-python",
                "keywords": ["technical analysis", "indicators", "TA-Lib"]
            }
        ]

    def search_projects(
        self,
        query: str = "quantitative trading",
        language: str = "Python",
        min_stars: int = 1000,
        limit: int = 10
    ) -> List[Dict]:
        """
        搜索 GitHub 上的量化项目
        
        Args:
            query: 搜索关键词
            language: 编程语言
            min_stars: 最小 stars 数量
            limit: 返回结果数量限制
            
        Returns:
            项目列表
        """
        # 使用预定义的项目列表（更可靠）
        projects = []
        
        for project_info in self.known_projects:
            try:
                project = self._get_project_info(
                    project_info["owner"],
                    project_info["repo"]
                )
                if project and project.get("stargazers_count", 0) >= min_stars:
                    project["keywords"] = project_info["keywords"]
                    projects.append(project)
            except Exception as e:
                print(f"⚠️ 获取项目 {project_info['owner']}/{project_info['repo']} 失败: {e}")
                continue
        
        # 按 stars 排序
        projects.sort(key=lambda x: x.get("stargazers_count", 0), reverse=True)
        
        self.search_results = projects[:limit]
        return self.search_results

    def _get_project_info(self, owner: str, repo: str) -> Optional[Dict]:
        """
        获取 GitHub 项目信息
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            
        Returns:
            项目信息字典
        """
        url = f"{self.github_api_base}/repos/{owner}/{repo}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ 获取 {owner}/{repo} 信息失败: {e}")
            return None

    def analyze_projects(self, projects: List[Dict]) -> Dict:
        """
        分析项目特点
        
        Args:
            projects: 项目列表
            
        Returns:
            分析结果字典
        """
        analysis = {
            "total_projects": len(projects),
            "total_stars": sum(p.get("stargazers_count", 0) for p in projects),
            "languages": {},
            "top_projects": [],
            "common_features": []
        }
        
        # 统计编程语言
        for project in projects:
            lang = project.get("language", "Unknown")
            if lang:
                analysis["languages"][lang] = analysis["languages"].get(lang, 0) + 1
        
        # 提取 top 项目
        for project in projects[:5]:
            analysis["top_projects"].append({
                "name": project.get("name"),
                "full_name": project.get("full_name"),
                "stars": project.get("stargazers_count"),
                "language": project.get("language"),
                "description": project.get("description", ""),
                "keywords": project.get("keywords", [])
            })
        
        # 提取常见特性
        all_keywords = []
        for project in projects:
            all_keywords.extend(project.get("keywords", []))
        
        # 统计关键词频率
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # 取出现频率最高的特性
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        analysis["common_features"] = [kw for kw, freq in sorted_keywords if freq >= 2]
        
        return analysis

    def generate_report(self, projects: List[Dict], analysis: Dict) -> str:
        """
        生成分析报告
        
        Args:
            projects: 项目列表
            analysis: 分析结果
            
        Returns:
            Markdown 格式的报告
        """
        report = f"""# GitHub 量化项目搜索报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📊 搜索结果概览

- **总项目数**: {analysis['total_projects']}
- **总 Stars**: {analysis['total_stars']:,}
- **编程语言分布**: {', '.join(f'{lang}({count})' for lang, count in analysis['languages'].items())}

---

## 🔥 热门项目列表

"""
        
        # 添加项目详情
        for i, project in enumerate(projects, 1):
            report += f"""
### {i}. {project.get('full_name', 'Unknown')}

- 🌟 **Stars**: {project.get('stargazers_count', 0):,}
- 💻 **语言**: {project.get('language', 'Unknown')}
- 📝 **描述**: {project.get('description', '无描述')}
- 🔗 **链接**: {project.get('html_url', '#')}
- 🏷️ **关键词**: {', '.join(project.get('keywords', []))}
- 📅 **创建时间**: {project.get('created_at', 'Unknown')}
- 🔄 **最后更新**: {project.get('updated_at', 'Unknown')}

"""
        
        # 添加分析部分
        report += f"""
---

## 📈 特性分析

### 常见特性
{chr(10).join(f'- {feature}' for feature in analysis['common_features'])}

### Top 5 项目
"""
        for i, project in enumerate(analysis['top_projects'], 1):
            report += f"""
{i}. **{project['full_name']}**
   - Stars: {project['stars']:,}
   - 语言: {project['language']}
   - 关键词: {', '.join(project['keywords'])}
"""

        # 添加借鉴建议
        report += """

---

## 💡 借鉴建议

### 值得学习的设计

1. **事件驱动架构** (backtrader)
   - 清晰的数据-策略-分析分离
   - 易于扩展的插件系统

2. **向量化编程** (vectorbt)
   - 使用 Numba 加速关键计算
   - 性能比传统回测快 10-100 倍

3. **完整流程** (machine-learning-for-trading)
   - 数据预处理 → 特征工程 → 模型训练 → 回测 → 部署
   - 风险管理嵌入每个环节

### 推荐集成的功能

1. **TA-Lib** - 150+ 技术指标
   ```bash
   pip install TA-Lib
   ```

2. **pandas-ta** - 130+ 技术指标（纯 Python）
   ```bash
   pip install pandas-ta
   ```

3. **Numba** - JIT 编译器，加速计算
   ```bash
   pip install numba
   ```

4. **Plotly** - 交互式可视化
   ```bash
   pip install plotly
   ```

### 应用到当前项目的建议

#### 阶段 1: 技术指标扩展
- 集成 TA-Lib，指标数量从 10+ 扩展到 150+
- 添加 Numba 加速关键计算

#### 阶段 2: 性能优化
- 使用向量化操作替代循环
- 批量处理多只股票

#### 阶段 3: 风险管理增强
- 参考回测框架的风险管理模块
- 添加仓位控制和回撤限制

---

## 📚 参考资源

- [awesome-quant](https://github.com/wilsonfreitas/awesome-quant) - 量化金融资源大全
- [machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) - 机器学习交易教程
- [backtrader 文档](https://www.backtrader.com/docu/) - 回测框架文档

---

*报告由 QuantProjectSearcher 自动生成*
"""
        
        return report

    def save_report(self, report: str, filename: str = "quant_projects_report.md"):
        """
        保存报告到文件
        
        Args:
            report: 报告内容
            filename: 文件名
        """
        filepath = f"/data/fortune/output/{filename}"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ 报告已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return None

    def run_search(
        self,
        query: str = "quantitative trading",
        min_stars: int = 1000,
        limit: int = 10,
        save: bool = True,
        filename: str = "quant_projects_report.md"
    ) -> Dict:
        """
        执行完整的搜索和分析流程
        
        Args:
            query: 搜索关键词
            min_stars: 最小 stars 数量
            limit: 返回结果数量限制
            save: 是否保存报告
            filename: 报告文件名
            
        Returns:
            包含项目列表、分析结果和报告的字典
        """
        print("=" * 70)
        print("GitHub 量化项目搜索和分析")
        print("=" * 70)
        print(f"搜索关键词: {query}")
        print(f"最小 Stars: {min_stars}")
        print(f"结果数量: {limit}")
        print()
        
        # 搜索项目
        print("🔍 正在搜索项目...")
        projects = self.search_projects(query, min_stars=min_stars, limit=limit)
        print(f"✅ 找到 {len(projects)} 个项目\n")
        
        # 分析项目
        print("📊 正在分析项目...")
        analysis = self.analyze_projects(projects)
        print(f"✅ 分析完成\n")
        
        # 生成报告
        print("📝 正在生成报告...")
        report = self.generate_report(projects, analysis)
        print("✅ 报告生成完成\n")
        
        # 保存报告
        if save:
            saved_path = self.save_report(report, filename)
            if saved_path:
                print(f"📁 报告路径: {saved_path}\n")
        
        print("=" * 70)
        print("✅ 搜索和分析完成！")
        print("=" * 70)
        
        return {
            "projects": projects,
            "analysis": analysis,
            "report": report
        }


def main():
    """主函数"""
    searcher = QuantProjectSearcher()
    
    # 执行搜索
    result = searcher.run_search(
        query="quantitative trading",
        min_stars=1000,
        limit=10,
        save=True,
        filename="quant_projects_search_report.md"
    )
    
    # 打印简要信息
    print("\n📊 搜索结果摘要:")
    for project in result["projects"][:5]:
        print(f"  - {project['full_name']}: {project['stargazers_count']:,} stars")


if __name__ == "__main__":
    main()