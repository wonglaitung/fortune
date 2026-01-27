#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化项目搜索工具使用示例
"""

from quant_project_searcher import QuantProjectSearcher


def example_basic_search():
    """基础搜索示例"""
    print("=" * 70)
    print("示例 1: 基础搜索")
    print("=" * 70)
    
    searcher = QuantProjectSearcher()
    result = searcher.run_search(
        query="quantitative trading",
        min_stars=1000,
        limit=5,
        save=True
    )
    
    print(f"\n✅ 找到 {len(result['projects'])} 个项目")
    print(f"📊 总 Stars: {result['analysis']['total_stars']:,}")


def example_custom_search():
    """自定义搜索示例"""
    print("\n" + "=" * 70)
    print("示例 2: 自定义搜索")
    print("=" * 70)
    
    searcher = QuantProjectSearcher()
    
    # 只搜索 Python 项目，stars > 5000
    result = searcher.run_search(
        query="backtesting",
        min_stars=5000,
        limit=3,
        save=True,
        filename="backtesting_projects.md"
    )
    
    # 打印 top 3 项目
    print("\n🏆 Top 3 回测项目:")
    for i, project in enumerate(result['projects'][:3], 1):
        print(f"  {i}. {project['full_name']}: {project['stargazers_count']:,} stars")


def example_analysis_only():
    """仅分析示例"""
    print("\n" + "=" * 70)
    print("示例 3: 仅分析不保存")
    print("=" * 70)
    
    searcher = QuantProjectSearcher()
    
    # 搜索项目
    projects = searcher.search_projects(
        query="machine learning trading",
        min_stars=1000,
        limit=5
    )
    
    # 分析项目
    analysis = searcher.analyze_projects(projects)
    
    # 生成报告但不保存
    report = searcher.generate_report(projects, analysis)
    
    print(f"\n📊 分析结果:")
    print(f"  - 项目数: {analysis['total_projects']}")
    print(f"  - 总 Stars: {analysis['total_stars']:,}")
    print(f"  - 编程语言: {', '.join(analysis['languages'].keys())}")
    print(f"  - 常见特性: {', '.join(analysis['common_features'])}")


def example_programmatic_use():
    """编程式使用示例"""
    print("\n" + "=" * 70)
    print("示例 4: 编程式使用")
    print("=" * 70)
    
    searcher = QuantProjectSearcher()
    
    # 获取项目列表
    projects = searcher.search_projects(
        query="reinforcement learning trading",
        min_stars=1000,
        limit=5
    )
    
    # 提取特定信息
    project_info = []
    for project in projects:
        project_info.append({
            'name': project['full_name'],
            'stars': project['stargazers_count'],
            'language': project['language'],
            'keywords': project.get('keywords', [])
        })
    
    # 转换为 DataFrame
    import pandas as pd
    df = pd.DataFrame(project_info)
    
    print("\n📊 项目信息 DataFrame:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # 运行所有示例
    example_basic_search()
    example_custom_search()
    example_analysis_only()
    example_programmatic_use()
    
    print("\n" + "=" * 70)
    print("✅ 所有示例运行完成！")
    print("=" * 70)
    print("\n💡 提示:")
    print("  - 查看 output/ 目录获取生成的报告")
    print("  - 参考 example_usage.py 了解更多使用方法")
    print("  - 直接导入 QuantProjectSearcher 类使用")