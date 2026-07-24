#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股新闻获取脚本 - 通过东方财富API获取个股资讯

数据源：东方财富个股资讯接口
- URL: https://np-listapi.eastmoney.com/comm/wap/getListInfo
- 深市secid格式: 0.{6位代码}
- 沪市secid格式: 1.{6位代码}

功能：
1. 获取A股自选股新闻
2. 集成情感分析（复用 llm_services/sentiment_analyzer.py）
3. 保存到 data/a_stock_news_records.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse

from a_stock_config import A_STOCK_TRAINING_LIST, get_market_code
from llm_services.sentiment_analyzer import batch_analyze_sentiment, get_sentiment_statistics


def get_a_stock_news(stock_code, stock_name="", size=5):
    """
    通过东方财富API获取A股个股新闻

    Args:
        stock_code (str): 股票代码，如 "300440"
        stock_name (str): 股票名称，如 "运达科技"
        size (int): 获取新闻条数，默认5

    Returns:
        list: 新闻列表 [{title, summary, url, publishedAt}]
    """
    try:
        # 构建secid：深市0.代码，沪市1.代码
        market = get_market_code(stock_code)
        market_prefix = '1' if market == 'sh' else '0'
        secid = f"{market_prefix}.{stock_code}"

        url = (
            f"https://np-listapi.eastmoney.com/comm/wap/getListInfo"
            f"?client=wap&type=1&mTypeAndCode={secid}"
            f"&pageSize={size}&pageNo=1"
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://guba.eastmoney.com/',
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        if data.get('code') != 1 or 'data' not in data:
            return []

        news_list = data['data'].get('list', [])
        if not news_list:
            return []

        # 计算一个月前的日期
        one_month_ago = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=30)

        articles = []
        for item in news_list:
            # 解析发布时间
            pub_time_str = item.get('Art_ShowTime', '')
            pub_datetime = None
            if pub_time_str:
                try:
                    pub_datetime = datetime.strptime(pub_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass

            # 只保留一个月内的新闻
            if pub_datetime and pub_datetime < one_month_ago:
                continue

            title = item.get('Art_Title', '').strip()
            if not title:
                continue

            articles.append({
                'title': title,
                'summary': '',  # 东方财富API不返回摘要
                'url': item.get('Art_Url', ''),
                'publishedAt': pub_time_str,
                'media': item.get('Art_MediaName', ''),
            })

        # 按时间由近到远排序
        sorted_articles = sorted(
            articles, key=lambda x: x['publishedAt'], reverse=True
        )
        return sorted_articles[:size]

    except requests.exceptions.Timeout:
        print(f"  ⚠️ 获取 {stock_name}({stock_code}) 新闻超时")
        return []
    except Exception as e:
        print(f"  ⚠️ 获取 {stock_name}({stock_code}) 新闻失败: {e}")
        return []


def fetch_all_a_stock_news(analyze_sentiment=True):
    """
    批量获取A股自选股新闻

    Args:
        analyze_sentiment (bool): 是否执行情感分析（默认True）

    Returns:
        DataFrame: 新闻数据
    """
    print("=" * 60)
    print("📰 批量获取A股自选股新闻")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_news_data = []

    for code, name in A_STOCK_TRAINING_LIST.items():
        print(f"\n🔍 正在获取 {name} ({code}) 的新闻...")

        articles = get_a_stock_news(code, name, size=5)

        if articles:
            print(f"  ✅ 获取到 {len(articles)} 条相关新闻")
            for article in articles:
                media_info = f" [{article['media']}]" if article.get('media') else ''
                all_news_data.append({
                    "股票名称": name,
                    "股票代码": code,
                    "新闻时间": article.get('publishedAt', ''),
                    "新闻标题": article.get('title', ''),
                    "简要内容": article.get('summary', ''),
                    "查询时间": query_time,
                })
                print(f"    • {article['title']}{media_info}")
        else:
            print(f"  ⚠️ 未获取到 {name} 的新闻")

        # 短暂休眠避免请求过于频繁
        time.sleep(0.5)

    if not all_news_data:
        print("\n❌ 未获取到任何新闻数据")
        return None

    # 保存新闻数据
    df = pd.DataFrame(all_news_data)

    # 确保股票代码保持字符串格式（防止前导零丢失）
    df['股票代码'] = df['股票代码'].astype(str).apply(lambda x: x.zfill(6))

    # 添加情感分析相关列（初始为空）
    df['情感分数'] = None
    df['相关性'] = None
    df['影响度'] = None
    df['预期差'] = None
    df['情感方向'] = None
    df['情感分析时间'] = None

    # 确保data目录存在
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    csv_file = os.path.join(data_dir, "a_stock_news_records.csv")

    # 合并旧数据（去重）
    if os.path.exists(csv_file):
        try:
            old_df = pd.read_csv(csv_file)
            merged_df = pd.concat([old_df, df], ignore_index=True)

            # 按股票代码、新闻时间、新闻标题去重，优先保留有情感分数的记录
            def keep_best_record(group):
                group = group.sort_values(
                    by=['情感分数'], na_position='last'
                )
                return group.iloc[[0]]

            merged_df = merged_df.groupby(
                ['股票代码', '新闻时间', '新闻标题'],
                as_index=False
            ).apply(keep_best_record).reset_index(drop=True)

            # 按时间排序
            merged_df['新闻时间'] = pd.to_datetime(merged_df['新闻时间'])
            merged_df = merged_df.sort_values('新闻时间', ascending=False)
            df = merged_df
        except Exception as e:
            print(f"⚠️ 合并旧数据失败: {e}，使用新数据")

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 新闻数据已保存到 {csv_file}")

    # 执行情感分析
    if analyze_sentiment:
        print("\n🤖 开始执行情感分析...")
        try:
            df = batch_analyze_sentiment(
                df, days_limit=3,
                save_path='data/a_stock_news_records.csv'
            )

            stats = get_sentiment_statistics(df)
            print(f"\n📊 情感分析统计:")
            print(f"  总新闻数: {stats['total']}")
            print(f"  已分析: {stats['analyzed']}")
            print(f"  未分析: {stats['unanalyzed']}")
            if stats['analyzed'] > 0:
                print(f"  平均情感分数: {stats['sentiment_score_mean']:.2f}")
                print(f"  正面新闻: {stats['positive_count']}")
                print(f"  负面新闻: {stats['negative_count']}")
                print(f"  中性新闻: {stats['neutral_count']}")

            print("\n✅ 情感分析完成")
        except Exception as e:
            print(f"⚠️ 情感分析失败: {e}")
            print("💡 提示：请检查 QWEN_API_KEY 环境变量是否设置")

    print(f"\n📊 总共处理了 {len(A_STOCK_TRAINING_LIST)} 只股票")
    print("=" * 60)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量获取A股自选股新闻')
    parser.add_argument(
        '--no-sentiment', action='store_true',
        help='跳过情感分析（默认：执行情感分析）'
    )

    args = parser.parse_args()
    analyze_sentiment = not args.no_sentiment
    fetch_all_a_stock_news(analyze_sentiment=analyze_sentiment)
