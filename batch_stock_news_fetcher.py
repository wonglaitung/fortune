#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取自选股新闻脚本
作者：AI 助手
日期：2025-10-25
更新：集成情感分析功能
"""

import yfinance as yf
from datetime import datetime, timedelta
import os
import csv
import time
import argparse
import schedule
import pandas as pd

# 导入hk_smart_money_tracker.py中的WATCHLIST
import sys
sys.path.append('/data/fortune')
from hk_smart_money_tracker import WATCHLIST

# 导入情感分析模块
from llm_services.sentiment_analyzer import batch_analyze_sentiment, get_sentiment_statistics



def get_stock_news(symbol, stock_name="", size=3):
    """
    通过yfinance获取个股新闻，只返回一个月内的新闻
    :param symbol: 股票代码 (例如: "0700.HK" for 腾讯控股)
    :param stock_name: 股票名称 (例如: "腾讯控股")
    :param size: 获取新闻条数
    :return: 新闻列表
    """
    try:
        # 使用yfinance获取个股新闻
        ticker = yf.Ticker(symbol)
        news_data = ticker.news
        
        if not news_data:
            return []
        
        articles = []
        
        # 计算一个月前的日期
        one_month_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None) - timedelta(days=30)
        
        # 处理新闻数据
        for item in news_data:
            # 从content字段获取新闻数据
            content = item.get("content", {})
            
            # 格式化发布时间
            pub_time_str = content.get("pubDate", "")
            pub_datetime = None
            pub_time = pub_time_str  # 默认使用原始时间字符串
            if pub_time_str:
                try:
                    # 将ISO格式时间字符串转换为datetime对象
                    pub_datetime = datetime.fromisoformat(pub_time_str.replace('Z', '+00:00'))
                    # 移除时区信息以避免比较错误
                    pub_datetime = pub_datetime.replace(tzinfo=None)
                    pub_time = pub_datetime.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    # 如果解析失败，保持使用原始时间字符串
                    pass
            
            # 只获取一个月内的新闻
            if pub_datetime and pub_datetime < one_month_ago:
                continue
            
            title = content.get("title", "").strip()
            summary = content.get("summary", "").strip()
            
            # 获取新闻链接
            url = ""
            canonical_url = content.get("canonicalUrl", {})
            click_through_url = content.get("clickThroughUrl", {})
            
            if isinstance(canonical_url, dict):
                url = canonical_url.get("url", "")
            elif isinstance(click_through_url, dict):
                url = click_through_url.get("url", "")
            
            articles.append({
                "title": title[:80] + ("..." if len(title) > 80 else ""),
                "summary": summary[:120] + ("..." if len(summary) > 120 else ""),
                "url": url,
                "publishedAt": pub_time
            })
        
        # 按时间由近到远排序，然后返回指定数量的新闻
        sorted_articles = sorted(articles, key=lambda x: x['publishedAt'], reverse=True)
        return sorted_articles[:size]
    except Exception as e:
        print(f"⚠️ 获取个股新闻失败: {e}")
        return []



def fetch_all_stock_news(analyze_sentiment=True):
    """
    获取watch list中所有股票的新闻
    
    Args:
        analyze_sentiment (bool): 是否执行情感分析（默认True）
    """
    print("=" * 60)
    print("📈 批量获取自选股新闻")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 获取当前查询时间
    query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_news_data = []
    
    # 删除超时限制，处理所有股票
    stock_count = 0
    
    for code, name in WATCHLIST.items():
        print(f"\n🔍 正在获取 {name} ({code}) 的新闻...")
        
        # 为yfinance使用完整的股票代码（包含后缀如.HK）
        symbol_code = code
        
        # 获取新闻，每只股票获取3条新闻
        articles = get_stock_news(symbol_code, name, size=3)
        
        if articles:
            print(f"  ✅ 获取到 {len(articles)} 条相关新闻")
            # 添加到总数据中
            for article in articles:
                all_news_data.append({
                    "stock_name": name,
                    "stock_code": code,
                    "publishedAt": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "query_time": query_time
                })
        else:
            print(f"  ⚠️ 未获取到 {name} 的新闻")
            
        stock_count += 1
        # 短暂休眠以避免请求过于频繁
        time.sleep(1)
    
    # 保存所有新闻数据到CSV文件
    if all_news_data:
        # 确保data目录存在
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # CSV文件路径
        csv_file = os.path.join(data_dir, "all_stock_news_records.csv")
        
        # 使用pandas保存数据（支持情感分析列）
        df = pd.DataFrame(all_news_data)
        df.columns = ["股票名称", "股票代码", "新闻时间", "新闻标题", "简要内容", "查询时间"]
        
        # 检查是否存在旧数据，如果存在则合并
        if os.path.exists(csv_file):
            try:
                old_df = pd.read_csv(csv_file)
                # 合并新旧数据
                merged_df = pd.concat([old_df, df], ignore_index=True)
                
                # 去重逻辑：保留有情感分数的记录
                # 如果新旧数据中有相同新闻，优先保留已有情感分数的记录
                def keep_best_record(group):
                    # 按情感分数是否为空排序，有情感分数的优先
                    group = group.sort_values(
                        by=['情感分数'],
                        na_position='last'  # 情感分数为空的排在最后
                    )
                    # 返回第一条（有情感分数的）
                    return group.iloc[[0]]
                
                # 按股票代码、新闻时间、新闻标题分组，每组保留最好的记录
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
        
        # 保存数据
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 所有新闻数据已保存到 {csv_file}")
        
        # 显示汇总信息
        print("\n📋 新闻汇总:")
        for news in all_news_data:
            print(f"  • {news['stock_name']} ({news['stock_code']}) | {news['title']}")
        
        # 执行情感分析
        if analyze_sentiment:
            print("\n🤖 开始执行情感分析...")
            try:
                # 只分析最近3天的新闻
                df = batch_analyze_sentiment(df, days_limit=3)
                
                # 显示统计信息
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
        
    else:
        print("\n❌ 未获取到任何新闻数据")
        
    print(f"\n📊 总共处理了 {stock_count} 只股票")
    print("=" * 60)

def run_scheduler():
    """设置定时任务"""
    # 设置香港时间上午9点和下午1点半运行
    schedule.every().day.at("09:00").do(fetch_all_stock_news)
    #schedule.every().day.at("13:30").do(fetch_all_stock_news)
    
    print("⏰ 定时任务已设置完成")
    print("📌 每天香港时间上午9:00和下午13:30将自动运行")
    print("📌 按 Ctrl+C 停止程序")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='批量获取自选股新闻')
    parser.add_argument('--schedule', '-s', action='store_true', 
                        help='启用定时任务模式（默认：单次运行）')
    parser.add_argument('--no-sentiment', action='store_true',
                        help='跳过情感分析（默认：执行情感分析）')
    
    # 解析参数
    args = parser.parse_args()
    
    if args.schedule:
        # 启用定时任务模式
        run_scheduler()
    else:
        # 单次运行模式
        analyze_sentiment = not args.no_sentiment
        fetch_all_stock_news(analyze_sentiment=analyze_sentiment)
