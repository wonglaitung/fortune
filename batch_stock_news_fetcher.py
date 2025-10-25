#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取自选股新闻脚本
作者：AI 助手
日期：2025-10-25
"""

import akshare as ak
import yfinance as yf
from datetime import datetime, time as dt_time
import json
import os
import csv
import time
import argparse
import schedule
from llm_services.qwen_engine import chat_with_llm

# 导入hk_smart_money_tracker.py中的WATCHLIST
import sys
sys.path.append('/data/fortune')
from hk_smart_money_tracker import WATCHLIST

def filter_news_with_llm(news_list, stock_name, stock_code, max_news=3):
    """
    使用大模型过滤新闻，评估新闻与股票的相关性，并按时间排序
    """
    if not news_list:
        return []
    
    # 准备新闻数据用于发送给大模型
    news_texts = []
    for i, news in enumerate(news_list):
        news_text = f"{i+1}. 标题: {news['title']}\n   内容: {news['summary']}\n   发布时间: {news['publishedAt']}\n"
        news_texts.append(news_text)
    
    news_data = "\n".join(news_texts)
    
    # 构建大模型查询
    prompt = f"""
请分析以下新闻与股票 \"{stock_name}\"（代码：{stock_code}）的相关性，并综合考虑相关性和时效性（发布时间越近越重要），按综合评分从高到低排序，返回前{max_news}条新闻的序号列表。

股票信息：
- 股票名称：{stock_name}
- 股票代码：{stock_code}

新闻列表：
{news_data}

请按照以下JSON格式返回结果：
{{
    \"relevant_news_indices\": [1, 3, 5, 2, 4]
}}

只返回JSON格式结果，不要添加其他解释。
"""
    
    try:
        # 调用大模型
        response = chat_with_llm(prompt)
        
        # 解析大模型返回的JSON
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            relevant_indices = result.get("relevant_news_indices", [])
            
            # 根据大模型返回的序号获取相关新闻
            filtered_news = []
            for idx in relevant_indices[:max_news]:
                if 1 <= idx <= len(news_list):
                    filtered_news.append(news_list[idx-1])
            
            # 按时间由近到远排序
            filtered_news.sort(key=lambda x: x['publishedAt'], reverse=True)
            
            return filtered_news
        else:
            # 如果没有找到有效的JSON，返回原始数据并按时间排序
            sorted_news = sorted(news_list[:max_news], key=lambda x: x['publishedAt'], reverse=True)
            return sorted_news
    except Exception as e:
        print(f"⚠️ 大模型过滤失败: {e}")
        # 如果大模型调用失败，返回原始数据并按时间排序
        sorted_news = sorted(news_list[:max_news], key=lambda x: x['publishedAt'], reverse=True)
        return sorted_news

def get_stock_news(symbol, stock_name="", size=3):
    """
    通过AKShare获取个股新闻，并使用大模型进行相关性过滤
    :param symbol: 股票代码 (例如: "1810" for 小米集团)
    :param stock_name: 股票名称 (例如: "小米集团")
    :param size: 获取新闻条数
    :return: 新闻列表
    """
    try:
        # 使用AKShare获取个股新闻
        df = ak.stock_news_em(symbol=symbol)
        articles = []
        
        # 处理新闻数据
        for _, item in df.iterrows():
            # 格式化发布时间
            pub_time = item.get("发布时间", "")
            if pub_time:
                # 将时间字符串转换为标准格式
                try:
                    pub_time = datetime.strptime(pub_time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            title = item.get("新闻标题", "").strip()
            summary = item.get("新闻内容", "").strip()
            
            articles.append({
                "title": title[:80] + ("..." if len(title) > 80 else ""),
                "summary": summary[:120] + ("..." if len(summary) > 120 else ""),
                "url": item.get("新闻链接", ""),
                "publishedAt": pub_time
            })
        
        # 如果没有指定股票名称，直接返回原始数据
        if not stock_name:
            return articles[:size]
        
        # 使用大模型过滤相关新闻
        filtered_articles = filter_news_with_llm(articles, stock_name, symbol, size)
        return filtered_articles
    except Exception as e:
        print(f"⚠️ 获取个股新闻失败: {e}")
        return []

def get_stock_info(symbol):
    """获取股价与基本面数据"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", "N/A"),
            "price": info.get("currentPrice", "N/A"),
            "currency": info.get("currency", "HKD"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "change": info.get("regularMarketChangePercent", "N/A")
        }
    except Exception as e:
        print(f"⚠️ 获取股价失败: {e}")
        return {}

def fetch_all_stock_news():
    """获取watch list中所有股票的新闻"""
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
        
        # 根据股票代码确定AKShare使用的代码格式
        if "." in code:
            symbol_code = code.split(".")[0]
        else:
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
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["股票名称", "股票代码", "新闻时间", "新闻标题", "简要内容", "查询时间"])
            # 写入数据
            for news in all_news_data:
                writer.writerow([
                    news["stock_name"],
                    news["stock_code"],
                    news["publishedAt"],
                    news["title"],
                    news["summary"],
                    news["query_time"]
                ])
        
        print(f"\n✅ 所有新闻数据已保存到 {csv_file}")
        
        # 显示汇总信息
        print("\n📋 新闻汇总:")
        for news in all_news_data:
            print(f"  • {news['stock_name']} ({news['stock_code']}) | {news['title']}")
    else:
        print("\n❌ 未获取到任何新闻数据")
        
    print(f"\n📊 总共处理了 {stock_count} 只股票")
    print("=" * 60)

def run_scheduler():
    """设置定时任务"""
    # 设置香港时间上午9点和下午1点半运行
    schedule.every().day.at("09:00").do(fetch_all_stock_news)
    schedule.every().day.at("13:30").do(fetch_all_stock_news)
    
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
    
    # 解析参数
    args = parser.parse_args()
    
    if args.schedule:
        # 启用定时任务模式
        run_scheduler()
    else:
        # 单次运行模式
        fetch_all_stock_news()
