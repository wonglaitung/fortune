"""
情感分析模块
使用大模型对新闻进行四维情感评分
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_services.qwen_engine import chat_with_llm

def analyze_news_sentiment(stock_name, stock_code, news_title, news_content):
    """
    使用大模型分析新闻的情感影响

    Args:
        stock_name (str): 股票名称
        stock_code (str): 股票代码
        news_title (str): 新闻标题
        news_content (str): 新闻内容

    Returns:
        dict: 包含四维情感评分的字典
            - relevance: 相关性 (0-1)
            - impact: 影响度 (0-1)
            - expectation_gap: 预期差 (-1到1)
            - sentiment_direction: 情感方向 (-1到1)
            - sentiment_score: 综合情感分数
    """

    prompt = f"""你是一位专业的金融分析师。请分析以下新闻对股票的情感影响。

股票：{stock_name} ({stock_code})
新闻标题：{news_title}
新闻内容：{news_content}

请从以下四个维度进行评分（每个维度都要给出理由和分数）：

1. **相关性 (Relevance, 0-1分)**：新闻与该股票的直接相关程度
   - 0-0.3: 低相关性（如宏观新闻、行业趋势）
   - 0.3-0.6: 中等相关性（如同行业新闻）
   - 0.6-1.0: 高相关性（如公司专属新闻）

2. **影响度 (Impact, 0-1分)**：新闻对股价的潜在影响程度
   - 0-0.3: 低影响（如常规公告）
   - 0.3-0.6: 中等影响（如业绩公告）
   - 0.6-1.0: 高影响（如重大利好/利空）

3. **预期差 (Expectation Gap, -1到1分)**：新闻是否超出市场预期
   - -1到-0.3: 低于预期（负面惊喜）
   - -0.3到0.3: 符合预期（中性）
   - 0.3到1: 超出预期（正面惊喜）

4. **情感方向 (Sentiment Direction, -1到1分)**：新闻的情感倾向
   - -1到-0.3: 负面（如业绩下滑、监管处罚）
   - -0.3到0.3: 中性（如常规公告）
   - 0.3到1: 正面（如业绩增长、重大合作）

请以JSON格式返回分析结果，格式如下：
{{
  "relevance": 分数,
  "impact": 分数,
  "expectation_gap": 分数,
  "sentiment_direction": 分数,
  "reasoning": "简要说明分析理由"
}}

注意：
- 分数必须是数值类型，不要是字符串
- 理由要简洁明了，不超过100字
- 只返回JSON，不要有其他内容
"""

    try:
        response = chat_with_llm(prompt, enable_thinking=False)

        # 尝试解析JSON响应
        result = json.loads(response)

        # 验证必要字段
        required_fields = ['relevance', 'impact', 'expectation_gap', 'sentiment_direction']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"缺少必要字段: {field}")

        # 计算综合情感分数
        # 公式：相关性 × 影响度 × (预期差 + 情感方向) × 5
        # 结果截断到 [-5, +5] 范围，避免极端值
        raw_score = (
            result['relevance'] *
            result['impact'] *
            (result['expectation_gap'] + result['sentiment_direction']) *
            5
        )
        
        # 截断到 [-5, +5] 范围
        result['sentiment_score'] = max(-5.0, min(5.0, raw_score))

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败: {e}")
        print(f"响应内容: {response}")
        return None
    except Exception as e:
        print(f"⚠️ 情感分析失败: {e}")
        return None


def batch_analyze_sentiment(news_df, days_limit=3):
    """
    批量分析新闻情感

    Args:
        news_df (DataFrame): 新闻数据
        days_limit (int): 只分析最近N天未分析的新闻（默认3天）

    Returns:
        DataFrame: 包含情感分析的新闻数据
    """
    # 转换日期
    news_df['新闻时间'] = pd.to_datetime(news_df['新闻时间'])
    news_df['日期'] = news_df['新闻时间'].dt.date

    # 只分析最近N天的未分析新闻
    cutoff_date = (datetime.now() - timedelta(days=days_limit)).date()
    recent_news = news_df[news_df['日期'] >= cutoff_date].copy()

    # 筛选未分析的新闻
    unanalyzed = recent_news[
        recent_news['情感分数'].isna() |
        recent_news['情感分数'].isnull()
    ].copy()

    if len(unanalyzed) == 0:
        print(f"✅ 所有最近{days_limit}天的新闻已分析")
        # 统计已分析的新闻总数
        analyzed_count = len(news_df[news_df['情感分数'].notna()])
        print(f"📊 已有 {analyzed_count} 条新闻完成情感分析")
        return news_df

    print(f"📊 开始分析 {len(unanalyzed)} 条新闻的情感...")

    # 逐条分析
    for idx, row in unanalyzed.iterrows():
        try:
            result = analyze_news_sentiment(
                row['股票名称'],
                row['股票代码'],
                row['新闻标题'],
                row['简要内容']
            )

            if result:
                # 更新数据
                news_df.loc[idx, '情感分数'] = result['sentiment_score']
                news_df.loc[idx, '相关性'] = result['relevance']
                news_df.loc[idx, '影响度'] = result['impact']
                news_df.loc[idx, '预期差'] = result['expectation_gap']
                news_df.loc[idx, '情感方向'] = result['sentiment_direction']
                news_df.loc[idx, '情感分析时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                print(f"✅ [{row['股票代码']}] 情感分数: {result['sentiment_score']:.2f}")
            else:
                print(f"⚠️ [{row['股票代码']}] 分析失败")

        except Exception as e:
            print(f"❌ [{row['股票代码']}] 分析异常: {e}")
            continue

    # 保存更新后的数据
    news_df.to_csv('data/all_stock_news_records.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 情感分析完成，数据已保存")

    return news_df


def get_sentiment_statistics(news_df):
    """
    获取情感分析统计信息

    Args:
        news_df (DataFrame): 新闻数据

    Returns:
        dict: 统计信息
    """
    # 筛选已分析的新闻
    analyzed = news_df[news_df['情感分数'].notna()].copy()

    if len(analyzed) == 0:
        return {
            'total': 0,
            'analyzed': 0,
            'unanalyzed': len(news_df)
        }

    stats = {
        'total': len(news_df),
        'analyzed': len(analyzed),
        'unanalyzed': len(news_df) - len(analyzed),
        'sentiment_score_mean': analyzed['情感分数'].mean(),
        'sentiment_score_std': analyzed['情感分数'].std(),
        'sentiment_score_min': analyzed['情感分数'].min(),
        'sentiment_score_max': analyzed['情感分数'].max(),
        'positive_count': len(analyzed[analyzed['情感分数'] > 0]),
        'negative_count': len(analyzed[analyzed['情感分数'] < 0]),
        'neutral_count': len(analyzed[analyzed['情感分数'] == 0])
    }

    return stats


if __name__ == '__main__':
    # 测试代码
    print("=== 情感分析测试 ===")

    # 读取新闻数据
    df = pd.read_csv('data/all_stock_news_records.csv')

    # 批量分析最近3天的新闻
    result_df = batch_analyze_sentiment(df, days_limit=3)

    # 显示统计信息
    stats = get_sentiment_statistics(result_df)
    print("\n=== 情感分析统计 ===")
    print(f"总新闻数: {stats['total']}")
    print(f"已分析: {stats['analyzed']}")
    print(f"未分析: {stats['unanalyzed']}")
    if stats['analyzed'] > 0:
        print(f"平均情感分数: {stats['sentiment_score_mean']:.2f}")
        print(f"正面新闻: {stats['positive_count']}")
        print(f"负面新闻: {stats['negative_count']}")
        print(f"中性新闻: {stats['neutral_count']}")