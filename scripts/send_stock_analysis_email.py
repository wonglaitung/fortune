#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分析邮件发送脚本

功能：
- 从综合分析报告中提取指定股票的分析信息（使用大模型智能提取）
- 支持多只股票同时分析
- 使用标准化 HTML 模板发送邮件
- 专业术语附带简明解释

用法：
    python3 scripts/send_stock_analysis_email.py --stocks 2318.HK
    python3 scripts/send_stock_analysis_email.py --stocks 2318.HK,0700.HK,2800.HK
    python3 scripts/send_stock_analysis_email.py --stocks 2318.HK --recipient user@example.com
"""

import argparse
import sys
import os
import re
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from message_services.email_sender import send_email
from data_services.calendar_features import get_last_trading_day
from config import STOCK_SECTOR_MAPPING
from llm_services.qwen_engine import chat_with_llm


# ============================================================================
# 大模型提取 Prompt 模板
# ============================================================================

STOCK_ANALYSIS_PROMPT = """你是一个股票分析专家。请从以下综合分析报告中提取指定股票的分析信息。

# 股票代码
{stock_code}

# 综合分析报告
{report_content}

# 提取要求

请提取以下信息，以 JSON 格式返回：

## 个股关键数据
1. **stock_name**: 股票名称
2. **current_price**: 当前价格（数值）
3. **price_change**: 涨跌幅（百分比数值，如 -2.5）
4. **catboost_prob_20d**: CatBoost 20天上涨概率（百分比数值，如 37.12）
5. **catboost_prob_5d**: CatBoost 5天上涨概率
6. **catboost_prob_1d**: CatBoost 1天上涨概率
7. **three_period_pattern**: 三周期模式（如"下跌中继⭐(001)"）
8. **transmission_mode**: 传导模式验证结果（如"✅传导(1天↓✓ 5天↓✓ 20天↑⏳)"）
9. **chip_resistance**: 筹码阻力（低/中/高）
10. **risk_score**: 风险得分（数值）
11. **return_score**: 回报得分（数值）
12. **total_score**: 综合得分（数值）
13. **risk_advice**: 风险建议（推荐/观察/暂缓/优选）

## 网络洞察
14. **community**: 网络社区归属（如"社区5"）
15. **hub_type**: 枢纽类型（低枢纽/中枢纽/高枢纽）
16. **is_bridge**: 是否为桥梁股（是/否）

## 大模型建议
17. **short_term_advice**: 大模型短期建议（买入/观察/卖出）
18. **mid_term_advice**: 大模型中期建议（买入/观察/卖出）
19. **position_advice**: 建议仓位（百分比数值）
20. **stop_loss**: 止损位（价格数值或 null）
21. **target_price**: 目标价（价格数值或 null）

## 技术指标
22. **rsi**: RSI值（数值）
23. **rsi_status**: RSI状态（超买/正常/超卖）
24. **macd_status**: MACD状态（金叉/死叉/弱金叉等）
25. **bb_position**: 布林带位置（百分比数值）
26. **ma_status**: 均线排列状态（多头排列/空头排列/震荡整理）

## 板块信息
27. **sector**: 所属板块名称
28. **sector_rank**: 板块排名（数值）
29. **sector_type**: 板块类型（周期/防御）
30. **sector_change**: 板块5日涨跌幅（百分比数值）

## 异常检测
31. **anomaly_alert**: 是否有异常检测警报（是/否）
32. **anomaly_reason**: 异常原因（如有）

## 股息信息
33. **dividend_info**: 股息信息（如"即将除净：2026-05-25，每股分红0.5元"或 null）

## 市场环境信息（所有股票共用，从报告开头提取）
34. **hsi_price**: 恒生指数价格（数值）
35. **hsi_change**: 恒生指数涨跌幅（百分比数值）
36. **market_status**: 市场状态（牛市/震荡偏涨/震荡市/震荡偏跌/熊市）
37. **market_duration**: 市场状态持续天数（数值）
38. **market_stability**: 市场状态稳定性（稳定/中等稳定/不稳定）
39. **vix**: VIX指数（数值）
40. **market_sentiment**: 市场情绪（正常/谨慎/暂停）
41. **modularity**: 模块度（数值）

# 返回格式

请以 JSON 格式返回，包含以上所有字段：
{{
    "stock_name": "...",
    "current_price": ...,
    "catboost_prob_20d": ...,
    ...
}}

**重要**：
1. 如果报告中没有某项信息，请填写 null
2. 数值字段请返回数值类型，不要返回字符串
3. 百分比字段请返回数值（如 37.12 表示 37.12%）
4. 确保返回的是纯 JSON，不要包含 Markdown 代码块标记
"""


# ============================================================================
# 大模型综合分析 Prompt 模板（集成 stock-analysis 技能的分析框架）
# ============================================================================

COMPREHENSIVE_ANALYSIS_PROMPT = """你是一个专业的股票分析师。请基于以下股票数据，按照分析框架生成详细的综合分析。

# 股票数据
{stock_data}

# 分析框架

请按以下步骤进行综合分析：

## 第一步：硬约束检查
- CatBoost 20天上涨概率 ≤ 50% → **禁止买入**
- CatBoost 20天上涨概率 ≥ 60% → 高置信度，进入下一步
- CatBoost 20天上涨概率 50-60% → 中等置信度，需其他信号确认

## 第二步：方向一致性检查（关键）
- 短期建议 + 中期建议 + CatBoost 三者方向一致 → **可买入**
- 短期建议"观察" + 中期建议"买入" → **观望**（方向冲突）
- 短期建议"买入" + 中期建议"观察" → **观望**（方向冲突）
- 短期建议"卖出" + 中期建议任何 → **禁止买入**

**重要规则**：即使CatBoost概率>60%，如果短期/中期方向不一致，也应归入"观望"，而非"强烈买入"。

## 第三步：市场环境检查
- 市场状态为"熊市" → 提高阈值至0.70
- 市场状态为"震荡市" → 提高阈值至0.65
- 市场状态稳定性<5天 → 提示风险，建议降低仓位

## 第四步：异常检测检查
- 有超买异常（RSI>70）→ 提示短期回调风险
- 有超卖异常（RSI<30）→ 提示可能存在反弹机会
- 有成交量异常 → 提示资金异动

## 第五步：网络洞察检查
- 模块度<0.20 → 市场同涨同跌，降低仓位30%
- 为桥梁股 → 提示跨社区风险传导

## 分类标准

| 分类 | CatBoost概率 | 短期建议 | 中期建议 | 说明 |
|------|-------------|---------|---------|------|
| ⭐强烈买入 | ≥60% | 买入 | 买入 | 三重确认 |
| 🟢买入 | 50-60% | 买入 | 买入 | 需其他信号确认 |
| 🟡观望 | ≥50% | 观察 | 买入 | 方向冲突，等待确认 |
| 🟡观望 | ≥50% | 买入 | 观察 | 方向冲突，等待确认 |
| 🔴禁止买入 | ≤50% | 任何 | 任何 | 硬约束 |
| 🔴禁止买入 | 任何 | 卖出 | 任何 | 方向明确看跌 |

# 返回格式

请以 JSON 格式返回以下字段：
{{
    "recommendation": "综合建议（如：⭐强烈买入 / 🟢买入 / 🟡观望 / 🔴禁止买入）",
    "recommendation_class": "CSS类名（rec-strong-buy / rec-buy / rec-hold / rec-sell）",
    "consistency": "方向一致性分析（如：一致 / 方向冲突，短期中期不一致）",
    "analysis_summary": "分析摘要（一句话说明判断依据）",
    "operation_advice": "操作建议（详细说明，包括仓位、止损等）",
    "risk_warnings": ["风险提示1", "风险提示2", ...]
}}

**重要**：
1. risk_warnings 必须是数组，列出3-5个主要风险因素
2. operation_advice 要具体，包括仓位建议、止损设置等
3. 确保返回的是纯 JSON，不要包含 Markdown 代码块标记
"""


# ============================================================================
# 大模型持货人建议 Prompt 模板
# ============================================================================

HOLDER_ADVICE_PROMPT = """你是一个专业的股票分析师。请基于以下股票分析数据，为已持有该股票的投资者提供操作建议。

# 股票数据
{stock_data}

# 分析框架

## 第一步：趋势判断
- CatBoost 20天概率 > 60% 且短中期建议"买入" → 趋势向上，考虑持有或加仓
- CatBoost 20天概率 50-60% → 趋势不明，谨慎持有
- CatBoost 20天概率 < 50% → 趋势向下，考虑减仓或止损

## 第二步：止盈止损判断
- 当前价格接近目标价 → 考虑分批止盈
- RSI > 70 或布林带位置 > 80% → 短期超买，考虑减仓锁利
- RSI < 30 或布林带位置 < 20% → 短期超卖，若趋势仍向上可持有
- MACD 死叉 → 警惕回调

## 第三步：市场环境调整
- 熊市环境 → 降低持仓，严格止损
- 市场情绪"暂停" → 建议减仓避险
- 市场状态不稳定（<5天）→ 降低仓位

## 第四步：综合建议
基于以上分析，给出持货人的具体操作建议。

# 返回格式

请以 JSON 格式返回：
{{
    "holder_action": "继续持有 / 加仓 / 减仓 / 止盈离场 / 止损离场",
    "holder_reason": "操作理由（1-2句话）",
    "key_level": "关键价位说明（如：跌破XX止损，突破XX加仓）",
    "risk_level": "低 / 中 / 高",
    "time_horizon": "短期（1-5天）/ 中期（5-20天）"
}}

**重要**：
1. 确保返回纯 JSON，不要包含 Markdown 代码块标记
2. holder_action 只能是上述5种之一
3. holder_reason 要具体，引用具体指标数据
"""

# 持货人操作建议样式映射
HOLDER_ACTION_STYLES = {
    '继续持有': ('holder-action-hold', '#d4edda', '#155724'),
    '加仓': ('holder-action-add', '#cce5ff', '#004085'),
    '减仓': ('holder-action-reduce', '#fff3cd', '#856404'),
    '止盈离场': ('holder-action-exit-profit', '#f8d7da', '#721c24'),
    '止损离场': ('holder-action-exit-loss', '#f8d7da', '#721c24'),
}


# ============================================================================
# HTML 邮件模板
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background-color: #ffffff;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 15px;
            margin-top: 0;
            font-size: 24px;
        }}
        .date-info {{
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 12px;
            font-size: 18px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-good {{
            color: #28a745;
            font-weight: bold;
        }}
        .metric-bad {{
            color: #dc3545;
            font-weight: bold;
        }}
        .metric-neutral {{
            color: #ffc107;
            font-weight: bold;
        }}
        .recommendation {{
            font-size: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: bold;
        }}
        .rec-strong-buy {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .rec-buy {{
            background-color: #cce5ff;
            color: #004085;
            border: 1px solid #b8daff;
        }}
        .rec-hold {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        .rec-sell {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .warning-box {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #dc3545;
        }}
        .info-box {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #17a2b8;
        }}
        .success-box {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 8px 0;
        }}
        .disclaimer {{
            font-size: 12px;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
            line-height: 1.8;
        }}
        .arrow-up {{
            color: #28a745;
            font-weight: bold;
        }}
        .arrow-down {{
            color: #dc3545;
            font-weight: bold;
        }}
        .stock-section {{
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .stock-section:last-child {{
            border-bottom: none;
        }}
        .holder-action-hold {{
            background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-add {{
            background-color: #cce5ff; color: #004085; border: 1px solid #b8daff;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-reduce {{
            background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
        .holder-action-exit-profit, .holder-action-exit-loss {{
            background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>【股票分析报告】{date}</h1>
        <p class="date-info"><strong>分析日期</strong>：{date}</p>

        {market_section}

        {stock_sections}

        <div class="disclaimer">
            <p><strong>免责声明</strong>：以上建议仅供参考，不构成投资建议，投资有风险，决策需谨慎。</p>
            <p>本报告由港股智能分析系统自动生成，分析日期：{date}</p>
        </div>
    </div>
</body>
</html>
"""

MARKET_SECTION_TEMPLATE = """
        <h2>市场环境概览</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>说明</th></tr>
            <tr><td>恒生指数</td><td>{hsi_price}</td><td>{hsi_change_display}</td></tr>
            <tr><td>市场状态</td><td>{market_status}</td><td>{market_status_desc}</td></tr>
            <tr><td>状态持续时间</td><td>{market_duration}天</td><td>{market_stability}</td></tr>
            <tr><td>VIX（恐慌指数）</td><td>{vix}</td><td>{vix_desc}</td></tr>
            <tr><td>市场情绪</td><td>{market_sentiment}</td><td>{sentiment_desc}</td></tr>
            <tr><td>模块度</td><td>{modularity}</td><td>{modularity_desc}</td></tr>
        </table>
"""

STOCK_SECTION_TEMPLATE = """
        <div class="stock-section">
            <h2>{stock_name}（{stock_code}）</h2>

            <div class="recommendation {rec_class}">
                综合建议：{recommendation}
            </div>

            <h3>一、核心指标</h3>
            <table>
                <tr><th>指标</th><th>数值</th><th>说明</th></tr>
                <tr><td>CatBoost 20天上涨概率</td><td class="{prob_class}">{catboost_prob_20d}%</td><td>{prob_desc}</td></tr>
                <tr><td>当前价格</td><td>HK${current_price}</td><td>{price_change_display}</td></tr>
                <tr><td>建议仓位</td><td>{position_advice}%</td><td></td></tr>
                <tr><td>止损位</td><td>{stop_loss}</td><td>最大亏损控制在-8%以内</td></tr>
                <tr><td>目标价</td><td>{target_price}</td><td></td></tr>
            </table>

            <h3>二、三周期预测</h3>
            <table>
                <tr><th>周期</th><th>预测概率</th><th>方向</th></tr>
                <tr><td>1天</td><td>{catboost_prob_1d}%</td><td class="{dir_1d_class}">{dir_1d}</td></tr>
                <tr><td>5天</td><td>{catboost_prob_5d}%</td><td class="{dir_5d_class}">{dir_5d}</td></tr>
                <tr><td>20天</td><td>{catboost_prob_20d}%</td><td class="{dir_20d_class}">{dir_20d}</td></tr>
            </table>
            <p><strong>三周期模式</strong>：{three_period_pattern}</p>
            <p><strong>传导模式</strong>：{transmission_mode}</p>

            <h3>三、大模型建议</h3>
            <ul>
                <li><strong>短期建议</strong>：{short_term_advice}</li>
                <li><strong>中期建议</strong>：{mid_term_advice}</li>
                <li><strong>一致性</strong>：{consistency}</li>
            </ul>

            <h3>四、技术指标</h3>
            <table>
                <tr><th>指标</th><th>数值</th><th>状态</th></tr>
                <tr><td>RSI（相对强弱指数）</td><td>{rsi}</td><td>{rsi_status_display}</td></tr>
                <tr><td>MACD</td><td>-</td><td>{macd_status}</td></tr>
                <tr><td>布林带位置</td><td>{bb_position}%</td><td>{bb_status}</td></tr>
                <tr><td>均线排列</td><td>-</td><td>{ma_status}</td></tr>
                <tr><td>筹码阻力</td><td>-</td><td>{chip_resistance}</td></tr>
            </table>

            <h3>五、风险评分</h3>
            <table>
                <tr><th>指标</th><th>得分</th></tr>
                <tr><td>风险得分</td><td>{risk_score}</td></tr>
                <tr><td>回报得分</td><td>{return_score}</td></tr>
                <tr><td>综合得分</td><td>{total_score}</td></tr>
                <tr><td>风险建议</td><td>{risk_advice}</td></tr>
            </table>

            <h3>六、网络洞察</h3>
            <ul>
                <li><strong>社区归属</strong>：{community}</li>
                <li><strong>枢纽类型</strong>：{hub_type}</li>
                <li><strong>是否桥梁股</strong>：{is_bridge}</li>
            </ul>

            <h3>七、板块表现</h3>
            <ul>
                <li><strong>所属板块</strong>：{sector}</li>
                <li><strong>板块排名</strong>：第{sector_rank}名</li>
                <li><strong>板块类型</strong>：{sector_type}</li>
            </ul>

            {anomaly_section}

            {dividend_section}

            {operation_box}

            {holder_advice_section}

            {risk_warnings_section}
        </div>
"""


# ============================================================================
# 核心函数
# ============================================================================

def extract_json_from_response(response: str) -> dict:
    """从大模型响应中提取 JSON"""
    try:
        # 尝试提取 Markdown 代码块中的 JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接匹配 JSON 对象
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def extract_stock_data_with_llm(stock_code: str, report_content: str) -> dict:
    """
    使用大模型从综合报告中提取指定股票的分析数据

    参数：
    - stock_code: 股票代码（如 "2318.HK"）
    - report_content: 综合报告内容

    返回：
    - dict: 股票分析数据
    """
    print(f"📊 正在使用大模型提取 {stock_code} 的分析数据...")

    # 构建提取 Prompt
    prompt = STOCK_ANALYSIS_PROMPT.format(
        stock_code=stock_code,
        report_content=report_content
    )

    # 调用大模型提取
    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return None

    # 解析 JSON 响应
    stock_data = extract_json_from_response(response)
    if stock_data:
        stock_data['stock_code'] = stock_code
        print(f"✅ 成功提取 {stock_code} 的分析数据")
        return stock_data
    else:
        print(f"❌ 无法从大模型响应中提取 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None


def comprehensive_analyze_with_llm(stock_data: dict) -> dict:
    """
    使用大模型进行综合分析（集成 stock-analysis 技能的分析框架）

    参数：
    - stock_data: 股票数据

    返回：
    - dict: 综合分析结果，包含 recommendation, operation_advice, risk_warnings 等
    """
    print(f"🤖 正在使用大模型进行综合分析...")

    # 构建分析 Prompt
    prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
        stock_data=json.dumps(stock_data, ensure_ascii=False, indent=2)
    )

    # 调用大模型分析
    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return None

    # 解析 JSON 响应
    analysis_result = extract_json_from_response(response)
    if analysis_result:
        print(f"✅ 成功生成综合分析")
        return analysis_result
    else:
        print(f"❌ 无法从大模型响应中提取 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None


def get_holder_advice_with_llm(stock_data: dict) -> dict:
    """
    使用大模型为已持货人生成操作建议

    参数：
    - stock_data: 股票数据

    返回：
    - dict: 持货人操作建议
    """
    print(f"📊 正在生成持货人操作建议...")

    prompt = HOLDER_ADVICE_PROMPT.format(
        stock_data=json.dumps(stock_data, ensure_ascii=False, indent=2)
    )

    try:
        response = chat_with_llm(prompt, enable_thinking=True)
    except Exception as e:
        print(f"❌ 持货人建议生成失败: {e}")
        return None

    result = extract_json_from_response(response)
    if result:
        print(f"✅ 成功生成持货人操作建议")
        return result
    else:
        print(f"❌ 无法从大模型响应中提取持货人建议 JSON")
        print(f"   响应内容: {response[:500]}...")
        return None
    """
    分析多只股票

    参数：
    - stock_codes: 股票代码列表
    - report_path: 综合报告路径

    返回：
    - tuple: (股票分析结果列表, 市场环境信息)
    """
    results = []
    market_info = {}

    # 读取报告内容
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
    except FileNotFoundError:
        print(f"❌ 报告文件不存在: {report_path}")
        return results, market_info

    for stock_code in stock_codes:
        # 第一步：提取股票数据
        stock_data = extract_stock_data_with_llm(stock_code, report_content)
        if not stock_data:
            continue

        # 第二步：使用大模型进行综合分析
        analysis_result = comprehensive_analyze_with_llm(stock_data)
        if analysis_result:
            # 合并分析结果到股票数据
            stock_data['analysis'] = analysis_result

        # 第三步：生成持货人操作建议
        holder_advice = get_holder_advice_with_llm(stock_data)
        if holder_advice:
            stock_data['holder_advice'] = holder_advice

        results.append(stock_data)

        # 提取市场环境信息（从第一个成功的股票数据中获取）
        if not market_info:
            market_info = {
                'hsi_price': stock_data.get('hsi_price'),
                'hsi_change': stock_data.get('hsi_change'),
                'market_status': stock_data.get('market_status'),
                'market_duration': stock_data.get('market_duration'),
                'market_stability': stock_data.get('market_stability'),
                'vix': stock_data.get('vix'),
                'market_sentiment': stock_data.get('market_sentiment'),
                'modularity': stock_data.get('modularity'),
            }

    return results, market_info


def generate_recommendation(stock_data: dict) -> tuple:
    """
    根据分析数据生成综合建议

    返回：
    - tuple: (建议文本, CSS类名)
    """
    prob_20d = stock_data.get('catboost_prob_20d', 0) or 0
    short_term = stock_data.get('short_term_advice', '观察') or '观察'
    mid_term = stock_data.get('mid_term_advice', '观察') or '观察'

    # 硬约束检查
    if prob_20d <= 50:
        return "🔴 禁止买入", "rec-sell"

    # 方向一致性检查
    if short_term == '买入' and mid_term == '买入' and prob_20d >= 60:
        return "⭐ 强烈买入", "rec-strong-buy"
    elif short_term == '买入' and mid_term == '买入':
        return "🟢 买入", "rec-buy"
    elif short_term == '卖出' or mid_term == '卖出':
        return "🔴 卖出", "rec-sell"
    else:
        return "🟡 观望", "rec-hold"


def format_value(value, default="-"):
    """格式化数值，处理 None 值"""
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def generate_stock_section(stock_data: dict) -> str:
    """生成单只股票的 HTML 部分（使用大模型综合分析结果）"""
    stock_code = stock_data.get('stock_code', '')
    stock_name = stock_data.get('stock_name', stock_code)

    # 获取大模型综合分析结果
    analysis = stock_data.get('analysis', {})

    # 综合建议（优先使用大模型分析结果）
    if analysis:
        recommendation = analysis.get('recommendation', '🟡 观望')
        rec_class = analysis.get('recommendation_class', 'rec-hold')
        consistency = analysis.get('consistency', '一致')
        operation_advice = analysis.get('operation_advice', '')
        risk_warnings = analysis.get('risk_warnings', [])
    else:
        # 如果没有大模型分析结果，使用默认逻辑
        recommendation, rec_class = generate_recommendation(stock_data)
        consistency = "一致"
        operation_advice = ""
        risk_warnings = []

    # CatBoost 概率
    prob_20d = stock_data.get('catboost_prob_20d', 0) or 0
    prob_5d = stock_data.get('catboost_prob_5d', 0) or 0
    prob_1d = stock_data.get('catboost_prob_1d', 0) or 0

    # 概率样式
    if prob_20d >= 60:
        prob_class = "metric-good"
        prob_desc = "高置信度，>60%"
    elif prob_20d >= 50:
        prob_class = "metric-neutral"
        prob_desc = "中等置信度，50-60%"
    else:
        prob_class = "metric-bad"
        prob_desc = "看跌，<50%硬约束禁止买入"

    # 方向判断
    dir_1d = "↑ 上涨" if prob_1d >= 50 else "↓ 下跌"
    dir_1d_class = "arrow-up" if prob_1d >= 50 else "arrow-down"
    dir_5d = "↑ 上涨" if prob_5d >= 50 else "↓ 下跌"
    dir_5d_class = "arrow-up" if prob_5d >= 50 else "arrow-down"
    dir_20d = "↑ 上涨" if prob_20d >= 50 else "↓ 下跌"
    dir_20d_class = "arrow-up" if prob_20d >= 50 else "arrow-down"

    # 涨跌幅显示
    price_change = stock_data.get('price_change', 0) or 0
    if price_change > 0:
        price_change_display = f"📈 +{price_change:.2f}%"
    elif price_change < 0:
        price_change_display = f"📉 {price_change:.2f}%"
    else:
        price_change_display = "-"

    # RSI 状态显示
    rsi = stock_data.get('rsi', 50) or 50
    rsi_status = stock_data.get('rsi_status', '正常') or '正常'
    if rsi > 70:
        rsi_status_display = f"超买（{rsi:.1f}）"
    elif rsi < 30:
        rsi_status_display = f"超卖（{rsi:.1f}）"
    else:
        rsi_status_display = f"{rsi_status}（{rsi:.1f}）"

    # 布林带状态
    bb_position = stock_data.get('bb_position', 50) or 50
    if bb_position > 80:
        bb_status = "接近上轨（超买）"
    elif bb_position < 20:
        bb_status = "接近下轨（超卖）"
    else:
        bb_status = "中性"

    # 止损位和目标价
    stop_loss = stock_data.get('stop_loss')
    if stop_loss:
        stop_loss = f"HK${stop_loss:.2f}"
    else:
        stop_loss = "-"

    target_price = stock_data.get('target_price')
    if target_price:
        target_price = f"HK${target_price:.2f}"
    else:
        target_price = "-"

    # 异常检测部分
    anomaly_alert = stock_data.get('anomaly_alert', '否') or '否'
    anomaly_reason = stock_data.get('anomaly_reason') or ''
    if anomaly_alert == '是':
        anomaly_section = f"""
            <h3>七、异常检测</h3>
            <div class="warning-box">
                ⚠️ 检测到异常：{anomaly_reason}
            </div>
        """
    else:
        anomaly_section = """
            <h3>七、异常检测</h3>
            <p>当日无异常检测警报</p>
        """

    # 股息信息
    dividend_info = stock_data.get('dividend_info')
    if dividend_info:
        dividend_section = f"""
            <h3>八、股息提醒</h3>
            <div class="info-box">
                📅 {dividend_info}
            </div>
        """
    else:
        dividend_section = ""

    # 操作建议框（使用大模型分析结果）
    if operation_advice:
        if "强烈买入" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="success-box">
                    {operation_advice}
                </div>
            """
        elif "买入" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="info-box">
                    {operation_advice}
                </div>
            """
        elif "观望" in recommendation:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="info-box">
                    {operation_advice}
                </div>
            """
        else:
            operation_box = f"""
                <h3>九、操作建议</h3>
                <div class="warning-box">
                    {operation_advice}
                </div>
            """
    else:
        # 默认操作建议
        if "强烈买入" in recommendation:
            operation_box = """
                <h3>九、操作建议</h3>
                <div class="success-box">
                    <strong>操作建议</strong>：可考虑分批建仓，首次建仓30%，回调补仓20%。<br>
                    <strong>止损设置</strong>：建议设置止损位，最大亏损控制在-8%以内。
                </div>
            """
        elif "买入" in recommendation:
            operation_box = """
                <h3>九、操作建议</h3>
                <div class="info-box">
                    <strong>操作建议</strong>：谨慎买入，建议仓位控制在20%以内。<br>
                    <strong>止损设置</strong>：建议设置止损位，最大亏损控制在-8%以内。
                </div>
            """
        elif "观望" in recommendation:
            operation_box = """
                <h3>九、操作建议</h3>
                <div class="info-box">
                    <strong>操作建议</strong>：暂时观望，等待更明确的买入信号。
                </div>
            """
        else:
            operation_box = """
                <h3>九、操作建议</h3>
                <div class="warning-box">
                    <strong>操作建议</strong>：不建议买入，或考虑减仓/清仓。
                </div>
            """

    # 持货人操作建议部分
    holder_advice = stock_data.get('holder_advice')
    if holder_advice:
        holder_action = holder_advice.get('holder_action', '继续持有')
        holder_reason = holder_advice.get('holder_reason', '')
        key_level = holder_advice.get('key_level', '')
        risk_level = holder_advice.get('risk_level', '中')
        time_horizon = holder_advice.get('time_horizon', '中期（5-20天）')

        # 根据操作类型选择样式
        style_info = HOLDER_ACTION_STYLES.get(holder_action, HOLDER_ACTION_STYLES['继续持有'])
        holder_action_class = style_info[0]

        holder_advice_section = f"""
            <h3>十、持货人操作建议</h3>
            <div class="{holder_action_class}">
                <strong>建议操作</strong>：{holder_action}<br>
                <strong>风险等级</strong>：{risk_level} | <strong>适用周期</strong>：{time_horizon}
            </div>
            <p><strong>理由</strong>：{holder_reason}</p>
            <p><strong>关键价位</strong>：{key_level}</p>
        """
    else:
        holder_advice_section = ""

    # 风险提示部分（使用大模型分析结果）
    if risk_warnings:
        risk_warnings_list = "".join([f"<li>{w}</li>" for w in risk_warnings])
        risk_warnings_section = f"""
            <h3>十、风险提示</h3>
            <ul>
                {risk_warnings_list}
            </ul>
        """
    else:
        risk_warnings_section = ""

    return STOCK_SECTION_TEMPLATE.format(
        stock_name=stock_name,
        stock_code=stock_code,
        rec_class=rec_class,
        recommendation=recommendation,
        prob_class=prob_class,
        catboost_prob_20d=format_value(prob_20d, "0"),
        prob_desc=prob_desc,
        current_price=format_value(stock_data.get('current_price'), "-"),
        price_change_display=price_change_display,
        position_advice=format_value(stock_data.get('position_advice', 0), "0"),
        stop_loss=stop_loss,
        target_price=target_price,
        catboost_prob_1d=format_value(prob_1d, "0"),
        catboost_prob_5d=format_value(prob_5d, "0"),
        dir_1d=dir_1d,
        dir_1d_class=dir_1d_class,
        dir_5d=dir_5d,
        dir_5d_class=dir_5d_class,
        dir_20d=dir_20d,
        dir_20d_class=dir_20d_class,
        three_period_pattern=stock_data.get('three_period_pattern') or "-",
        transmission_mode=stock_data.get('transmission_mode') or "-",
        short_term_advice=stock_data.get('short_term_advice') or "观察",
        mid_term_advice=stock_data.get('mid_term_advice') or "观察",
        consistency=consistency,
        rsi=format_value(rsi, "-"),
        rsi_status_display=rsi_status_display,
        macd_status=stock_data.get('macd_status') or "-",
        bb_position=format_value(bb_position, "-"),
        bb_status=bb_status,
        ma_status=stock_data.get('ma_status') or "-",
        chip_resistance=stock_data.get('chip_resistance') or "-",
        risk_score=format_value(stock_data.get('risk_score'), "-"),
        return_score=format_value(stock_data.get('return_score'), "-"),
        total_score=format_value(stock_data.get('total_score'), "-"),
        risk_advice=stock_data.get('risk_advice') or "-",
        community=stock_data.get('community') or "-",
        hub_type=stock_data.get('hub_type') or "-",
        is_bridge=stock_data.get('is_bridge') or "-",
        sector=stock_data.get('sector') or "-",
        sector_rank=format_value(stock_data.get('sector_rank'), "-"),
        sector_type=stock_data.get('sector_type') or "-",
        anomaly_section=anomaly_section,
        dividend_section=dividend_section,
        operation_box=operation_box,
        holder_advice_section=holder_advice_section,
        risk_warnings_section=risk_warnings_section,
    )


def generate_recommendation(stock_data: dict) -> tuple:
    """
    根据分析数据生成综合建议（备用函数，当大模型分析失败时使用）

    返回：
    - tuple: (建议文本, CSS类名)
    """
    prob_20d = stock_data.get('catboost_prob_20d', 0) or 0
    short_term = stock_data.get('short_term_advice', '观察') or '观察'
    mid_term = stock_data.get('mid_term_advice', '观察') or '观察'

    # 硬约束检查
    if prob_20d <= 50:
        return "🔴 禁止买入", "rec-sell"

    # 方向一致性检查
    if short_term == '买入' and mid_term == '买入' and prob_20d >= 60:
        return "⭐ 强烈买入", "rec-strong-buy"
    elif short_term == '买入' and mid_term == '买入':
        return "🟢 买入", "rec-buy"
    elif short_term == '卖出' or mid_term == '卖出':
        return "🔴 卖出", "rec-sell"
    else:
        return "🟡 观望", "rec-hold"


def generate_market_section(market_info: dict) -> str:
    """生成市场环境 HTML 部分"""
    if not market_info:
        return ""

    # 涨跌幅显示
    hsi_change = market_info.get('hsi_change', 0) or 0
    if hsi_change > 0:
        hsi_change_display = f"📈 +{hsi_change:.2f}%"
    elif hsi_change < 0:
        hsi_change_display = f"📉 {hsi_change:.2f}%"
    else:
        hsi_change_display = "-"

    # 市场状态说明
    market_status = market_info.get('market_status') or "-"
    market_status_desc = {
        '牛市': '持续上涨趋势',
        '震荡偏涨': '震荡中偏强',
        '震荡市': '横盘整理',
        '震荡偏跌': '震荡中偏弱',
        '熊市': '持续下跌趋势',
    }.get(market_status, '-')

    # VIX 说明
    vix = market_info.get('vix', 15) or 15
    if vix < 15:
        vix_desc = "市场平静"
    elif vix < 20:
        vix_desc = "正常水平"
    elif vix < 30:
        vix_desc = "市场恐慌"
    else:
        vix_desc = "极度恐慌"

    # 市场情绪说明
    sentiment = market_info.get('market_sentiment') or '正常'
    sentiment_desc = {
        '正常': '可正常交易',
        '谨慎': '建议降低仓位',
        '暂停': '建议暂停交易',
    }.get(sentiment, '-')

    # 模块度说明
    modularity = market_info.get('modularity', 0.4) or 0.4
    if modularity > 0.4:
        modularity_desc = "分化明显，选股有效"
    elif modularity > 0.2:
        modularity_desc = "中等分化"
    else:
        modularity_desc = "同涨同跌，系统性风险高"

    return MARKET_SECTION_TEMPLATE.format(
        hsi_price=format_value(market_info.get('hsi_price'), "-"),
        hsi_change_display=hsi_change_display,
        market_status=market_status,
        market_status_desc=market_status_desc,
        market_duration=format_value(market_info.get('market_duration'), "-"),
        market_stability=market_info.get('market_stability') or "-",
        vix=format_value(vix, "-"),
        vix_desc=vix_desc,
        market_sentiment=sentiment,
        sentiment_desc=sentiment_desc,
        modularity=format_value(modularity, "-"),
        modularity_desc=modularity_desc,
    )


def generate_html_email(stock_results: list, market_info: dict, date_str: str) -> str:
    """生成完整的 HTML 邮件内容"""
    # 生成市场环境部分
    market_section = generate_market_section(market_info)

    # 生成每只股票的分析部分
    stock_sections = ""
    for stock_data in stock_results:
        stock_sections += generate_stock_section(stock_data)

    return HTML_TEMPLATE.format(
        date=date_str,
        market_section=market_section,
        stock_sections=stock_sections,
    )


def main():
    parser = argparse.ArgumentParser(description='股票分析邮件发送脚本')
    parser.add_argument('--stocks', required=True,
                        help='股票代码，逗号分隔（如 2318.HK,0700.HK）')
    parser.add_argument('--email', action='store_true',
                        help='发送邮件')
    parser.add_argument('--recipient', default=None,
                        help='收件人邮箱（可选，默认使用环境变量）')
    parser.add_argument('--date', default=None,
                        help='分析日期（可选，默认使用最近交易日）')
    args = parser.parse_args()

    # 解析股票代码
    stock_codes = [code.strip() for code in args.stocks.split(',') if code.strip()]
    if not stock_codes:
        print("❌ 未指定股票代码")
        sys.exit(1)

    # 获取分析日期
    if args.date:
        date_str = args.date
    else:
        date_str = get_last_trading_day()

    print(f"📅 分析日期: {date_str}")
    print(f"📊 分析股票: {', '.join(stock_codes)}")

    # 报告路径
    report_path = f'output/comprehensive_reports/{date_str}.md'

    # 分析股票
    print("\n" + "="*50)
    print("🔍 开始分析...")
    print("="*50)

    stock_results, market_info = analyze_multiple_stocks(stock_codes, report_path)

    if not stock_results:
        print("❌ 未能提取任何股票的分析数据")
        sys.exit(1)

    print(f"\n✅ 成功分析 {len(stock_results)} 只股票")

    # 生成 HTML 邮件
    print("\n" + "="*50)
    print("📝 生成邮件内容...")
    print("="*50)

    html_content = generate_html_email(stock_results, market_info, date_str)

    # 发送邮件
    if args.email:
        print("\n" + "="*50)
        print("📧 发送邮件...")
        print("="*50)

        # 构建邮件主题
        if len(stock_results) == 1:
            stock_name = stock_results[0].get('stock_name', stock_results[0].get('stock_code'))
            subject = f"【股票分析】{stock_name} - {date_str}"
        else:
            subject = f"【股票分析报告】{len(stock_results)}只股票 - {date_str}"

        # 发送邮件
        recipients = [args.recipient] if args.recipient else None
        success = send_email(subject, "请查看股票分析报告", html_content, recipients)

        if success:
            print("✅ 邮件发送成功")
        else:
            print("❌ 邮件发送失败")
            sys.exit(1)
    else:
        # 不发送邮件，输出到文件
        output_path = f'output/stock_analysis_email_{date_str}.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n📄 邮件内容已保存到: {output_path}")

    print("\n" + "="*50)
    print("✅ 完成！")
    print("="*50)


if __name__ == "__main__":
    main()
