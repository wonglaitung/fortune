---
description: 发送股票分析邮件 - 将股票分析报告发送到指定邮箱
allowed-tools: read_file, bash
---

# 发送股票分析邮件技能

将股票分析报告以邮件形式发送到指定邮箱。

## 触发条件

用户请求发送分析报告时触发：
- "把分析发给我"
- "发送分析邮件"
- "把刚才的分析发到邮箱"
- "发送报告到 xxx@example.com"
- "邮件发送分析结果"

## 环境配置

邮件配置在 `set_key.sh` 文件中，发送邮件前需要先加载环境变量：

```bash
source set_key.sh
```

**配置的环境变量**：
- `SMTP_SERVER` - SMTP 服务器地址
- `EMAIL_SENDER` - 发件人邮箱
- `EMAIL_PASSWORD` - 邮箱密码/应用密码
- `RECIPIENT_EMAIL` - 默认收件人（多个用逗号分隔）

**注意**：敏感信息（密码等）保存在 `set_key.sh` 中，不要在技能文件中硬编码。

## 执行步骤

### 0. 加载环境变量

发送邮件前，必须先加载环境变量：

```bash
source /data/fortune/set_key.sh
```

### 1. 确认分析内容

首先确认用户要发送的分析内容：
- 如果用户刚刚查询过某只股票，使用该分析结果
- 如果用户没有指定，询问用户要发送哪只股票的分析

### 2. 获取收件人邮箱

按以下优先级获取收件人邮箱：
1. 用户在命令中指定的邮箱（如"发送到 xxx@example.com"）
2. 环境变量 `RECIPIENT_EMAIL` 中配置的默认收件人
3. 如果都没有，询问用户收件人邮箱

### 3. 构建邮件内容

使用标准化 HTML 模板渲染邮件内容。

**邮件主题格式**：
- 个股分析：`【股票分析】XXX（XXXX.HK）- YYYY-MM-DD`
- 综合推荐：`【投资建议】今日可买入股票推荐 - YYYY-MM-DD`

### 4. 发送邮件

调用邮件发送模块发送邮件。

### 5. 确认发送结果

- 发送成功：告知用户邮件已发送
- 发送失败：告知用户失败原因，并提供可能的解决方案

---

## 标准化 HTML 模板

使用以下模板渲染邮件，确保每次发送的格式一致。

### HTML 模板

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .container {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a1a1a;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 15px;
            margin-top: 0;
            font-size: 24px;
        }
        .date-info {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }
        h2 {
            color: #2c3e50;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 12px;
            font-size: 18px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: left;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .metric-good {
            color: #28a745;
            font-weight: bold;
        }
        .metric-bad {
            color: #dc3545;
            font-weight: bold;
        }
        .metric-neutral {
            color: #ffc107;
            font-weight: bold;
        }
        .recommendation {
            font-size: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: bold;
        }
        .rec-strong-buy {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .rec-buy {
            background-color: #cce5ff;
            color: #004085;
            border: 1px solid #b8daff;
        }
        .rec-hold {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }
        .rec-sell {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .warning-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #dc3545;
        }
        .info-box {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #17a2b8;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
        }
        ul {
            margin: 10px 0;
            padding-left: 25px;
        }
        li {
            margin: 8px 0;
        }
        .disclaimer {
            font-size: 12px;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
            line-height: 1.8;
        }
        .arrow-up {
            color: #28a745;
            font-weight: bold;
        }
        .arrow-down {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>【股票分析】{stock_name}（{stock_code}）</h1>
        <p class="date-info"><strong>分析日期</strong>：{analysis_date}</p>

        <div class="recommendation {rec_class}">
            综合建议：{recommendation}
        </div>

        <h2>一、核心指标</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>说明</th></tr>
            <tr><td>CatBoost 20天上涨概率</td><td class="{prob_class}">{probability}%</td><td>{prob_desc}</td></tr>
            <tr><td>当前价格</td><td>HK${price}</td><td>{price_change}</td></tr>
            <tr><td>建议仓位</td><td>{position}%</td><td></td></tr>
            <tr><td>止损位</td><td>{stop_loss}</td><td>最大亏损控制在-8%以内</td></tr>
            <tr><td>目标价</td><td>{target_price}</td><td></td></tr>
        </table>

        <h2>二、三周期预测</h2>
        <table>
            <tr><th>周期</th><th>预测概率</th><th>方向</th></tr>
            <tr><td>1天</td><td>{prob_1d}%</td><td class="{dir_1d_class}">{dir_1d}</td></tr>
            <tr><td>5天</td><td>{prob_5d}%</td><td class="{dir_5d_class}">{dir_5d}</td></tr>
            <tr><td>20天</td><td>{prob_20d}%</td><td class="{dir_20d_class}">{dir_20d}</td></tr>
        </table>
        <p><strong>三周期模式</strong>：{pattern}</p>
        <p><strong>传导模式</strong>：{conduction}</p>

        <h2>三、大模型建议</h2>
        <ul>
            <li><strong>短期建议</strong>：{short_term}</li>
            <li><strong>中期建议</strong>：{mid_term}</li>
            <li><strong>一致性</strong>：{consistency}</li>
        </ul>

        <h2>四、技术指标</h2>
        <table>
            <tr><th>指标</th><th>数值</th><th>状态</th></tr>
            <tr><td>RSI</td><td>{rsi}</td><td>{rsi_status}</td></tr>
            <tr><td>MACD</td><td>{macd}</td><td>{macd_status}</td></tr>
            <tr><td>布林带位置</td><td>{bb_pos}%</td><td>{bb_status}</td></tr>
            <tr><td>均线排列</td><td>-</td><td>{ma_status}</td></tr>
            <tr><td>筹码阻力</td><td>-</td><td>{chip_resistance}</td></tr>
        </table>

        <h2>五、风险评分</h2>
        <table>
            <tr><th>指标</th><th>得分</th></tr>
            <tr><td>风险得分</td><td>{risk_score}</td></tr>
            <tr><td>回报得分</td><td>{return_score}</td></tr>
            <tr><td>综合得分</td><td>{total_score}</td></tr>
            <tr><td>风险建议</td><td>{risk_advice}</td></tr>
        </table>

        <h2>六、市场环境</h2>
        <ul>
            <li><strong>恒生指数</strong>：{hsi_price}（{hsi_status}）</li>
            <li><strong>市场状态</strong>：{market_status}</li>
            <li><strong>状态持续时间</strong>：{duration}天（{stability}）</li>
            <li><strong>VIX</strong>：{vix}（{vix_status}）</li>
            <li><strong>市场情绪</strong>：{sentiment}</li>
        </ul>

        <h2>七、网络洞察</h2>
        <ul>
            <li><strong>社区归属</strong>：社区{community}</li>
            <li><strong>枢纽类型</strong>：{hub_type}</li>
            <li><strong>是否桥梁股</strong>：{is_bridge}</li>
            <li><strong>模块度</strong>：{modularity}（市场分化程度）</li>
        </ul>

        <h2>八、板块表现</h2>
        <ul>
            <li><strong>所属板块</strong>：{sector}</li>
            <li><strong>板块排名</strong>：第{sector_rank}名（共16个板块）</li>
            <li><strong>5日涨跌幅</strong>：{sector_change}%</li>
            <li><strong>板块类型</strong>：{sector_type}</li>
        </ul>

        <h2>九、操作建议</h2>
        {operation_box}

        <h2>十、风险提示</h2>
        <ul>
            {risk_warnings}
        </ul>

        <h2>十一、股息提醒</h2>
        <ul>
            {dividend_info}
        </ul>

        <div class="disclaimer">
            <p><strong>免责声明</strong>：以上建议仅供参考，不构成投资建议，投资有风险，决策需谨慎。</p>
            <p>本报告由港股智能分析系统自动生成，分析日期：{analysis_date}</p>
        </div>
    </div>
</body>
</html>
```

### 模板变量说明

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `{stock_name}` | 股票名称 | 中国平安 |
| `{stock_code}` | 股票代码 | 2318.HK |
| `{analysis_date}` | 分析日期 | 2026-05-20 |
| `{recommendation}` | 综合建议 | 🟡 观望 |
| `{rec_class}` | 建议样式类 | rec-hold |
| `{probability}` | CatBoost概率 | 37.12 |
| `{prob_class}` | 概率样式类 | metric-bad |
| `{prob_desc}` | 概率描述 | 看跌，<50% |
| `{price}` | 当前价格 | 61.40 |
| `{position}` | 建议仓位 | 0 |
| `{stop_loss}` | 止损位 | - |
| `{target_price}` | 目标价 | - |
| `{prob_1d}` | 1天概率 | 44 |
| `{prob_5d}` | 5天概率 | 43 |
| `{prob_20d}` | 20天概率 | 37 |
| `{dir_1d}` | 1天方向 | ↓ 下跌 |
| `{dir_1d_class}` | 方向样式 | arrow-down |
| `{pattern}` | 三周期模式 | 一致看跌(000) |
| `{conduction}` | 传导模式 | ✅ 传导 |
| `{short_term}` | 短期建议 | 观察 |
| `{mid_term}` | 中期建议 | 观察 |
| `{consistency}` | 一致性 | 一致看空 |
| `{rsi}` | RSI值 | 53.85 |
| `{rsi_status}` | RSI状态 | 中性 |
| `{macd}` | MACD值 | 0.30 |
| `{macd_status}` | MACD状态 | 弱金叉 |
| `{bb_pos}` | 布林带位置 | 28.7 |
| `{ma_status}` | 均线状态 | 震荡整理 |
| `{chip_resistance}` | 筹码阻力 | 🔴 高 |
| `{risk_score}` | 风险得分 | 67.5 |
| `{return_score}` | 回报得分 | 51.0 |
| `{total_score}` | 综合得分 | 59.2 |
| `{risk_advice}` | 风险建议 | 🟡 观察 |
| `{hsi_price}` | 恒指价格 | 25797.85 |
| `{hsi_status}` | 恒指状态 | 震荡偏跌 |
| `{market_status}` | 市场状态 | 震荡偏跌 |
| `{duration}` | 持续时间 | 5 |
| `{stability}` | 稳定性 | 中等稳定 |
| `{vix}` | VIX值 | 18.11 |
| `{vix_status}` | VIX状态 | 正常 |
| `{sentiment}` | 市场情绪 | 正常 |
| `{community}` | 社区编号 | 5 |
| `{hub_type}` | 枢纽类型 | 高枢纽 |
| `{is_bridge}` | 是否桥梁股 | ⚠️ 是 |
| `{modularity}` | 模块度 | 0.5038 |
| `{sector}` | 所属板块 | 保险股 |
| `{sector_rank}` | 板块排名 | 12 |
| `{sector_change}` | 板块涨跌 | -6.20 |
| `{sector_type}` | 板块类型 | 防御 |
| `{operation_box}` | 操作建议框 | `<div class="warning-box">...</div>` |
| `{risk_warnings}` | 风险提示列表 | `<li>...</li>` |
| `{dividend_info}` | 股息信息列表 | `<li>...</li>` |

### 样式类说明

| 类名 | 用途 |
|------|------|
| `metric-good` | 好指标（绿色） |
| `metric-bad` | 差指标（红色） |
| `metric-neutral` | 中等指标（黄色） |
| `rec-strong-buy` | 强烈买入样式（绿色背景） |
| `rec-buy` | 买入样式（蓝色背景） |
| `rec-hold` | 观望样式（黄色背景） |
| `rec-sell` | 卖出样式（红色背景） |
| `warning-box` | 警告框（红色边框） |
| `success-box` | 成功框（绿色边框） |
| `info-box` | 信息框（蓝色边框） |
| `arrow-up` | 上涨箭头（绿色） |
| `arrow-down` | 下跌箭头（红色） |

---

## 发送邮件代码示例

```bash
source /data/fortune/set_key.sh

python3 << 'EOF'
import sys
sys.path.insert(0, '/data/fortune')
from message_services.email_sender import send_email

# 替换模板变量生成 HTML
html_template = """上面的 HTML 模板"""

# 替换变量
html_content = html_template.replace("{stock_name}", "中国平安")
html_content = html_content.replace("{stock_code}", "2318.HK")
# ... 替换其他变量

subject = "【股票分析】中国平安（2318.HK）- 2026-05-20"
content = "请查看股票分析报告"

send_email(subject, content, html_content, recipients=["user@example.com"])
EOF
```

---

## 注意事项

1. **邮件配置**：确保环境变量已正确配置（SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, RECIPIENT_EMAIL）
2. **HTML 格式**：邮件使用 HTML 格式，确保邮件客户端支持 HTML 渲染
3. **编码问题**：使用 UTF-8 编码，确保中文正确显示
4. **发送失败处理**：如果发送失败，检查邮件配置和网络连接
5. **隐私保护**：不要在日志中记录邮件内容和收件人地址
6. **模板一致性**：每次发送使用相同的 HTML 模板，确保格式统一

## 示例用法

用户说："把刚才的平安分析发到我的邮箱"

执行：
1. 加载环境变量 `source set_key.sh`
2. 获取刚才的中国平安分析内容
3. 使用标准化模板渲染 HTML
4. 发送邮件
5. 确认发送结果
