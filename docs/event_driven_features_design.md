# 事件驱动特征设计方案

## 概述
基于现有代码结构，设计15-20个事件驱动特征，用于捕捉财报公告、分红事件、重大公告等对股价的影响。

## 数据源
1. **现有数据源**
   - AKShare：财报指标数据（`ak.stock_hk_financial_indicator_em`）
   - AKShare：股息数据（`ak.stock_hk_dividend_payout_em`）
   - 新闻情感分析（`llm_services/sentiment_analyzer.py`）

2. **待补充数据源**
   - 财报公告日数据：雅虎财经/东方财富/新浪财经
   - 重大事件公告：新闻关键词识别 + NLP分类

## 特征分类

### 1. 财报公告日特征（2-3个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Earnings_Announcement_In_7d` | Binary | 未来7天内是否有财报公告（0/1） | 财报公告日API |
| `Earnings_Announcement_In_30d` | Binary | 未来30天内是否有财报公告（0/1） | 财报公告日API |
| `Days_Since_Last_Earnings` | Continuous | 距离上次财报公告的天数（1-120+） | 财报公告日API |

### 2. 财报超预期/不及预期特征（3-4个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Earnings_Surprise_Score` | Continuous | 财报超预期评分（-1到+1，基于新闻情感） | 新闻情感分析 |
| `Earnings_Surprise_Trend` | Continuous | 近期财报超预期趋势（过去3次平均） | 新闻情感历史 |
| `Post_Earnings_Price_Change_3d` | Continuous | 财报后3日股价变化率（%） | 历史股价数据 |
| `Post_Earnings_Price_Change_10d` | Continuous | 财报后10日股价变化率（%） | 历史股价数据 |

### 3. 除净日和分红公告特征（3-4个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Ex_Dividend_In_7d` | Binary | 未来7天内是否有除净日（0/1） | AKShare股息数据 |
| `Ex_Dividend_In_30d` | Binary | 未来30天内是否有除净日（0/1） | AKShare股息数据 |
| `Dividend_Yield_Ratio` | Continuous | 股息率相对于板块平均值（0.5-2.0） | 财务数据 + 板块平均 |
| `Dividend_Frequency_12m` | Continuous | 过去12个月分红次数（1-12） | AKShare股息历史 |

### 4. 重大事件公告特征（4-5个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Major_Event_In_7d` | Binary | 未来7天内重大事件（0/1） | 新闻关键词识别 |
| `M_A_Announcement` | Binary | 近30天并购事件公告（0/1） | 新闻NLP分类 |
| `Buyback_Announcement` | Binary | 近30天股票回购公告（0/1） | 新闻NLP分类 |
| `Governance_Event` | Binary | 近30天公司治理事件（0/1） | 新闻NLP分类 |
| `Regulatory_Event` | Binary | 近30天监管/合规事件（0/1） | 新闻NLP分类 |

## 实现方案

### 阶段3.1：基于现有数据源的特征（优先级：高）
先实现可直接从现有数据源获取的特征：
1. 除净日和分红公告特征（AKShare股息数据）
2. 财报超预期评分特征（新闻情感分析）

### 阶段3.2：需补充数据源的特征（优先级：中）
实现需要补充数据源的特征：
1. 财报公告日特征（雅虎财经/东方财富API）
2. 财报后股价变化特征（历史股价数据）

### 阶段3.3：NLP分类特征（优先级：低）
实现需要NLP分类的特征：
1. 重大事件公告特征（新闻关键词识别 + LLM分类）

## 数据获取策略

### 财报公告日数据
**优先方案**：雅虎财经API
```python
import yfinance as yf

def get_earnings_dates(stock_code):
    """
    使用雅虎财经获取财报公告日
    """
    symbol = stock_code.replace('.HK', '').lstrip('0') + '.HK'
    ticker = yf.Ticker(symbol)
    # 尝试获取财报日历
    calendar = ticker.calendar
    return calendar
```

**备选方案1**：东方财富网爬虫
```python
import akshare as ak

def get_earnings_announcements_eastmoney(stock_code):
    """
    使用东方财富网获取财报公告
    """
    # 东方财富可能提供财报公告日API
    # 需要进一步调研
    pass
```

**备选方案2**：新浪财经爬虫
```python
import requests
from bs4 import BeautifulSoup

def get_earnings_announcements_sina(stock_code):
    """
    使用新浪财经获取财报公告
    """
    # 新浪财经财经日历可能提供财报信息
    # 需要进一步调研
    pass
```

### 重大事件公告识别
**方案**：关键词匹配 + LLM分类
```python
from llm_services.qwen_engine import QwenEngine

MA_KEYWORDS = ['收购', '并购', 'M&A', 'acquisition', 'merger']
BUYBACK_KEYWORDS = ['回购', 'buyback', 'share repurchase']
GOVERNANCE_KEYWORDS = ['罢免', '董事会', '治理', 'governance', 'board']
REGULATORY_KEYWORDS = ['处罚', '监管', '合规', 'regulatory', 'compliance']

def classify_major_event(news_text):
    """
    使用LLM分类重大事件类型
    """
    prompt = f"""
    判断以下新闻属于哪种类型：
    1. 并购（M&A）
    2. 股票回购（Buyback）
    3. 公司治理（Governance）
    4. 监管/合规（Regulatory）
    5. 其他

    新闻内容：{news_text}

    返回类型编号。
    """
    # 使用QwenEngine进行分类
    pass
```

## 特征工程实现

### 在ml_trading_model.py中添加方法

```python
def create_event_driven_features(self, code, df):
    """
    创建事件驱动特征（15-20个）

    参数:
    - code: 股票代码
    - df: 股票数据DataFrame

    返回:
    - df: 添加事件驱动特征的DataFrame
    """
    if len(df) < 30:  # 需要足够的历史数据
        return df

    # ========== 阶段3.1：除净日和分红特征 ==========
    try:
        # 获取股息信息
        dividend_info = self._get_dividend_calendar(code)

        if dividend_info is not None and not dividend_info.empty:
            df = self._add_dividend_features(df, dividend_info)
    except Exception as e:
        print(f"  ⚠️ 添加分红特征失败: {e}")

    # ========== 阶段3.1：财报超预期特征 ==========
    try:
        # 获取财报新闻情感
        earnings_sentiment = self._get_earnings_sentiment(code)

        if earnings_sentiment is not None:
            df = self._add_earnings_surprise_features(df, earnings_sentiment)
    except Exception as e:
        print(f"  ⚠️ 添加财报超预期特征失败: {e}")

    # ========== 阶段3.2：财报公告日特征 ==========
    try:
        # 获取财报公告日
        earnings_calendar = self._get_earnings_calendar(code)

        if earnings_calendar is not None and not earnings_calendar.empty:
            df = self._add_earnings_date_features(df, earnings_calendar)
    except Exception as e:
        print(f"  ⚠️ 添加财报公告日特征失败: {e}")

    # ========== 阶段3.3：重大事件特征 ==========
    try:
        # 获取重大事件
        major_events = self._get_major_events(code)

        if major_events is not None:
            df = self._add_major_event_features(df, major_events)
    except Exception as e:
        print(f"  ⚠️ 添加重大事件特征失败: {e}")

    return df


def _get_dividend_calendar(self, code):
    """
    获取股息日历（复用hsi_email.py的get_dividend_info方法）
    """
    try:
        import akshare as ak

        symbol = code.replace('.HK', '')
        if len(symbol) < 5:
            symbol = symbol.zfill(5)
        elif len(symbol) > 5:
            symbol = symbol[-5:]

        df_dividend = ak.stock_hk_dividend_payout_em(symbol=symbol)

        if df_dividend is None or df_dividend.empty:
            return None

        # 提取关键列
        result = []
        for _, row in df_dividend.iterrows():
            ex_date = row.get('除净日', None)
            if pd.notna(ex_date):
                result.append({
                    '除净日': ex_date,
                    '分红方案': row.get('分红方案', None),
                    '财政年度': row.get('财政年度', None)
                })

        return pd.DataFrame(result)
    except Exception as e:
        print(f"  ⚠️ 获取股息日历失败: {e}")
        return None


def _add_dividend_features(self, df, dividend_info):
    """
    添加除净日和分红特征（3-4个特征）
    """
    # 确保日期索引是datetime类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 转换除净日为datetime
    dividend_info['除净日'] = pd.to_datetime(dividend_info['除净日'])

    # 特征1：未来7天内是否有除净日
    df['Ex_Dividend_In_7d'] = df.index.to_series().apply(
        lambda x: any(0 <= (date - x).days <= 7 for date in dividend_info['除净日'])
    ).astype(int)

    # 特征2：未来30天内是否有除净日
    df['Ex_Dividend_In_30d'] = df.index.to_series().apply(
        lambda x: any(0 <= (date - x).days <= 30 for date in dividend_info['除净日'])
    ).astype(int)

    # 特征3：股息率（从基本面数据获取）
    # 这里需要与基本面数据集成
    # 暂时使用占位符
    df['Dividend_Yield'] = 0.0

    # 特征4：过去12个月分红次数
    df['Dividend_Frequency_12m'] = df.index.to_series().apply(
        lambda x: sum(1 for date in dividend_info['除净日'] if -365 <= (date - x).days <= 0)
    )

    return df


def _get_earnings_calendar(self, code):
    """
    获取财报公告日（使用雅虎财经）
    """
    try:
        import yfinance as yf

        symbol = code.replace('.HK', '').lstrip('0') + '.HK'
        ticker = yf.Ticker(symbol)

        # 尝试获取财报日历
        calendar = ticker.calendar

        if calendar is None or calendar.empty:
            return None

        return calendar
    except Exception as e:
        print(f"  ⚠️ 获取财报公告日失败: {e}")
        return None


def _add_earnings_date_features(self, df, earnings_calendar):
    """
    添加财报公告日特征（2-3个特征）
    """
    # 特征1：未来7天内是否有财报公告
    if 'Earnings Date' in earnings_calendar.columns:
        df['Earnings_Announcement_In_7d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 7 for date in earnings_calendar['Earnings Date'])
        ).astype(int)

        # 特征2：未来30天内是否有财报公告
        df['Earnings_Announcement_In_30d'] = df.index.to_series().apply(
            lambda x: any(0 <= (date - x).days <= 30 for date in earnings_calendar['Earnings Date'])
        ).astype(int)

    # 特征3：距离上次财报公告的天数
    if 'Earnings Date' in earnings_calendar.columns:
        df['Days_Since_Last_Earnings'] = df.index.to_series().apply(
            lambda x: min(
                [(x - date).days for date in earnings_calendar['Earnings Date'] if (x - date).days >= 0],
                default=120
            )
        )

    return df
```

## 实现优先级

### 第一优先级（立即实现）
1. 除净日和分红特征（已有数据源）
   - `Ex_Dividend_In_7d`
   - `Ex_Dividend_In_30d`
   - `Dividend_Frequency_12m`

### 第二优先级（调研后实现）
2. 财报公告日特征（需补充数据源）
   - `Earnings_Announcement_In_7d`
   - `Earnings_Announcement_In_30d`
   - `Days_Since_Last_Earnings`

3. 财报超预期特征（需新闻情感分析）
   - `Earnings_Surprise_Score`
   - `Earnings_Surprise_Trend`

### 第三优先级（可选实现）
4. 重大事件特征（需NLP分类）
   - `M_A_Announcement`
   - `Buyback_Announcement`
   - `Governance_Event`
   - `Regulatory_Event`

## 预期效果

1. **除净日特征**：
   - 捕捉除净日前的买入机会（高股息策略）
   - 预期提升银行股、能源股的预测准确率

2. **财报公告日特征**：
   - 捕捉财报前的波动性
   - 预期提升财报季期间的模型稳定性

3. **重大事件特征**：
   - 识别并购、回购等催化剂
   - 预期提升突发事件的预测能力

## 风险与挑战

1. **数据源可靠性**：
   - 雅虎财经数据可能不全
   - 需要多个数据源备份

2. **数据泄漏风险**：
   - 必须使用历史数据
   - 避免使用未来信息

3. **特征稀疏性**：
   - 事件特征多为稀疏特征（0/1）
   - 需要与连续特征结合使用

4. **NLP分类准确性**：
   - 重大事件分类依赖LLM
   - 需要建立标注数据集

## 下一步行动

1. 实现第一优先级特征（除净日和分红特征）
2. 测试雅虎财经API获取财报公告日
3. 集成新闻情感分析计算财报超预期评分
4. 运行Walk-forward验证评估新特征效果