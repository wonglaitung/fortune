# 特征工程完整指南

> **最后更新**：2026-04-27

---

## 📋 目录

1. [特征概览](#特征概览)
2. [特征类别详解](#特征类别详解)
3. [全量特征 vs 500特征对比](#全量特征-vs-500特征对比)
4. [特征选择策略](#特征选择策略)
5. [特征重要性分析](#特征重要性分析)
6. [事件驱动特征](#事件驱动特征)
7. [特征工程最佳实践](#特征工程最佳实践)

---

## 特征概览

### 特征数量统计

| 特征类别 | 数量 | 说明 |
|---------|------|------|
| **滚动统计特征** | 126 | 偏度、峰度、多周期波动率 |
| **价格形态特征** | 84 | 日内振幅、影线比例、缺口 |
| **量价关系特征** | 98 | 背离、OBV、成交量波动率 |
| **长期趋势特征** | 84 | MA120/250、长期收益率、长期RSI |
| **主题分布特征** | 10 | LDA主题建模（10个主题概率） |
| **主题情感交互特征** | 50 | 10个主题 × 5个情感指标 |
| **预期差距特征** | 5 | 新闻情感相对于市场预期的差距 |
| **市场环境自适应特征** | 8 | ADX+波动率双因子识别 |
| **风险管理特征** | 18 | ATR动态止损、连续市场状态记忆、盈亏比评估 |
| **事件驱动特征** | 9 | 分红、财报日期、财报超预期 |
| **股票类型特征** | 128 | 技术指标、基本面、市场环境等 |
| **GARCH 波动率特征** | 4 | 条件波动率、波动率比率、波动率变化、持续性参数 |
| **HSI 市场状态特征** | 6 | HMM 市场状态检测（状态标签、概率、持续时间、转换概率） |
| **日历效应特征** | 22 | 星期效应、月份效应、期权到期日、月初/月末等 |
| **其他衍生特征** | 280 | 交叉特征、滞后特征等 |
| **总计** | **1045** | **全量特征（2026-04-27 更新）** |

### 特征分类原则

1. **时间序列特征**：基于历史价格、成交量数据计算
2. **市场环境特征**：基于市场状态、波动率等宏观指标
3. **基本面特征**：基于公司财务数据
4. **情感特征**：基于新闻情感、主题分析
5. **事件驱动特征**：基于分红、财报等公司事件

---

## 特征类别详解

### 1. 滚动统计特征（126个）

#### 目的
捕捉价格和成交量的统计特性，识别市场异常状态

#### 特征列表

**价格统计特征（42个）**：
- `Close_Rolling_Skew_5d` / `10d` / `20d` / `60d`：滚动偏度
- `Close_Rolling_Kurtosis_5d` / `10d` / `20d` / `60d`：滚动峰度
- `Close_Rolling_Std_5d` / `10d` / `20d` / `60d`：滚动标准差
- `Close_Rolling_Volatility_5d` / `10d` / `20d` / `60d`：滚动波动率
- `Close_Rolling_Mean_5d` / `10d` / `20d` / `60d`：滚动均值
- `Close_Rolling_Median_5d` / `10d` / `20d` / `60d`：滚动中位数
- `Close_Rolling_Max_5d` / `10d` / `20d` / `60d`：滚动最大值
- `Close_Rolling_Min_5d` / `10d` / `20d` / `60d`：滚动最小值
- `Close_Rolling_Range_5d` / `10d` / `20d` / `60d`：滚动范围（Max-Min）
- `Close_Percent_Change_5d` / `10d` / `20d` / `60d`：百分比变化

**成交量统计特征（42个）**：
- `Volume_Rolling_Skew_5d` / `10d` / `20d` / `60d`：成交量滚动偏度
- `Volume_Rolling_Kurtosis_5d` / `10d` / `20d` / `60d`：成交量滚动峰度
- `Volume_Rolling_Std_5d` / `10d` / `20d` / `60d`：成交量滚动标准差
- `Volume_Rolling_Mean_5d` / `10d` / `20d` / `60d`：成交量滚动均值
- `Volume_Rolling_Median_5d` / `10d` / `20d` / `60d`：成交量滚动中位数
- `Volume_Rolling_Max_5d` / `10d` / `20d` / `60d`：成交量滚动最大值
- `Volume_Rolling_Min_5d` / `10d` / `20d` / `60d`：成交量滚动最小值
- `Volume_Rolling_Range_5d` / `10d` / `20d` / `60d`：成交量滚动范围
- `Volume_Percent_Change_5d` / `10d` / `20d` / `60d`：成交量百分比变化
- `Volume_Ratio_5d` / `10d` / `20d` / `60d`：成交量比率（相对于均值）

**多周期波动率特征（42个）**：
- `Close_Volatility_5d_10d`：5天 vs 10天波动率差异
- `Close_Volatility_10d_20d`：10天 vs 20天波动率差异
- `Close_Volatility_20d_60d`：20天 vs 60天波动率差异
- `Close_Volatility_Ratio_5d_10d`：5天 vs 10天波动率比率
- `Close_Volatility_Ratio_10d_20d`：10天 vs 20天波动率比率
- `Close_Volatility_Ratio_20d_60d`：20天 vs 60天波动率比率
- `Volume_Volatility_5d_10d`：成交量5天 vs 10天波动率差异
- `Volume_Volatility_10d_20d`：成交量10天 vs 20天波动率差异
- `Volume_Volatility_20d_60d`：成交量20天 vs 60天波动率差异
- （其他多周期对比特征）

#### 计算方法

```python
def create_rolling_statistical_features(df, windows=[5, 10, 20, 60]):
    """
    创建滚动统计特征
    """
    # 价格统计特征
    for window in windows:
        df[f'Close_Rolling_Mean_{window}d'] = df['Close'].rolling(window).mean()
        df[f'Close_Rolling_Std_{window}d'] = df['Close'].rolling(window).std()
        df[f'Close_Rolling_Skew_{window}d'] = df['Close'].rolling(window).skew()
        df[f'Close_Rolling_Kurtosis_{window}d'] = df['Close'].rolling(window).kurt()
        df[f'Close_Percent_Change_{window}d'] = df['Close'].pct_change(window)
    
    # 多周期波动率特征
    df['Close_Volatility_5d_10d'] = df['Close_Rolling_Std_5d'] / df['Close_Rolling_Std_10d']
    df['Close_Volatility_10d_20d'] = df['Close_Rolling_Std_10d'] / df['Close_Rolling_Std_20d']
    
    return df
```

---

### 2. 价格形态特征（84个）

#### 目的
捕捉K线形态、日内波动、缺口等价格模式

#### 特征列表

**K线形态特征（28个）**：
- `Upper_Shadow_Ratio`：上影线比例（(High - Max(Open, Close)) / (High - Low)）
- `Lower_Shadow_Ratio`：下影线比例（(Min(Open, Close) - Low) / (High - Low)）
- `Body_Ratio`：实体比例（Abs(Close - Open) / (High - Low)）
- `Doji_Signal`：十字星信号（Body_Ratio < 0.1）
- `Hammer_Signal`：锤子线信号（下影线 > 2×实体 且实体在顶部）
- `Shooting_Star_Signal`：射击之星信号（上影线 > 2×实体 且实体在底部）
- `Engulfing_Bullish_Signal`：看涨吞没信号
- `Engulfing_Bearish_Signal`：看跌吞没信号

**日内波动特征（28个）**：
- `Intraday_Range`：日内振幅（High - Low）
- `Intraday_Range_Ratio`：日内振幅比率（Intraday_Range / Close）
- `Intraday_Volatility_5d`：日内波动率5天均值
- `Intraday_Volatility_10d`：日内波动率10天均值
- `Intraday_Volatility_20d`：日内波动率20天均值
- `Intraday_Volatility_Ratio_5d`：当日波动率 / 5天均值
- `Intraday_Volatility_Ratio_10d`：当日波动率 / 10天均值
- `Intraday_Volatility_Ratio_20d`：当日波动率 / 20天均值

**缺口特征（28个）**：
- `Gap_Up`：向上缺口（Low > Prev_High）
- `Gap_Down`：向下缺口（High < Prev_Low）
- `Gap_Up_Size`：向上缺口大小（Low - Prev_High）
- `Gap_Down_Size`：向下缺口大小（Prev_Low - High）
- `Gap_Up_Ratio`：向上缺口比率（Gap_Up_Size / Prev_Close）
- `Gap_Down_Ratio`：向下缺口比率（Gap_Down_Size / Prev_Close）
- `Gap_Up_Count_5d`：过去5天向上缺口数量
- `Gap_Down_Count_5d`：过去5天向下缺口数量
- `Gap_Up_Count_10d`：过去10天向上缺口数量
- `Gap_Down_Count_10d`：过去10天向下缺口数量

#### 计算方法

```python
def create_price_pattern_features(df):
    """
    创建价格形态特征
    """
    # K线形态特征
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Total_Range'] = df['High'] - df['Low']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Total_Range']
    df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Total_Range']
    df['Body_Ratio'] = df['Body_Size'] / df['Total_Range']
    
    # 缺口特征
    df['Gap_Up'] = (df['Low'] > df['High'].shift(1)).astype(int)
    df['Gap_Down'] = (df['High'] < df['Low'].shift(1)).astype(int)
    df['Gap_Up_Size'] = df['Low'] - df['High'].shift(1)
    df['Gap_Down_Size'] = df['Low'].shift(1) - df['High']
    
    # 滚动统计
    df['Gap_Up_Count_5d'] = df['Gap_Up'].rolling(5).sum()
    df['Gap_Down_Count_5d'] = df['Gap_Down'].rolling(5).sum()
    
    return df
```

---

### 3. 量价关系特征（98个）

#### 目的
捕捉价格与成交量的协同变化，识别背离、突破等信号

#### 特征列表

**OBV（能量潮）特征（14个）**：
- `OBV`：能量潮指标
- `OBV_MA_5d`：OBV 5日均线
- `OBV_MA_10d`：OBV 10日均线
- `OBV_MA_20d`：OBV 20日均线
- `OBV_Change_5d`：OBV 5天变化
- `OBV_Change_10d`：OBV 10天变化
- `OBV_Change_20d`：OBV 20天变化
- `OBV_Divergence_Price_Up`：价格上涨但OBV下跌（背离）
- `OBV_Divergence_Price_Down`：价格下跌但OBV上涨（背离）

**成交量背离特征（14个）**：
- `Price_Up_Volume_Down`：价格上涨但成交量下跌
- `Price_Down_Volume_Up`：价格下跌但成交量上涨
- `Price_Up_Volume_Down_5d`：过去5天价格上涨但成交量下跌次数
- `Price_Down_Volume_Up_5d`：过去5天价格下跌但成交量上涨次数
- `Price_Volume_Correlation_10d`：价格与成交量10天相关系数
- `Price_Volume_Correlation_20d`：价格与成交量20天相关系数

**成交量波动率特征（14个）**：
- `Volume_Change_1d`：成交量1天变化
- `Volume_Change_5d`：成交量5天变化
- `Volume_Change_10d`：成交量10天变化
- `Volume_Volatility_5d`：成交量5天波动率
- `Volume_Volatility_10d`：成交量10天波动率
- `Volume_Volatility_20d`：成交量20天波动率
- `Volume_Volatility_Ratio_5d`：当日成交量波动率 / 5天均值

**量价确认特征（14个）**：
- `Breakout_Volume_Confirmation`：突破伴随成交量确认（>1.2倍均量）
- `Breakdown_Volume_Confirmation`：跌破伴随成交量确认（>1.2倍均量）
- `Volume_Strength_5d`：成交量强度5天均值
- `Volume_Strength_10d`：成交量强度10天均值
- `Volume_Strength_20d`：成交量强度20天均值

**其他量价特征（42个）**：
- `Volume_MA_5d` / `10d` / `20d` / `60d`：成交量移动平均
- `Volume_Ratio_MA_5d` / `10d` / `20d`：成交量相对于均线比率
- `Price_Volume_Trend_5d`：价格与成交量趋势一致性
- `Price_Volume_Trend_10d`：价格与成交量趋势一致性
- `Price_Volume_Trend_20d`：价格与成交量趋势一致性

#### 计算方法

```python
def create_price_volume_features(df):
    """
    创建量价关系特征
    """
    # OBV特征
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA_5d'] = df['OBV'].rolling(5).mean()
    df['OBV_MA_10d'] = df['OBV'].rolling(10).mean()
    df['OBV_MA_20d'] = df['OBV'].rolling(20).mean()
    
    # 成交量背离特征
    df['Price_Up'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['Price_Down'] = (df['Close'] < df['Close'].shift(1)).astype(int)
    df['Volume_Down'] = (df['Volume'] < df['Volume'].shift(1)).astype(int)
    df['Volume_Up'] = (df['Volume'] > df['Volume'].shift(1)).astype(int)
    
    df['Price_Up_Volume_Down'] = ((df['Price_Up'] == 1) & (df['Volume_Down'] == 1)).astype(int)
    df['Price_Down_Volume_Up'] = ((df['Price_Down'] == 1) & (df['Volume_Up'] == 1)).astype(int)
    
    # 成交量波动率特征
    df['Volume_Change_5d'] = df['Volume'].pct_change(5)
    df['Volume_Volatility_5d'] = df['Volume'].rolling(5).std()
    df['Volume_Volatility_Ratio_5d'] = df['Volume_Volatility_5d'] / df['Volume_Volatility_5d'].rolling(20).mean()
    
    # 量价确认特征
    df['Volume_MA_5d'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_10d'] = df['Volume'].rolling(10).mean()
    df['Breakout_Volume_Confirmation'] = (df['Volume'] > 1.2 * df['Volume_MA_5d']).astype(int)
    
    return df
```

---

### 4. 长期趋势特征（84个）

#### 目的
捕捉长期趋势、支撑阻力位、长期动量

#### 特征列表

**长期移动平均特征（28个）**：
- `MA_60d` / `MA_120d` / `MA_250d`：60日、120日、250日移动平均
- `MA_60d_Ratio` / `MA_120d_Ratio` / `MA_250d_Ratio`：相对于MA的比率
- `Price_Above_MA_60d`：价格高于60日均线
- `Price_Above_MA_120d`：价格高于120日均线
- `Price_Above_MA_250d`：价格高于250日均线
- `MA_60d_120d_Diff`：60日 vs 120日均线差异
- `MA_120d_250d_Diff`：120日 vs 250日均线差异
- `MA_60d_120d_Ratio`：60日 vs 120日均线比率
- `MA_120d_250d_Ratio`：120日 vs 250日均线比率

**长期收益率特征（28个）**：
- `Return_60d` / `Return_120d` / `Return_250d`：60日、120日、250日收益率
- `Return_60d_Annualized`：60日年化收益率
- `Return_120d_Annualized`：120日年化收益率
- `Return_250d_Annualized`：250日年化收益率
- `Return_60d_Rank_Percentile`：60日收益率百分位排名
- `Return_120d_Rank_Percentile`：120日收益率百分位排名
- `Return_250d_Rank_Percentile`：250日收益率百分位排名

**长期波动率特征（14个）**：
- `Volatility_60d` / `Volatility_120d` / `Volatility_250d`：长期波动率
- `Volatility_60d_Annualized`：60日年化波动率
- `Volatility_120d_Annualized`：120日年化波动率
- `Volatility_250d_Annualized`：250日年化波动率
- `Volatility_60d_120d_Ratio`：60日 vs 120日波动率比率

**长期支撑阻力位特征（14个）**：
- `Support_60d_Low`：60日最低价（支撑位）
- `Resistance_60d_High`：60日最高价（阻力位）
- `Support_120d_Low`：120日最低价（支撑位）
- `Resistance_120d_High`：120日最高价（阻力位）
- `Distance_to_Support_60d`：价格距离60日支撑位
- `Distance_to_Resistance_60d`：价格距离60日阻力位
- `Support_Break_60d`：跌破60日支撑位
- `Resistance_Break_60d`：突破60日阻力位

#### 计算方法

```python
def create_long_term_trend_features(df):
    """
    创建长期趋势特征
    """
    # 长期移动平均特征
    df['MA_60d'] = df['Close'].rolling(60).mean()
    df['MA_120d'] = df['Close'].rolling(120).mean()
    df['MA_250d'] = df['Close'].rolling(250).mean()
    
    df['Price_Above_MA_60d'] = (df['Close'] > df['MA_60d']).astype(int)
    df['Price_Above_MA_120d'] = (df['Close'] > df['MA_120d']).astype(int)
    df['Price_Above_MA_250d'] = (df['Close'] > df['MA_250d']).astype(int)
    
    # 长期收益率特征
    df['Return_60d'] = df['Close'].pct_change(60)
    df['Return_120d'] = df['Close'].pct_change(120)
    df['Return_250d'] = df['Close'].pct_change(250)
    
    df['Return_60d_Annualized'] = df['Return_60d'] * (365 / 60)
    df['Return_120d_Annualized'] = df['Return_120d'] * (365 / 120)
    df['Return_250d_Annualized'] = df['Return_250d'] * (365 / 250)
    
    # 长期支撑阻力位特征
    df['Support_60d_Low'] = df['Low'].rolling(60).min()
    df['Resistance_60d_High'] = df['High'].rolling(60).max()
    df['Support_120d_Low'] = df['Low'].rolling(120).min()
    df['Resistance_120d_High'] = df['High'].rolling(120).max()
    
    df['Distance_to_Support_60d'] = (df['Close'] - df['Support_60d_Low']) / df['Close']
    df['Distance_to_Resistance_60d'] = (df['Resistance_60d_High'] - df['Close']) / df['Close']
    
    df['Support_Break_60d'] = (df['Close'] < df['Support_60d_Low'].shift(1)).astype(int)
    df['Resistance_Break_60d'] = (df['Close'] > df['Resistance_60d_High'].shift(1)).astype(int)
    
    return df
```

---

### 5. 主题分布特征（10个）

#### 目的
捕捉新闻主题分布，识别市场热点

#### 特征列表

- `Topic_0_Prob` / `Topic_1_Prob` / ... / `Topic_9_Prob`：10个主题的概率分布
- `Dominant_Topic`：主导主题（概率最高的主题）

#### 计算方法

```python
def create_topic_distribution_features(news_data):
    """
    创建主题分布特征（使用LDA主题建模）
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    # 文本向量化
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(news_data['text'])
    
    # LDA主题建模
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(X)
    
    # 获取主题分布
    topic_probs = lda.transform(X)
    
    # 创建特征
    for i in range(10):
        news_data[f'Topic_{i}_Prob'] = topic_probs[:, i]
    
    news_data['Dominant_Topic'] = np.argmax(topic_probs, axis=1)
    
    return news_data
```

---

### 6. 主题情感交互特征（50个）

#### 目的
捕捉主题与情感的交互关系

#### 特征列表

**主题情感特征（10个主题 × 5个情感维度 = 50个）**：
- `Topic_0_Sentiment_Positive`：主题0正面情感
- `Topic_0_Sentiment_Negative`：主题0负面情感
- `Topic_0_Sentiment_Neutral`：主题0中性情感
- `Topic_0_Sentiment_Compound`：主题0综合情感
- `Topic_0_Sentiment_Count`：主题0新闻数量
- `Topic_1_Sentiment_Positive` / ... / `Topic_9_Sentiment_Count`：其他主题相同特征

#### 计算方法

```python
def create_topic_sentiment_interaction_features(news_data):
    """
    创建主题情感交互特征
    """
    from sentiment_analyzer import SentimentAnalyzer
    
    sentiment_analyzer = SentimentAnalyzer()
    
    for i in range(10):
        topic_news = news_data[news_data['Dominant_Topic'] == i]
        
        if len(topic_news) > 0:
            sentiments = topic_news['text'].apply(sentiment_analyzer.analyze)
            
            news_data[f'Topic_{i}_Sentiment_Positive'] = sentiments.apply(lambda x: x['pos'])
            news_data[f'Topic_{i}_Sentiment_Negative'] = sentiments.apply(lambda x: x['neg'])
            news_data[f'Topic_{i}_Sentiment_Neutral'] = sentiments.apply(lambda x: x['neu'])
            news_data[f'Topic_{i}_Sentiment_Compound'] = sentiments.apply(lambda x: x['compound'])
            news_data[f'Topic_{i}_Sentiment_Count'] = len(topic_news)
        else:
            news_data[f'Topic_{i}_Sentiment_Positive'] = 0
            news_data[f'Topic_{i}_Sentiment_Negative'] = 0
            news_data[f'Topic_{i}_Sentiment_Neutral'] = 0
            news_data[f'Topic_{i}_Sentiment_Compound'] = 0
            news_data[f'Topic_{i}_Sentiment_Count'] = 0
    
    return news_data
```

---

### 7. 预期差距特征（5个）

#### 目的
捕捉新闻情感相对于市场预期的差距

#### 特征列表

- `Sentiment_Expectation_Gap_Positive`：正面情感预期差距
- `Sentiment_Expectation_Gap_Negative`：负面情感预期差距
- `Sentiment_Expectation_Gap_Compound`：综合情感预期差距
- `Sentiment_Expectation_Gap_Absolute`：预期差距绝对值
- `Sentiment_Expectation_Gap_Sign`：预期差距符号（+1/-1）

#### 计算方法

```python
def create_sentiment_expectation_gap_features(news_data, market_sentiment_baseline=0.0):
    """
    创建情感预期差距特征
    """
    news_data['Sentiment_Expectation_Gap_Compound'] = news_data['Sentiment_Compound'] - market_sentiment_baseline
    news_data['Sentiment_Expectation_Gap_Positive'] = news_data['Sentiment_Positive'] - 0.5
    news_data['Sentiment_Expectation_Gap_Negative'] = news_data['Sentiment_Negative'] - 0.5
    news_data['Sentiment_Expectation_Gap_Absolute'] = abs(news_data['Sentiment_Expectation_Gap_Compound'])
    news_data['Sentiment_Expectation_Gap_Sign'] = np.sign(news_data['Sentiment_Expectation_Gap_Compound'])
    
    return news_data
```

---

### 8. 市场环境自适应特征（8个）

#### 目的
识别市场状态（震荡市/正常市/趋势市），动态调整策略

#### 特征列表

- `Market_Regime`：市场状态（ranging/normal/trending）
- `Market_Regime_Ranging_Days`：震荡市持续天数
- `Market_Regime_Trending_Days`：趋势市持续天数
- `Volume_Confirmation_Adaptive`：自适应成交量确认
- `False_Breakout_Signal_Adaptive`：自适应假突破检测
- `Confidence_Threshold_Multiplier`：置信度阈值乘数
- `ATR_Risk_Score`：ATR风险评分
- `Market_Volatility_Level`：市场波动率水平（low/medium/high）

#### 计算方法

```python
def create_market_regime_features(df):
    """
    创建市场环境自适应特征
    """
    # 计算ADX（Average Directional Index）
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14d'] = df['True_Range'].rolling(14).mean()
    
    # 计算市场状态
    df['ATR_Ratio'] = df['ATR_14d'] / df['Close']
    df['Market_Regime'] = 'normal'
    df.loc[df['ATR_Ratio'] < 0.015, 'Market_Regime'] = 'ranging'
    df.loc[df['ATR_Ratio'] > 0.025, 'Market_Regime'] = 'trending'
    
    # 市场状态持续天数
    df['Market_Regime_Ranging_Days'] = 0
    df['Market_Regime_Trending_Days'] = 0
    
    for i in range(1, len(df)):
        if df['Market_Regime'].iloc[i] == 'ranging':
            df['Market_Regime_Ranging_Days'].iloc[i] = df['Market_Regime_Ranging_Days'].iloc[i-1] + 1
        else:
            df['Market_Regime_Ranging_Days'].iloc[i] = 0
        
        if df['Market_Regime'].iloc[i] == 'trending':
            df['Market_Regime_Trending_Days'].iloc[i] = df['Market_Regime_Trending_Days'].iloc[i-1] + 1
        else:
            df['Market_Regime_Trending_Days'].iloc[i] = 0
    
    # 其他自适应特征
    df['Volume_Confirmation_Adaptive'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['Confidence_Threshold_Multiplier'] = 1.0
    df.loc[df['Market_Regime'] == 'ranging', 'Confidence_Threshold_Multiplier'] = 1.05
    df.loc[df['Market_Regime'] == 'trending', 'Confidence_Threshold_Multiplier'] = 0.95
    
    df['ATR_Risk_Score'] = df['ATR_14d'] / df['Close'].rolling(60).mean()
    
    # 市场波动率水平
    df['Market_Volatility_Level'] = 'medium'
    df.loc[df['ATR_Ratio'] < 0.015, 'Market_Volatility_Level'] = 'low'
    df.loc[df['ATR_Ratio'] > 0.025, 'Market_Volatility_Level'] = 'high'
    
    return df
```

---

### 9. 风险管理特征（18个）

#### 目的
提供风险管理相关的特征

#### 特征列表

**ATR动态止损特征（6个）**：
- `ATR_Stop_Loss_Distance`：ATR止损距离
- `ATR_Change_5d`：ATR 5天变化
- `ATR_Change_10d`：ATR 10天变化
- `ATR_Change_20d`：ATR 20天变化
- `ATR_Ratio_5d`：当日ATR / 5天均值
- `ATR_Ratio_20d`：当日ATR / 20天均值

**连续市场状态记忆（6个）**：
- `Consecutive_Ranging_Days`：连续震荡市天数
- `Consecutive_Trending_Days`：连续趋势市天数
- `Ranging_Fatigue_Index`：震荡市疲劳指数
- `Trending_Strength_Index`：趋势强度指数
- `Market_State_Change_Signal`：市场状态变化信号
- `State_Transition_Probability`：状态转移概率

**盈亏比与交易质量（6个）**：
- `Risk_Reward_Ratio`：盈亏比
- `Expected_Value_Score`：期望值评分
- `Trade_Quality_Score`：交易质量评分
- `Win_Rate_Estimate`：胜率估计
- `Max_Drawdown_Estimate`：最大回撤估计
- `Position_Size_Risk`：仓位风险

#### 计算方法

```python
def create_risk_management_features(df):
    """
    创建风险管理特征
    """
    # ATR动态止损特征
    df['ATR_14d'] = df['True_Range'].rolling(14).mean()
    df['ATR_Stop_Loss_Distance'] = 2 * df['ATR_14d']  # 2倍ATR作为止损距离
    df['ATR_Change_5d'] = df['ATR_14d'].pct_change(5)
    df['ATR_Change_10d'] = df['ATR_14d'].pct_change(10)
    df['ATR_Change_20d'] = df['ATR_14d'].pct_change(20)
    df['ATR_Ratio_5d'] = df['ATR_14d'] / df['ATR_14d'].rolling(5).mean()
    df['ATR_Ratio_20d'] = df['ATR_14d'] / df['ATR_14d'].rolling(20).mean()
    
    # 连续市场状态记忆
    df['Consecutive_Ranging_Days'] = 0
    df['Consecutive_Trending_Days'] = 0
    
    for i in range(1, len(df)):
        if df['Market_Regime'].iloc[i] == 'ranging':
            df['Consecutive_Ranging_Days'].iloc[i] = df['Consecutive_Ranging_Days'].iloc[i-1] + 1
            df['Consecutive_Trending_Days'].iloc[i] = 0
        elif df['Market_Regime'].iloc[i] == 'trending':
            df['Consecutive_Trending_Days'].iloc[i] = df['Consecutive_Trending_Days'].iloc[i-1] + 1
            df['Consecutive_Ranging_Days'].iloc[i] = 0
        else:
            df['Consecutive_Ranging_Days'].iloc[i] = 0
            df['Consecutive_Trending_Days'].iloc[i] = 0
    
    # 震荡市疲劳指数
    df['Ranging_Fatigue_Index'] = df['Consecutive_Ranging_Days'] / 20  # 归一化到0-1
    df['Trending_Strength_Index'] = df['Consecutive_Trending_Days'] / 20  # 归一化到0-1
    
    # 盈亏比与交易质量
    df['Risk_Reward_Ratio'] = (df['Close'].rolling(20).max() - df['Close']) / df['ATR_Stop_Loss_Distance']
    df['Expected_Value_Score'] = 0.6 * df['Risk_Reward_Ratio'] - 0.4  # 假设胜率60%
    df['Trade_Quality_Score'] = (df['Expected_Value_Score'] + 1) / 2  # 归一化到0-1
    
    return df
```

---

### 10. 事件驱动特征（9个）

#### 目的
捕捉分红、财报等公司事件的影响

#### 特征列表

**分红特征（3个）**：
- `Ex_Dividend_In_7d`：未来7天内是否有除净日
- `Ex_Dividend_In_30d`：未来30天内是否有除净日
- `Dividend_Frequency_12m`：过去12个月分红次数

**财报公告日特征（3个）**：
- `Earnings_Announcement_In_7d`：未来7天内是否有财报公告
- `Earnings_Announcement_In_30d`：未来30天内是否有财报公告
- `Days_Since_Last_Earnings`：距离上次财报公告的天数

**财报超预期特征（3个）**：
- `Earnings_Surprise_Score`：最新财报超预期评分（基于Surprise(%)）
- `Earnings_Surprise_Avg_3`：过去3次财报超预期平均
- `Earnings_Surprise_Trend`：近期财报超预期趋势

#### 计算方法

```python
def create_event_driven_features(df, dividend_data, earnings_data):
    """
    创建事件驱动特征
    """
    # 分红特征
    df['Ex_Dividend_In_7d'] = 0
    df['Ex_Dividend_In_30d'] = 0
    
    if dividend_data is not None:
        for idx, row in dividend_data.iterrows():
            ex_date = pd.to_datetime(row['ex_dividend_date'])
            for i in range(len(df)):
                current_date = df.index[i]
                if 0 < (ex_date - current_date).days <= 7:
                    df['Ex_Dividend_In_7d'].iloc[i] = 1
                if 0 < (ex_date - current_date).days <= 30:
                    df['Ex_Dividend_In_30d'].iloc[i] = 1
    
    # 分红频率
    df['Dividend_Frequency_12m'] = 0
    if dividend_data is not None:
        for i in range(len(df)):
            current_date = df.index[i]
            one_year_ago = current_date - pd.Timedelta(days=365)
            recent_dividends = dividend_data[
                (pd.to_datetime(dividend_data['ex_dividend_date']) >= one_year_ago) &
                (pd.to_datetime(dividend_data['ex_dividend_date']) <= current_date)
            ]
            df['Dividend_Frequency_12m'].iloc[i] = len(recent_dividends)
    
    # 财报公告日特征
    df['Earnings_Announcement_In_7d'] = 0
    df['Earnings_Announcement_In_30d'] = 0
    df['Days_Since_Last_Earnings'] = 999
    
    if earnings_data is not None:
        for i in range(len(df)):
            current_date = df.index[i]
            
            # 未来公告
            future_earnings = earnings_data[
                (pd.to_datetime(earnings_data['earnings_date']) > current_date) &
                (pd.to_datetime(earnings_data['earnings_date']) <= current_date + pd.Timedelta(days=7))
            ]
            df['Earnings_Announcement_In_7d'].iloc[i] = 1 if len(future_earnings) > 0 else 0
            
            future_earnings_30d = earnings_data[
                (pd.to_datetime(earnings_data['earnings_date']) > current_date) &
                (pd.to_datetime(earnings_data['earnings_date']) <= current_date + pd.Timedelta(days=30))
            ]
            df['Earnings_Announcement_In_30d'].iloc[i] = 1 if len(future_earnings_30d) > 0 else 0
            
            # 距离上次财报
            past_earnings = earnings_data[
                pd.to_datetime(earnings_data['earnings_date']) <= current_date
            ]
            if len(past_earnings) > 0:
                last_earnings_date = pd.to_datetime(past_earnings['earnings_date']).max()
                df['Days_Since_Last_Earnings'].iloc[i] = (current_date - last_earnings_date).days
    
    # 财报超预期特征
    df['Earnings_Surprise_Score'] = 0
    df['Earnings_Surprise_Avg_3'] = 0
    df['Earnings_Surprise_Trend'] = 0
    
    if earnings_data is not None and 'surprise_percent' in earnings_data.columns:
        for i in range(len(df)):
            current_date = df.index[i]
            recent_earnings = earnings_data[
                pd.to_datetime(earnings_data['earnings_date']) <= current_date
            ].tail(3)
            
            if len(recent_earnings) > 0:
                latest_surprise = recent_earnings.iloc[-1]['surprise_percent']
                df['Earnings_Surprise_Score'].iloc[i] = latest_surprise
                
                if len(recent_earnings) >= 3:
                    df['Earnings_Surprise_Avg_3'].iloc[i] = recent_earnings['surprise_percent'].mean()
                    
                    # 趋势：最新 > 平均
                    if latest_surprise > recent_earnings['surprise_percent'].mean():
                        df['Earnings_Surprise_Trend'].iloc[i] = 1
                    elif latest_surprise < recent_earnings['surprise_percent'].mean():
                        df['Earnings_Surprise_Trend'].iloc[i] = -1
    
    return df
```

---

### 11. 股票类型特征（128个）

#### 目的
基于股票类型（银行股、科技股等）创建特征

#### 特征列表

**技术指标特征（80个）**：
- 移动平均（MA5、MA10、MA20、MA60）及变化率、比率、交叉信号
- RSI（5日、10日、20日）及超买超卖信号
- MACD（DIF、DEA、MACD）及金叉死叉信号
- 布林带（上轨、下轨、带宽）及突破信号
- ATR（5日、10日、20日）及变化率
- 成交量比率（5日、10日、20日）

**基本面特征（8个）**：
- PE、PB、ROE、ROA、股息率、EPS、净利率、毛利率

**市场环境特征（3个）**：
- 恒生指数收益率、相对表现、市场状态

**资金流向特征（5个）**：
- 价格位置、成交量信号、动量信号

**股票类型特征（18个）**：
- 13种行业类型 + 5个衍生评分（防御性、成长性、周期性、流动性、风险）

**关键流动性特征（3个）**：
- VIX_Level、成交额变化率、换手率变化率

**美股市场特征（11个）**：
- 标普500/纳斯达克1/5/20日收益率
- VIX绝对值/变化率/比率
- 美国10年期国债收益率及变化率

#### 计算方法

```python
def create_stock_type_features(df, stock_code, stock_type_mapping):
    """
    创建股票类型特征
    """
    # 获取股票类型信息
    if stock_code in stock_type_mapping:
        stock_info = stock_type_mapping[stock_code]
        stock_type = stock_info['type']
        
        # 股票类型特征（13个行业类型）
        stock_types = ['bank', 'tech', 'semiconductor', 'ai', 'energy', 'shipping',
                      'exchange', 'insurance', 'biotech', 'new_energy', 'environmental',
                      'real_estate', 'index']
        
        for st in stock_types:
            df[f'Stock_Type_{st}'] = 1 if stock_type == st else 0
        
        # 衍生评分特征（5个）
        df['Stock_Defensive_Score'] = stock_info.get('defensive', 50) / 100
        df['Stock_Growth_Score'] = stock_info.get('growth', 50) / 100
        df['Stock_Cyclical_Score'] = stock_info.get('cyclical', 50) / 100
        df['Stock_Liquidity_Score'] = stock_info.get('liquidity', 50) / 100
        df['Stock_Risk_Score'] = stock_info.get('risk', 50) / 100
    
    else:
        # 默认值
        stock_types = ['bank', 'tech', 'semiconductor', 'ai', 'energy', 'shipping',
                      'exchange', 'insurance', 'biotech', 'new_energy', 'environmental',
                      'real_estate', 'index']
        
        for st in stock_types:
            df[f'Stock_Type_{st}'] = 0
        
        df['Stock_Defensive_Score'] = 0.5
        df['Stock_Growth_Score'] = 0.5
        df['Stock_Cyclical_Score'] = 0.5
        df['Stock_Liquidity_Score'] = 0.5
        df['Stock_Risk_Score'] = 0.5
    
    return df
```

---

### 12. GARCH 波动率特征（4个）

#### 目的
使用 GARCH(1,1) 模型捕捉波动率聚类特性，提供更精确的风险评估

#### 特征列表

| 特征名称 | 说明 | 计算方法 |
|---------|------|---------|
| `GARCH_Conditional_Vol` | 条件波动率 | GARCH(1,1) 模型拟合后的条件标准差 |
| `GARCH_Vol_Ratio` | 波动率比率 | 当前条件波动率 / 历史均值 |
| `GARCH_Vol_Change_5d` | 5日波动率变化 | 条件波动率的5日变化率 |
| `GARCH_Persistence` | 波动率持续性参数 | α + β（GARCH 参数之和，反映波动率持续性） |

#### 计算方法

```python
from arch import arch_model

def calculate_garch_features(returns):
    """
    计算 GARCH 波动率特征
    """
    # 使用 GARCH(1,1) 模型
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    
    # 条件波动率
    conditional_vol = results.conditional_volatility / 100
    
    # 波动率比率
    vol_ratio = conditional_vol.iloc[-1] / conditional_vol.mean()
    
    # 波动率变化
    vol_change = conditional_vol.pct_change(5).iloc[-1]
    
    # 持续性参数
    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']
    persistence = alpha + beta
    
    return {
        'GARCH_Conditional_Vol': conditional_vol.iloc[-1],
        'GARCH_Vol_Ratio': vol_ratio,
        'GARCH_Vol_Change_5d': vol_change,
        'GARCH_Persistence': persistence
    }
```

#### 实现文件
- `data_services/volatility_model.py`

---

### 13. HSI 市场状态特征（6个）

#### 目的
使用 HMM（隐马尔可夫模型）识别恒生指数市场状态，辅助个股预测

#### 特征列表

| 特征名称 | 说明 | 取值范围 |
|---------|------|---------|
| `HSI_Market_Regime` | 市场状态标签 | 0=震荡, 1=牛市, 2=熊市 |
| `HSI_Regime_Prob_0` | 震荡市概率 | 0-1 |
| `HSI_Regime_Prob_1` | 牛市概率 | 0-1 |
| `HSI_Regime_Prob_2` | 熊市概率 | 0-1 |
| `HSI_Regime_Duration` | 当前状态持续时间 | 1-100+ 天 |
| `HSI_Regime_Transition_Prob` | 状态转换概率 | 0-1 |

#### 计算方法

```python
from hmmlearn import hmm

def calculate_hsi_regime_features(hsi_returns):
    """
    计算 HSI 市场状态特征（HMM）
    """
    # 准备数据
    returns = hsi_returns.values.reshape(-1, 1)
    
    # 训练 HMM 模型（3个状态）
    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type='full',
        n_iter=100,
        random_state=42
    )
    model.fit(returns)
    
    # 预测状态
    hidden_states = model.predict(returns)
    state_probs = model.predict_proba(returns)
    
    # 当前状态
    current_state = hidden_states[-1]
    
    # 状态持续时间
    duration = 1
    for i in range(len(hidden_states) - 2, -1, -1):
        if hidden_states[i] == current_state:
            duration += 1
        else:
            break
    
    # 转换概率
    trans_prob = model.transmat_[current_state, current_state]
    
    return {
        'HSI_Market_Regime': current_state,
        'HSI_Regime_Prob_0': state_probs[-1, 0],
        'HSI_Regime_Prob_1': state_probs[-1, 1],
        'HSI_Regime_Prob_2': state_probs[-1, 2],
        'HSI_Regime_Duration': duration,
        'HSI_Regime_Transition_Prob': trans_prob
    }
```

#### 特征重要性（2026-04-27 验证）

| 特征 | 重要性排名 | 重要性得分 |
|------|-----------|-----------|
| HSI_Regime_Duration | 第2 | 3.95 |
| HSI_Regime_Prob_1 | 第3 | 2.44 |
| HSI_Regime_Prob_0 | 第10 | 1.44 |
| HSI_Market_Regime | 第20 | 1.00 |

#### 实现文件
- `data_services/regime_detector.py`

---

### 14. 日历效应特征（22个）

#### 目的
捕捉周期性市场规律，包括星期效应、月份效应、节假日效应等

#### 特征列表

**周期性编码特征（4个）**：
- `Month_Sin` / `Month_Cos`：月份周期性编码
- `DOW_Sin` / `DOW_Cos`：星期周期性编码

**星期效应特征（5个）**：
- `Day_of_Week`：星期几（0-4）
- `Is_Monday` / `Is_Friday`：周一/周五效应
- `Is_Week_End`：是否临近周末

**月份效应特征（4个）**：
- `Month`：月份（1-12）
- `Is_Month_Start` / `Is_Month_End`：月初/月末效应
- `Is_Quarter_End`：是否季末

**节假日效应特征（5个）**：
- `Days_to_Holiday`：距离最近假期天数
- `Is_Pre_Holiday`：是否假期前
- `Is_Post_Holiday`：是否假期后
- `Is_Typhoon_Season`：是否台风季（7-9月）
- `Is_Golden_Week`：是否黄金周前后

**期权到期特征（4个）**：
- `Days_to_Options_Expiry`：距离期权到期天数
- `Is_Options_Expiry_Week`：是否期权到期周
- `Is_Weekly_Options_Day`：是否周期权到期日
- `Is_Quarterly_Expiry`：是否季度期权到期

#### 实现文件
- `data_services/calendar_features.py`

---

### 15. 其他衍生特征（280个）

#### 目的
创建交叉特征、滞后特征等衍生特征

#### 特征列表

**交叉特征（100个）**：
- 价格 × 成交量交互特征
- 技术指标 × 市场环境交互特征
- 情感 × 技术指标交互特征
- 板块 × 技术指标交互特征

**滞后特征（60个）**：
- 1天、3天、5天、10天、20天、60天滞后特征
- 滞后收益率、滞后波动率、滞后成交量等

**差分特征（60个）**：
- 收益率的一阶、二阶差分
- 波动率的一阶、二阶差分
- 成交量的一阶、二阶差分

**比率特征（60个）**：
- 不同技术指标的比率
- 不同时间窗口特征的比率
- 滞后特征的比率

#### 计算方法

```python
def create_derived_features(df):
    """
    创建衍生特征
    """
    # 交叉特征
    df['Price_Volume_Interaction'] = df['Close'] * df['Volume']
    df['RSI_Volume_Interaction'] = df['RSI_14d'] * df['Volume_Ratio_5d']
    df['MACD_Volume_Interaction'] = df['MACD'] * df['Volume_Ratio_5d']
    df['Sentiment_RSI_Interaction'] = df['Sentiment_Compound'] * df['RSI_14d']
    
    # 滞后特征
    for lag in [1, 3, 5, 10, 20, 60]:
        df[f'Close_Lag_{lag}d'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}d'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}d'] = df['Return'].shift(lag)
    
    # 差分特征
    df['Return_Diff_1'] = df['Return'].diff(1)
    df['Return_Diff_2'] = df['Return'].diff(2)
    df['Volatility_Diff_1'] = df['Volatility_5d'].diff(1)
    df['Volume_Diff_1'] = df['Volume'].diff(1)
    
    # 比率特征
    df['RSI_MA5_Ratio'] = df['RSI_14d'] / df['RSI_14d'].rolling(5).mean()
    df['MACD_Signal_Ratio'] = df['MACD'] / df['MACD_Signal']
    df['Price_MA5_MA10_Ratio'] = df['MA_5d'] / df['MA_10d']
    
    return df
```

---

## 全量特征 vs 500特征对比

### 验证背景
- **目的**：对比全量特征（892个）和500个精选特征的性能差异
- **验证方法**：业界标准 Walk-forward 验证（12个Fold）
- **测试对象**：银行股板块（6只股票）
- **测试周期**：2024-01-01 至 2025-12-31
- **置信度阈值**：0.60（确保公平对比）

### 验证结果

| 指标 | 全量特征（892个） | 500特征 | 改进幅度 |
|------|------------------|---------|---------|
| **年化收益率** | **40.42%** | 30.28% | **+10.14%** ✅ |
| **索提诺比率** | **1.9023** | 1.0400 | **+83%** ✅ |
| **夏普比率** | -0.0235 | -0.0501 | +53% ✅ |
| **平均收益率** | **3.21%** | 2.40% | +34% ✅ |
| **买入信号胜率** | 49.60% | 49.13% | +0.47% ✅ |
| **准确率** | 61.90% | 62.13% | -0.23% ⚠️ |
| **最大回撤** | -13.08% | -12.73% | +2.7% ⚠️ |
| **交易次数** | 714 | 716 | -0.3% |

### 为什么全量特征更好？

1. **CatBoost 的自动特征选择机制**：
   - CatBoost 内置 L2 正则化和自动特征重要性计算
   - 能够自动降权不重要特征，无需预先筛选
   - Ordered Boosting 算法减少训练集-验证集信息泄露

2. **信息保留完整**：
   - 892个特征包含所有信息，避免特征选择导致的信息丢失
   - 520个交叉特征可能包含重要的非线性关系
   - 减少预选误差，避免人为特征选择引入的偏差

3. **业界实践更新**：
   - 之前认为"业界90%使用300-500特征"不适用于 CatBoost
   - CatBoost 的优化策略与 LightGBM/XGBoost 不同
   - 在 CatBoost 场景下，保留所有特征比预先筛选更优

### 可接受的代价
- 准确率略降0.23%（61.90% vs 62.13%），但在可接受范围内
- 最大回撤略增0.35%（-13.08% vs -12.73%），仍符合银行股特性
- 训练时间稍长，但性能提升显著，值得投入

### 最终推荐

✅ **使用全量特征（892个）是当前最优方案**
- 年化收益率提升10.14%，索提诺比率提升83%
- CatBoost 的自动特征选择机制足够强大，不需要预先进行特征选择
- 避免信息丢失，保留所有潜在有用的特征

❌ **废弃特征选择方法（2026-03-27）**
- 统计方法（F-test+互信息）：已被全量特征方法取代
- 模型重要性法：不再使用
- 累积重要性法：实验已证明不如全量特征
- 固定500特征策略：已被全量特征方法取代

---

## 特征选择策略

### 统一策略：使用全量特征（892个）

**推荐配置**：
```bash
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

**不推荐使用**：
- `--use-feature-selection` 参数（已废弃）
- 任何预先特征选择的方法

### 特征选择方法对比（已废弃）

| 方法 | 原理 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **全量特征** | 保留所有特征，依赖模型自动选择 | 信息保留完整，避免偏差 | 训练时间稍长 | ⭐⭐⭐⭐⭐ |
| 统计方法（已废弃） | F-test + 互信息混合 | 理论基础扎实 | 可能丢失重要特征 | ⭐ |
| 模型重要性法（已废弃） | 基于模型预测贡献 | 直观易懂 | 可能过拟合 | ⭐ |
| 累积重要性法（已废弃） | 基于累积重要性阈值 | 自动决定特征数量 | 阈值设定主观 | ⭐ |

---

## 特征重要性分析

### CatBoost 特征重要性排名（Top 20）

| 排名 | 特征名称 | 重要性 | 说明 |
|------|---------|--------|------|
| 1 | `RSI_14d` | 8.52% | 相对强弱指标 |
| 2 | `MACD` | 7.34% | 平滑异同移动平均线 |
| 3 | `MA_5d` | 6.89% | 5日移动平均 |
| 4 | `MA_20d` | 6.54% | 20日移动平均 |
| 5 | `Volume_Ratio_5d` | 5.98% | 5日成交量比率 |
| 6 | `Return_5d` | 5.32% | 5日收益率 |
| 7 | `ATR_14d` | 4.87% | 14日平均真实波幅 |
| 8 | `Sentiment_Compound` | 4.21% | 综合情感得分 |
| 9 | `Price_Above_MA_20d` | 3.98% | 价格高于20日均线 |
| 10 | `Market_Regime` | 3.76% | 市场状态（震荡/正常/趋势） |
| 11 | `Topic_3_Prob` | 3.45% | 主题3概率分布 |
| 12 | `OBV_MA_5d` | 3.21% | OBV 5日均线 |
| 13 | `Close_Rolling_Std_20d` | 2.98% | 20日滚动标准差 |
| 14 | `Volume_Volatility_5d` | 2.76% | 5日成交量波动率 |
| 15 | `Support_60d_Low` | 2.54% | 60日最低价（支撑位） |
| 16 | `Resistance_60d_High` | 2.32% | 60日最高价（阻力位） |
| 17 | `Ex_Dividend_In_7d` | 2.15% | 未来7天内除净日 |
| 18 | `Earnings_Announcement_In_7d` | 1.98% | 未来7天内财报公告 |
| 19 | `Stock_Type_Bank` | 1.87% | 银行股类型 |
| 20 | `VIX_Level` | 1.65% | VIX波动率水平 |

### 特征重要性分析

**高重要性特征类别**：
1. **技术指标**（RSI、MACD、MA、ATR）：占比约40%
2. **量价关系**（成交量比率、OBV）：占比约15%
3. **情感特征**（Sentiment_Compound）：占比约5%
4. **市场环境**（Market_Regime、VIX）：占比约10%
5. **事件驱动**（分红、财报）：占比约5%

**低重要性特征类别**：
1. **滚动统计特征**（偏度、峰度）：占比约5%
2. **长期趋势特征**（MA120、MA250）：占比约5%
3. **主题分布特征**（Topic_Prob）：占比约5%
4. **衍生特征**（交叉、滞后、差分）：占比约10%

### 特征重要性标准差分析

**高稳定性特征**（标准差 < 1%）：
- `RSI_14d`（0.45%）
- `MACD`（0.52%）
- `MA_5d`（0.61%）
- `Volume_Ratio_5d`（0.78%）

**中稳定性特征**（标准差 1-2%）：
- `Return_5d`（1.23%）
- `ATR_14d`（1.45%）
- `Sentiment_Compound`（1.67%）

**低稳定性特征**（标准差 > 2%）：
- `Topic_3_Prob`（2.34%）
- `Support_60d_Low`（2.56%）
- `Ex_Dividend_In_7d`（2.78%）

---

## 事件驱动特征

### 验证结果（Walk-forward，阈值0.6）

| 指标 | 无事件驱动特征 | 有事件驱动特征 | 改进幅度 |
|------|-------------|-------------|---------|
| **索提诺比率** | 0.8125 | **1.2251** | **+51%** ✅ |
| **震荡市胜率** | 45.62% | **50.60%** | **+4.98%** ✅ |
| **准确率** | 61.16% | **61.90%** | +0.74% ✅ |
| **正确决策比例** | 82.47% | **83.10%** | +0.63% ✅ |
| **年化收益率** | 31.52% | 30.71% | -0.81% ⚠️ |
| **夏普比率** | -0.0345 | -0.0235 | +32% ✅ |
| **最大回撤** | -12.73% | -13.08% | -2.7% ⚠️ |

### 关键发现

1. **震荡市表现显著改善**：
   - Fold 7 (+4.18%)：震荡市胜率从46.44%提升至50.62%
   - Fold 9 (+5.98%)：震荡市胜率从44.26%提升至50.24%

2. **风险控制能力提升**：
   - 索提诺比率+51%（只考虑下行风险）
   - 夏普比率+32%（风险调整后收益）

3. **可接受的代价**：
   - 年化收益率轻微下降：-0.81%
   - 最大回撤略增：-2.7%
   - 但整体风险调整收益显著提升

### 结论

✅ **事件驱动特征在风险控制和震荡市表现优异，值得保留**

---

## 特征工程最佳实践

### 1. 使用全量特征（892个）

**推荐命令**：
```bash
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
```

**不推荐**：
```bash
# 不要使用这个参数（已废弃）
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost --use-feature-selection
```

### 2. 固定随机种子

**重要性**：确保结果可重现

**配置**：
```python
random.seed(42)
np.random.seed(42)
```

### 3. 数据泄漏防护

**原则**：所有特征必须使用滞后数据

**示例**：
```python
# ❌ 错误：使用当日数据
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

# ✅ 正确：使用滞后数据
df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(5).mean()
```

### 4. 特征标准化

**CatBoost 不需要标准化**：
- CatBoost 内置特征缩放
- 不需要手动标准化或归一化

### 5. 分类特征编码

**使用 LabelEncoder**：
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Market_Regime_Encoded'] = le.fit_transform(df['Market_Regime'])
```

### 6. 特征缺失值处理

**CatBoost 自动处理缺失值**：
- CatBoost 可以直接处理缺失值
- 不需要手动填充

### 7. 特征重要性监控

**定期检查特征重要性**：
```python
model = CatBoostModel()
model.train(X, y)

# 获取特征重要性
feature_importance = model.catboost_model.get_feature_importance()

# 保存特征重要性
pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False).to_csv('output/feature_importance.csv')
```

### 8. 特征工程版本控制

**记录特征工程版本**：
- 保存特征工程代码版本
- 记录特征数量和类型
- 保存特征重要性排名

---

## 相关文件

- **特征工程实现**：`ml_services/ml_trading_model.py` 中的 `FeatureEngineer` 类
- **特征重要性输出**：`output/ml_trading_model_catboost_20d_importance.csv`
- **全量特征验证报告**：`output/feature_comparison_final_20260327.md`
- **事件驱动特征验证报告**：`output/event_driven_features_validation_20260329.md`

---

## 参考资料

- **CatBoost 官方文档**：https://catboost.ai/docs/
- **特征工程最佳实践**：https://www.kaggle.com/learn/feature-engineering
- **时间序列特征工程**：https://machinelearningmastery.com/feature-engineering-for-time-series/
- **金融时间序列特征**：https://www.investopedia.com/terms/t/technical-analysis.asp

---

**最后更新**：2026-04-03

---

## 事件驱动特征设计方案

> **来源**：event_driven_features_design.md（合并于 2026-04-27）

### 概述

事件驱动特征用于捕捉财报公告、分红事件、重大公告等对股价的影响。

### 数据源

1. **现有数据源**
   - AKShare：财报指标数据（`ak.stock_hk_financial_indicator_em`）
   - AKShare：股息数据（`ak.stock_hk_dividend_payout_em`）
   - 新闻情感分析（`llm_services/sentiment_analyzer.py`）

2. **待补充数据源**
   - 财报公告日数据：雅虎财经/东方财富/新浪财经
   - 重大事件公告：新闻关键词识别 + NLP分类

### 特征分类

#### 1. 财报公告日特征（2-3个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Earnings_Announcement_In_7d` | Binary | 未来7天内是否有财报公告（0/1） | 财报公告日API |
| `Earnings_Announcement_In_30d` | Binary | 未来30天内是否有财报公告（0/1） | 财报公告日API |
| `Days_Since_Last_Earnings` | Continuous | 距离上次财报公告的天数（1-120+） | 财报公告日API |

#### 2. 财报超预期/不及预期特征（3-4个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Earnings_Surprise_Score` | Continuous | 财报超预期评分（-1到+1，基于新闻情感） | 新闻情感分析 |
| `Earnings_Surprise_Trend` | Continuous | 近期财报超预期趋势（过去3次平均） | 新闻情感历史 |
| `Post_Earnings_Price_Change_3d` | Continuous | 财报后3日股价变化率（%） | 历史股价数据 |
| `Post_Earnings_Price_Change_10d` | Continuous | 财报后10日股价变化率（%） | 历史股价数据 |

#### 3. 除净日和分红公告特征（3-4个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Ex_Dividend_In_7d` | Binary | 未来7天内是否有除净日（0/1） | AKShare股息数据 |
| `Ex_Dividend_In_30d` | Binary | 未来30天内是否有除净日（0/1） | AKShare股息数据 |
| `Dividend_Yield_Ratio` | Continuous | 股息率相对于板块平均值（0.5-2.0） | 财务数据 + 板块平均 |
| `Dividend_Frequency_12m` | Continuous | 过去12个月分红次数（1-12） | AKShare股息历史 |

#### 4. 重大事件公告特征（4-5个）

| 特征名称 | 类型 | 说明 | 数据来源 |
|---------|------|------|---------|
| `Major_Event_In_7d` | Binary | 未来7天内重大事件（0/1） | 新闻关键词识别 |
| `M_A_Announcement` | Binary | 近30天并购事件公告（0/1） | 新闻NLP分类 |
| `Buyback_Announcement` | Binary | 近30天股票回购公告（0/1） | 新闻NLP分类 |
| `Governance_Event` | Binary | 近30天公司治理事件（0/1） | 新闻NLP分类 |
| `Regulatory_Event` | Binary | 近30天监管/合规事件（0/1） | 新闻NLP分类 |

### 实现优先级

#### 第一优先级（立即实现）
1. 除净日和分红特征（已有数据源）
   - `Ex_Dividend_In_7d`
   - `Ex_Dividend_In_30d`
   - `Dividend_Frequency_12m`

#### 第二优先级（调研后实现）
2. 财报公告日特征（需补充数据源）
   - `Earnings_Announcement_In_7d`
   - `Earnings_Announcement_In_30d`
   - `Days_Since_Last_Earnings`

3. 财报超预期特征（需新闻情感分析）
   - `Earnings_Surprise_Score`
   - `Earnings_Surprise_Trend`

#### 第三优先级（可选实现）
4. 重大事件特征（需NLP分类）
   - `M_A_Announcement`
   - `Buyback_Announcement`
   - `Governance_Event`
   - `Regulatory_Event`

### 预期效果

1. **除净日特征**：
   - 捕捉除净日前的买入机会（高股息策略）
   - 预期提升银行股、能源股的预测准确率

2. **财报公告日特征**：
   - 捕捉财报前的波动性
   - 预期提升财报季期间的模型稳定性

3. **重大事件特征**：
   - 识别并购、回购等催化剂
   - 预期提升突发事件的预测能力

### 风险与挑战

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

---

## 不同股票类型分析框架对比

> **来源**：不同股票类型分析框架对比.md（合并于 2026-04-27）

### 概述

不同类型的股票需要采用不同的分析框架和预测策略。本节对比银行股、科技股、半导体股等主要股票类型的特点和最佳分析方法。

### 股票类型分类

| 类型 | 代表股票 | 特点 | 最佳预测周期 |
|------|----------|------|-------------|
| **银行股** | 汇丰、工行、建行 | 趋势稳定、分红高、波动低 | 20天（70-82%） |
| **科技股** | 腾讯、阿里、美团 | 波动大、成长性强、噪音高 | 不推荐（<55%） |
| **半导体股** | 中芯、华虹 | 周期性强、受政策影响 | 20天（约50%） |
| **新能源股** | 比亚迪、宁德时代 | 成长性强、政策驱动 | 1-5天（60%+） |
| **公用事业股** | 中移动、中电信 | 防御性强、分红稳定 | 20天（约70%） |
| **保险股** | 友邦、中国平安 | 周期性与防御性兼具 | 1天（约52%） |
| **能源股** | 中海油、中石油 | 周期性强、受油价影响 | 5天（约50%） |

### 各类型特征重要性差异

#### 银行股（趋势型）

**核心特征**：
- MA250_Slope（长期趋势斜率）
- Price_Distance_MA250（价格距离年线）
- MA250_Slope_5d（趋势斜率变化）

**特点**：
- 趋势性强，长期预测准确率高
- 对技术指标响应稳定
- 分红公告影响显著

**策略建议**：使用20天周期，关注MA250系统

#### 科技股（波动型）

**核心特征**：
- Momentum_Accel_5d（动量加速）
- Volume_Ratio（成交量比率）
- RSI_ROC（RSI变化率）

**特点**：
- 噪音大，难以预测
- 对市场情绪敏感
- 短期波动剧烈

**策略建议**：不推荐纯技术预测，需结合基本面和情绪指标

#### 半导体股（周期型）

**核心特征**：
- CMF_Signal（资金流信号）
- ATR_Ratio（波动率比率）
- MACD_Signal（MACD信号）

**特点**：
- 受政策和行业周期影响大
- 中期预测能力中等
- 需关注行业景气度

**策略建议**：20天周期，结合行业景气度指标

#### 新能源股（成长型）

**核心特征**：
- Return_1d（短期收益）
- Momentum_Accel_5d（动量加速）
- Price_Distance_MA60（价格距离60日线）

**特点**：
- 短期动量有效
- 政策敏感度高
- 成长性驱动

**策略建议**：1-5天短期交易，关注政策动态

#### 公用事业股（防御型）

**核心特征**：
- MACD_Signal（MACD信号）
- MA250_Slope_20d（长期趋势变化）
- Volume_MA250（长期成交量）

**特点**：
- 防御性强
- 波动率低
- 分红稳定

**策略建议**：20天周期，关注MACD信号

### 板块轮动分析

不同板块在不同市场环境下表现差异显著：

| 市场环境 | 推荐板块 | 原因 |
|----------|----------|------|
| **牛市** | 科技股、半导体股、新能源股 | 成长性受益 |
| **熊市** | 银行股、公用事业股 | 防御性强 |
| **震荡市** | 银行股、保险股 | 分红收益 |
| **政策刺激** | 新能源股、半导体股 | 政策驱动 |

### 特征工程优化方向

```
银行股特征优化：
├── 核心：MA系统（MA20/60/120/250）
├── 增强：Price_Distance系列
└── 可删：动量变化率

科技股特征优化：
├── 核心：Momentum_Accel、Volume_Ratio
├── 增强：波动率系列（ATR_Ratio、BB_Width）
└── 可删：长期均线

新能源股特征优化：
├── 核心：动量加速（Momentum_Accel）
├── 增强：成交量相关
└── 可删：无

公用事业股特征优化：
├── 核心：MACD信号系统
├── 增强：趋势斜率
└── 可删：短期动量
```