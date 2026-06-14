# <img src="assets/icon.svg" width="40" height="48" alt="Financial Intelligence Analysis" style="vertical-align: middle; margin-right: 10px;"> Financial Asset & Hong Kong Stock Intelligent Analysis System

**[中文版 (Chinese Version)](README_CN.md)**

**⭐ If you find this project useful, please Star and Fork to support its development! ⭐**

Implementing the concept of **Human-AI Hybrid Intelligence**, developing a financial asset intelligent quantitative analysis assistant with monetization capabilities. The system integrates **Large Language Model reasoning** with **Machine Learning prediction models**, monitoring cryptocurrency, Hong Kong stocks, gold, and other financial markets in real-time.

---

## 📄 Documentation

- Daily updated [Hong Kong Stock Trading Recommendations](output/comprehensive_reports)

---

## I. Core Features

### 1.1 Project Advantages

**Human-AI Hybrid Intelligence**: Fuses LLM reasoning capabilities with ML prediction precision, maintaining quantitative analysis objectivity while understanding market context flexibly. Compared to pure quantitative strategies, it better handles market events and irrational behaviors.

**Validated Strategies**: All trading strategies undergo at least two years of historical backtesting and Walk-forward validation. Core strategies like False Breakout Long (87% win rate) and Z-Score Bottom Fishing (72% win rate) have been verified in actual trading.

**Full Automation**: From data collection, feature computation, model prediction to email delivery - fully automated via GitHub Actions scheduled workflows. No manual intervention needed, ensuring no trading opportunity is missed.

**Hong Kong Market Focus**: Optimized specifically for HK market characteristics, including southbound capital tracking, HSI correlation analysis, sector rotation research. Better captures HK market patterns than general quantitative tools.

**Multi-dimensional Cross-validation**: Single indicators may fail, but multi-dimensional signal resonance significantly improves reliability. System integrates four dimensions: three-horizon prediction, anomaly detection, LLM analysis, sector rotation - strong recommendations only when signals align.

**Transparent Performance Monitoring**: Daily automatic prediction accuracy evaluation with monthly/quarterly/yearly statistics. No hiding failed predictions, continuous iterative improvement.

---

### 1.2 Hang Seng Index Three-Horizon Prediction System

**Core Philosophy**: Simultaneously predicting 1-day, 5-day, and 20-day horizons captures market trends at different time scales, supporting short-term, medium-term, and long-term trading decisions. Three-horizon cross-validation significantly improves prediction reliability.

**Multi-Horizon Predictions** (Validated 2026-05-18):

| Horizon | Accuracy | Characteristics | Usage |
|---------|----------|-----------------|-------|
| 1-day | 51.49% | High noise, reference only | Intraday trading |
| 5-day | 65.86% | Trend confirmation, auxiliary | Weekly holding decisions |
| **20-day** | **81.22%** | **Most reliable, primary decision** | Monthly investment direction |

**Eight Trading Patterns** (HSI Enhanced Model, validated 2026-05-18):

Based on three-horizon prediction results (1/0 indicates up/down), forming 8 trading signals. E.g., "110" means 1-day up, 5-day up, 20-day down.

| Pattern | Description | 20-day Accuracy | Strategy |
|---------|-------------|-----------------|----------|
| **101** | False Breakout (1-up 5-down 20-up) | **87.32%** | ⭐ Best long entry, short pullback then medium-term rise |
| **111** | Consistent Bullish | **86.26%** | ⭐⭐⭐⭐⭐ Second-best long entry |
| **001** | Downward Continuation | **81.05%** | ⭐⭐⭐⭐ Long signal |
| 000 | Consistent Bearish | **79.80%** | Strong sell, three-horizon resonance down |
| 010 | Failed Rebound | **77.78%** | Short signal |

**Four Trading Rules** (HSI Enhanced Model):

| Rule | Condition | Accuracy | Application |
|------|-----------|----------|-------------|
| **False Breakout Long** | 1-up 5-down 20-up (101) | **87.32%** | Long after short pullback |
| Consistent Bullish Buy | All three up (111) | **86.26%** | Add position after trend confirmation |
| Downward Continuation Long | 1-down 5-down 20-up (001) | **81.05%** | Buy after decline |
| Consistent Bearish Short | All three down (000) | 79.80% | Reduce position or short after trend confirmation |

### 1.3 Hong Kong Stock CatBoost Machine Learning Model

**Core Advantage**: Uses CatBoost gradient boosting algorithm, integrating 1023 technical indicators, fundamental data, market state, network features, and sentiment indicators for multi-horizon up/down prediction. ML models automatically discover complex market patterns beyond traditional technical analysis.

**Performance Metrics**:

| Validation Method | Horizon | Accuracy | IC | Rank IC | Recommendation |
|------------------|---------|----------|------|---------|----------------|
| **Walk-forward** (12 folds, 57 stocks) | **20-day** | **55.04%** | **0.205** | **0.231** | ⭐⭐⭐⭐ Recommended |
| Walk-forward (12 folds) | 5-day | ~50% | - | - | ⭐⭐⭐ Use cautiously |
| Training 5-fold CV | 20-day | 64.67% | - | - | Reference |

> ⚠️ Note: Training CV accuracy higher than Walk-forward is normal, not data leakage. See `docs/VALIDATION_GUIDE.md`.

**Walk-forward Validation Results** (57 stocks, 12 folds, Top 500 features, Market Sentiment Filter enabled, 2026-05-23):

| Metric | Value | Industry Standard | Assessment |
|--------|-------|-------------------|------------|
| Composite Score | **90/100** | - | Excellent |
| Average Sharpe Ratio | **5.33** | >0.5 | ✅ Excellent |
| Average Max Drawdown | **-1.04%** | <-20% | ✅ Excellent |
| Average Accuracy | 55.04% | >50% | ✅ |
| Average IC | **0.205** | >0.05 | ✅ Excellent |
| Average Rank IC | **0.231** | >0.05 | ✅ Excellent |
| Average Return | **+5.08%** | >0% | ✅ Positive |

**Feature System (1023 features)**:

| Category | Feature Examples | Purpose |
|----------|------------------|---------|
| Technical Indicators | MA, RSI, MACD, Bollinger Bands, KDJ | Capture price trends and momentum |
| Price Patterns | Candlestick patterns, support/resistance | Identify classic trading signals |
| Fundamentals | PE, PB, ROE, Market Cap | Assess intrinsic value |
| Market Sentiment | HSI trend, sector strength | Reflect overall market environment |
| Capital Flow | Southbound capital, institutional inflow | Track smart money |
| **Interest Rate Features** | US-CN rates, term spreads, CN-US spread | HK capital flow key drivers |
| **GARCH Volatility** | Conditional volatility, volatility ratio, persistence | Capture volatility clustering |
| **LSTM-GARCH Hybrid** | Hybrid volatility, uncertainty, trend signal | Fuse econometrics with deep learning |
| **HSI Market Regime** | HMM market state, state probability, duration | Identify bull/bear/range markets |
| **Calendar Effects** | Day-of-week, month effects, option expiry | Capture cyclical market patterns |
| **Network Features** | Community membership, centrality, bridge stocks | Reflect stock correlations |
| **Network Cross Features** | Market-level features × Network community | Different communities respond differently to market signals |

> **Market-level Feature Handling**: 60 market-level features (same value for all stocks) are crossed with network communities, enabling differentiated responses to the same market signal. Interest rate features distinguish stocks through this mechanism.

**Feature Importance** (Stock 20-day Model, Top 10, 2026-05-23):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | Volatility_30pct | 2.01 | Volatility |
| 2 | MA250_Slope | 1.83 | Trend |
| 3 | Volatility_30d | 1.79 | Volatility |
| 4 | BB_Width_MA60 | 1.76 | Technical |
| 5 | net_cohesion_HSI_Regime_Duration | 1.51 | **Network Cross** |
| 6 | Volatility_70pct | 1.49 | Volatility |
| 7 | Distance_Support_120d | 1.43 | Technical |
| 8 | net_cohesion_per_GARCH_Conditional_Vol | 1.36 | **Network Cross** |
| 9 | Stock_Price_Stability_Score | 1.34 | Risk |
| 10 | 60d_Trend_HSI_Return_60d | 1.31 | **Network Cross** |

> **Key Finding**: Network cross features (`net_cohesion_*`, `net_constraint_*`) occupy 3 of Top 10, proving market-level features crossed with network communities have significant predictive value.

**Model Configuration**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Prediction Threshold** | 0.5 | Probability > 0.5 predicts up, ≤ 0.5 predicts down |
| Confidence Levels | 0.65 / 0.55 | High (>0.65), Medium (0.55-0.65), Low (<0.55) |
| Feature Cache | 7-day validity | Feature computation cache, **170x speedup** |
| Random Seed | 42 (fixed) | Ensures reproducibility |

**Dual-Mode Prediction System**:

System distinguishes two prediction scenarios for train-predict consistency:

| Scenario | Feature Timestamp | `mode` Parameter | Application |
|----------|------------------|------------------|-------------|
| Post-market Prediction | Current day data | `production` | Real trading decisions |
| Walk-forward Validation | T-1 data | `backtest` | Model validation, prevent leakage |

**Practical Application**:
- Daily automatic prediction of watchlist stocks' 5-day and 20-day up/down probabilities
- Position sizing based on confidence levels (high confidence = larger position, low confidence = observe)
- Email delivery of predictions for timely decisions

**⚠️ Risk Warning**:

High confidence prediction errors still carry significant loss risk:

| Metric | Value |
|--------|-------|
| High confidence (>=0.65) error samples | 1,539 |
| Average loss | **-6.91%** |
| Maximum loss | **-72.96%** |
| Loss <= -5% | **49.4%** |
| Loss <= -10% | **24.7%** |

**Must use stop-loss strategy**: Recommend 3-5% stop-loss, can improve expected return by 30%.

### 1.4 Market Sentiment Filter

**Core Principle**: Market up ratio has strong autocorrelation (lag=1 autocorrelation 0.929), lag-1 day data effectively identifies extreme market environments, dynamically adjusting prediction thresholds.

**Threshold Layers**:

| Layer | Up Ratio | Dynamic Threshold | Action |
|-------|----------|-------------------|--------|
| extreme_bear | <20% | 1.0 | Pause trading |
| bear | 20-30% | 0.70 | High confidence required |
| weak | 30-40% | 0.65 | Cautious |
| normal | >40% | 0.50 | Standard |

**Validation Results** (12 folds, 57 stocks):

| Metric | Before Filter | After Filter | Change |
|--------|--------------|--------------|--------|
| Accuracy | 62.0% | 70.7% | **+8.7%** |
| Total Return | 242.13 | 305.57 | **+63.44** |
| FP | 2217 | 1424 | **-793** |

**Key Finding**: The problem is "model remains over-optimistic when market broadly declines", not "wrong stock selection". Market environment perception is more effective than individual stock filtering.

### 1.5 Hong Kong Stock Anomaly Detection

**Core Value**: Alerts when market shows abnormal fluctuations, helping investors timely avoid risks or seize opportunities. Anomaly signals often indicate important market turning points.

**Dual-Layer Detection Mechanism**:

| Layer | Method | Detection Target | Advantage |
|-------|--------|------------------|-----------|
| First Layer | Z-Score | Price/volume deviation from mean | Quick statistical anomaly identification |
| Second Layer | Isolation Forest | Multi-dimensional feature space outliers | Capture complex anomaly patterns |

**Validation Strategy** (Two-year historical backtesting):

| Anomaly Type | Strategy | 5-day Return | Win Rate | Application |
|--------------|----------|--------------|----------|-------------|
| **Price anomaly + same-day down** | 🟢 Bottom Fishing | +4.12% | **72%** | Oversold rebound opportunity, left-side trading |
| Price anomaly + same-day up | ⚠️ Observe | +1.96% | 54% | Chase risk, wait for confirmation |
| IF high anomaly | 🔴 Reduce Position | -3.04% | 43% | Multi-dimensional warning, reduce exposure |

**Usage Scenarios**:
- Pre-market overnight anomaly detection, predict intraday trend
- Intraday real-time monitoring, detect abnormal stocks immediately
- Combine with other analysis tools, improve decision reliability

⚠️ **Important Warning**: Stock anomaly strategies **NOT applicable to cryptocurrency markets**. Cryptocurrency characteristics differ, requiring specialized strategies.

### 1.6 Large Language Model Intelligent Decision

**Core Philosophy**: Uses LLM (Qwen) reasoning capabilities, integrating multi-dimensional information to generate trading recommendations. Compared to traditional quantitative strategies, LLM understands market context, providing targeted advice.

**Six-Layer Analysis Framework**:

| Layer | Analysis Dimension | Output Content | Priority |
|-------|--------------------|----------------|----------|
| 1️⃣ | Risk Control | Position suggestion, stop-loss point | Highest priority |
| 2️⃣ | Market Environment | Index trend, macro factors | Overall direction |
| 3️⃣ | Fundamentals | Financial health, valuation level | Medium-long term value judgment |
| 4️⃣ | Technical Analysis | Trend, support/resistance, patterns | Entry timing |
| 5️⃣ | Signal Recognition | Anomaly signals, capital flow | Short-term opportunity capture |
| 6️⃣ | Comprehensive Decision | Final buy/sell recommendation | Synthesizes above five layers |

**Sector Rotation Analysis**:

| Analysis Content | Output | Application |
|------------------|--------|-------------|
| 16 sector rankings | Strong sectors → Weak sectors | Select hot sectors |
| Leader stock identification | Sector leading stocks | Select specific targets |
| Cyclical/Defensive rotation | Market style judgment | Adjust portfolio |
| Institutional capital tracking | Build/distribute signals | Follow smart money |

**Output Examples**:
- Buy/Sell/Hold recommendation for each stock
- Specific position allocation suggestions
- Risk warnings and stop-loss levels

### 1.7 Stock Analysis Skill

**Core Value**: When users ask about stock trading recommendations, automatically queries comprehensive analysis reports, providing 12-dimension professional analysis.

**Trigger Methods**:
- "Is XXX stock a good buy today?"
- "XXX stock analysis"
- "Is XXX stock worth buying?"

**Analysis Dimensions** (12):

| Dimension | Content | Purpose |
|-----------|---------|---------|
| Core Metrics | CatBoost probability, price, position, stop-loss | Decision basis |
| Three-Horizon Prediction | 1-day/5-day/20-day prediction probability | Trend judgment |
| LLM Recommendation | Short-term/medium-term suggestions | Intelligent reference |
| Technical Indicators | RSI/MACD/Bollinger/chip resistance | Entry timing |
| Risk Score | Risk/return/composite score | Risk assessment |
| Market Environment | HSI/market state/VIX | Environment awareness |
| Anomaly Detection | Overbought/oversold/volume anomaly | Risk warning |
| Network Insights | Community membership/bridge stocks/modularity | Correlation analysis |
| Sector Performance | Sector ranking/price change | Sector rotation |
| Operation Suggestions | Staged entry/stop-profit-stop-loss | Specific operations |
| Risk Alerts | Major risk factors | Warning |
| Dividend Reminder | Ex-dividend date/dividend scheme | Income supplement |

**Hard Constraints**:
- CatBoost 20-day up probability ≤ 50% → Prohibit buy recommendation
- Market state is bear → Raise threshold to 0.70
- Market state is range-bound → Raise threshold to 0.65

### 1.8 Risk-Reward Ratio Analysis

**Core Value**: When selecting stocks, not only consider expected return, but also assess potential risk. Helps investors choose optimal risk-reward ratio among multiple candidates, achieving "maximum return, minimum risk".

**Three Investment Styles**:

| Style | Risk Weight | Return Weight | Application Scenario | Suitable For |
|-------|-------------|---------------|---------------------|--------------|
| Conservative | 60% | 40% | Defensive assets, bear market | Risk-averse investors |
| **Balanced** | **50%** | **50%** | **Steady investing, range-bound market** | **Most investors** |
| Aggressive | 30% | 70% | High growth targets, bull market | Risk-seeking investors |

**Risk Metrics** (Measuring downside risk):

| Metric | Meaning | Application |
|--------|---------|-------------|
| VaR (Value at Risk) | Maximum possible loss at 95% confidence | Assess extreme risk |
| Maximum Drawdown | Historical maximum decline | Psychological tolerance test |
| Volatility | Price fluctuation degree | Measure stability |
| Beta | Sensitivity relative to index | Judge systematic risk |
| Liquidity | Average daily turnover | Assess liquidation capability |

**Return Metrics** (Measuring upside potential):

| Metric | Meaning | Application |
|--------|---------|-------------|
| Trend Score | Current trend strength | Trend-following reference |
| Momentum Score | Price momentum strength | Judge sustainability |
| Sharpe Ratio | Risk-adjusted return | Comprehensive efficiency |
| Technical Pattern | Classic bullish/bearish patterns | Entry timing |
| Real-time Status | Intraday price change | Short-term timing |

**Output Reports**:
- Composite score and ranking for each stock
- Risk-reward radar chart
- Specific buy/sell recommendations

### 1.9 Simulated Trading System

**Core Value**: Test strategy effectiveness without real capital, accumulate trading experience. Supports multiple risk preferences, helping investors find suitable trading style.

**Three Risk Preferences**:

| Type | Characteristics | Stop-Loss | Suitable Scenario |
|------|-----------------|-----------|-------------------|
| Aggressive | Pursue high return, accept high risk | -8% | Bull market, growth stocks |
| Balanced | Balance risk and return | -6% | Range-bound market, blue chips |
| Conservative | Asset preservation primary | -4% | Bear market, defensive stocks |

**Core Functions**:

| Function | Description | Purpose |
|----------|-------------|---------|
| Automatic stop-loss tracking | Dynamically adjust stop-loss after price rise | Lock profit, control drawdown |
| Decision consistency protection | Avoid frequent reverse operations within 3h/24h | Prevent emotional trading |
| Trading log | Record decision process for each trade | Review and improve |
| Return statistics | Calculate total return, win rate, max drawdown | Evaluate strategy effectiveness |

---

## II. Quick Start

```bash
# Hang Seng Index prediction
python3 hsi_prediction.py --no-email

# Comprehensive analysis (includes three-horizon prediction and risk-reward analysis)
./scripts/run_comprehensive_analysis.sh

# Risk-reward analysis (stock selection assistance)
python3 ml_services/risk_reward_analyzer.py --stocks watchlist --style moderate

# Hong Kong stock anomaly detection
python3 detect_stock_anomalies.py --mode standalone --mode-type deep

# Model training
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost

# Walk-forward validation
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20

# Clear feature cache
rm -rf data/feature_cache/*.pkl
```

---

## III. Technical Architecture

```
External Data Sources → data_services/ → Analysis Layer → ml_services/ → Output
         ↓                    ↓                ↓              ↓            ↓
    Tencent Finance    Technical Indicators  Anomaly Detection  CatBoost    Email Reports
    yfinance           Fundamental Data      Comprehensive      Walk-forward  JSON Files
    AKShare            Southbound Capital    Analysis           Performance
                       Capital Tracking                         Monitoring
```

### 3.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Collection Layer                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ yfinance │  │  Tencent │  │  AKShare │  │  HSI Data│  │Southbound│         │
│  │          │  │  Finance │  │          │  │          │  │  Capital │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │             │             │             │                │
│       └─────────────┴─────────────┴──────┬──────┴─────────────┘                │
│                                          ▼                                      │
│                              ┌────────────────────┐                             │
│                              │  data/stock_cache/ │ ← Raw data cache (7 days)  │
│                              └────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Processing Layer (data_services/)             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │ Technical       │    │  Fundamental    │    │   Market Data   │             │
│  │ Indicators      │    │  Data           │    │   Integration   │             │
│  │ MA/RSI/MACD    │    │  PE/PB/ROE      │    │  HSI/Sector/    │             │
│  │ Bollinger/KDJ  │    │  Revenue/Profit │    │  Capital Flow   │             │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘             │
│           │                      │                      │                       │
│           └──────────────────────┼──────────────────────┘                       │
│                                  ▼                                              │
│                       ┌─────────────────────┐                                   │
│                       │ 1023 Feature        │                                   │
│                       │ Engineering         │                                   │
│                       └──────────┬──────────┘                                   │
│                                  ▼                                              │
│                       ┌─────────────────────┐                                   │
│                       │ data/feature_cache/ │ ← Feature cache (7 days, 170x)   │
│                       └─────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Analysis Layer                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────┐    ┌─────────────────────────────┐    │
│  │      anomaly_detector/              │    │    comprehensive_analysis   │    │
│  │      Anomaly Detection Module       │    │    Comprehensive Analysis   │    │
│  │  ┌─────────────┐ ┌───────────────┐  │    │  ┌───────────────────────┐  │    │
│  │  │  Z-Score    │ │ Isolation     │  │    │  │ Sector Rotation       │  │    │
│  │  │  Layer 1    │ │ Forest Layer 2│  │    │  │ Capital Tracking      │  │    │
│  │  │  (Real-time)│ │ (Deep)        │  │    │  │ Risk-Reward Analysis  │  │    │
│  │  └─────────────┘ └───────────────┘  │    │  └───────────────────────┘  │    │
│  └─────────────────────────────────────┘    └─────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Machine Learning Layer (ml_services/)                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        CatBoost Prediction Model                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │  1-day      │  │  5-day      │  │  20-day     │  │ Three-Horizon│      │  │
│  │  │  (High Noise│  │  (Caution)  │  │  (Recommended│  │ Signal       │      │  │
│  │  │  Reference) │  │             │  │  Primary)   │  │ Combination  │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────┐  ┌───────────────────────────┐                  │
│  │  Walk-forward Validation  │  │  Performance Monitoring   │                  │
│  │  12-fold Time Series CV   │  │  Prediction Accuracy      │                  │
│  └───────────────────────────┘  └───────────────────────────┘                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LLM Decision Layer (llm_services/)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     Qwen Large Language Model                            │    │
│  │                                                                          │    │
│  │   Input: Price Data + Technical Indicators + Anomaly Signals +          │    │
│  │          ML Predictions + Sector Analysis + Capital Flow                 │    │
│  │                                                                          │    │
│  │   Six Layers: Risk Control → Market Environment → Fundamentals →        │    │
│  │                Technical Analysis → Signal Recognition → Decision       │    │
│  │                                                                          │    │
│  │   Output: Buy/Sell Recommendations + Position Allocation +              │    │
│  │           Risk Warnings + Stop-Loss Levels                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Output Layer                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Email Reports  │  │   JSON Data     │  │ Markdown Reports│                 │
│  │  HSI Prediction │  │ prediction_     │  │  output/*.md   │                 │
│  │  Anomaly Alerts │  │ history.json    │  │ Comprehensive   │                 │
│  │  Comprehensive  │  │ model_accuracy  │  │ Backtest        │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      GitHub Actions Automated Scheduling                 │    │
│  │  HSI Prediction(06:00) → Anomaly Detection(02:00/hourly) →              │    │
│  │  Comprehensive Analysis(16:00) → Performance Monitoring(monthly)         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Cache Mechanism**:

| Cache Type | Location | Validity | Speedup |
|------------|----------|----------|---------|
| Raw Data | `data/stock_cache/` | 7 days | - |
| Feature Cache | `data/feature_cache/` | 7 days | **170x** |

---

## IV. Project Structure

```
fortune/
├── Core Scripts
│   ├── comprehensive_analysis.py       # Comprehensive analysis
│   ├── hsi_prediction.py               # HSI three-horizon prediction
│   ├── detect_stock_anomalies.py       # Anomaly detection
│   └── simulation_trader.py            # Simulated trading
├── ml_services/                        # Machine Learning module
│   ├── ml_trading_model.py             # CatBoost model
│   └── walk_forward_validation.py      # Walk-forward validation
├── data_services/                      # Data services
├── anomaly_detector/                   # Anomaly detection
├── llm_services/                       # LLM services
├── docs/                               # Detailed documentation
└── data/                               # Data and cache
    ├── stock_cache/                    # Raw data cache
    └── feature_cache/                  # Feature cache
```

---

## V. Automated Scheduling

| Workflow | Function | Execution Time |
|----------|----------|----------------|
| `hsi-prediction.yml` | Hang Seng Index prediction | Weekdays 06:00 |
| `comprehensive-analysis.yml` | Comprehensive analysis | Weekdays 16:00 |
| `stock-anomaly-detection.yml` | HK stock anomaly detection | Daily 02:00 |
| `hourly-stock-monitor.yml` | Trading hours monitoring | 10:00-15:00 hourly |
| `performance-monitor.yml` | Performance report | Monthly 1st |

### 5.1 Command Summary

```bash
# Core Prediction and Analysis
python3 hsi_prediction.py --no-email                    # HSI prediction
python3 comprehensive_analysis.py                        # Comprehensive analysis
python3 detect_stock_anomalies.py --mode standalone     # Anomaly detection

# Model Training and Validation
python3 ml_services/ml_trading_model.py --mode train --horizon 20 --model-type catboost
python3 ml_services/walk_forward_validation.py --model-type catboost --horizon 20
python3 ml_services/walk_forward_by_sector.py --sector bank --horizon 20

# Analysis Tools
python3 ml_services/risk_reward_analyzer.py --stocks watchlist --style moderate
python3 ml_services/performance_monitor.py --mode all --no-email
python3 ml_services/analyze_causal_chain.py             # Causal chain analysis

# Cache Management
rm -rf data/feature_cache/*.pkl                         # Clear feature cache
```

---

## VI. Installation

```bash
# 1. Clone project
git clone https://github.com/wonglaitung/fortune.git
cd fortune

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp set_key.sh.sample set_key.sh
# Edit set_key.sh, fill in email and API keys
source set_key.sh

# 4. Verify installation
python hsi_email.py --no-email
```

**Required Environment Variables**:

| Variable Name | Description |
|---------------|-------------|
| `SMTP_SERVER` | SMTP server address |
| `EMAIL_SENDER` | Sender email |
| `EMAIL_PASSWORD` | Email authorization code |
| `RECIPIENT_EMAIL` | Recipient email list |
| `QWEN_API_KEY` | Qwen API key |

---

## VII. Core Warnings

| Warning | Description |
|---------|-------------|
| **Data Leakage** | Walk-forward accuracy >65% (individual stocks) or >80% (HSI) usually indicates data leakage |
| **Prediction Threshold** | Direction judgment uses **0.5**, not 0.65 |
| **CatBoost 1-day** | High noise, reference only |
| **Deep Learning** | LSTM/Transformer F1≈0, not recommended |
| **Walk-forward** | Only trustworthy validation method |
| **Cryptocurrency** | Stock strategies NOT applicable |
| **HSI vs Individual Stocks** | HSI accuracy significantly higher than stocks (81% vs 57%), stock prediction needs caution |
| **Feature Cache Version** | Clear cache when invalidated (`rm -rf data/feature_cache/*.pkl`) |
| **Categorical Feature NaN** | CatBoost prediction must handle categorical NaN, training and prediction preprocessing must be consistent |
| **High Confidence Risk** | High confidence errors can lose -73%, must use stop-loss |
| **IC ≠ Return** | High IC doesn't guarantee high return, need loss distribution analysis |
| **Dual-Mode Prediction** | Post-market uses current day data, Walk-forward uses T-1 data to prevent leakage |

---

## VIII. Documentation

- **[CLAUDE.md](CLAUDE.md)** - Quick reference guide
- **[lessons.md](lessons.md)** - Lessons learned
- **[progress.txt](progress.txt)** - Project progress
- **[docs/](docs/)** - Detailed documentation
  - [THREE_HORIZON_ANALYSIS.md](docs/THREE_HORIZON_ANALYSIS.md) - Three-horizon analysis
  - [FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) - Feature engineering (includes GARCH/HSI Regime)
  - [FEATURE_IMPORTANCE_ANALYSIS.md](docs/FEATURE_IMPORTANCE_ANALYSIS.md) - Feature importance analysis
  - [VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md) - Validation method guide
  - [SECTOR_ROTATION_TRADING_RULES.md](docs/SECTOR_ROTATION_TRADING_RULES.md) - Sector rotation
  - [programmer_skill.md](docs/programmer_skill.md) - Development standards

---

## IX. Dependencies

`yfinance` `catboost` `akshare` `pandas` `scikit-learn` `lightgbm` `jieba` `hmmlearn` `arch`

---

## X. License

MIT License

---

## XI. Contact

- Issues: https://github.com/wonglaitung/fortune/issues
- Email: wonglaitung@gmail.com

---

## XII. Star History

![Star History Chart](https://api.star-history.com/svg?repos=wonglaitung/fortune&type=Date)