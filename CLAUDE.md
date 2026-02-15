# Intraday Trading System — Strategy Research Playbook

## Vision
Fast-iteration trading strategy research. Test hypotheses quickly, kill bad ideas fast, promote winners to live trading. The goal is **alpha generation**, not infrastructure building.

## Core Philosophy
- **Good data layer is foundational** — QuantConnect handles this for us
- **Fast backtesting = fast learning** — test a thesis, throw away what isn't promising immediately
- Build **multiple uncorrelated/negatively correlated value streams**, not one strategy
- **Position sizing is where risk lives**, not in the signal (ref: Tom Bassos)
- Indicators (RSI, MAs, BBs, Donchian, Keltner) all do the same thing — use the past to make a statement about the present. Focus on whether they predict anything.
- **Simplicity wins** — threshold-based strategies built on trend probabilities can outperform complex RL models
- Ref books: Van Tharp "Trade Your Way to Financial Freedom", Brian Shannon "Technical Analysis Using Multiple Timeframes"

## Platform: QuantConnect
We use QuantConnect as our end-to-end platform: research → backtest → paper trade → live trade via IBKR.

### Why QuantConnect
- Data included (equities, options, futures, forex, crypto, alternative data)
- Backtesting with realistic slippage/fees/spreads
- Walk-forward validation built in
- Direct IBKR integration for live trading
- Same code for backtest and live (critical — no separate codepaths)
- LEAN engine is open source (can self-host later if needed)
- Congressional trading data available via Quiver Quantitative dataset

### Cost
- **Free tier**: Unlimited backtesting, minute/hour/daily data, no live trading — use this to validate first
- **Researcher ($10/mo)**: Tick/second data, paper trading, IBKR live, local VSCode coding
- **Live trading node (~$20/mo per algo)**: Required to run strategy 24/7
- **Alternative datasets**: Add-on pricing (e.g., Quiver congressional trades)
- **Realistic total**: $30-50/mo once running one live strategy

## Repo Structure
This repo is a **strategy research workspace**, not infrastructure code.

```
/
├── CLAUDE.md                    # This file — project context
├── notebooks/                   # Jupyter notebooks for research & exploration
│   ├── 01_data_exploration/     # Understanding available data
│   ├── 02_feature_engineering/  # Feature experiments
│   ├── 03_model_training/       # Model development & evaluation
│   └── 04_alternative_data/     # NLP, sentiment, semantic search experiments
├── strategies/                  # QuantConnect algorithm files (Python)
│   ├── congressional_trading/   # Hypothesis 1: Congressional copy-trading
│   ├── directional_classifier/  # Hypothesis 2: Global directional model
│   └── combined_signal/         # Hypothesis 3: Combined signals
├── research/                    # Analysis results, backtest reports
│   ├── backtest_results/
│   └── performance_analysis/
├── data/                        # Local data for notebook experiments
│   ├── external/                # Downloaded datasets (news, sentiment, etc.)
│   └── processed/               # Cleaned/transformed data
├── models/                      # Trained model artifacts
├── config/                      # Strategy parameters, feature configs
│   └── features.yaml            # Feature registry
└── utils/                       # Shared utility code
    ├── features.py              # Feature computation functions
    ├── evaluation.py            # Performance metrics & analysis
    └── data_loaders.py          # Custom data loading utilities
```

## Research Insights — Dual Stream Paper
A master's thesis comparing two prediction-based trading approaches. Key takeaways:

### Validated Principles
- **Decouple prediction from decision-making**: Modular pipeline (predict → decide) beats end-to-end
- **Directional classification > return regression**: Binary UP/DOWN is more tractable than exact returns
- **Simple threshold rules > complex RL**: Threshold trading (confidence > per-stock τ*) returned 40% vs DQN's 14.8%
- **Global models > per-stock models**: One model across all stocks with embeddings captures cross-asset patterns
- **Walk-forward validation is mandatory**: Prevents data leakage between prediction and trading stages

### Ideas to Test
1. Global directional classifier (LightGBM/TabNet) for binary UP/DOWN with per-stock thresholds
2. Cross-stock features (returns of other stocks as inputs)
3. Accumulative learning (transfer body weights across walk-forward folds, reset head)

### What NOT to Copy
- Don't start with LLM/StockTime — heavy, slow to iterate
- Don't start with DQN/RL — needs more data/compute than we have initially
- Don't trust 40% return at face value — single bull year, daily frequency, no live validation

## Strategy Hypotheses

### Hypothesis 1: Congressional Copy-Trading (START HERE — fastest to validate)
- **Signal**: When high-performing politicians disclose a BUY (STOCK Act filings), buy on disclosure date +1
- **Data**: Quiver Quantitative congressional trading dataset (available in QuantConnect)
- **Filters**: Trade size >$50K, top performers only (Pelosi, etc.), buys only initially
- **Holding periods to test**: 7, 14, 30, 60 days
- **Variants**: Long-short, committee-weighted (finance committee on financial stocks)
- **Known edge/risk**: 45-day disclosure delay, but disclosure itself often moves price. Bull market bias in historical data.
- **Success criteria**: Positive Sharpe ratio, outperforms S&P 500 buy-and-hold over test period

### Hypothesis 2: Global Directional Classifier
- **Signal**: LightGBM binary UP/DOWN prediction across multiple stocks
- **Inputs**: Multi-timeframe returns (1, 5, 10, 21 day), cross-stock features, stock identity
- **Decision rule**: Per-stock confidence thresholds — only trade when confidence > optimized τ*
- **Validation**: Walk-forward (train pre-2020, optimize thresholds 2021-2022, test 2023+)
- **Start with**: 19-stock subset across diverse sectors, expand to 99
- **Success criteria**: >60% directional accuracy, positive risk-adjusted returns

### Hypothesis 3: Combined Signal
- **Signal**: Congressional buy + directional classifier agrees = high confidence trade
- **Rationale**: Uncorrelated signals should improve precision
- **Test**: Does combining reduce drawdown and improve Sharpe vs either signal alone?

## Alternative Data Pipeline (Future — Notebook-based)
For adding custom data sources like news sentiment, semantic search over articles, etc.

### Approach
1. **Collect**: Scrape/API news articles, SEC filings, earnings transcripts, Reddit, Twitter/X
2. **Embed**: Sentence transformers or OpenAI embeddings → vector representations
3. **Store**: Vector DB (ChromaDB/FAISS locally, Pinecone for production)
4. **Query**: "What's the sentiment around NVDA this week?" → numeric score
5. **Integrate**: Pass score as custom data feature into QuantConnect algorithm via Custom Data API
6. **Validate**: Does adding sentiment improve Sharpe over base strategy?

### Data Sources to Explore
- Financial news APIs (NewsAPI, Benzinga, Alpha Vantage news)
- SEC EDGAR filings (10-K, 10-Q, 8-K)
- Earnings call transcripts
- Reddit (r/wallstreetbets, r/stocks) via PRAW
- Twitter/X financial accounts
- Analyst reports and ratings changes

### Tools
- Sentence Transformers (all-MiniLM-L6-v2 or FinBERT for finance-specific)
- ChromaDB or FAISS for local vector storage
- LangChain for RAG pipeline if needed
- QuantConnect Custom Data API to feed signals into live strategies

## Workflow: Idea → Live Trading

```
1. EXPLORE (notebooks)
   └── Explore data, test feature ideas, train models locally

2. BACKTEST (QuantConnect)
   └── Write strategy as QuantConnect algorithm
   └── Run walk-forward backtest with realistic costs
   └── Analyze: Sharpe, Sortino, max drawdown, win rate, profit factor

3. PAPER TRADE (QuantConnect + IBKR Paper)
   └── Deploy to paper trading for 2-4 weeks
   └── Compare live predictions vs actuals
   └── Monitor for data issues, execution problems

4. EVALUATE
   └── Does it beat benchmark?
   └── Is the edge real or was it overfitting?
   └── Is drawdown acceptable?
   └── Are signals uncorrelated with existing live strategies?

5. LIVE TRADE (QuantConnect + IBKR Real)
   └── Deploy with small position sizes initially
   └── Scale up as confidence grows
   └── Monitor daily, review weekly

6. ITERATE
   └── Log everything
   └── Add new features/data sources
   └── Kill strategies that stop working
   └── Promote new ideas from notebook stage
```

## Evaluation Metrics
Every strategy must be evaluated on:
- **Sharpe Ratio**: Risk-adjusted return (target: >1.0)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough loss (target: <20%)
- **Win Rate**: % of trades profitable
- **Profit Factor**: Gross profit / gross loss (target: >1.5)
- **Annual Return**: Absolute performance
- **vs Benchmark**: Must outperform S&P 500 buy-and-hold
- **Correlation**: With existing strategies (want low/negative correlation)

## Old Repo
Old repo: `github.com/chenshoval/intraday-trading-system` (private). Used Azure AKS, Service Bus, IBKR, FinRL. We may cherry-pick useful utility code but the approach has fundamentally changed.

## Code Conventions
- Python 3.11+
- Jupyter notebooks for exploration, .py files for QuantConnect algorithms
- Type hints in utility code
- Feature registry pattern (YAML config + computation functions)
- All strategy parameters in config files, no hardcoded values
- Document every experiment and result in notebooks
