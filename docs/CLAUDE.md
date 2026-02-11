# Intraday Trading System — Project Context

## Vision
Fast-iteration intraday trading agent. The system exists to **test hypotheses quickly and kill bad ideas fast**. Adding a new data source, feature, or model should take <1 day.

## Core Philosophy (from r/algotrading discussion)
- You need a **good data layer** — foundational, not optional
- You need a **fast backtesting system** — quickly test a thesis, throw away what isn't promising immediately
- Indicators (RSI, MAs, BBs, Donchian, Keltner) are all the same thing — they use the past to make a statement about the present. Don't obsess over which ones; obsess over whether they predict anything
- Build **multiple uncorrelated/negatively correlated value streams**, not one strategy
- **Position sizing is where risk lives**, not in the signal (ref: Tom Bassos)
- Ref books: Van Tharp "Trade Your Way to Financial Freedom", Brian Shannon "Technical Analysis Using Multiple Timeframes"

## Architecture — 6 Layers

### 1. Data Ingestion
- **Market Data Gateway**: IBKR TWS API (ib_insync), persistent connection in dedicated pod
- **Data Normalizer**: Raw IBKR → unified schema (OHLCV, bid/ask, volume profiles), publish to message bus
- **Message Bus**: Redis Streams for hot path (<1ms), Azure Service Bus for cold path (audit, alerts)
- **Feature Store**: Redis (hot, for inference) + TimescaleDB (warm, for backtesting). Same feature pipeline for both.

### 2. Feature Engineering
- **Feature Registry**: Central YAML-driven registry. Each feature: name, computation function, dependencies, lookback window. Add new features by adding to registry.
- **Technical Indicators**: ta-lib/pandas-ta, computed incrementally. Configurable periods.
- **Microstructure Features** (phase 2): Order flow imbalance, trade arrival rate, bid-ask bounce, volume clock
- **Alternative Data** (phase 2): Pluggable adapter interface for news sentiment, social signals, sector momentum

### 3. Model Layer
- **Model Registry**: MLflow or custom. Version models, track experiments, store artifacts.
- **Signal Models**: Start with LightGBM (fast, interpretable). Output: directional signal + confidence score, NOT trade decisions. Add RL/DL later.
- **Ensemble / Meta-Learner** (phase 2): Weighted avg → stacking
- **Regime Detector** (phase 2): HMM/Clustering for trending/mean-reverting/volatile/quiet

### 4. Execution Engine
- **Risk Manager**: HARD STOPS — position limits, drawdown limits, exposure limits, correlation checks. Runs BEFORE any order. Cannot be overridden by models.
- **Order Manager**: Signals → orders. Order types, partial fills, cancellations, position tracking. State machine per order.
- **Position Tracker**: Real-time state in Redis, periodic reconciliation with IBKR.
- **Execution Optimizer** (phase 2): TWAP/VWAP algos, smart order routing

### 5. Evaluation Loop (BUILD THIS FIRST)
- **Backtester**: Same feature code as live. Start with vectorbt for speed, build custom event-driven backtester for accuracy (slippage modeling).
- **Paper Trading**: IBKR Paper + shadow mode (new model runs alongside production)
- **Performance Analytics**: Sharpe, Sortino, max drawdown, win rate, profit factor, avg trade duration. Daily P&L attribution.
- **Feedback Logger**: Log EVERY prediction, trade, feature value to TimescaleDB. This is gold for debugging, retraining, feature importance.

### 6. Infrastructure
- **AKS Cluster**: Separate node pools — data (always-on), compute (spot for training), execution (dedicated low-latency)
- **Config Management**: All params in YAML, secrets in Azure Key Vault. Change behavior without code deploys.
- **Monitoring**: Prometheus + Grafana. Alert on data gaps, model drift, risk breaches, latency spikes.
- **CI/CD**: GitHub Actions + ArgoCD. Automated: lint → test → backtest → paper → promote to live. Backtest gate required.

## Key Decisions
| Decision | Choice | Why |
|---|---|---|
| FinRL vs Custom | Custom | FinRL adds abstraction that hurts intraday latency and debugging. Borrow ideas (env design, reward shaping), own the code. |
| Service Bus vs Redis | Redis Streams hot path, Service Bus cold | Intraday needs <10ms. Redis ~1ms. Keep SB for audit/alerts. |
| First model | LightGBM | Working baseline in days not weeks. Interpretable. Add RL once eval pipeline is solid. |
| Backtester | Vectorbt → custom | Vectorbt for speed now. Custom event-driven later for accurate slippage/fill modeling. |

## Build Priority
1. **Week 1-2**: Data pipeline + feature foundation (IBKR connection, normalizer, Redis feature store, 5-10 core features, basic backtester)
2. **Week 3**: First model + evaluation (LightGBM signal, backtest framework, performance metrics, paper trading)
3. **Week 4**: Execution + risk (hard stops, order manager, position tracker, monitoring)
4. **Week 5-6**: Close the loop (feedback logging, retraining pipeline, feature registry, 2nd model, A/B testing)
5. **Phase 2**: Microstructure features, regime detection, RL models, execution optimization, alternative data

## Iteration Cycle
Every new idea goes through: Ingest Data → Engineer Features → Train Model → Backtest → Paper Trade → Evaluate → Deploy/Iterate

## Existing Repo
Old repo: `github.com/chenshoval/intraday-trading-system` (private). Uses Azure AKS, Service Bus, IBKR, FinRL. Starting fresh repo but copying useful pieces from old one.

## Code Conventions
- Python 3.11+
- Adapter/interface pattern for data sources (easy to add new ones)
- Feature registry pattern (YAML config + computation functions)
- Same code path for live and backtest (critical — no separate backtest code)
- All config in YAML, no hardcoded params
- Type hints everywhere
- Tests required for risk management code
