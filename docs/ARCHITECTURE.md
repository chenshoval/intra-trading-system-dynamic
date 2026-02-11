# Intraday Trading System — Architecture & Implementation Blueprint

> **Goal**: Fast-iteration intraday trading. Test hypotheses quickly, kill bad ideas fast.
> Adding a new data source, feature, or model should take **< 1 day**.

---

## Table of Contents
1. [Philosophy](#1-philosophy)
2. [Directory Structure](#2-directory-structure)
3. [Phase 1 — Data Layer](#3-phase-1--data-layer-build-first)
4. [Phase 2 — Feature Engineering](#4-phase-2--feature-engineering)
5. [Phase 3 — Fast Backtesting & Modeling](#5-phase-3--fast-backtesting--modeling)
6. [Phase 4 — Execution & Paper Trading](#6-phase-4--execution--paper-trading)
7. [Infrastructure & Deployment](#7-infrastructure--deployment)
8. [Migration Map — Old Repo → New](#8-migration-map--old-repo--new)
9. [Model Layer — Input/Output Contract & Swappability](#9-model-layer--inputoutput-contract--swappability)
10. [Research & References — Paper-to-Code](#10-research--references--paper-to-code)
11. [Infrastructure Gaps & Deployment Details Per Phase](#11-infrastructure-gaps--deployment-details-per-phase)

---

## 1. Philosophy

From the [r/algotrading discussion](https://reddit.com/r/algotrading) that shaped this system:

- **Good data layer** — foundational, not optional
- **Fast backtesting** — quickly test a thesis, throw away what isn't promising immediately
- **Indicators are all the same** — they use the past to make a statement about the present. Don't obsess over which ones; obsess over **whether they predict anything**
- **Multiple uncorrelated value streams** — not one strategy
- **Position sizing is where risk lives** — not in the signal (ref: Tom Bassos)
- **No FinRL** — FinRL adds abstraction that hurts intraday latency and debugging. Borrow ideas, own the code.
- **LightGBM first** — working baseline in days not weeks. Interpretable. Add RL/DL once eval pipeline is solid.

### Key Technical Decisions

| Decision | Choice | Why |
|---|---|---|
| First model | LightGBM | Fast to train (~seconds), interpretable feature importance, works great on tabular data |
| Backtester | vectorbt → custom | vectorbt for speed now. Custom event-driven later for accurate slippage/fill modeling |
| Hot path messaging | Redis Streams | Intraday needs <10ms. Redis ~1ms |
| Cold path messaging | Azure Service Bus | Audit, alerts, pipeline orchestration |
| Feature store | Redis (hot) + Azure Blob (cold) | Hot for live inference, cold for backtesting |
| Target variable | Configurable via YAML | N-bar forward return, direction classification, multi-class — swap in minutes |
| Live vs backtest code | Same code path | Critical — no separate backtest code. Feature pipeline is identical. |

---

## 2. Directory Structure

```
intra-trading-system-dynamic/
│
├── docs/
│   ├── CLAUDE.md                    # Project vision & principles
│   └── ARCHITECTURE.md              # This file — implementation blueprint
│
├── config/
│   ├── system.yaml                  # Global config (tickers, intervals, Azure resources)
│   ├── features.yaml                # Feature registry (YAML-driven)
│   ├── targets.yaml                 # Target variable definitions
│   ├── models.yaml                  # Model hyperparameters & experiment configs
│   └── risk.yaml                    # Risk limits (hard stops, position sizing)
│
├── src/
│   ├── data/                        # Layer 1: Data Ingestion
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract DataSource interface
│   │   ├── ibkr_adapter.py          # IBKR implementation (ported from old repo)
│   │   ├── normalizer.py            # Raw data → unified OHLCV schema
│   │   ├── csv_adapter.py           # Local CSV loader (for backtesting)
│   │   └── tickers.py               # Ticker universes (SP500, DOW30, etc.)
│   │
│   ├── features/                    # Layer 2: Feature Engineering
│   │   ├── __init__.py
│   │   ├── registry.py              # YAML-driven feature registry
│   │   ├── engine.py                # Feature computation engine
│   │   ├── targets.py               # Target variable generation
│   │   └── indicators/              # Individual indicator implementations
│   │       ├── __init__.py
│   │       ├── moving_averages.py   # SMA, EMA
│   │       ├── momentum.py          # RSI, MACD, ROC
│   │       ├── volatility.py        # ATR, Bollinger, True Range
│   │       ├── volume.py            # VWAP, volume ratios
│   │       ├── price_action.py      # Gap, price position, candlestick
│   │       ├── structure.py         # Support/resistance, swing detection
│   │       └── time_features.py     # Session, calendar features
│   │
│   ├── models/                      # Layer 3: Model Layer
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract SignalModel interface
│   │   ├── lightgbm_model.py        # LightGBM signal model
│   │   ├── registry.py              # Model registry (experiment tracking)
│   │   └── ensemble.py              # Multi-model ensemble (Phase 2+)
│   │
│   ├── execution/                   # Layer 4: Execution Engine
│   │   ├── __init__.py
│   │   ├── risk_manager.py          # HARD STOPS — cannot be overridden
│   │   ├── order_manager.py         # Signal → order state machine
│   │   └── position_tracker.py      # Real-time position state
│   │
│   ├── evaluation/                  # Layer 5: Evaluation Loop
│   │   ├── __init__.py
│   │   ├── backtester.py            # vectorbt-based backtester
│   │   ├── analytics.py             # Sharpe, Sortino, drawdown, profit factor
│   │   ├── experiment.py            # Hypothesis → result workflow
│   │   └── feedback_logger.py       # Log predictions, trades, features
│   │
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── azure_blob.py            # Azure Blob read/write (ported)
│       ├── service_bus.py           # Azure Service Bus (ported)
│       ├── logger.py                # Logging with timezone support (ported)
│       ├── dates.py                 # Date/quarter utilities (ported)
│       └── env.py                   # Environment variable helpers (ported)
│
├── services/                        # Deployable microservices
│   ├── fetcher/                     # Data fetching service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py                  # Entrypoint: fetch → blob → notify
│   │
│   ├── feature_engine/              # Feature engineering service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py                  # Entrypoint: raw data → features → blob
│   │
│   └── model_trainer/               # Model training service (Phase 3+)
│       ├── Dockerfile
│       ├── requirements.txt
│       └── main.py
│
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_backtest_hypothesis.ipynb
│
├── scripts/                         # Local development & one-off scripts
│   ├── fetch_local.py               # Run fetcher locally (no K8s needed)
│   ├── backtest.py                  # Run a backtest from CLI
│   └── feature_importance.py        # Analyze feature predictive power
│
├── deployment/                      # Infrastructure & deployment
│   ├── deploy.sh                    # Master deployment orchestrator
│   ├── deploy-infrastructure.sh     # ARM template deployment
│   ├── setup-service-bus.sh         # Create Service Bus queues
│   ├── build-and-push.sh            # Build Docker images → ACR
│   ├── manage-cluster.sh            # Start/stop AKS cluster
│   ├── trading-system-infra.json    # ARM template
│   └── helm/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── fetcher/
│           ├── feature-engine/
│           ├── ibkr-gateway/
│           └── shared/
│
├── tests/
│   ├── test_data/                   # Test fixtures (small CSV samples)
│   ├── test_features.py
│   ├── test_targets.py
│   ├── test_backtester.py
│   └── test_risk_manager.py         # Required by code conventions
│
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 3. Phase 1 — Data Layer (BUILD FIRST)

**Goal**: Deploy the IBKR data pipeline to Azure, fetch real market data, verify it lands in Blob Storage.

### 3.1 Data Source Interface

```python
# src/data/base.py — Abstract interface, all data sources implement this
class DataSource(ABC):
    @abstractmethod
    def fetch(self, tickers: list[str], start: datetime, end: datetime,
              interval: str = "5m") -> pd.DataFrame:
        """Returns DataFrame with unified schema: date, open, high, low, close, volume, ticker"""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Is the data source available?"""
```

**Why adapter pattern?** Tomorrow you might want to add Polygon.io, Alpaca, or a CSV replay adapter. Same interface, plug it in.

### 3.2 IBKR Adapter — Port from Old Repo

**Source**: `c:\ML\intraday-trading-system\fetcher\data_fetcher\IBKRDataFetcher.py` (659 lines)

**What to keep**:
- `ib_insync` connection with retry logic (3 retries, client ID rotation)
- 29-day chunk fetching (IBKR API limit)
- Progress tracking via metadata.json on PVC (resume on crash)
- UTC timestamp standardization
- Deduplication by date

**What to change**:
- Implement `DataSource` interface
- Remove hardcoded paths → read from `config/system.yaml`
- Add type hints everywhere
- Output unified schema (see normalizer below)

### 3.3 Data Normalizer

All data sources produce the same schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64[ns, UTC] | UTC timezone-aware timestamp |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | float64 | Trade volume (-1 if unavailable from IBKR) |
| `ticker` | str | Ticker symbol (renamed from `tic` in old repo) |

### 3.4 Ticker Universes

**Port from**: `c:\ML\intraday-trading-system\fetcher\ConfigTickers.py`

| Universe | Count | Config Key |
|----------|-------|------------|
| `SINGLE_TICKER` | 1 | AAPL (testing) |
| `DOW_30` | 30 | Blue chips |
| `SP_500` | ~503 | Full S&P 500 |
| `NAS_100` | 86 | NASDAQ-100 |

### 3.5 Azure Blob Storage

**Port from**: `c:\ML\intraday-trading-system\utils\azureBlobReader.py` + `azureBlobWriter.py`

Functions to port:
- `upload_to_blob(df, filename, container, account_url)` — chunked block upload for large DataFrames
- `download_blob_to_dataframe(container, account_url, blob_name)` → DataFrame
- `list_blob_files(container, account_url, prefix)` → list of blob names
- `blob_exists(container, account_url, blob_name)` → bool
- `discover_raw_data_files(container, account_url)` → list of unprocessed files
- `find_missing_processed_files(container, account_url)` → gap analysis

All use `DefaultAzureCredential` (Azure Workload Identity in K8s, `az login` locally).

### 3.6 Service Bus (Cold Path) — Topics, Not Queues

**Port from**: `c:\ML\intraday-trading-system\utils\serviceBusWriter.py`
**Breaking change**: Old repo used **queues** (point-to-point). New repo uses **topics + subscriptions** (pub/sub with fan-out). See [Section 11.1](#111-service-bus-queues--topics-migration) for full migration details.

4 topics with subscriptions:

| Topic | Publisher | Subscriptions | Message |
|-------|-----------|---------------|---------|
| `raw-data-updates` | Fetcher | `feature-engine`, `audit-log` | `{"event_type": "raw_data_available", "data": {"file_path": "...", "quarter": "...", ...}}` |
| `processed-data-updates` | Feature Engine | `model-trainer` | `{"event_type": "processed_data_available", ...}` |
| `model-creation` | Trainer | `model-evaluator` | `{"event_type": "model_trained", ...}` |
| `model-updates` | Evaluator | `paper-trader` | `{"event_type": "model_validated", ...}` |

### 3.7 Fetcher Service

**File**: `services/fetcher/main.py`

Orchestrates: read config → connect IBKR → fetch by quarter → upload to blob → notify via Service Bus.

Two modes (same as old repo):
- **Daily**: Fetch previous trading day, append to quarterly file
- **Quarterly**: Fetch a date range (historical backfill)

### 3.8 Deployment — Reuse Existing Azure Resources

| Resource | Name | Already Exists |
|----------|------|----------------|
| Resource Group | `trading-system-rg` | ✅ |
| ACR | `tradingsystemacr.azurecr.io` | ✅ |
| Storage Account | `tradingsystemsa` | ✅ |
| Service Bus | `trading-system-service-bus` | ✅ |
| Key Vault | `trading-system-plat-kv` | ✅ |
| Managed Identity | `trading-system-mi` | ✅ |
| AKS Cluster | `trading-system-aks` | ✅ |

**Deployment scripts** (adapted from old repo):
- `deployment/build-and-push.sh` — Build Docker image, push to ACR
- `deployment/manage-cluster.sh` — Start/stop AKS (`./manage-cluster.sh up|down`)
- `deployment/deploy.sh` — Deploy Helm chart to AKS

**Helm chart** (adapted from old repo):
- IBKR Gateway deployment (always-on, ports 4001/4004)
- Fetcher CronJob (daily 23:00 UTC or on-demand quarterly)
- Feature Engine deployment (KEDA-scaled 0→10 on Service Bus queue depth)
- Shared ServiceAccount with Azure Workload Identity

### 3.9 Phase 1 Verification

```
End-to-end test:
1. Start AKS cluster: ./deployment/manage-cluster.sh up
2. Deploy Helm chart: ./deployment/deploy.sh
3. Trigger fetcher (quarterly mode, small date range, 5 tickers)
4. Verify: az storage blob list --container trading-data-blob --prefix "SP_500/"
5. Verify: Service Bus message received on "raw-data-updates" queue
6. Check data schema matches normalizer spec (7 columns, UTC dates)
```

---

## 4. Phase 2 — Feature Engineering

**Goal**: YAML-driven feature registry. Same code path for live and backtest. Configurable targets.

### 4.1 Feature Registry — The Core Innovation

```yaml
# config/features.yaml
features:
  sma_5:
    function: moving_averages.sma
    params: { window: 5, column: close }
    lookback: 5
    category: trend

  rsi_14:
    function: momentum.rsi
    params: { window: 14 }
    lookback: 14
    category: momentum

  atr_14:
    function: volatility.atr
    params: { window: 14 }
    lookback: 15  # needs previous close
    category: volatility
    dependencies: [true_range]

  volume_ratio:
    function: volume.volume_ratio
    params: { short_window: 5, long_window: 20 }
    lookback: 20
    category: volume

# Adding a new feature = adding 4 lines of YAML + a computation function
```

**Registry engine** (`src/features/registry.py`):
- Reads `features.yaml` at startup
- Resolves dependency graph (topological sort)
- Computes features in correct order
- Validates lookback requirements
- **Same code runs for both live inference and backtest** — the engine doesn't know which mode it's in

### 4.2 Features to Port (43 from old repo)

All computation logic from `c:\ML\intraday-trading-system\data_engineering\data_processor\feature_engineer.py`, split into modules:

| Module | Features | Count |
|--------|----------|-------|
| `moving_averages.py` | sma_5/10/20/50, ema_5/10/20, sma_5_20_ratio, price_sma_20_ratio | 9 |
| `momentum.py` | returns, log_returns, roc_5/10, rsi_14, macd, macd_signal, macd_histogram | 8 |
| `volatility.py` | volatility_5/20, true_range, atr_14 | 4 |
| `volume.py` | volume_sma_5/20, volume_ratio, price_volume, vwap_5 | 5 |
| `price_action.py` | price_position, gap, intraday_range, body_range_ratio | 4 |
| `structure.py` | local_max/min, higher_high, lower_low | 4 |
| `time_features.py` | hour, minute, day_of_week, day_of_month, month, quarter, market_open_minutes, is_market_open, is_pre_market, is_after_hours | 10 |

### 4.3 Target Variable System

```yaml
# config/targets.yaml
targets:
  # Test all of these — obsess over which predicts, not which indicator
  forward_return_5:
    type: regression
    function: targets.forward_return
    params: { bars_ahead: 5 }
    description: "5-bar forward return (25 min at 5m bars)"

  direction_5:
    type: classification
    function: targets.direction
    params: { bars_ahead: 5, threshold: 0.001 }  # 0.1% threshold
    classes: [short, hold, long]
    description: "Direction in 5 bars with dead zone"

  direction_12:
    type: classification
    function: targets.direction
    params: { bars_ahead: 12, threshold: 0.002 }
    classes: [short, hold, long]
    description: "Direction in 1 hour with 0.2% dead zone"

  # Easy to add more — this is the "obsess over whether they predict" part
  volatility_regime:
    type: classification
    function: targets.volatility_regime
    params: { window: 20, threshold_low: 0.5, threshold_high: 1.5 }
    classes: [low_vol, normal, high_vol]
```

**Why configurable?** The Reddit wisdom: don't obsess over indicators, obsess over whether they predict. We need to rapidly test: "Does RSI predict 5-bar returns? What about 12-bar? What about with a 0.2% dead zone?" Change the YAML, re-run, compare.

### 4.4 Feature Engine Service

**File**: `services/feature_engine/main.py`

Two modes:
- **Service Bus listener** (production): KEDA-scaled, processes `raw-data-updates` messages
- **Direct mode** (backtesting): `engine.compute_features(df)` — same code, no Service Bus

### 4.5 Phase 2 Verification

```
1. Load a raw CSV from blob storage
2. Run feature engine: python scripts/feature_importance.py --data SP_500/2025Q1/...csv
3. Verify: 43 features computed, no NaN in output (after lookback warmup)
4. Verify: target variables generated for all configured targets
5. Run feature importance analysis: which features predict which targets?
```

---

## 5. Phase 3 — Fast Backtesting & Modeling

**Goal**: Hypothesis → result in minutes. The core value proposition.

### 5.1 The Hypothesis Workflow

```
1. Have an idea: "RSI divergence + volume spike predicts 5-bar reversal"
2. Add features to features.yaml (if new ones needed)
3. Define target in targets.yaml (if new one needed)
4. Run: python scripts/backtest.py --features rsi_14,volume_ratio --target direction_5 --tickers DOW_30
5. Get results: Sharpe, win rate, drawdown, equity curve — in under 5 minutes
6. If promising → iterate. If not → kill it immediately and move on.
```

**This is the < 1 day cycle the system is designed for.**

### 5.2 LightGBM Signal Model

```python
# src/models/lightgbm_model.py
class LightGBMSignal(SignalModel):
    """
    Outputs: directional signal + confidence score
    NOT trade decisions — that's the execution engine's job
    """
    def train(self, X_train, y_train, X_val, y_val) -> dict:
        # Train LightGBM classifier/regressor
        # Return: training metrics, feature importance

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Return: signal (-1 to 1), confidence (0 to 1)
        # Signal > 0 = bullish, < 0 = bearish
        # Confidence = model's certainty (used for position sizing)
```

**Why LightGBM first?**
- Trains in seconds (not hours like RL)
- Built-in feature importance → tells you which features actually predict
- Handles missing values natively
- No GPU needed
- Interpretable → you can debug why it's wrong

### 5.3 Vectorbt Backtester

```python
# src/evaluation/backtester.py
class Backtester:
    """
    Vectorized backtesting via vectorbt.
    Same feature code as live — features are pre-computed, not computed during backtest.
    """
    def run(self, signals: pd.DataFrame, prices: pd.DataFrame,
            config: BacktestConfig) -> BacktestResult:
        # signals: output from model.predict()
        # prices: OHLCV data
        # config: position sizing, costs, slippage
        # Returns: BacktestResult with full analytics
```

### 5.4 Performance Analytics

**Port metrics from**: `c:\ML\intraday-trading-system\model_training\trainer\model_evaluator.py`

| Metric | Category | Formula |
|--------|----------|---------|
| Sharpe Ratio | Risk-adjusted | `mean(returns) / std(returns) * sqrt(252)` |
| Sortino Ratio | Risk-adjusted | `mean(returns) / downside_std * sqrt(252)` |
| Max Drawdown | Risk | `max(peak - trough) / peak` |
| Win Rate | Trading | `winning_trades / total_trades` |
| Profit Factor | Trading | `gross_profits / gross_losses` |
| Avg Trade Duration | Trading | `mean(exit_time - entry_time)` |
| Daily P&L Attribution | Analysis | P&L broken down by feature contribution |
| Calmar Ratio | Risk-adjusted | `annualized_return / max_drawdown` |

### 5.5 Experiment Tracking

```python
# src/evaluation/experiment.py
class Experiment:
    """
    Every hypothesis is an experiment. Log everything.
    """
    def __init__(self, name: str, config: dict):
        self.name = name          # e.g., "rsi_divergence_volume_spike_v1"
        self.features = config['features']
        self.target = config['target']
        self.model_params = config['model']
        self.backtest_config = config['backtest']

    def run(self) -> ExperimentResult:
        # 1. Load data
        # 2. Compute features (via registry)
        # 3. Generate targets
        # 4. Train model
        # 5. Backtest
        # 6. Log everything to feedback logger
        # 7. Return results
```

### 5.6 Phase 3 Verification

```
1. Run end-to-end experiment:
   python scripts/backtest.py \
     --features sma_5,sma_20,rsi_14,volume_ratio,atr_14 \
     --target direction_5 \
     --tickers AAPL,MSFT,GOOGL \
     --start 2024-01-01 --end 2024-12-31

2. Verify output:
   - Sharpe ratio, win rate, max drawdown printed
   - Equity curve plotted
   - Feature importance ranked
   - Total runtime < 5 minutes for 3 tickers × 1 year × 5-min bars

3. Run a second experiment with different features/target — verify < 5 min turnaround
```

---

## 6. Phase 4 — Execution & Paper Trading

**Goal**: Connect signals to IBKR paper trading. Risk management is non-negotiable.

### 6.1 Risk Manager — HARD STOPS

```python
# src/execution/risk_manager.py — Runs BEFORE any order. Cannot be overridden.
class RiskManager:
    """
    All limits from config/risk.yaml. No model can bypass these.
    """
    def check_order(self, order: Order, portfolio: Portfolio) -> RiskDecision:
        # Position limits: max shares per stock, max positions
        # Drawdown limits: daily, weekly, total
        # Exposure limits: max sector exposure, max correlation
        # Time limits: no trades in last 15 min of session
```

```yaml
# config/risk.yaml
limits:
  max_position_pct: 0.05        # Max 5% of portfolio in one stock
  max_positions: 20              # Max 20 concurrent positions
  daily_loss_limit_pct: 0.02    # Stop trading if down 2% today
  weekly_loss_limit_pct: 0.05   # Stop trading if down 5% this week
  max_drawdown_pct: 0.10        # Hard stop at 10% drawdown
  no_trade_before_close_min: 15  # No new trades in last 15 minutes
```

### 6.2 Paper Trading Integration

- IBKR Paper account (port 4004, already configured in old Helm chart)
- Shadow mode: new model runs alongside production, signals logged but not executed
- 3-day validation gate before promoting to live

---

## 7. Infrastructure & Deployment

### 7.1 Existing Azure Resources (REUSE)

```
Resource Group: trading-system-rg (East US)
├── ACR: tradingsystemacr.azurecr.io
├── Storage: tradingsystemsa.blob.core.windows.net
│   └── Container: trading-data-blob
├── Service Bus: trading-system-service-bus
│   ├── Topic: raw-data-updates        (was: queue — migrate)
│   ├── Topic: processed-data-updates   (was: queue — migrate)
│   ├── Topic: model-creation           (was: queue — migrate)
│   └── Topic: model-updates            (was: queue — migrate)
├── Key Vault: trading-system-plat-kv
│   ├── Secret: ibusername
│   └── Secret: ibpassword
├── Managed Identity: trading-system-mi
│   └── Client ID: 1d7b3a21-f6e5-4cb5-8eb1-009472652db5
└── AKS: trading-system-aks
    ├── System pool: 2 × Standard_D2a_v4
    └── Worker pool: 0-1 × Standard_D4a_v4 (autoscale)
```

### 7.2 Deployment Scripts

| Script | Purpose | Ported From |
|--------|---------|-------------|
| `deployment/deploy.sh` | Master orchestrator | `Deployment/deploy.sh` |
| `deployment/build-and-push.sh` | Build Docker → push to ACR | `Deployment/DeployFetcherToAcr.sh` etc. |
| `deployment/manage-cluster.sh` | Start/stop AKS | `Deployment/ManageCluster.sh` |
| `deployment/setup-service-bus.sh` | Create SB queues | `Deployment/setup-service-bus.sh` |

### 7.3 Helm Chart Structure

```yaml
# deployment/helm/values.yaml (key sections)
global:
  image:
    registry: tradingsystemacr.azurecr.io

ibkrGateway:
  tradingMode: paper
  port: 4004

fetcher:
  schedule: "0 23 * * *"
  tickerGroup: SP_500
  interval: 5m
  mode: daily

featureEngine:
  processorMode: scanning_gaps
  autoscaling:
    minReplicas: 0
    maxReplicas: 10
```

### 7.4 Local Development

For running without K8s/Azure:

```bash
# Fetch data locally (requires IBKR Gateway running)
python scripts/fetch_local.py --tickers AAPL,MSFT --start 2024-01-01 --end 2024-03-31

# Run backtest on local CSV
python scripts/backtest.py --data ./data/local/ --features all --target direction_5

# Feature importance analysis
python scripts/feature_importance.py --data ./data/local/
```

---

## 8. Migration Map — Old Repo → New

### Files to Port (with modifications)

| Old File | New Location | Changes |
|----------|-------------|---------|
| `fetcher/data_fetcher/IBKRDataFetcher.py` | `src/data/ibkr_adapter.py` | Implement `DataSource` interface, type hints, config from YAML |
| `fetcher/ConfigTickers.py` | `src/data/tickers.py` | Clean up, add type hints |
| `data_engineering/data_processor/feature_engineer.py` | `src/features/indicators/*.py` | Split into modules, wrap in registry pattern |
| `utils/azureBlobReader.py` + `azureBlobWriter.py` | `src/utils/azure_blob.py` | Combine, add type hints, use structured logging |
| `utils/serviceBusWriter.py` | `src/utils/service_bus.py` | Add type hints, use structured logging |
| `utils/logger.py` | `src/utils/logger.py` | Minimal changes |
| `utils/dateUtils.py` | `src/utils/dates.py` | Add type hints |
| `utils/envUtils.py` | `src/utils/env.py` | Add type hints |
| `model_training/trainer/model_evaluator.py` | `src/evaluation/analytics.py` | Extract metric formulas, drop RL-specific code |
| `helm/*` | `deployment/helm/*` | Restructure, remove model-training (replace with LightGBM) |
| `Deployment/*.sh` | `deployment/*.sh` | Adapt paths, simplify |

### Files to Write Fresh

| File | Purpose |
|------|---------|
| `src/data/base.py` | Abstract DataSource interface |
| `src/data/normalizer.py` | Unified schema normalization |
| `src/data/csv_adapter.py` | Local CSV loader for backtesting |
| `src/features/registry.py` | YAML-driven feature registry |
| `src/features/engine.py` | Feature computation engine |
| `src/features/targets.py` | Configurable target variables |
| `src/models/base.py` | Abstract SignalModel interface |
| `src/models/lightgbm_model.py` | LightGBM signal model |
| `src/evaluation/backtester.py` | vectorbt backtester |
| `src/evaluation/experiment.py` | Hypothesis workflow |
| `src/execution/risk_manager.py` | Hard stops |
| `config/*.yaml` | All configuration files |
| `scripts/*.py` | CLI tools for local dev |
| `pyproject.toml` | Project dependencies |

### Files NOT Carried Over (Dead Code)

| Old File | Reason |
|----------|--------|
| `ExtendedStockTradingEnv.py` | FinRL — abandoned |
| `ExtendedDRLAgent.py` | FinRL — abandoned |
| `ModelTrainer.py` | Old monolithic trainer |
| `fetcher/YahooDownloader.py` | Deprecated |
| `model_training/trainer/model_interface.py` | SB3/RL-specific |
| `model_training/trainer/storage_manager.py` | RL model storage, not needed for LightGBM |
| `model_training/trainer/data_loader.py` | RL-specific data loading |
| `model_training/config/model_configs.yaml` | PPO/DQN/A2C configs |

---

## 9. Model Layer — Input/Output Contract & Swappability

### 9.1 The Universal Model Interface

Every model — LightGBM, XGBoost, Transformer, RL — implements the same interface. Swapping models is changing one line in `config/models.yaml`.

```python
# src/models/base.py
class SignalModel(ABC):
    """
    Universal interface. ALL models output the same thing.
    The execution engine doesn't know or care what model produced the signal.
    """

    # ── INPUT ──
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              config: dict) -> TrainResult:
        """
        X:      Feature matrix (N rows × F features). Columns match features.yaml.
        y:      Target vector (N rows). From targets.yaml.
        X_val:  Validation features (for early stopping / hyperparameter tuning).
        y_val:  Validation targets.
        config: Model-specific hyperparameters from models.yaml.

        Returns TrainResult:
            - metrics: dict (train_loss, val_loss, etc.)
            - feature_importance: dict[str, float]
            - training_time_seconds: float
        """

    # ── OUTPUT ──
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> SignalOutput:
        """
        X: Feature matrix (same schema as training).

        Returns SignalOutput:
            - signal: float [-1.0 to 1.0]
                -1.0 = strong short, 0.0 = neutral, 1.0 = strong long
            - confidence: float [0.0 to 1.0]
                Model's certainty. Used by execution engine for position sizing.
            - metadata: dict (optional — model-specific debug info)
        """

    @abstractmethod
    def save(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'SignalModel': ...

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Model name, version, framework, parameter count, training date."""
```

### 9.2 SignalOutput — The Universal Output

This is the contract between the model layer and the execution engine:

```python
@dataclass
class SignalOutput:
    signal: float       # -1.0 (strong short) to 1.0 (strong long)
    confidence: float   # 0.0 (no idea) to 1.0 (certain)
    metadata: dict      # Model-specific debug info

    # Execution engine uses signal + confidence for position sizing:
    # position_size = base_size * abs(signal) * confidence
    # direction = sign(signal)
```

**Why this contract matters:**
- The execution engine, risk manager, and backtester all consume `SignalOutput`
- They never touch model internals
- You can swap LightGBM → Transformer → RL without changing anything downstream
- You can ensemble models by averaging their `SignalOutput` values

### 9.3 Model Implementations

```yaml
# config/models.yaml
models:
  # ── Phase 1: Start here ──
  lgbm_direction:
    framework: lightgbm
    class: LightGBMSignal
    task: classification          # or "regression"
    target: direction_5           # references targets.yaml
    features: all                 # or explicit list
    hyperparameters:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 500
      min_child_samples: 20
      subsample: 0.8
      colsample_bytree: 0.8
      early_stopping_rounds: 50
    enabled: true

  # ── Phase 2: Add when eval pipeline is solid ──
  xgboost_direction:
    framework: xgboost
    class: XGBoostSignal
    task: classification
    target: direction_5
    hyperparameters:
      max_depth: 6
      learning_rate: 0.05
      n_estimators: 500
      subsample: 0.8
    enabled: false

  # ── Phase 3: Transformer for sequence modeling ──
  temporal_transformer:
    framework: pytorch
    class: TemporalTransformerSignal
    task: classification
    target: direction_12
    hyperparameters:
      d_model: 64
      nhead: 4
      num_layers: 2
      sequence_length: 60         # 60 bars = 5 hours at 5m
      dropout: 0.1
      learning_rate: 0.0001
      batch_size: 256
      epochs: 50
    enabled: false

  # ── Phase 4: RL when everything else works ──
  ppo_agent:
    framework: stable_baselines3
    class: RLSignalModel           # Wraps RL output into SignalOutput
    task: rl
    target: null                   # RL uses reward, not target
    hyperparameters:
      algorithm: PPO
      n_steps: 2048
      batch_size: 128
      learning_rate: 0.00025
      gamma: 0.99
      total_timesteps: 500000
    environment:
      reward: sharpe               # sharpe, pnl, sortino
      initial_capital: 1000000
      transaction_cost_pct: 0.001
    enabled: false
```

### 9.4 How Fast Can We Swap?

| Switch | What Changes | What Stays the Same | Time |
|--------|-------------|---------------------|------|
| LightGBM → XGBoost | `models.yaml`: set `xgboost_direction.enabled: true` | Features, targets, backtester, execution, risk | 5 minutes |
| Classification → Regression | `models.yaml`: change `task` + `targets.yaml` | Features, model code, backtester | 10 minutes |
| Add Transformer | Write `src/models/transformer_model.py` implementing `SignalModel` | Everything else | 1 day |
| Add RL agent | Write `src/models/rl_model.py` + Gymnasium env wrapping `SignalOutput` | Features, execution, risk, backtester | 2-3 days |
| Ensemble 2 models | `models.yaml`: enable both, set `ensemble.enabled: true` | Individual models, execution | 30 minutes |

### 9.5 RL Re-Integration Path

When RL makes sense (after eval pipeline is proven):

```python
# src/models/rl_model.py
class RLSignalModel(SignalModel):
    """
    Wraps any RL agent (PPO, DQN, SAC) into the SignalOutput contract.
    The RL agent sees features as observation, outputs action.
    We map action → signal + confidence.
    """
    def predict(self, X: pd.DataFrame) -> SignalOutput:
        observation = X.values[-1]  # Latest feature vector
        action, _states = self.agent.predict(observation, deterministic=True)

        # Map RL action space to SignalOutput
        # Action 0=sell, 1=hold, 2=buy (discrete)
        # Or continuous action in [-1, 1]
        signal = self._action_to_signal(action)
        confidence = self._action_to_confidence(action, _states)

        return SignalOutput(signal=signal, confidence=confidence, metadata={})
```

**Key insight**: RL doesn't replace the interface — it plugs into it. The backtester, risk manager, and execution engine never know it's RL underneath.

---

## 10. Research & References — Paper-to-Code

### 10.1 Supervised Learning for Intraday (What We're Building First)

| Paper / Source | Key Idea | How We Use It |
|---|---|---|
| **"Machine Learning for Trading"** (Marcos López de Prado, 2018-2020) | Triple barrier method for labeling: set profit-taking, stop-loss, and time barriers. Label = which barrier is hit first. | Use as an advanced target in `targets.yaml` — better than simple N-bar forward return because it accounts for path dependency. |
| **"Advances in Financial Machine Learning"** (López de Prado) | Meta-labeling: first model predicts direction, second model predicts whether to act on the signal (confidence). | Phase 2: Train LightGBM for direction, then a second LightGBM as meta-model for confidence. Maps directly to our `signal + confidence` output. |
| **"Financial Machine Learning"** (Kaastra & Boyd → modern successors) | Feature importance via permutation, not just model-internal importance. Cross-validated purged walk-forward. | Use `sklearn.inspection.permutation_importance` in our experiment workflow. Use purged k-fold instead of naive train/test split. |
| **Gradient Boosting frameworks comparison** (various Kaggle competitions) | LightGBM beats XGBoost on speed for tabular data. CatBoost handles categoricals natively. | Start LightGBM, add XGBoost/CatBoost via `models.yaml` swap. |

**Implementation: Triple Barrier Labeling**
```yaml
# config/targets.yaml — add this
triple_barrier_5:
  type: classification
  function: targets.triple_barrier
  params:
    bars_ahead: 5              # max holding period
    profit_take_pct: 0.005     # 0.5% profit target
    stop_loss_pct: 0.003       # 0.3% stop loss
  classes: [stop_loss, timeout, profit_take]
  description: "Triple barrier: 0.5% PT / 0.3% SL / 5 bar timeout"
```

### 10.2 Walk-Forward Validation (Critical for Time Series)

Standard train/test split leaks future information. We need **purged walk-forward**:

```
──────────────────────────────────────────── Time →
[████ Train ████][Gap][██ Test ██]
                 [████ Train ████][Gap][██ Test ██]
                                  [████ Train ████][Gap][██ Test ██]

Gap = purge window (remove data too close to test boundary to prevent leakage)
```

```python
# src/evaluation/backtester.py — walk-forward method
class Backtester:
    def walk_forward(self, data, model_class, config,
                     train_window=252*78,    # 1 year of 5-min bars
                     test_window=20*78,      # 1 month of 5-min bars
                     purge_window=12,        # 1 hour purge gap
                     step=20*78              # step forward 1 month
                     ) -> list[BacktestResult]:
        """
        Sliding window backtest. Retrain model at each step.
        Returns list of BacktestResult for each test window.
        """
```

### 10.3 Transformer / Deep Learning (Phase 3)

| Paper | Key Idea | Implementation Path |
|---|---|---|
| **"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"** (Lim et al., 2021) | TFT: attention-based model that outputs variable importance + temporal patterns. Multi-horizon prediction. | `src/models/tft_model.py` — predicts multiple horizons at once (5-bar, 12-bar, 24-bar). Interpretable via attention weights. |
| **"Attention Is All You Need"** applied to finance (various 2023-2025) | Standard transformer encoder on price sequences. Positional encoding captures time-of-day patterns. | `src/models/transformer_model.py` — sequence input (last 60 bars), output `SignalOutput`. |
| **"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"** (Zhou et al.) | ProbSparse attention for long sequences (efficient for intraday with many bars). | Use Informer architecture if standard transformer is too slow on full-day sequences. |
| **"PatchTST: A Time Series is Worth 64 Words"** (Nie et al., 2023) | Patch-based tokenization of time series. State-of-art on many benchmarks. | Consider as transformer variant — patches of 12 bars (1 hour) as tokens. |

### 10.4 Reinforcement Learning (Phase 4 — When Eval Pipeline Is Proven)

| Paper | Key Idea | Implementation Path |
|---|---|---|
| **"Deep Reinforcement Learning for Automated Stock Trading"** (Yang et al., 2020) — the FinRL paper | Ensemble of PPO/A2C/DDPG. Custom trading env with Sharpe reward. | We already had this in old repo. Re-implement as `RLSignalModel` wrapping SB3 into `SignalOutput`. |
| **"FinRL-Meta: Market Environments and Benchmarks"** (Liu et al., 2022) | Standardized market environments. Data-centric approach. | Borrow env design (state=features+positions, action=portfolio weights) but own the code. |
| **"Model-Based RL for Trading"** (various 2024-2025) | Learn a world model of market dynamics, then plan in the model. More sample-efficient than model-free RL. | Phase 5+. Use MuZero/Dreamer-style approach. Train world model on our feature data. |
| **"Multi-Agent RL for Market Making"** (various 2024) | Multiple agents for different strategies (trend, mean-revert, momentum). Emergent behavior. | Phase 5+. Each agent outputs `SignalOutput`, ensemble combines. |

### 10.5 Microstructure & Alternative Features (Phase 2+)

| Source | Features | Implementation |
|---|---|---|
| **Order flow imbalance** (Cont et al.) | `bid_volume - ask_volume` at top of book. Strong short-term predictor. | Requires Level 2 data from IBKR. Add `src/features/indicators/microstructure.py`. |
| **VPIN** (Easley, López de Prado) | Volume-synchronized probability of informed trading. Measures toxicity of order flow. | Compute on volume-bucketed bars instead of time bars. |
| **Kyle's Lambda** | Price impact per unit of order flow. Measures market depth. | Add to microstructure features module. |
| **Sector momentum** | Relative strength of stock vs. sector ETF. | Add `src/features/indicators/cross_asset.py`. Compare stock to SPY/QQQ/sector ETFs. |
| **News sentiment** (FinBERT, various) | NLP on financial news → sentiment score per ticker. | Pluggable adapter: `src/data/news_adapter.py`. Feature: `news_sentiment_1h`. |

### 10.6 Open-Source Repos to Study

| Repo | What It Does | What to Borrow |
|---|---|---|
| **vectorbt** (`polakowo/vectorbt`) | Vectorized backtesting framework. Extremely fast. | Our backtester wraps this. Use `vbt.Portfolio.from_signals()`. |
| **QLib** (`microsoft/qlib`) | Microsoft's quant framework. Model zoo, data layer, backtest. | Study their feature expression engine and rolling training approach. |
| **mlfinlab** (`hudson-and-thames/mlfinlab`) | Implements López de Prado's book. Triple barrier, meta-labeling, purged k-fold. | Port triple barrier labeling and purged walk-forward validation. |
| **FinRL** (`AI4Finance-Foundation/FinRL`) | RL trading framework. The one we're NOT using but borrowing from. | Env design patterns (state representation, reward shaping). |
| **zipline-reloaded** | Event-driven backtester (Quantopian heritage). | Reference for custom event-driven backtester in Phase 5. |
| **LightGBM** (`microsoft/LightGBM`) | The model we're using. | Built-in feature importance, early stopping, GPU support. |

---

## 11. Infrastructure Gaps & Deployment Details Per Phase

### 11.1 Service Bus: Queues → Topics Migration

**Current state (old repo)**: Uses 4 **queues** with partitioning.
**Problem**: Queues are point-to-point (1 producer → 1 consumer). As the system grows, we want one event to trigger multiple consumers (e.g., "raw data available" should trigger both feature engineering AND a data quality monitor).
**Solution**: Migrate to **topics + subscriptions**.

```
OLD (Queues — point to point):
  Fetcher ──→ [raw-data-updates queue] ──→ Feature Engine

NEW (Topics — pub/sub with fan-out):
  Fetcher ──→ [raw-data-updates topic]
                 ├──→ [feature-engine subscription] ──→ Feature Engine
                 ├──→ [data-quality subscription]   ──→ Data Quality Monitor
                 └──→ [audit-log subscription]      ──→ Audit Logger
```

**Migration script** (`deployment/setup-service-bus.sh`):

```bash
#!/bin/bash
RESOURCE_GROUP="trading-system-rg"
NAMESPACE_NAME="trading-system-service-bus"

# ── Delete old queues (if migrating) ──
echo "🗑️ Removing old queues..."
for QUEUE in raw-data-updates processed-data-updates model-creation model-updates; do
    az servicebus queue delete \
        --resource-group $RESOURCE_GROUP \
        --namespace-name $NAMESPACE_NAME \
        --name $QUEUE 2>/dev/null
done

# ── Create Topics ──
echo "📨 Creating topics..."
for TOPIC in raw-data-updates processed-data-updates model-creation model-updates; do
    az servicebus topic create \
        --resource-group $RESOURCE_GROUP \
        --namespace-name $NAMESPACE_NAME \
        --name $TOPIC \
        --default-message-time-to-live PT2H \
        --enable-partitioning true \
        --max-size 1024
    echo "  ✅ Topic: $TOPIC"
done

# ── Create Subscriptions ──
echo "📬 Creating subscriptions..."

# raw-data-updates topic
az servicebus topic subscription create \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $NAMESPACE_NAME \
    --topic-name raw-data-updates \
    --name feature-engine \
    --lock-duration PT30S \
    --max-delivery-count 10 \
    --default-message-time-to-live PT2H

az servicebus topic subscription create \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $NAMESPACE_NAME \
    --topic-name raw-data-updates \
    --name audit-log \
    --lock-duration PT10S \
    --max-delivery-count 3

# processed-data-updates topic
az servicebus topic subscription create \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $NAMESPACE_NAME \
    --topic-name processed-data-updates \
    --name model-trainer \
    --lock-duration PT30S \
    --max-delivery-count 10

# model-creation topic
az servicebus topic subscription create \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $NAMESPACE_NAME \
    --topic-name model-creation \
    --name model-evaluator \
    --lock-duration PT60S \
    --max-delivery-count 5

# model-updates topic
az servicebus topic subscription create \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $NAMESPACE_NAME \
    --topic-name model-updates \
    --name paper-trader \
    --lock-duration PT30S \
    --max-delivery-count 5

echo "✅ Service Bus topics and subscriptions setup complete!"
```

**Code change** (`src/utils/service_bus.py`):
```python
# Change from queue sender to topic sender
def send_message(namespace: str, topic_name: str, message_data: dict):
    """Send message to a Service Bus topic (replaces old queue-based send)."""
    credential = DefaultAzureCredential()
    client = ServiceBusClient(
        fully_qualified_namespace=f"{namespace}.servicebus.windows.net",
        credential=credential,
    )
    with client:
        sender = client.get_topic_sender(topic_name=topic_name)  # was: get_queue_sender
        with sender:
            message = ServiceBusMessage(
                body=json.dumps(message_data).encode('utf-8'),
                content_type='application/json'
            )
            sender.send_messages(message)

def receive_messages(namespace: str, topic_name: str, subscription_name: str, ...):
    """Receive messages from a Service Bus topic subscription."""
    # ... uses get_subscription_receiver instead of get_queue_receiver
```

### 11.2 Full Infrastructure Gap Analysis

| Component | Current State | Required State | Action | Phase |
|-----------|--------------|----------------|--------|-------|
| **Service Bus queues** | 4 queues exist | Need topics + subscriptions | Run `setup-service-bus.sh` to migrate | Phase 1 |
| **KEDA ScaledObject** | Scales on queue depth | Must scale on topic subscription depth | Update Helm template: `trigger.type: azure-servicebus` with `topicName` + `subscriptionName` | Phase 1 |
| **Service Bus utils code** | `send_message_to_queue()` | `send_message()` using topic sender | Port + modify `src/utils/service_bus.py` | Phase 1 |
| **Blob container structure** | `trading-data-blob/SP_500/` | Same (no change needed) | None | — |
| **Blob container: processed** | `trading-data-blob/processed/SP_500/` | Same (no change needed) | None | — |
| **Blob container: models** | `models/` container (RL models) | Same container, but LightGBM artifacts (`.txt` files, not `.zip`) | No infra change, just different files | Phase 3 |
| **AKS node pools** | System (2×D2a) + Worker (0-1×D4a) | Same — adequate for LightGBM (trains on CPU) | None | — |
| **AKS namespace** | `trading-system` | Same namespace (reuse) | None | — |
| **IBKR Gateway config** | Paper mode, port 4004 | Same | None | — |
| **Key Vault secrets** | `ibusername`, `ibpassword` | Same | None | — |
| **Managed Identity RBAC** | Has `ServiceBusDataOwner` on SB | Topics use same RBAC as queues — no change | None | — |
| **Docker images** | `fetcher:2.4`, `data-engineering:1.1`, `model-training:1.0` | New images: `fetcher:3.0`, `feature-engine:1.0` | Build & push via `build-and-push.sh` | Phase 1, 2 |
| **Helm chart: model-training** | Deploys SB3/RL trainer | Replace with LightGBM trainer (different deps, smaller image) | Rewrite Helm template + Dockerfile | Phase 3 |
| **Federated Identity for KEDA** | Configured for queue-based scaling | Topic-based scaling uses same federated identity | None | — |
| **Redis** | Not deployed | Need Redis for hot feature store (Phase 2+) | Add Redis deployment to Helm chart OR use Azure Cache for Redis | Phase 2+ |
| **Monitoring** | None | Prometheus + Grafana (Phase 4+) | Add Helm charts for monitoring stack | Phase 4+ |

### 11.3 Deployment Scripts — Detailed Per Phase

#### Phase 1 Deployment Scripts

**`deployment/build-and-push.sh`** — Build Docker images and push to ACR:
```bash
#!/bin/bash
# Usage: ./build-and-push.sh [fetcher|feature-engine|all] [tag]

REGISTRY="tradingsystemacr.azurecr.io"
SERVICE=${1:-"all"}
TAG=${2:-"latest"}

build_and_push() {
    local service=$1
    local dockerfile="services/${service}/Dockerfile"
    local image="${REGISTRY}/${service}:${TAG}"

    echo "🔨 Building ${image}..."
    docker build -t ${image} -f ${dockerfile} .
    echo "📤 Pushing to ACR..."
    az acr login --name tradingsystemacr
    docker push ${image}
    echo "✅ ${image} pushed successfully"
}

case $SERVICE in
    fetcher)        build_and_push fetcher ;;
    feature-engine) build_and_push feature-engine ;;
    all)
        build_and_push fetcher
        build_and_push feature-engine
        ;;
    *) echo "❌ Unknown service: $SERVICE" && exit 1 ;;
esac
```

**`deployment/manage-cluster.sh`** — Start/stop AKS:
```bash
#!/bin/bash
# Usage: ./manage-cluster.sh [up|down]
RESOURCE_GROUP="trading-system-rg"
CLUSTER_NAME="trading-system-aks"

case $1 in
    up)
        echo "🚀 Starting AKS cluster..."
        az aks start --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
        az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing
        echo "✅ Cluster started. kubectl configured."
        ;;
    down)
        echo "🛑 Stopping AKS cluster..."
        az aks stop --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
        echo "✅ Cluster stopped. No charges for compute."
        ;;
    *) echo "Usage: $0 [up|down]" && exit 1 ;;
esac
```

**`deployment/deploy.sh`** — Deploy Helm chart:
```bash
#!/bin/bash
# Usage: ./deploy.sh [install|upgrade|uninstall]
NAMESPACE="trading-system"
RELEASE_NAME="trading-platform"
CHART_DIR="deployment/helm"

case $1 in
    install)
        echo "📦 Installing Helm chart..."
        kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
        helm install $RELEASE_NAME $CHART_DIR \
            --namespace $NAMESPACE \
            --values $CHART_DIR/values.yaml
        ;;
    upgrade)
        echo "🔄 Upgrading Helm chart..."
        helm upgrade $RELEASE_NAME $CHART_DIR \
            --namespace $NAMESPACE \
            --values $CHART_DIR/values.yaml
        ;;
    uninstall)
        echo "🗑️ Uninstalling Helm chart..."
        helm uninstall $RELEASE_NAME --namespace $NAMESPACE
        ;;
    *) echo "Usage: $0 [install|upgrade|uninstall]" && exit 1 ;;
esac
```

**`deployment/verify-data.sh`** — Verify data pipeline end-to-end:
```bash
#!/bin/bash
# Verify data landed in blob storage after a fetch run
CONTAINER="trading-data-blob"
ACCOUNT_URL="https://tradingsystemsa.blob.core.windows.net"
PREFIX=${1:-"SP_500/"}

echo "🔍 Checking blob storage for data..."
az storage blob list \
    --container-name $CONTAINER \
    --account-name tradingsystemsa \
    --prefix "$PREFIX" \
    --auth-mode login \
    --output table

echo ""
echo "📊 Service Bus topic message counts:"
for TOPIC in raw-data-updates processed-data-updates model-creation model-updates; do
    COUNT=$(az servicebus topic show \
        --resource-group trading-system-rg \
        --namespace-name trading-system-service-bus \
        --name $TOPIC \
        --query "countDetails.activeMessageCount" -o tsv 2>/dev/null)
    echo "  $TOPIC: ${COUNT:-N/A} active messages"
done
```

#### Phase 2 Deployment Additions

- Update `build-and-push.sh`: add `feature-engine` service
- Update Helm `values.yaml`: add feature-engine deployment with KEDA ScaledObject (scaling on `raw-data-updates` topic, `feature-engine` subscription)
- Add `deployment/helm/templates/feature-engine/` templates

#### Phase 3 Deployment Additions

- Update `build-and-push.sh`: add `model-trainer` service (LightGBM, NOT RL)
- Update Helm: add model-trainer deployment (KEDA-scaled on `processed-data-updates` topic)
- Model-trainer image is much smaller than old RL image (no torch, no gymnasium)

---

## Build Priority Timeline

```
Phase 1 (Week 1-2): Data Layer
  ├── Project scaffolding (pyproject.toml, directory structure, configs)
  ├── Port IBKR adapter + normalizer + blob utils + service bus
  ├── Port Helm charts + deployment scripts
  ├── Deploy to AKS, fetch real data, verify in blob storage
  └── Gate: "I can run a script and data appears in Azure Blob"

Phase 2 (Week 2-3): Feature Engineering
  ├── Build YAML feature registry
  ├── Port 43 features into registry pattern
  ├── Build configurable target system
  ├── Deploy feature engine service (KEDA-scaled)
  └── Gate: "I can load raw data and get features + targets out"

Phase 3 (Week 3-4): Fast Backtesting
  ├── Build LightGBM signal model
  ├── Build vectorbt backtester
  ├── Build experiment workflow
  ├── Build performance analytics
  └── Gate: "I can test a hypothesis and get Sharpe/drawdown in < 5 minutes"

Phase 4 (Week 5-6): Close the Loop
  ├── Risk manager (hard stops)
  ├── Paper trading integration
  ├── Feedback logger
  ├── Multiple value streams (2nd model, ensemble)
  └── Gate: "I have signals trading on IBKR Paper with risk controls"
```
