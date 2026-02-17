# Experiment Log

## Session: February 15-17, 2026

### Strategy Evolution

---

## Strategy 1: Congressional Copy-Trading
**Hypothesis**: Copy stock purchases disclosed by US politicians (STOCK Act filings).

### Setup
- **Data**: Quiver Quantitative US Congress Trading ($5/mo on QC)
- **Signal**: Politician discloses BUY → we buy next day
- **Hold**: 30 days fixed
- **Universe**: All stocks that politicians trade

### Iterations
1. **v1 (debug)**: First attempt. Couldn't get data flowing — wrong class name (`QuiverCongressTrading`). Found correct name: `QuiverQuantCongressUniverse` + `add_universe(Type, name, callback)` pattern.
2. **v1 (working)**: Data flowing but 0 trades. `OrderDirection.Buy` → `OrderDirection.BUY` (uppercase). Top performers filter too restrictive + min_trade_value $50K too high (amounts are lower bounds of ranges like "$1,001-$15,000").
3. **v2 (tuned)**: Removed top_performers filter (accept all politicians). Lowered min_trade_value to $1K. 70% return but 7,000+ pending signals accumulated (never cleared).
4. **v2 (clean)**: Added signal expiry (3 days max). **66% return, 596 trades, $162K final.**

### Results
| Period | Return | Notes |
|---|---|---|
| 2020-2024 | 66% | 596 trades, 11,241 signals |

### Deployed
- **Paper trading**: DU7678510 (IBKR paper), L-MICRO node
- Later stopped to free node for other strategies

### Lessons
- QC dataset class names are not obvious — use `pip install QuantConnect-stubs` and grep `.pyi` files
- `OrderDirection` uses UPPERCASE enum values (BUY, SELL, HOLD)
- `self.debug()` shows in Cloud Terminal, `print()` does not, `self.log()` goes to Logs tab
- `add_universe(Type, name, callback)` — must pass all 3 args
- Congressional data subscription required ($5/mo extra)
- Amounts in filings are range lower bounds, not exact values

---

## Strategy 2: Sentiment + Price History
**Hypothesis**: Combine Tiingo News sentiment with price action (SMA, RSI) to trade.

### Setup
- **Data**: Tiingo News (free on QC) + standard OHLCV
- **Signal**: Keyword-based sentiment scoring + bullish price action (SMA20 > SMA50, RSI)
- **Hold**: 5 days
- **Universe**: Started with 20 tickers, expanded to 50

### Iterations
1. **v1 (20 tickers)**: 57% return, 1,283 trades, 4.4M articles processed
2. **v2 (50 tickers)**: 93% return, 4,982 trades, 6.8M articles processed

### Results
| Period | Tickers | Return | Trades | Articles |
|---|---|---|---|---|
| 2020-2024 | 20 | 57% | 1,283 | 4.4M |
| 2020-2024 | 50 | 93% | 4,982 | 6.8M |

### Analysis
- PSR 37.7% — only 37% probability the Sharpe is positive
- ~80% of return is beta (riding the bull market), not alpha
- Keyword matching ("beats", "strong", "growth") is too crude
- Fees: $10K+ on the 50-ticker version
- **Verdict**: Doesn't beat SPY buy-and-hold after fees. Abandoned.

### Deployed
- **Paper trading**: DU7678510, later stopped
- Decision: move to event-driven approach instead

---

## Strategy 3: Event-Driven News Reactor (CURRENT)
**Hypothesis**: Buy quality stocks on specific high-impact catalysts (earnings beats, analyst upgrades, price gaps with volume). Hold 3-5 days for post-event drift.

### Academic Backing
- Bernard & Thomas (1989) — Post-Earnings Announcement Drift: stocks that beat earnings continue drifting up for 60 days
- Womack (1996) — Analyst upgrades generate 2-3% drift over weeks
- Jegadeesh & Titman (1993) — Momentum: winners keep winning

### Setup
- **Data**: Tiingo News (free) + Price/Volume (free) + Morningstar Fundamentals (free)
- **Signal**: Two detection methods:
  1. **Price gaps**: Stock opens >X% higher than yesterday on Y× average volume
  2. **News events**: Article matches specific catalyst keywords (earnings beat, analyst upgrade, guidance raise)
- **Universe**: 50 pre-selected quality large-cap stocks across sectors
- **Hold**: 3-5 days depending on event type
- **Fractional shares**: `round(qty, 4)` for small accounts

### Event Patterns
```python
"earnings_beat": ["beats estimates", "tops estimates", "better than expected", ...]  → hold 5 days
"analyst_upgrade": ["upgraded to buy", "price target raised", ...]  → hold 3 days
"guidance_raise": ["raises guidance", "raises outlook", ...]  → hold 4 days
"price_gap": gap >= threshold% AND volume >= ratio × avg  → hold 5 days
```

### Iterations

#### v1-v2: Dynamic universe (FAILED)
- Tried `add_universe(coarse_filter, fine_filter)` with Morningstar quality screen
- Morningstar filter worked (300 quality stocks passed), but `add_data(TiingoNews)` inside `fine_filter` and `on_securities_changed` didn't deliver news data
- Kept getting 0 trades despite correct universe selection
- API `&` character error on some tickers
- **Lesson**: Dynamic universe + Tiingo `add_data` in callbacks doesn't work in QC

#### v3: Fixed ticker list (WORKING)
- Switched to same pattern as working sentiment strategy: fixed ticker list + `add_data` in `initialize()`
- 50 quality stocks hard-coded across all sectors
- Parameters: gap 3%, volume 1.5x, 15 positions, 6% per position

##### v3 Results (experiment_17_2_2026_1204)
| Period | Return | CAGR | Sharpe | Trades | Fees | Alpha | Win Rate |
|---|---|---|---|---|---|---|---|
| 2016-2020 | 57% | 9.5% | 0.54 | 3,981 | $10.2K | 0.002 | 54% |
| 2018-2021 | 28% | 8.5% | 0.39 | 2,669 | $5.8K | -0.004 | 54% |
| 2020-2024 | 71% | 11.3% | 0.49 | 4,268 | $8.7K | 0.008 | 53% |
| **2022-2023** | **19%** | **8.9%** | **0.27** | **1,637** | **$3.3K** | **0.040** | **52%** |

**Key finding**: Consistent 8.5-11.3% CAGR across ALL periods. **Positive alpha in 2022-2023 bear market** (19% when SPY was flat). Win rate 52-54%, profit factor 1.17-1.22. Beta ~0.5.

**Problems identified**:
- Too many trades (800+/year) → high fees
- Avg P&L per trade only $13-19 (barely above commission)
- Sharpe < 1.0 everywhere
- Many trades are noise (gap detection catches normal volatility)

#### v4: Higher thresholds + stop-loss (TESTING)
Changes from v3:
- `min_gap_pct`: 3% → **5%** (only strongest gaps)
- `min_volume_ratio`: 1.5x → **2.0x** (confirms institutional interest)
- `stop_loss_pct`: NEW **3%** (exit if down 3% from entry — cut losers fast)
- `max_positions`: 15 → **20** (more diversification)
- `position_size_pct`: 6% → **4%** (less risk per trade)
- Track entry prices for stop-loss
- Added `check_stop_losses` scheduled event

**Expected impact**: fewer trades → lower fees, stop-loss → smaller losses, higher thresholds → better signal quality.

**Status**: Running backtests for all 4 periods. Results pending.

---

## Infrastructure & Costs

### QuantConnect
- **Plan**: Researcher ($60/mo)
- **Data subscriptions**: Quiver Congress ($5/mo) — cancelled after stopping congressional strategy
- **Free data used**: Tiingo News, Morningstar Fundamentals, Price/Volume
- **Live nodes**: 1 × L-MICRO (included in Researcher plan)

### Interactive Brokers
- **Account**: U12654641 (real), DU7678510 (paper)
- **Pricing**: Switched from Fixed to **Tiered** ($0.35 min/trade instead of $1.00)
- **Fractional shares**: Enabled and tested (0.2 shares of AAPL confirmed)
- **Capital**: ~$500 deposited, planning $20K if strategy proves itself

### QC API Lessons (documented in docs/quantconnect-api-reference.md)
- `QuiverQuantCongressUniverse` for congress data
- `TiingoNews` for news data (added per-ticker via `add_data`)
- `OrderDirection.BUY` (uppercase)
- `add_universe(Type, name, callback)` — 3 args required
- `self.debug()` for terminal output
- Dynamic universe + `add_data` in callbacks = broken
- Fixed ticker list + `add_data` in `initialize()` = works

---

## Datasets Evaluated

| Dataset | Provider | Cost | Used? | Notes |
|---|---|---|---|---|
| US Congress Trading | Quiver | $5/mo | Yes (cancelled) | Congressional copy-trading |
| Tiingo News | Tiingo | Free | Yes | Article-level news data |
| Morningstar Fundamentals | Morningstar | Free | Attempted | ROE, gross margin, market cap — worked in fine_filter but couldn't combine with Tiingo add_data |
| Brain Sentiment Indicator | Brain | $10/mo | Not yet | AI-scored NLP sentiment (7-day and 30-day). Would replace our keyword matching. Consider if v4 results aren't good enough. |
| Brain ML Stock Ranking | Brain | $10/mo | No | ML-based stock rankings |
| US Equity Price/Volume | QC Built-in | Free | Yes | Core data for gap detection |

---

## Next Steps

1. **Complete v4 backtests** (4 periods) — compare to v3
2. **If v4 better**: deploy to paper trading, monitor 2 weeks
3. **If v4 same/worse**: try Brain Sentiment ($10/mo) or cross-sectional momentum strategy
4. **Paper trade 2 weeks** → if positive → deploy real money ($500-1000 initially)
5. **Scale to $20K** if real money trading is positive for 1 month

## Future Strategy Ideas (from research)
- **Cross-sectional momentum**: 12-1 month momentum, long top quintile. Sharpe ~0.5-0.6.
- **Trend following (Faber 10m SMA)**: If price > 10-month SMA → hold, else cash. Reduces drawdown 50%.
- **Multi-factor (momentum + quality)**: Combine momentum with profitability. Sharpe ~1.0.
- **Time-series momentum (multi-asset ETFs)**: Trend following across SPY, TLT, GLD, etc.
