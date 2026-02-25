# Intraday Trading System — Project Progress

## Current Status: Monthly Rotator v2 LIVE on IBKR (Feb 2026)

### What's Deployed
- **Strategy:** `strategies/monthly_rotator/main_v2.py` (MonthlyRotatorV2)
- **Platform:** QuantConnect → IBKR paper/live, $500 capital, fractional shares
- **First rebalance:** March 1, 2026
- **Status:** Monitoring for mechanical issues only. Do NOT tweak for 6 months.

### What Monthly Rotator v2 Does
- Monthly rebalance on first trading day
- Scores 50 stocks on 5 signals: momentum (35%), trend (25%), recent strength (20%), volatility (10%), news events (10%)
- Buys top 15 equal-weight, holds full month
- SPY 10/50 MA trend gate: reduces to top 5 in downtrend
- Wednesday mid-week emergency check: if SPY flips to downtrend, reduce to 5 positions
- ~300 trades/year, ~$1K fees (vs 4,000 trades/$11K fees in old event reactor)

### Backtest Results (v2 — the deployed strategy)
| Period | CAR | Sharpe | PSR | Max DD | Alpha | Final Equity |
|--------|-----|--------|-----|--------|-------|-------------|
| 2016-2020 | 43.4% | 1.45 | 80.1% | 35.8% | 0.183 | $608K |
| 2018-2021 | 35.8% | 1.14 | 56.9% | 35.8% | 0.116 | $340K |
| 2020-2024 | 26.4% | 0.85 | 39.2% | 32.4% | 0.090 | $323K |
| 2022-2023 | 10.0% | 0.28 | 18.6% | 27.1% | 0.056 | $121K |

### Known Weakness
- Max drawdown 32-36% in bull periods, 27% in bear (2022-2023)
- High beta (~1.0-1.1) — concentrated long-only equity exposure
- Building v3-v6 variants to address this with short/hedge legs

---

## Strategy Evolution (What We Tried)

### 1. Event-Driven Reactor v4 (Baseline)
- `strategies/event_driven/main.py`
- News keyword scanning + gap detection, 3-5 day holds
- Results: 8-11% CAR, Sharpe 0.27-0.54, DD 14-17%
- **Problem:** Thin edge, high fees (~$10K/year), 41% of trades were noise

### 2. Event-Driven Reactor v5 (LightGBM filter)
- Added ML model to filter trades based on confidence
- **FAILED:** Model killed winners more than losers. Returns halved.
- Root cause: Features were trade metadata (stock_id, log_price), not market features. Created feedback loop via rolling features.

### 3. Trend+Events v1
- `strategies/trend_events/main.py`
- SPY MA overlay + ATR stops + seasonality scaling
- **FAILED:** Same pattern as v5 — reduced exposure but didn't add alpha. More trades, more fees.

### 4. Combined Dual-Engine v1
- `strategies/combined_dual/main.py`
- Event reactor (50% capital) + momentum (50% capital)
- **Mixed:** Sharpe 1.28/PSR 88.5% on 2016-2020, but short-hold event trades (0-7d) bled -$37K while monthly holds made +$40K. Capital split diluted both engines.

### 5. Pure Momentum
- `strategies/pure_momentum/main.py`
- Same scoring as v2 but NO event engine, NO news
- **Good returns but worse risk-adjusted:** 486% return but 35% DD, negative Sharpe in 2022-2023

### 6. Monthly Rotator v1
- `strategies/monthly_rotator/main.py`
- Multi-signal scoring, monthly rebalance, 15 stocks
- Also ran as combined_dual (which was actually v1 + event engine)
- v1's dual-engine had Sharpe 1.28 but event trades dragged returns

### 7. Monthly Rotator v2 ← WINNER (deployed)
- `strategies/monthly_rotator/main_v2.py`
- Same as v1 but events used as scoring BOOST (5th signal) not separate trades
- Won 13/20 metric contests across all strategies
- Beat pure momentum on Sharpe in all 4 periods

---

## Research Papers (in `docs/offline-articles/`)

| Paper | Key Insight for Us |
|-------|-------------------|
| Warsaw (Castellano & Slepaczuk) | MA crossover IR*=0.68, robust parameter selection, portfolio of uncorrelated strategies |
| Network Forecasting (Baitinger, SSRN-3370098) | Cross-asset correlation features predict returns |
| Financial ML (Kelly & Xiu, SSRN-4501707) | Long-short decile Sharpe 1.72, regularization mandatory, GBT best for cross-section |
| Inflation Strategies (Harvey/Man Group, SSRN-3813202) | Trend-following works in ALL regimes (95 years), momentum +8% real in inflation |
| HRP (López de Prado, SSRN-2708678) | Hierarchical allocation beats Markowitz OOS |

---

## Key Lessons Learned

1. **Adding overlays that REDUCE exposure without ADDING alpha always fails** (v5, trend+events v1)
2. **Monthly holds >> daily trading** — 14-20 day holds had 76% WR vs 26% for 0-1 day
3. **Events work as scoring signal, NOT as trade triggers** — v2 proved this
4. **Two uncorrelated streams > one complex strategy** (Warsaw paper principle)
5. **Don't filter trades, SIZE them** (or in our case, select better stocks)
6. **Fees kill** — $11K/year in fees on a $100K account is 11% drag

---

## Next Steps (In Progress)

### Bear Market Variants (v3-v6)
Building 4 variants to reduce drawdown while keeping v2's returns:
- **v3:** Long-short (short bottom 5 from scoring) — Kelly/Xiu approach
- **v4:** SPY hedge (short SPY in downtrends only) — Warsaw approach
- **v5:** Sector ETF rotation long-short — Harvey/Man Group approach
- **v6:** Trend overlay (70% v2 + 30% SPY trend following) — Man Group approach

### After Backtesting Variants
- Deploy best variant as strategy #2 alongside v2
- 6-month review of v2 live performance (Sept 2026)
- Consider HRP allocation between strategies if running 2+

### Longer Term
- Congressional trading hypothesis (CLAUDE.md Hypothesis 1)
- Global directional classifier with proper features (Hypothesis 2)
- Futures trend-following as separate asset class strategy
