# Intraday Trading System — Project Progress

## Current Status: Monthly Rotator v2 LIVE on IBKR (March 2026)

### What's Deployed
- **Strategy:** `strategies/monthly_rotator/main_v2.py` (MonthlyRotatorV2)
- **Platform:** QuantConnect → IBKR paper/live, $500 capital, fractional shares
- **First rebalance:** March 3, 2026 (first trading day)
- **Status:** Paper test March 3 → verify clean → switch to live immediately
- **DO NOT TWEAK** for 6 months (until Sept 2026)

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
- Static 50-stock universe has survivorship bias (UBER, ABNB didn't exist in 2016)
- Missing fundamental signals (value, quality, leverage) that papers show improve robustness

## Live Deployment Scaling Ladder

| Capital | Strategy | File | Stocks | Downtrend | Notes |
|:-------:|----------|------|:------:|:---------:|-------|
| **$500-$2K** | v10 Small | `main_v10_small.py` | Top 15 (buy affordable) | 5 | Cheap stocks added, skip-and-log |
| **$2K-$20K** | v11 Medium | `main_v11_medium.py` | Top 10 | 5 | Original 50 universe, most stocks affordable |
| **$20K-$100K** | v2 Full | `main_v2.py` | Top 15 | 5 | Full diversification, all stocks affordable |
| **$100K+** | v2 Full | `main_v2.py` | Top 15 | 5 | Same as above, more shares per position |

**How to scale up:**
1. Start with v10 at $500
2. When account hits $2K → stop v10, deploy v11 medium
3. When account hits $20K → stop v11, deploy v2 full
4. All use same scoring engine, same signals, same trend gate
5. Only difference is how many stocks are held and position sizing

**At $100K+ consider:**
- Adding VIX-based hedge or defensive sector rotation for geopolitical risk
- The SPY trend gate (MA 10/50) catches sustained declines (worked in 2020 COVID, 2022 bear)
- But doesn't protect against overnight gap-down shocks (e.g., sudden military escalation)
- Man Group paper: trend-following protects in every crisis over 95 years because crises develop over weeks
- At larger capital, adding a SPY put hedge or sector rotation into defense stocks (LMT, RTX, XOM) during elevated geopolitical risk could reduce max drawdown
- This is a $100K+ problem — at $500-$20K, the trend gate is sufficient

**Current deployment:** v10 on IBKR with $515 (March 2026)

## How the 50-Stock Universe Was Picked

The universe was NOT from any academic paper. It was hand-curated based on:
1. **Sector diversification** — intentional balance across 8 sectors
2. **Heavy news coverage** — needed for Tiingo event scoring (earnings beats, upgrades)
3. **High liquidity** — all mega-cap, easy to enter/exit
4. **Personal familiarity** — companies you can reason about fundamentally

### Why it works despite survivorship bias:
- The scoring engine picks the top 15 from this pool each month
- Bad-momentum stocks just don't get selected — they sit idle in the universe
- The universe is the "candidate pool," not the portfolio
- Backtests show hand-picked beats dynamic selection (v9/v9b experiments)

### When to refresh:
- Every 6 months: check for acquisitions, delistings, market cap drops below $10B
- Every 12 months: consider adding new mega-caps, rebalancing sector weights
- If a stock is removed, replace with the next-largest in the same sector

---

## Maintenance Schedule

### Deployment Rules
- **CRITICAL: Any live deployment MUST trigger an immediate rebalance on startup.** Do NOT rely on the monthly schedule — if deployed mid-month, the strategy will sit in cash until the next month_start event. Add `self.monthly_rebalance()` at the end of `initialize()` for any live deployment, or use a one-time scheduled event for the deployment day. Remove the one-time trigger after the first rebalance fires.
- Lesson learned: March 2026 deployment sat in cash for a full month because the month_start schedule had already passed.

### Monthly (every rebalance)
- Check QC logs for errors after each month's rebalance
- Verify orders filled correctly
- Glance at portfolio — do the top 15 picks make sense?

### Every 6 Months (next: September 2026)
- **Universe review:** Scan the 50-stock list. Did any stock get acquired, delisted, or drop below $10B market cap? If yes, swap it for the next largest in that sector.
- **Performance review:** Calculate live Sharpe. Compare to backtest expectations (target >0.5 annualized). If live Sharpe >0.5, consider scaling capital from $500 to $2-5K.
- **Sector balance check:** Are the sector weights still reasonable? (14 tech, 7 finance, 7 healthcare, 7 consumer, 5 industrial, 3 energy, 2 media, 4 fintech)

### Every 12 Months (next: March 2027)
- **Full strategy review:** Run fresh backtest on most recent 2-3 years. Does performance still hold?
- **Universe refresh:** Check if new mega-caps emerged that should be added. Check if any current stocks fundamentally changed (e.g., major business model shift).
- **Consider v8 upgrade:** If v8 (fundamentals-enhanced) backtests well, consider upgrading from v2 to v8 for live deployment. v8 adds value (E/P), quality (ROE), and leverage (D/E) signals.
- **Consider dynamic universe:** Evaluate switching from static 50 to dynamic top-50-by-market-cap (already prototyped in v3b).
- **Re-read CLAUDE.md hypotheses:** Are there new strategies worth pursuing? (Congressional trading, directional classifier, etc.)

---

## Stock Universe: 50 Hand-Picked Large-Caps

### Why these 50
Chosen for sector diversification + heavy news coverage (needed for event scoring).
All are large/mega-cap with high liquidity and analyst coverage.

### Sector Breakdown
| Sector | Count | Tickers |
|--------|:-----:|---------|
| Tech | 14 | AAPL, MSFT, NVDA, GOOGL, AMZN, META, AVGO, CRM, ADBE, ORCL, AMD, QCOM, NFLX, INTU |
| Finance | 7 | JPM, GS, MS, V, MA, AXP, BLK |
| Healthcare | 7 | UNH, LLY, ABBV, MRK, TMO, ABT, ISRG |
| Consumer | 7 | COST, HD, MCD, NKE, PG, KO, PEP |
| Industrial/Defense | 5 | CAT, HON, UPS, GE, LMT, RTX |
| Energy | 3 | XOM, CVX, COP |
| Media | 2 | DIS, CMCSA |
| Fintech | 4 | PYPL, SQ, ABNB, UBER |

### Known Issues
- **Survivorship bias:** These are 2026 winners picked with hindsight. UBER (IPO 2019) and ABNB (IPO 2020) didn't exist for early backtest periods.
- **Tech-heavy:** 14/50 = 28% tech. In a tech crash (2022), the strategy is overexposed.
- **No small/mid caps:** Momentum factor is academically stronger in smaller stocks, but we can't capture that.

### Refresh Rules
- Every 6 months: check for acquisitions, delistings, or market cap drops below $10B
- Every 12 months: consider adding new mega-caps, rebalancing sector weights
- If a stock is removed, replace with the next-largest in the same sector to maintain balance

---

## Bear Market Variants Tested (v3-v7) — None Beat v2 Overall
| Variant | Approach | Bull Performance | Bear Performance | Verdict |
|---------|----------|:---:|:---:|---------|
| v3 (LS-50) | Short bottom 5 of 50 stocks | Worse | Negative Sharpe | KILL — shorting quality stocks doesn't work |
| v3b (LS-500) | S&P 500 long-short, threshold | 2.9% CAR 2016-2020 | OK (5% CAR) | Dangerous — 49% DD on 2020-2024 |
| v4 (SPY hedge) | Short SPY in downtrend | OK but weaker | Lost money (-1.3%) | Mediocre — hedge didn't help enough |
| v5 (sectors) | 11 sector ETFs long-short | Weak (5.7% CAR) | **Best: 12% CAR, Sharpe 0.56** | Bear champion but too weak in bulls |
| v6 (trend) | 70% stocks + 30% SPY trend | OK but lower | Lost money (-3.7%) | SPY sat as dead weight, whipsawed |
| v7 (dual) | 70% v2 + 30% v5 sectors | Halved v2's returns | Worse than v2 | Capital split dilutes again |

**Conclusion: v2 remains the best overall strategy. No variant improved on it without sacrificing more than it gained.**

## Other Strategies Tested
| Strategy | Type | Result | Verdict |
|----------|------|--------|---------|
| Forex Zone Bounce | EUR/USD support/resistance mean-reversion | 7% in 4 years, Sharpe -0.93 | KILLED — too few trades, too many filters |

---

## v8 Fundamentals Results — Promising for Bear Markets

### v2 vs v8 Head-to-Head: v2 wins 12/20, v8 wins 8/20

| Period | v2 CAR | v8 CAR | v2 Sharpe | v8 Sharpe | v2 DD | v8 DD |
|--------|:------:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2016-2020 | **43.4%** | 30.7% | **1.45** | 1.13 | 35.8% | **31.2%** |
| 2018-2021 | **35.8%** | 29.1% | **1.14** | 1.00 | 35.8% | **31.1%** |
| 2020-2024 | **26.4%** | 22.0% | **0.85** | 0.73 | 32.4% | **31.3%** |
| 2022-2023 | 10.0% | **10.5%** | 0.28 | **0.31** | 27.1% | **23.3%** |

**Key findings:**
- v2 wins on returns and Sharpe in bull markets (momentum dominance)
- v8 wins on drawdown in ALL periods (fundamentals add stability)
- v8 beats v2 in bear market (2022-2023) on every metric — fundamentals shine when momentum fails
- **Verdict:** Fundamentals help most when it matters most (bear markets). Worth incorporating into v9 with dynamic universe.

## Chart Reading Study & v12 (March 2026)

### What We Did
1. **Transcribed Micha.Stocks YouTube channel** (Hebrew) using Whisper AI
   - Audio files in `audio/` folder, transcripts in `transcripts_1.txt`, `transcripts_2.txt`
   - Extracted chart reading techniques: Inverse H&S, rounding bottom, support/resistance, MA150, volume confirmation, insider buying signals

2. **Built chart reading lessons** — visual tutorials with fake data
   - `notebooks/05_chart_reading/chart_reading_101.py` → 7 lesson PNGs (candlesticks, S/R, patterns, MAs, volume, full trade setup, cheat sheet)
   - `notebooks/05_chart_reading/chart_calculations.py` → explains the actual algorithms (swing point detection, S/R clustering, MA formulas)
   - Output in `notebooks/05_chart_reading/output/`

3. **Gap analysis: chart reading vs our strategies**
   - ✅ Already using: MA50 trend filter, swing points (forex), MA crossover (SPY gate)
   - ❌ Missing: volume confirmation, HH/HL/LH/LL swing trend, breakout detection, chart patterns

4. **Built v12 (chart-enhanced scoring)** — `strategies/monthly_rotator/main_v12_chart_enhanced.py`
   - Adds 3 new signals to v2's 5: volume confirmation (0.15), swing trend structure (0.10), breakout detection (0.10)
   - Class: `MonthlyRotatorV12ChartEnhanced`
   - Also built `main_v2_chart_test.py` (MonthlyRotatorV2ChartTest) — exact v2 at $100K with chart signals added for fair A/B test

### v2 vs v2+chart A/B Test Results (March 2026)

Tested at $100K, 15 stocks, same deploy/rebalance logic. Only difference: 8 signals vs 5.
Results in `results_from_quant_connect/monthlyrotatorv2charttest/`.

| Period | v2 CAGR | v2+chart CAGR | v2 Sharpe | v2+chart Sharpe | v2 MaxDD | v2+chart MaxDD | Winner |
|--------|---------|---------------|-----------|-----------------|----------|----------------|--------|
| 2016-2020 | **43.4%** | 33.4% | **1.45** | 0.98 | **35.8%** | 51.0% | **v2** |
| 2018-2021 | **35.8%** | 31.1% | **1.14** | 0.87 | **35.8%** | 51.0% | **v2** |
| 2020-2024 | **26.4%** | 25.2% | **0.85** | 0.79 | **32.4%** | 35.1% | **v2** |
| 2022-2023 | **10.0%** | 4.3% | **0.28** | 0.06 | **27.1%** | 30.2% | **v2** |
| 2025-2026 | 8.8% | **12.9%** | 0.12 | **0.28** | **21.5%** | 24.0% | **v2+chart** |

**Scoreboard: v2 wins 4, v2+chart wins 1**
**Averages: v2 Sharpe 0.77 / MaxDD 30.5% vs v2+chart Sharpe 0.60 / MaxDD 38.3%**

**Verdict: Chart signals KILLED.** Adding volume, swing trend, and breakout signals diluted momentum's weight (0.35→0.25) which is v2's strongest signal. Worse Sharpe, worse drawdown in 4/5 periods. The one win (2025-2026) is interesting but insufficient. v2 remains king.

**Lesson learned:** Chart reading concepts (S/R, HH/HL, volume) are useful for *understanding* markets but don't improve *systematic monthly rotation scoring*. They may be better suited for a separate short-term breakout strategy, not as additional signals in a momentum rotator.

### Full Strategy Comparison (all $100K results from `results_from_quant_connect/`)

Generated March 2026 by reading `totalPerformance.portfolioStatistics` from all JSON files.
Each strategy folder contains: `.json` (full stats), `_orders.csv`, `_trades.csv`, and `.png` (equity curve) per period.

| Strategy | Period | CAGR | Sharpe | Sortino | MaxDD | Net Profit | Start$ |
|----------|--------|------|--------|---------|-------|------------|--------|
| **MonthlyRotatorV2** | 2016-2020 | **43.4%** | **1.45** | 1.32 | 35.8% | 508.0% | $100K |
| **MonthlyRotatorV2** | 2018-2021 | **35.8%** | **1.14** | 1.07 | 35.8% | 240.2% | $100K |
| **MonthlyRotatorV2** | 2020-2024 | **26.4%** | **0.85** | 0.87 | 32.4% | 223.1% | $100K |
| **MonthlyRotatorV2** | 2022-2023 | 10.0% | 0.28 | 0.32 | 27.1% | 21.0% | $100K |
| PureMomentum | 2016-2020 | 42.4% | 1.39 | 1.27 | 35.0% | 485.5% | $100K |
| PureMomentum | 2022-2023 | 2.6% | -0.01 | -0.01 | 27.5% | 5.3% | $100K |
| monthlyrotatorv8 | 2016-2020 | 30.7% | 1.13 | 1.04 | 31.2% | 281.6% | $100K |
| monthlyrotatorv8 | 2022-2023 | **10.5%** | **0.31** | **0.37** | **23.3%** | 22.1% | $100K |
| CombinedDualEngine | 2016-2020 | 14.8% | 1.28 | 1.31 | 7.1% | 99.5% | $100K |
| CombinedDualEngine | 2022-2023 | 5.7% | 0.09 | 0.10 | 7.0% | 11.7% | $100K |
| monthlyrotatorv9 | 2016-2020 | 30.3% | 0.80 | 0.73 | **51.5%** | 276.0% | $100K |
| monthlyrotatorv9 | 2022-2023 | **-17.4%** | -0.56 | -0.72 | **46.0%** | -31.7% | $100K |
| monthlyrotatorv10 | 2016-2020 | 43.9% | 1.10 | 1.03 | 45.2% | 519.1% | **$500** |
| monthlyrotatorv10 | 2022-2023 | 5.3% | 0.10 | 0.11 | 24.6% | 11.0% | **$500** |
| monthlyrotatorv10b | 2022-2023 | -15.9% | -0.75 | -0.83 | 38.5% | -29.2% | **$500** |
| monthlyrotatorv10small | 2022-2023 | -7.0% | -0.46 | -0.55 | 34.0% | -13.6% | **$500** |

**Key conclusion:** v2 at $100K remains the best overall. Small account variants (v10/v10b/v10small at $500) have 40-47% max drawdowns due to position concentration. Plan: accumulate $500/month in IBKR, run v2 from day 1 (it skips stocks it can't afford), and it naturally scales into full operation by ~$10-15K.

### Decision: Stick with v2, Accumulate to $100K
- v2 is deployed on IBKR with $500 (March 2026)
- Add $500/month manually
- v2 buys what it can afford (skips expensive stocks via `int(target_alloc/price) < 1 → continue`)
- By ~$10-15K all 15 positions are buyable
- v10/v11 scaling ladder exists but v2 handles low capital fine with its skip logic
- v12 (chart-enhanced) built but NOT yet backtested — will compare vs v2 when ready
- v12b (self-optimizing walk-forward weights) planned as next step after v12 baseline

### Monthly Deposit Projection (v2 at ~15% CAGR conservative estimate)
```
Month 0:   $500
Month 6:   ~$3,800   (v2 buys ~8 stocks)
Month 12:  ~$7,500   (v2 buys ~12 stocks)
Month 24:  ~$16,000  (v2 fully operational, all 15 positions)
Month 36:  ~$27,000
Month 48:  ~$42,000
Month 60:  ~$60,000
```
Note: Deposit math done via spreadsheet, NOT in backtest. Backtesting with deposits corrupts Sharpe/return metrics (deposits inflate equity curve). Test strategy quality with fixed capital, project growth with spreadsheet.

## Roadmap

### Multi-Strategy Path (March 2026 — the plan going forward)

The goal is 4 uncorrelated alpha streams combined with risk parity weighting.

**Stage 1: Prove v2 out-of-sample** (March–Sept 2026)
- v2 deployed on IBKR with $1,000. Accumulating $500/month.
- First rebalance: April 1, 2026 (month_start schedule).

**Stage 2: Strategy 2 — Commodity Momentum** ✅ VALIDATED
- Applies v2's scoring engine to 15 commodity/real-asset ETFs (GLD, SLV, PPLT, XLE, USO, UNG, DBA, MOO, WEAT, CORN, SOYB, CPER, DBC, PDBC, TLT).
- 4 signals (momentum 0.40, trend 0.25, recent 0.20, vol 0.15), top 4, monthly rebalance.
- Standalone results are mediocre (Sharpe 0.10–0.45) BUT correlation to v2 is only 0.11.
- **The 2022 bear market proves the thesis**: v2 alone LOST $7K, commodity MADE $34K, combined MADE $6K.
- Files: `strategies/commodity_momentum/main_v1.py`
- Results: `results_from_quant_connect/commoditymomentumv1/` (2016-2020, 2020-2023, 2023-2025)

**Stage 2b: Combined v2 + Commodity (75/25)** ✅ BACKTESTED
- Single algorithm running both sleeves with fixed 75/25 capital split.
- Results across all periods:
  - 2016-2020: CAR 32.6%, Sharpe 1.30, MaxDD 33.3%, PSR 74.6%
  - 2020-2023: CAR 21.5%, Sharpe 0.79, MaxDD 29.0%, PSR 34.9%
  - 2023-2025: CAR 19.5%, Sharpe 0.64, MaxDD 20.8%, PSR 47.9%
- **Key insight**: Combined has LOWER peak returns than v2 alone (32.6% vs 43.4%) but:
  - Turns bear market losses into profits (2022: +$6K vs -$7K)
  - Lower MaxDD in ALL periods
  - Lower beta in ALL periods (less market dependent)
  - More stable PSR (doesn't collapse like v2's 80%→18.6%)
- **v2's PSR is decaying over time**: 80.1% → 56.9% → 39.2% → 18.6% → 26.7%. The combined approach is insurance against this edge decay.
- Files: `strategies/combined_v2_commodity/main_v1.py`
- Results: `results_from_quant_connect/combinedv2commodity/` (2016-2020, 2020-2023, 2023-2025)

**Stage 3: Strategy 3 — Dividend Yield Rotation** ✅ VALIDATED
- 12 high-dividend ETFs, yield proxy scoring. PSR 61.1% in 2023-2025 (highest ever).
- Correlation to v2: 0.45 (borderline but acceptable). Correlation to commodity: 0.31 (good).
- Files: `strategies/dividend_yield/main_v1.py`

**Strategy 4 — Bond Momentum** ✅ VALIDATED
- 12 bond ETFs (TLT, ZROZ, IEF, SHY, etc.). Beta -0.21 to SPY (negative!).
- Correlation to v2: 0.15 (excellent). Only strategy that made money during COVID March 2020.
- Standalone Sharpe is weak (-0.97 to 0.47), but uncorrelation value is high.
- Files: `strategies/bond_momentum/main_v1.py`

**Full 4x4 correlation matrix (115 months of real data):**
```
              v2 Equity   Commodity   Dividend    Bond
v2 Equity     1.000       +0.340      +0.451      +0.153
Commodity     0.340       1.000       +0.308      +0.164
Dividend      0.451       0.308       1.000       +0.292
Bond          0.153       0.164       0.292       1.000
```
No pair above 0.6. Two pairs below 0.2. Matrix is solid.

Crisis coverage proven:
- COVID crash (2020-03): Bond was only positive strategy (+$602)
- Rate hike bear (2022): Commodity was savior (+$34K while v2 lost -$7K)
- Sector rotation (2022-01): Dividend went up while v2 dropped

**Stage 4: Combined 4-Strategy Algorithm** ✅ BUILT AND TESTED (v1→v5)

Five iterations of the combined algorithm:
| Version | Innovation | 2020-2023 Sharpe | 2020-2023 DD | Problem |
|---------|-----------|-----------------|-------------|---------|
| v1 | Raw risk parity | 0.38 | 18.4% | SPY gate killed all sleeves, noisy |
| v2 | Per-sleeve gates + TLT fix | 0.36 | 21.5% | More emergencies, flip-flopping |
| v3 | Symmetric smoothing (15%/mo) | 0.45 | 18.9% | Still noisy underlying solver |
| **v4** | **AI decision tree** | **0.72** | **17.4%** | **Best numbers but static thresholds** |
| v5 | Relative ranking + asymmetric smooth | 0.57 | 17.9% | Smoothest weights, live-ready |

**v4 is the backtest winner. v5 is the live deployment winner** (never goes stale).

Key fixes discovered through iteration:
1. Per-sleeve trend gates (SPY→equity, DBC→commodity, DVY→dividend, AGG→bond)
2. TLT belongs in bond sleeve only, not commodity
3. Fallback = keep previous weights, not reset to 25%
4. Asymmetric smoothing: slope UP fast (+20%), slope DOWN slow (-8%)
5. Relative ranking never needs retraining

**The honest verdict on combined vs v2 standalone:**
| | v2 standalone | Best combined (v4) |
|---|---|---|
| 2016-2020 CAR | **43.4%** | 18.4% |
| 2020-2023 CAR | **22.9%** | 13.6% |
| 2016-2020 MaxDD | 35.8% | **24.4%** |
| 2020-2023 MaxDD | 32.4% | **17.4%** |
| 2020-2023 Sharpe | 0.75 | **0.72** (nearly equal!) |
| 2020-2023 PSR | 29.5% | **36.8%** |

Combined cuts drawdown in half but also cuts returns in half.
The Sharpe ratio is nearly identical — same risk-adjusted return, different ride.

**Deployment plan:**
```
Now → $15K:     v2 standalone (maximize growth, need capital accumulation)
                $1K start + $500/month deposits
                Estimated timeline: ~19-23 months (Oct 2027 - Feb 2028)
$15K+:          Switch to v4 or v5 (protect capital, similar risk-adjusted returns)
                v4 if willing to retrain every 6 months
                v5 if want zero-maintenance live deployment
$50K+:          Combined is clearly better (35% DD on $50K = -$17.5K is devastating)
```

v2 is OK for now. The multi-strategy work is not wasted — it's the graduation strategy
waiting in the repo for when capital justifies the switch.

All strategy code and backtest results preserved:
- `strategies/commodity_momentum/main_v1.py`
- `strategies/dividend_yield/main_v1.py`
- `strategies/bond_momentum/main_v1.py`
- `strategies/combined_v2_commodity/main_v1.py`
- `strategies/combined_4strat_riskparity/main_v1.py` through `main_v5.py`
- `notebooks/05_weight_learning/train_weights.py`
- All results in `results_from_quant_connect/` with matching periods

### v8 is the upgrade path (when ready)
- v8 (fundamentals on static universe) is the best alternative to v2
- Lower drawdown in ALL periods, better bear market performance
- Deploy as v2 replacement after 6-month v2 live validation (Sept 2026)

### Dynamic universe — KILLED
- v9 (dollar volume): beta 2.0, 59% DD, -17% in 2022. Terrible.
- v9b (market cap + sector caps + positive EPS): better but still v2 wins 17/20
- **Conclusion: hand-picked universe beats dynamic selection.** The curated quality
  and intentional sector balance outweighs survivorship bias concerns.
- **Action: manually refresh the 50-stock list every 6 months instead.**

### v10 (future): Real Sentiment Analysis
- Replace keyword scanning with actual NLP sentiment (FinBERT or similar)
- Requires: model hosting, inference pipeline, custom data feed into QC
- Timeline: after v8 deployment validation (2027+)

### Overnight Hold Strategy — KILLED (March 2026)
- **Research basis**: Kakushadze 2014, Glasserman 2025, Knuteson 2020
- **Strategy**: Buy top 5 stocks at 3:45 PM, sell at 9:31 AM, sit in cash all day
- **v1 backtest (2016-2020, $100K)**: CAR -11.8%, Sharpe -0.42, MaxDD 61%, Final $53K
- **Root cause**: 5,399 trades × $11.90/trade = **$64K in fees** on a $100K account
- **The edge was real but tiny**: 52% win rate, $3.24 avg profit/trade, but fees were $11.90/trade
- **Lesson**: The overnight return premium exists (papers are correct) but it's a market microstructure observation, NOT a tradeable strategy. The papers didn't test with transaction costs. Same lesson as event-driven v5 (ML filter): thin edges get eaten by fees.
- **VERDICT: KILLED** — even weekly trading (v1b) doesn't fix the per-trade economics ($3.24 edge vs $11.90 cost)
- Files kept for reference: `strategies/overnight_hold/main_v1.py`, `main_v1_broad.py`, `main_v1b_weekly.py`

### v13 Regime-Aware Sector Rotation — TESTED, DEPRIORITIZED (March 2026)
- **Research basis**: RegimeFolio (Zhang 2025) — regime-aware sectoral portfolio, Sharpe 1.17
- **Key insight**: Don't short in bear, don't split capital — SWITCH what you hold based on regime
- **Result**: Regime switching ate bull returns with false signals. Only 40% return halfway through 2016-2020 when v2 does 43% for the full period. Not worth the complexity.
- **Lesson**: The combined multi-strategy approach (v2 + commodity at 75/25) achieves the same goal (bear protection) more reliably than regime switching.
- **Strategy file**: `strategies/monthly_rotator/main_v13_regime.py`
- **Status**: Deprioritized in favor of multi-strategy combination approach

### Other Future Ideas
1. **Global TabNet directional classifier**: From the dual-stream paper. Stream B achieved ~40% annual. Requires proper features (momentum, volatility, cross-stock, macro — NOT trade metadata like v5 used). Walk-forward validation mandatory. Could run as second uncorrelated strategy alongside momentum rotator.
2. **Anti-curve-fitting validation**: Monte Carlo randomization, noise testing, synthetic testing. We did WFA and regime testing — these 3 remain.
3. **Bi-weekly rebalance**: Catches momentum reversals faster. Worth testing at higher capital levels.
4. **Threshold-based long/neutral/short**: Score > 0.65 → long, 0.35-0.65 → neutral, < 0.35 → short.
5. **Congressional trading**: Quiver Quantitative dataset in QC (CLAUDE.md Hypothesis 1).
6. **Multi-strategy HRP allocation**: When running 2+ strategies, use Lopez de Prado's hierarchical allocation.
7. **Pairs trading (Kalman/Copula)**: Market-neutral mean-reversion on cointegrated pairs. Uncorrelated to momentum. See neural Kalman paper (Milstein 2022) and graph clustering paper (Korniejczuk 2024) in `docs/offline-articles/arxiv/`.
8. **Real Sentiment Analysis (v10 future)**: Replace keyword scanning with FinBERT NLP. After v8 deployment validation (2027+).

---

## How to Analyze Experiment Results

### Quick comparison script (run from repo root):
```python
python3 -c "
import json, os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Change these paths to compare any two experiments:
bases = {
    'Strategy A': 'results_from_quant_connect/FOLDER_A',
    'Strategy B': 'results_from_quant_connect/FOLDER_B',
}
periods = ['2016-2020', '2018-2021', '2020-2024', '2022-2023']

def get_data(base, period):
    pp = os.path.join(base, period)
    if not os.path.isdir(pp): return None
    jf = glob.glob(os.path.join(pp, '*.json'))
    if not jf: return None
    with open(jf[0], 'r') as f: return json.load(f)

def val(data, key):
    if data is None: return '?'
    return data.get('statistics', {}).get(key, '?')

for period in periods:
    print(f'\n--- {period} ---')
    for name, base in bases.items():
        d = get_data(base, period)
        car = val(d, 'Compounding Annual Return')
        sharpe = val(d, 'Sharpe Ratio')
        dd = val(d, 'Drawdown')
        psr = val(d, 'Probabilistic Sharpe Ratio')
        print(f'  {name}: CAR={car} Sharpe={sharpe} DD={dd} PSR={psr}')
"
```

### Multi-strategy correlation matrix script (run from repo root):
```python
python3 -c "
import json, os, glob, sys, io, numpy as np
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === CONFIGURE: strategy name -> list of JSON result files ===
strategies = {
    'v2 Equity': [
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2016-2020/Energetic Apricot Cobra.json',
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2020-2023/Casual Orange Alpaca.json',
        'results_from_quant_connect/MonthlyRotatorV2/experiment_19_2_2026_1645/2023-2025/Square Green Rhinoceros.json',
    ],
    'Commodity': [
        'results_from_quant_connect/commoditymomentumv1/2016-2020/Muscular Brown Gorilla.json',
        'results_from_quant_connect/commoditymomentumv1/2020-2023/Upgraded Brown Butterfly.json',
        'results_from_quant_connect/commoditymomentumv1/2023-2025/Virtual Apricot Owl.json',
    ],
    'Dividend': [
        'results_from_quant_connect/dividendyieldv1/2016-2020/Hyper Active Yellow Green Sheep.json',
        'results_from_quant_connect/dividendyieldv1/2020-2023/Swimming Fluorescent Yellow Rhinoceros.json',
        'results_from_quant_connect/dividendyieldv1/2023-2025/Ugly Light Brown Pig.json',
    ],
    'Bond': [
        'results_from_quant_connect/bondmomentumv1/2016-2020/Square Red Panda.json',
        'results_from_quant_connect/bondmomentumv1/2020-2023/Emotional Fluorescent Orange Eagle.json',
        'results_from_quant_connect/bondmomentumv1/2023-2025/Smooth Yellow Green Coyote.json',
    ],
}
# === END CONFIG ===

def get_monthly_pnl(json_file):
    with open(json_file) as f:
        d = json.load(f)
    pnl = d.get('profitLoss', {})
    monthly = defaultdict(float)
    for ts, val in pnl.items():
        monthly[ts[:7]] += val
    return dict(monthly)

# 1. Standalone stats
print('=== STANDALONE STATS ===')
for name, files in strategies.items():
    print(f'{name}:')
    for fname in files:
        period = fname.split('/')[-2]
        with open(fname) as f:
            s = json.load(f)['statistics']
        print(f'  {period}: CAR={s[\"Compounding Annual Return\"]:>8s}  Sharpe={s[\"Sharpe Ratio\"]:>6s}  PSR={s[\"Probabilistic Sharpe Ratio\"]:>8s}  DD={s[\"Drawdown\"]:>7s}  Beta={s[\"Beta\"]:>6s}')
    print()

# 2. Load all monthly P&L
all_pnl = {}
for name, files in strategies.items():
    all_pnl[name] = {}
    for f in files:
        all_pnl[name].update(get_monthly_pnl(f))

# 3. Correlation matrix
names = list(strategies.keys())
common = sorted(set.intersection(*[set(all_pnl[n].keys()) for n in names]))
print(f'=== CORRELATION MATRIX ({len(common)} months) ===')
arrays = [np.array([all_pnl[n][m] for m in common]) for n in names]
matrix = np.corrcoef(arrays)
header = ''.join(f'{n:>12s}' for n in names)
print(f'{\"\":>12s}{header}')
for i, n in enumerate(names):
    row = f'{n:>12s}'
    for j in range(len(names)):
        v = matrix[i,j]
        row += f'{v:>+12.3f}'
    print(row)

# 4. Crisis months
print()
print('=== CRISIS MONTHS ===')
for m in ['2020-03','2022-01','2022-06','2022-09','2022-10']:
    vals = '  '.join(f'{n}={all_pnl[n].get(m,0):>+8.0f}' for n in names)
    print(f'  {m}: {vals}')
"
```

### Trade-level analysis script (run from repo root):
```python
python3 -c "
import csv, sys, io
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === CONFIGURE: path to trades CSV ===
TRADES_FILE = 'results_from_quant_connect/FOLDER/PERIOD/NAME_trades.csv'
# === END CONFIG ===

trades = []
with open(TRADES_FILE) as f:
    for row in csv.DictReader(f):
        trades.append(row)

pnl_by_sym = defaultdict(float)
count_by_sym = defaultdict(int)
wins_by_sym = defaultdict(int)
for t in trades:
    sym = t['Symbols'].strip()
    pnl = float(t['P&L'])
    pnl_by_sym[sym] += pnl
    count_by_sym[sym] += 1
    if t['IsWin'].strip() == '1':
        wins_by_sym[sym] += 1

print(f'Total: {len(trades)} trades')
for sym, pnl in sorted(pnl_by_sym.items(), key=lambda x: x[1], reverse=True):
    wr = wins_by_sym[sym] / count_by_sym[sym] * 100
    print(f'  {sym:>6s}: PnL={pnl:>+10,.2f}  trades={count_by_sym[sym]:>3d}  WR={wr:.0f}%')

all_pnl = [float(t['P&L']) for t in trades]
wins = [p for p in all_pnl if p > 0]
losses = [p for p in all_pnl if p <= 0]
print(f'Avg win: {sum(wins)/len(wins):,.2f}  Avg loss: {sum(losses)/len(losses):,.2f}  Ratio: {abs(sum(wins)/len(wins)/(sum(losses)/len(losses))):.2f}')
"
```

### Result folder naming convention:
```
results_from_quant_connect/
├── MonthlyRotatorV2/experiment_19_2_2026_1645/   # v2 (deployed)
├── commoditymomentumv1/                           # Strategy 2: commodity momentum
├── dividendyieldv1/                               # Strategy 3: dividend yield rotation
├── bondmomentumv1/                                # Strategy 4: bond momentum
├── combinedv2commodity/                           # Combined v2+commodity (75/25)
├── PureMomentom/expreiment_19_2_2026_1616/       # pure momentum
├── CombinedDualEngine/experiment_19_2_2026_1240/ # combined dual (v1+events)
├── monthlyrotatorv3/                              # v3 long-short 50
├── monthlyrotatorv3b/                             # v3b S&P 500 long-short
├── monthlyrotator4/                               # v4 SPY hedge
├── MomthlyRotatorV5/                              # v5 sectors
├── monthlyrotatorv6/                              # v6 trend overlay
├── monthlyrotatorv7/                              # v7 dual (v2+v5)
├── monthlyrotatorv8/                              # v8 fundamentals
├── monthlyrotatorv9/                              # v9 dynamic universe (killed)
├── monthlyrotatorv9b/                             # v9b fixed dynamic
├── overnightholdv1/                               # overnight hold (killed)
├── Forext/                                        # forex zone bounce (killed)
├── trade-events/experiment_18_2_2026_1800/       # trend+events v1
└── event-driven new refactor/                     # baseline v4 + v5 ML
```

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

## Research Papers (in `docs/`)

### Equity Strategy Papers (`docs/offline-articles/`)
| Paper | Key Insight for Us |
|-------|-------------------|
| Warsaw (Castellano & Slepaczuk) | MA crossover IR*=0.68, robust parameter selection, portfolio of uncorrelated strategies |
| Network Forecasting (Baitinger, SSRN-3370098) | Cross-asset correlation features predict returns |
| Financial ML (Kelly & Xiu, SSRN-4501707) | Long-short decile Sharpe 1.72, regularization mandatory, GBT best for cross-section |
| Inflation Strategies (Harvey/Man Group, SSRN-3813202) | Trend-following works in ALL regimes (95 years), momentum +8% real in inflation |
| HRP (López de Prado, SSRN-2708678) | Hierarchical allocation beats Markowitz OOS |

### Forex Papers (`docs/forex papers/`)
| Paper | Key Insight |
|-------|-------------|
| Teeple (SSRN-3667920) | S/R levels are real equilibrium outcomes from coarse Bayesian updating, robust to arbitrage |
| Fritz & Weinhardt (SSRN-2788997) | Order book depth at S/R levels confirms institutional support; transitory pricing errors at S/R = mean-reversion signal |
| Bushee (SSRN-161739) | Not relevant (institutional earnings myopia) |

---

## Key Lessons Learned

1. **Adding overlays that REDUCE exposure without ADDING alpha always fails** (v5, trend+events v1)
2. **Monthly holds >> daily trading** — 14-20 day holds had 76% WR vs 26% for 0-1 day
3. **Events work as scoring signal, NOT as trade triggers** — v2 proved this
4. **Capital splits always dilute** — every combined strategy (dual engine, v7) performed worse than pure v2
5. **Don't filter trades, SIZE them** (or in our case, select better stocks)
6. **Fees kill** — $11K/year in fees on a $100K account is 11% drag
7. **Shorting bottom of a quality universe doesn't work** — need genuinely bad stocks for short leg
8. **SPY trend overlay as dead weight** — holding SPY long for 2 years adds beta not alpha
9. **PSR is misleading on short periods** — 2-year windows mechanically produce low PSR regardless of strategy quality
10. **QC "73 parameters" warning is noise** — most are structural choices (tickers, keywords), not tuned parameters
11. **Chart reading signals don't improve momentum rotation** — volume, swing trend (HH/HL), and breakout detection diluted momentum's weight and made v2 worse in 4/5 periods. Chart concepts are for understanding markets, not for systematic scoring.
