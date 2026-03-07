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

## Roadmap

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

### Other Future Ideas
1. **Global TabNet directional classifier** ← PRIORITY: From the dual-stream paper. Stream B achieved ~40% annual. Requires proper features (momentum, volatility, cross-stock, macro — NOT trade metadata like v5 used). Walk-forward validation mandatory. Could run as second uncorrelated strategy alongside momentum rotator.
2. **Anti-curve-fitting validation**: Monte Carlo randomization, noise testing, synthetic testing. We did WFA and regime testing — these 3 remain.
3. **Bi-weekly rebalance**: Catches momentum reversals faster. Worth testing at higher capital levels.
4. **Threshold-based long/neutral/short**: Score > 0.65 → long, 0.35-0.65 → neutral, < 0.35 → short.
5. **Congressional trading**: Quiver Quantitative dataset in QC (CLAUDE.md Hypothesis 1).
6. **Multi-strategy HRP allocation**: When running 2+ strategies, use López de Prado's hierarchical allocation.
7. **Pairs trading (Kalman/Copula)**: Market-neutral mean-reversion on cointegrated pairs. Uncorrelated to momentum.
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

### Result folder naming convention:
```
results_from_quant_connect/
├── MonthlyRotatorV2/experiment_19_2_2026_1645/   # v2 (deployed)
├── PureMomentom/expreiment_19_2_2026_1616/       # pure momentum
├── CombinedDualEngine/experiment_19_2_2026_1240/ # combined dual (v1+events)
├── monthlyrotatorv3/                              # v3 long-short 50
├── monthlyrotatorv3b/                             # v3b S&P 500 long-short
├── monthlyrotator4/                               # v4 SPY hedge
├── MomthlyRotatorV5/                              # v5 sectors
├── monthlyrotatorv6/                              # v6 trend overlay
├── monthlyrotatorv7/                              # v7 dual (v2+v5)
├── monthlyrotatorv8/                              # v8 fundamentals
├── monthlyrotatorv9/                              # v9 dynamic universe (killed — beta 2.0)
├── monthlyrotatorv9b/                             # v9b fixed dynamic (still worse than v2)
├── Forext/                                        # forex zone bounce (killed)
├── trade-events/experiment_18_2_2026_1800/       # trend+events v1
└── event-driven new refactor/                     # baseline v4 + v5 ML
    ├── experiment_17_2_2026_1204/                 # v4 baseline
    ├── experiment_17_2_2026_1300/                 # v4 variant
    └── experiment_17_2_2026_1440/                 # v5 ML filter
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
