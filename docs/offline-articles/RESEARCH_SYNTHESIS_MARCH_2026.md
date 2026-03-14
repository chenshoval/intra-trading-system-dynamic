# Paper Research Synthesis — March 2026
# Improving QC Strategies: Monthly, Intraday, Combinations, and Daily Routines

## Note on Sources
- **arXiv:** 14 papers found and saved as .md files in `docs/offline-articles/arxiv/`
- **SSRN/Semantic Scholar:** Rate-limited heavily during this session. Only 2 new papers saved to `docs/offline-articles/ssrn/`. For future research, get a free Semantic Scholar API key at https://www.semanticscholar.org/product/api#api-key-form to avoid this.
- **Existing SSRN papers in `docs/offline-articles/`** (from previous research, still relevant):
  - `ssrn-3370098.pdf` — Network Forecasting (Baitinger): Cross-asset correlation features predict returns
  - `ssrn-4501707.pdf` — Financial ML (Kelly & Xiu): Long-short decile Sharpe 1.72, GBT best for cross-section
  - `ssrn-3813202.pdf` — Inflation Strategies (Harvey/Man Group): Trend-following works ALL regimes, 95 years
  - `ssrn-2708678.pdf` — HRP (Lopez de Prado): Hierarchical allocation beats Markowitz OOS
  - `ssrn-3226776.pdf` — (Warsaw/Castellano & Slepaczuk): MA crossover strategies, portfolio of uncorrelated strategies

## Papers Found & Saved

### Category 1: Long-Range Monthly Strategy Improvement
| # | Paper | Year | Relevance | Ranking |
|---|-------|------|-----------|---------|
| 1 | **Dynamic Inclusion & Multi-Factor Tilts** (Garrone) | 2026 | ★★★★★ | #1 MUST READ |
| 2 | **Combining Smart Beta Strategies** (Maguire et al.) | 2018 | ★★★★★ | #2 ACTIONABLE |
| 3 | **Agnostic Allocation Portfolios** (CFM/Bouchaud) | 2019 | ★★★★☆ | #3 USEFUL |
| 4 | **DeepAries Adaptive Rebalancing** (Kim et al.) | 2025 | ★★★★☆ | #4 FUTURE |
| 5 | **Low-Volatility Anomaly & AMF** (Jarrow et al.) | 2020 | ★★★☆☆ | #5 VALIDATES |
| 6 | **Momentum Strategies L1 Filter** (Dao) | 2014 | ★★★☆☆ | #6 NICHE |
| 7 | **Kelly Rebalancing Frequency** (Hsieh et al.) | 2018 | ★★☆☆☆ | #7 THEORETICAL |

### Category 2: Short-Range Intraday Strategies
| # | Paper | Year | Relevance | Ranking |
|---|-------|------|-----------|---------|
| 1 | **Overnight News Explains Returns** (Glasserman) | 2025 | ★★★★★ | #1 CRITICAL |
| 2 | **4-Factor Overnight Returns** (Kakushadze) | 2014 | ★★★★☆ | #2 ACTIONABLE |
| 3 | **Overnight vs Intraday Returns** (Knuteson) | 2020 | ★★★★☆ | #3 FOUNDATIONAL |
| 4 | **Order-Flow Entropy** (Singha) | 2025 | ★★★☆☆ | #4 RESEARCH |
| 5 | **Decision Trees Intraday** (Naga et al.) | 2024 | ★★☆☆☆ | #5 MARGINAL |

### Category 3: Strategy Combination / Multi-Strategy
| # | Paper | Year | Relevance | Ranking |
|---|-------|------|-----------|---------|
| 1 | **Combining Smart Beta** (Maguire) | 2018 | ★★★★★ | #1 BLUEPRINT |
| 2 | **Stat Arb Graph Clustering** (Korniejczuk/Ślepaczuk) | 2024 | ★★★★★ | #2 ACTIONABLE |
| 3 | **Neural Kalman Pairs Trading** (Milstein et al.) | 2022 | ★★★★☆ | #3 PRACTICAL |
| 4 | **Robust Stat Arb Deep NN** (Neufeld et al.) | 2022 | ★★★☆☆ | #4 ADVANCED |
| 5 | **151 Trading Strategies** (Kakushadze) | 2019 | ★★★★☆ | REFERENCE |

### Category 4: Daily Routine / Multi-Strategy Daily
| # | Paper | Year | Relevance | Ranking |
|---|-------|------|-----------|---------|
| 1 | **4-Factor Overnight Returns** (Kakushadze) | 2014 | ★★★★★ | #1 DAILY STRATEGY |
| 2 | **Overnight News Returns** (Glasserman) | 2025 | ★★★★☆ | #2 NEWS ROUTINE |
| 3 | **Predictive Directional Trading** (Letteri) | 2025 | ★★★☆☆ | #3 LEAD-LAG |
| 4 | **AdaptiveTrend Multi-Component** (Bui/Nguyen) | 2026 | ★★★☆☆ | #4 FRAMEWORK |

### Additional Warning Paper
| # | Paper | Year | Relevance | Ranking |
|---|-------|------|-----------|---------|
| 1 | **Mean Reversion Fails in Practice** (Moon et al.) | 2019 | ★★★★★ | CRITICAL WARNING |

---

## SYNTHESIS: What This Means for Our QC Strategies

### 1. v2 Monthly Rotator Is Academically Validated ✅

The Garrone 2026 paper describes almost exactly what v2 does:
- Cross-sectional rankings (not return predictions) ✓
- Equal-weight baseline with factor tilts ✓
- Dynamic eligibility based on market conditions (SPY gate) ✓
- Hard bounds on concentration ✓
- Monthly deterministic rebalancing ✓

**v2 is not a hacky retail strategy — it's a formalized, academically-validated approach.**

### 2. Biggest Improvement Opportunity: ADD a Second Strategy (Not Improve v2)

The research overwhelmingly shows:
- Combining INDEPENDENT strategies > improving one strategy (Maguire: 0.61 + 0.90 → 0.96 Sharpe)
- Our v2 is a momentum+trend+quality strategy → adding MORE momentum signals (v12 chart) DILUTED it
- The right move is a SEPARATE, uncorrelated strategy running alongside v2

**Best second strategy candidates (ranked):**

| Rank | Strategy | Type | Correlation to v2 | Complexity | Papers Supporting |
|------|----------|------|-------------------|------------|-------------------|
| 1 | **Overnight hold** | Daily | Low (different timeframe) | Low | Kakushadze 2014, Glasserman 2025, Knuteson 2020 |
| 2 | **Pairs trading** | Market-neutral | Near-zero | Medium | Milstein 2022, Korniejczuk 2024, Neufeld 2022 |
| 3 | **Low-volatility** | Monthly | Moderate | Low | Jarrow 2020, Maguire 2018 |
| 4 | **Stat arb (graph)** | Daily | Low | High | Korniejczuk 2024 |

### 3. INTRADAY Strategies: DO NOT Build Pure Intraday Long

Three independent papers confirm:
- **Overnight returns are positive, intraday returns are negative** (Knuteson 2020)
- **News drives overnight returns** (Glasserman 2025)
- **4 factors predict overnight returns** (Kakushadze 2014)

**Implication: Any "intraday" strategy should actually be an OVERNIGHT strategy.**
- Buy at close → sell at open (capture overnight premium)
- Use news sentiment to pick WHICH stocks to hold overnight
- This is NOT fighting the structural headwind of negative intraday returns

### 4. Mean Reversion Warning ⚠️

Moon et al. (2019) show that mean reversion strategies:
- Look great on benchmark datasets
- FAIL on real S&P 500 data with transaction costs
- Are highly sensitive to costs

**Implication: If we do pairs trading, it must be robust to costs. Neural Kalman (Milstein 2022) or graph-based (Korniejczuk 2024) approaches handle this better than classical OU-process methods.**

### 5. What NOT to Change About v2

Based on the research:
- ✅ Keep cross-sectional ranking (not return prediction)
- ✅ Keep equal-weight allocation
- ✅ Keep monthly rebalancing
- ✅ Keep SPY trend gate (state-dependent constraint)
- ✅ Keep the 50-stock curated universe
- ❌ Don't add more signals to v2 (already proven with v12 chart test)
- ❌ Don't switch to dynamic universe (already proven with v9/v9b)
- ❌ Don't try daily rebalancing for v2 (increases costs, Kelly paper shows diminishing returns)

---

## RECOMMENDED ACTION PLAN

### Phase 1: Quick Win — Overnight Strategy (Q2 2026)
**Paper basis:** Kakushadze 2014, Glasserman 2025, Knuteson 2020

Build a simple overnight strategy on QC:
1. At 3:55 PM: Score stocks using 4 factors (size, volatility, momentum, liquidity) from OHLCV
2. Buy top N at close
3. Sell at open next day
4. This runs DAILY alongside monthly v2
5. Low correlation to v2 (different timeframe, different factors)
6. No fundamental data needed — pure OHLCV
7. Test with Tiingo news sentiment as 5th factor (Glasserman's finding)

**Why this first:**
- Simplest to implement on QC (no new data sources)
- 3 independent papers support it
- Uncorrelated to monthly v2
- Can start with $500 separate allocation

### Phase 2: Pairs Trading (Q3-Q4 2026)
**Paper basis:** Milstein 2022, Korniejczuk 2024

Build a neural-augmented Kalman filter pairs trading strategy:
1. Use graph clustering to select pairs from our 50-stock universe
2. Kalman filter to track spread
3. Bollinger Bands for signals
4. Neural augmentation for robustness
5. Kelly criterion for position sizing
6. Market-neutral → zero correlation to v2

**Why second:**
- More complex but higher expected alpha
- Market-neutral means zero beta → pure alpha
- Korniejczuk is from same Warsaw group as our existing MA crossover paper

### Phase 3: v2 Minor Improvements (Q1 2027, after 6-month live validation)
**Paper basis:** Garrone 2026, CFM/Bouchaud 2019

Consider:
1. Replace equal-weight with Agnostic Allocation (CFM paper)
2. Add formal concentration bounds (Garrone paper)
3. Test adaptive rebalancing (skip months when market is stable)
4. These are refinements, not redesigns

### Phase 4: Multi-Strategy Allocation (When running 2+ strategies)
**Paper basis:** Maguire 2018, HRP (López de Prado, already in docs/)

Use HRP (Hierarchical Risk Parity) to allocate between:
- v2 monthly momentum (long-only)
- Overnight strategy (daily, long-only)
- Pairs trading (daily, market-neutral)

Target: Combined Sharpe > 1.0 with max DD < 20%

---

## DAILY ROUTINE CONCEPT (from research)

If running all 3 strategies, the daily routine would be:

### Market Open (9:30 AM)
- **Overnight strategy**: Sell overnight positions at open
- **Pairs trading**: Check for new signals, manage existing positions

### Mid-Day (12:00 PM)
- **Pairs trading**: Monitor spread deviations, manage stops

### Market Close (3:55 PM)
- **Overnight strategy**: Score stocks, enter overnight positions
- **Pairs trading**: Enter/exit pair positions based on Bollinger signals

### Monthly (1st trading day)
- **v2 Rotator**: Full rebalance of monthly positions
- **HRP allocation**: Rebalance capital across 3 strategies

### Wednesday (mid-week)
- **v2**: Emergency SPY trend check (already built)

This creates a "business" of trading rather than a single strategy gamble.

---

## KEY PAPERS BY IMPACT ON OUR SYSTEM

### Tier 1: Must Read & Act On
1. 🏆 **Garrone 2026** — Validates v2's approach academically
2. 🏆 **Maguire 2018** — Blueprint for combining independent strategies
3. 🏆 **Glasserman 2025** — Overnight news → overnight returns (actionable alpha)
4. 🏆 **Kakushadze 2014** — 4-factor overnight model (implementable on QC)
5. 🏆 **Korniejczuk 2024** — Graph-based pairs trading with Kelly (same research group)

### Tier 2: Important Context
6. **Knuteson 2020** — Overnight premium is real (structural)
7. **Milstein 2022** — Neural Kalman pairs trading (pairs implementation guide)
8. **Jarrow 2020** — Low-vol anomaly explained (validates our volatility signal)
9. **Moon 2019** — Mean reversion FAILS with costs (critical warning)

### Tier 3: Reference / Future
10. **DeepAries 2025** — Adaptive rebalancing (future v2 enhancement)
11. **151 Trading Strategies** — Idea encyclopedia
12. **Neufeld 2022** — Robust stat arb (advanced, future)
13. **CFM/Bouchaud 2019** — Agnostic allocation (v2 refinement)
14. **Order-flow entropy 2025** — Volatility prediction (needs tick data)

---

## BOTTOM LINE

**v2 is solid. Don't touch it. Instead:**
1. Build an overnight strategy (Phase 1) — simplest, 3 papers support it, uncorrelated to v2
2. Build pairs trading (Phase 2) — market-neutral, robust to market regime
3. Combine with HRP allocation (Phase 4) — target Sharpe >1.0, DD <20%
4. The "daily routine of strategies" is: overnight trades daily + pairs daily + v2 monthly
5. Kill the idea of pure intraday long strategies — the research says overnight is where returns live

---

## FILE INDEX — All New Articles Saved This Session

### arxiv/ (14 papers)
| File | Paper | Year |
|------|-------|------|
| `DeepAries_adaptive_rebalancing_2025.md` | Adaptive rebalancing intervals with Transformer+PPO | 2025 |
| `overnight_vs_intraday_returns_2020.md` | Overnight positive, intraday negative pattern | 2020 |
| `decision_trees_intraday_2024.md` | Per-stock decision tree intraday rules | 2024 |
| `mean_reversion_strategies_empirical_2019.md` | Mean reversion fails with real costs | 2019 |
| `combining_smart_beta_strategies_2018.md` | Momentum + low-vol combination, Sharpe 0.96→1.35 live | 2018 |
| `neural_kalman_pairs_trading_2022.md` | Neural-augmented Kalman filter for pairs | 2022 |
| `stat_arb_graph_clustering_2024.md` | Graph clustering pairs, Kelly sizing (Warsaw group) | 2024 |
| `robust_stat_arb_deep_nn_2022.md` | 50-stock stat arb without cointegration requirement | 2022 |
| `dynamic_inclusion_multifactor_tilts_2026.md` | Validates v2 approach academically | 2026 |
| `momentum_L1_filter_2014.md` | L1 filter for trend detection in momentum | 2014 |
| `order_flow_entropy_intraday_2025.md` | Order-flow entropy predicts move magnitude | 2025 |
| `151_trading_strategies_2019.md` | Encyclopedia of 150+ strategies (reference) | 2019 |
| `overnight_news_explains_returns_2025.md` | News topics drive overnight returns (2.4M articles) | 2025 |
| `4_factor_overnight_returns_2014.md` | 4 OHLCV factors predict overnight returns | 2014 |
| `low_volatility_anomaly_AMF_2020.md` | Low-vol explained by factor loadings | 2020 |
| `agnostic_allocation_portfolios_2019.md` | CFM's risk-based portfolio construction | 2019 |

### ssrn/ (2 papers — Semantic Scholar rate limited)
| File | Paper | Year |
|------|-------|------|
| `adaptive_trend_following_2026.md` | Multi-component trend-following framework | 2026 |
| `predictive_directional_trading_volatility_2025.md` | Lead-lag causal inference for equities | 2025 |

### Previously existing SSRN PDFs (5 papers)
| File | Paper |
|------|-------|
| `ssrn-3370098.pdf` | Network Forecasting (Baitinger) |
| `ssrn-4501707.pdf` | Financial ML (Kelly & Xiu) |
| `ssrn-3813202.pdf` | Inflation Strategies (Harvey/Man Group) |
| `ssrn-2708678.pdf` | HRP (Lopez de Prado) |
| `ssrn-3226776.pdf` | Warsaw MA Crossover (Castellano & Slepaczuk) |
