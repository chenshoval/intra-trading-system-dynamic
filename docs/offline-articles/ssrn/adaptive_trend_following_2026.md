# Systematic Trend-Following with Adaptive Portfolio Construction (AdaptiveTrend)

**Source:** Semantic Scholar / SSRN
**Authors:** Duc-Quang Bui, T. Nguyen
**Year:** 2026
**URL:** https://www.semanticscholar.org/paper/e654e947db6539a7c9f567ea55e3b388b0964fd1

## Abstract (partial)
Cryptocurrency markets exhibit pronounced momentum effects and regime-dependent volatility. Proposes AdaptiveTrend: multi-component algorithmic trading framework integrating high-frequency trend-following on 6-hour intervals with monthly adaptive portfolio construction and asymmetric long-short capital allocation.

## Relevance to Our System
- **MODERATE** — Crypto-focused but the framework is transferable
- Multi-component: high-frequency signals + monthly portfolio construction
- Asymmetric long-short allocation — similar to our long-only with trend gate
- Regime-dependent volatility handling aligns with our SPY MA 10/50 trend gate
- 6-hour intervals + monthly construction = multi-timeframe approach
- Could inspire a version of v2 that uses intraday signals for monthly scoring

## Key Takeaways
1. Multi-timeframe approach (short signals + monthly allocation)
2. Adaptive to regime changes
3. Asymmetric allocation (more long in bull, reduce in bear)
4. Originally crypto but framework applies to equities
