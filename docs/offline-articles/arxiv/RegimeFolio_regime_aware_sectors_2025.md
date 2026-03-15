# RegimeFolio: Regime-Aware Sectoral Portfolio Optimization

**Authors:** Yiyao Zhang, Diksha Goel, Hussain Ahmad, Claudia Szabo
**Published:** 2025-09-14
**Categories:** q-fin.PM, cs.AI
**URL:** http://arxiv.org/abs/2510.14986v1
**PDF:** https://arxiv.org/pdf/2510.14986v1

## Abstract
Proposes RegimeFolio: regime-aware, sector-specialized framework. Combines: (i) VIX-based classifier for regime detection, (ii) regime+sector-specific ensemble learners (RF, GBT), (iii) dynamic mean-variance optimizer with shrinkage covariance. Tested on 34 large-cap US equities 2020-2024. Achieves 137% cumulative return, Sharpe 1.17, 12% lower max drawdown vs benchmarks.

## Relevance to Our System
- **HIGHLY RELEVANT** — This is the most promising approach for bear market protection
- VIX-based regime detection is simpler and more interpretable than our SPY MA 10/50 gate
- Sector-specific models mean different scoring in different regimes (not just "reduce to 5")
- In bear regime: could shift allocation to defensive sectors (utilities, staples, healthcare)
- In bull regime: could load up on tech/growth (what v2 already does naturally)
- The key difference from our v3-v7 failures: they didn't CHANGE which sectors to pick, just how many stocks. This paper says ROTATE INTO defensive sectors during bear.
- 12% lower max drawdown is exactly what we need

## Key Takeaways
1. VIX-based regime detection beats MA crossover for regime identification
2. Different sectors need different models in different regimes
3. In bear market: defensive sectors (utilities, staples) outperform
4. In bull market: growth sectors (tech) outperform
5. Dynamic mean-variance allocation adapts weights per regime
6. Sharpe 1.17 on 2020-2024 (includes COVID crash and 2022 bear) — impressive
