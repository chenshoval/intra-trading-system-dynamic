# Combining Independent Smart Beta Strategies for Portfolio Optimization

**Authors:** Phil Maguire, Karl Moffett, Rebecca Maguire
**Published:** 2018-08-07
**Categories:** q-fin.PM
**URL:** http://arxiv.org/abs/1808.02505v2
**PDF:** https://arxiv.org/pdf/1808.02505v2

## Abstract
Explores combining a long-short beta-neutral strategy (from AdaBoost on momentum indicators) with a minimized volatility portfolio. Market benchmark Sharpe 0.42 → market-neutral component Sharpe 0.61 → low-volatility Sharpe 0.90 → **combined leveraged strategy Sharpe 0.96**. In 6 months of live trading: **Sharpe 1.35**.

## Relevance to Our System
- **HIGHLY RELEVANT** — This is exactly the multi-strategy approach we need.
- Combines two independent factor strategies: momentum (long-short) + low-volatility
- Monthly reweighting — same frequency as our v2
- Uses AdaBoost classifier on momentum indicators — similar to our LightGBM directional classifier idea
- Shows combining uncorrelated strategies works: 0.61 + 0.90 → 0.96 (diversification benefit)
- Live trading confirmed the backtest (Sharpe 1.35 in 6 months)
- **Action items:**
  1. Our v2 (momentum) could be paired with a low-volatility strategy
  2. Low-vol is already partially in v2 (volatility signal at 10% weight) — but this paper suggests a SEPARATE low-vol strategy, not a signal
  3. The "bad beta" short portfolio is interesting — short the high-vol, high-momentum-losers

## Key Takeaways
1. Independent smart beta strategies combine well (diversification benefit confirmed)
2. Monthly reweighting is sufficient frequency
3. AdaBoost on momentum indicators works for long-short signal generation
4. Low-volatility anomaly is real and complementary to momentum
5. Combined Sharpe (0.96) > individual components (0.61, 0.90) — true diversification
6. Live trading validated backtest (critical — most papers don't do this)
