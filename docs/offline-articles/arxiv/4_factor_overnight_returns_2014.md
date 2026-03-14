# 4-Factor Model for Overnight Returns

**Authors:** Zura Kakushadze
**Published:** 2014-10-21
**Categories:** q-fin.PM, q-fin.ST
**URL:** http://arxiv.org/abs/1410.5513v2
**PDF:** https://arxiv.org/pdf/1410.5513v2

## Abstract
Proposes a 4-factor model for overnight returns using ONLY intraday price and volume data. Factors: size (price), volatility, momentum, liquidity (volume). Fundamental factors (value, growth) have NO predictive power for overnight returns. Fama-MacBeth regressions show sizable serial t-statistics. Uses 4-factor model in an explicit intraday mean-reversion alpha.

## Relevance to Our System
- **RELEVANT for daily/overnight strategy design**
- The 4 factors are all computable from OHLCV data (no alternative data needed)
- Fundamental factors DON'T predict overnight returns → confirms our v2 approach at monthly scale is different from daily
- Same author as 151 Trading Strategies (Kakushadze) — prolific and credible
- Could build a simple overnight alpha: size + volatility + momentum + liquidity → predict overnight return → buy at close, sell at open
- **Action item**: Test a simple daily overnight strategy on QC as a second uncorrelated strategy

## Key Takeaways
1. Overnight returns are predictable using 4 intraday factors
2. Fundamentals don't matter for overnight → different from monthly momentum
3. Factors: size (price), volatility, momentum, liquidity (volume) — all from OHLCV
4. Intraday mean-reversion alpha is real
5. Low correlation with monthly momentum → good diversification candidate
