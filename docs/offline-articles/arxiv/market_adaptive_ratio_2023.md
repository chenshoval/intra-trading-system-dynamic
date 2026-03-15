# Market-Adaptive Ratio for Portfolio Management

**Authors:** Ju-Hong Lee, Bayartsetseg Kalina, KwangTek Na
**Published:** 2023-12-21
**Categories:** q-fin.PM
**URL:** http://arxiv.org/abs/2312.13719v2
**PDF:** https://arxiv.org/pdf/2312.13719v2

## Abstract
Introduces Market-adaptive Ratio that adjusts risk preferences dynamically based on bull vs bear markets. Uses a rho parameter derived from historical data to differentiate regimes. Implemented in RL framework. Outperformed Sharpe Ratio for portfolio allocation.

## Relevance to Our System
- **MODERATE** — The concept is useful but the implementation is RL-heavy
- Key insight: you should have DIFFERENT risk preferences in bull vs bear
- In bull: accept more risk for more return (our v2 top 15 is correct)
- In bear: minimize risk even at cost of return (our downtrend_top_n = 5 is directionally correct)
- The rho parameter could replace our binary uptrend/downtrend gate with a continuous adjustment
- But RL framework is complex — may not be worth it at our stage

## Key Takeaways
1. Risk preferences should change with market regime (not just position count)
2. Bull market → accept more risk, bear market → minimize risk
3. Continuous regime parameter > binary bull/bear switch
4. RL framework for learning optimal rho parameter
