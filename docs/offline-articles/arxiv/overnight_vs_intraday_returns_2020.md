# Strikingly Suspicious Overnight and Intraday Returns

**Authors:** Bruce Knuteson
**Published:** 2020-10-05
**Categories:** q-fin.GN, econ.GN, q-fin.TR
**URL:** http://arxiv.org/abs/2010.01727v1
**PDF:** https://arxiv.org/pdf/2010.01727v1

## Abstract
The world's stock markets display a strikingly suspicious pattern of overnight and intraday returns. Overnight returns to major stock market indices over the past few decades have been wildly positive, while intraday returns have been disturbingly negative. The cause of these astonishingly consistent return patterns is unknown.

## Relevance to Our System
- **IMPORTANT FOR INTRADAY STRATEGY DESIGN:** If intraday returns are systematically negative, pure intraday long strategies face a structural headwind.
- Suggests an overnight-hold strategy (buy at close, sell at open) captures the bulk of equity returns.
- For any intraday strategy we build, we need to be aware that the intraday return distribution is unfavorable for longs.
- Could inform a simple "overnight gap capture" strategy: buy at close, sell at open.
- Our monthly rotator benefits from this since it holds overnight by default.

## Key Takeaways
1. Overnight returns are positive, intraday returns are negative (systematic pattern)
2. Intraday long-only strategies fight a structural headwind
3. Holding overnight captures the equity premium
4. Any intraday strategy should consider this asymmetry
