# Detecting Data-Driven Robust Statistical Arbitrage Strategies with Deep Neural Networks

**Authors:** Ariel Neufeld, Julian Sester, Daiying Yin
**Published:** 2022-03-07 (Updated: 2024-02-26)
**Categories:** q-fin.CP, cs.LG, q-fin.MF, q-fin.ST, q-fin.TR
**URL:** http://arxiv.org/abs/2203.03179v4
**PDF:** https://arxiv.org/pdf/2203.03179v4

## Abstract
Deep neural network approach for identifying robust statistical arbitrage strategies under model ambiguity. Does NOT require cointegrated pairs — works on high-dimensional markets (tested up to 50 securities simultaneously). Profitable even during financial crises and when cointegration breaks down. Entirely data-driven and model-free.

## Relevance to Our System
- **INTERESTING BUT COMPLEX** — Goes beyond simple pairs trading
- Handles 50 securities simultaneously (matches our 50-stock universe!)
- Robust to model ambiguity (doesn't assume specific statistical relationship)
- Works during financial crises (our biggest weakness is bear market drawdown)
- Doesn't need cointegration (classical pairs trading limitation)
- BUT: Deep NN approach = harder to implement and validate on QC
- Could be a future research direction after we prove simpler pairs trading works

## Key Takeaways
1. Statistical arbitrage doesn't require cointegration — neural nets can find non-linear relationships
2. Works in high dimensions (50 stocks) — could use our existing universe
3. Robust during crises — exactly what we need for bear market protection
4. Model-free, data-driven — no assumptions about relationships
5. Complexity trade-off: powerful but harder to implement and trust
