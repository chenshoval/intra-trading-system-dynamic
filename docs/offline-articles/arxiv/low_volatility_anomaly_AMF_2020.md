# The Low-Volatility Anomaly and the Adaptive Multi-Factor Model

**Authors:** Robert A. Jarrow, Rinald Murataj, Martin T. Wells, Liao Zhu
**Published:** 2020-03-16
**Categories:** q-fin.ST, stat.ML
**URL:** http://arxiv.org/abs/2003.08302v2
**PDF:** https://arxiv.org/pdf/2003.08302v2

## Abstract
Provides new explanation of the low-volatility anomaly using the Adaptive Multi-Factor (AMF) model. Low and high volatility portfolios load on very DIFFERENT factors — volatility is not independent risk but related to existing risk factors. Out-performance of low-vol portfolio is due to equilibrium performance of these loaded risk factors. AMF model outperforms Fama-French 5-factor model both in-sample and out-of-sample.

## Relevance to Our System
- **RELEVANT** — Explains WHY low-vol works (it's factor exposure, not "anomaly")
- Our v2 already uses volatility as 10% signal weight — this validates that
- Key insight: low-vol and high-vol stocks load on DIFFERENT factors
- Suggests we could improve v2 by understanding which factors drive our low-vol signal
- AMF model outperforms Fama-French 5-factor → more sophisticated factor model could help
- Confirms combining momentum + low-vol (as in the smart beta paper) is not double-counting — they're genuinely different factors

## Key Takeaways
1. Low-volatility anomaly is real but explained by factor loadings
2. Low-vol and high-vol stocks are fundamentally different (different factor exposures)
3. Momentum + low-vol combination is valid (they capture different things)
4. AMF model > Fama-French 5-factor for explaining this
5. Our 10% volatility weight in v2 scoring is directionally correct
