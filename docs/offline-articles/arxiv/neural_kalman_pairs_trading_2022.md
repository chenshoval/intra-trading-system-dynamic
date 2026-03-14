# Neural Augmented Kalman Filtering with Bollinger Bands for Pairs Trading (KBPT)

**Authors:** Amit Milstein, Haoran Deng, Guy Revach, Hai Morgenstern, Nir Shlezinger
**Published:** 2022-10-19
**Categories:** q-fin.TR, cs.LG, eess.SP
**URL:** http://arxiv.org/abs/2210.15448v2
**PDF:** https://arxiv.org/pdf/2210.15448v2

## Abstract
Proposes KBPT (KalmanNet-aided Bollinger Bands Pairs Trading), a deep learning approach that augments Kalman Filter + Bollinger Bands pairs trading. Models pairs as having "partial co-integration" via extended state space model. Two-stage training: (1) unsupervised tracking tuning, (2) adaptation for revenue maximization. Systematically yields improved revenue vs model-based and data-driven benchmarks across various assets.

## Relevance to Our System
- **RELEVANT for pairs trading hypothesis** (Roadmap item #7: Pairs trading Kalman/Copula)
- Neural-augmented Kalman filter handles the key problem: cointegration isn't perfect or constant
- "Partial co-integration" model is more realistic than assuming perfect cointegration
- Bollinger Bands for trade signals are simple and interpretable
- Two-stage training avoids overfitting to trading objective
- Could implement as a market-neutral strategy alongside v2 (uncorrelated to momentum)

## Key Takeaways
1. Pure Kalman Filter pairs trading is too simplistic — relationships drift
2. Neural augmentation handles model mismatch without abandoning interpretable structure
3. Still uses Bollinger Bands for signals (simple, proven)
4. Market-neutral → uncorrelated to our long-only momentum rotator
5. Good candidate for second strategy alongside v2
