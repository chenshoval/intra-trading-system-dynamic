# Hidden Order in Trades Predicts the Size of Price Moves

**Authors:** Mainak Singha
**Published:** 2025-12-02
**Categories:** q-fin.TR, q-fin.ST, stat.AP, stat.ME
**URL:** http://arxiv.org/abs/2512.15720v1
**PDF:** https://arxiv.org/pdf/2512.15720v1

## Abstract
Demonstrates that real-time order-flow entropy (15-state Markov transition matrix at second resolution) predicts the MAGNITUDE of intraday returns WITHOUT predicting direction. Low entropy (below 5th percentile) increases subsequent 5-minute absolute returns by 2.89x (t=12.41). But directional accuracy stays at 45% (random). Walk-forward validated. 38.5M SPY trades over 36 days.

## Relevance to Our System
- **INTERESTING FOR FUTURE INTRADAY** — But requires tick-level data
- Key insight: you can predict VOLATILITY (move size) without predicting direction
- For an intraday strategy, this means: when entropy is low, expect big moves → use options straddles or wider stops
- Could combine with directional signals: entropy predicts "something big is coming", other signals predict direction
- Limitation: only 36 days of data, single instrument (SPY) — needs more validation
- Requires QC Researcher tier ($10/mo) for tick/second data
- Not immediately actionable but a fascinating research direction

## Key Takeaways
1. Order-flow entropy predicts move magnitude, not direction
2. Low entropy = informed trading = big move incoming
3. Direction remains unpredictable (efficient market for direction, not volatility)
4. Walk-forward validated but short sample (36 days)
5. Potential use: volatility targeting, straddle timing, dynamic stop-loss widths
