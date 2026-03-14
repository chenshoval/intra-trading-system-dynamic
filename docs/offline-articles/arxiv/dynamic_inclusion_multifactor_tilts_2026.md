# Dynamic Inclusion and Bounded Multi-Factor Tilts for Robust Portfolio Construction

**Authors:** Roberto Garrone
**Published:** 2026-01-08
**Categories:** math.OC, cs.LG
**URL:** http://arxiv.org/abs/2601.05428v1
**PDF:** https://arxiv.org/pdf/2601.05428v1

## Abstract
Portfolio construction framework combining dynamic asset eligibility, deterministic rebalancing, and bounded multi-factor tilts on an equal-weight baseline. Uses cross-sectional rankings and hard structural bounds (not estimated returns/covariances). Robust under estimation error, non-stationarity, and realistic trading constraints. Fully algorithmic, transparent, and directly implementable.

## Relevance to Our System
- **EXTREMELY RELEVANT** — This describes almost exactly what our v2 does, but formalized academically!
- Equal-weight baseline + factor tilts = our equal-weight top-15 with scoring tilts
- Dynamic asset eligibility = our "skip stocks we can't afford" + scoring cutoffs
- Cross-sectional rankings (not return predictions) = our 5-signal scoring system
- Hard bounds on concentration/turnover = our top-15 cap + monthly rebalance
- State-dependent constraints (liquidity, volatility, breadth) = our SPY trend gate
- Published January 2026 — very recent, validates our approach academically
- **Action: Read this paper in full — it may provide theoretical justification for v2 AND improvement ideas**

## Key Takeaways
1. Cross-sectional rankings > predicted returns (what we already do!)
2. Equal-weight + bounded factor tilts is a robust framework
3. Dynamic eligibility based on market conditions (we do this with SPY gate)
4. Hard bounds on concentration prevent overfitting
5. Framework is transparent and implementable — not a black box
6. VERY recent (Jan 2026) — state-of-the-art thinking
