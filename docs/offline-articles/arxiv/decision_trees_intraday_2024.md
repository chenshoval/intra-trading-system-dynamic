# Decision Trees for Intuitive Intraday Trading Strategies

**Authors:** Prajwal Naga, Dinesh Balivada, Sharath Chandra Nirmala, Poornoday Tiruveedi
**Published:** 2024-05-22
**Categories:** q-fin.ST
**URL:** http://arxiv.org/abs/2405.13959v1
**PDF:** https://arxiv.org/pdf/2405.13959v1

## Abstract
Investigates the efficacy of decision trees in constructing intraday trading strategies using existing technical indicators for individual equities in the NIFTY50 index. Unlike conventional methods that rely on fixed rules, the approach leverages decision trees to create unique trading rules for each stock. While the method does not guarantee success for every stock, decision tree-based strategies outperform buy-and-hold for many stocks.

## Relevance to Our System
- **MODERATE:** Per-stock decision tree rules align with our "per-stock threshold" concept from the dual-stream paper.
- Each stock gets its own optimal trading rules — similar to per-stock confidence thresholds (τ*).
- Uses technical indicators as features (RSI, MACD, BBands) — same feature space we're familiar with.
- Could be a simpler alternative to LightGBM for the directional classifier (Hypothesis 2).
- Caution: NIFTY50 (India) results may not transfer to US markets.

## Key Takeaways
1. Per-stock rules via decision trees outperform fixed rules for many stocks
2. Not all stocks are predictable — need stock-specific evaluation
3. Decision trees provide interpretable rules (can audit what the model learned)
4. Simple ML (decision trees) can beat complex approaches for intraday
