# Multi-Factor Market-Neutral Strategy for NYSE Equities

**Authors:** Georgios M. Gkolemis, Adwin Richie Lee, Amine Roudani
**Published:** 2024-12-16
**Categories:** q-fin.TR
**URL:** http://arxiv.org/abs/2412.12350v1
**PDF:** https://arxiv.org/pdf/2412.12350v1

## Abstract
Systematic market-neutral, multi-factor strategy for NYSE equities. Combines momentum indicators, fundamental factors, and analyst recommendations. Risk parity portfolio construction demonstrated superior Sharpe ratio, lower beta, and smaller maximum drawdown vs S&P 500. Market-neutral = balanced long AND short positions.

## Relevance to Our System
- **RELEVANT** — Shows that market-neutral (long + short) CAN work if done properly
- Key difference from our failed v3: they use risk parity weighting, not equal weight
- Key difference from our failed v3: they short from a BROAD universe (NYSE), not our curated 50 quality stocks
- Risk parity produced best results (vs equal weight and min variance)
- Uses analyst recommendations as a factor — we could use Tiingo news as proxy
- The short leg needs genuinely weak/overvalued stocks, not just "lowest scored of our quality 50"

## Why Our v3 Failed But This Works
Our v3 shorted the bottom 5 of our 50 hand-picked quality large-caps. But those are ALL good companies — shorting the "worst" of a great pool means shorting decent stocks. This paper shorts from the full NYSE universe, finding actually weak stocks to short.

## Key Takeaways
1. Market-neutral requires shorting genuinely BAD stocks, not just lowest-ranked good ones
2. Risk parity weighting > equal weighting for long-short
3. Multiple factor types (momentum + fundamentals + analyst) beat single-factor
4. You need a BROAD short universe to find real short candidates
