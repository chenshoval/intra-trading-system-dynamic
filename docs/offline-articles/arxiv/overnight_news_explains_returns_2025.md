# Does Overnight News Explain Overnight Returns?

**Authors:** Paul Glasserman, Kriste Krstovski, Paul Laliberte, Harry Mamaysky
**Published:** 2025-07-06
**Categories:** q-fin.TR, cs.CL, stat.ML
**URL:** http://arxiv.org/abs/2507.04481v1
**PDF:** https://arxiv.org/pdf/2507.04481v1

## Abstract
Over the past 30 years, nearly all U.S. stock market gains have been earned overnight; average intraday returns are negative or flat. The paper explains this through news — 2.4M articles analyzed with supervised topic analysis. Overnight news topics and their responses explain the overnight return premium. Out-of-sample forecasts identify which stocks do well overnight and poorly intraday. Also explains continuation and reversal patterns.

## Relevance to Our System
- **HIGHLY RELEVANT** — Directly connects news sentiment to overnight returns
- Confirms: overnight returns carry the equity premium, intraday is flat/negative
- Our v2 holds positions overnight (monthly) → already captures this premium
- For any intraday strategy: we'd be fighting a structural headwind
- **Key insight for sentiment strategy (Hypothesis 4)**: News content predicts WHICH stocks benefit overnight
- Could enhance v2: use overnight news sentiment to predict next-day openers
- 2.4M articles, supervised topic analysis — serious methodology
- Out-of-sample forecasting works → this is tradeable alpha

## Key Takeaways
1. Overnight returns = where the money is (30 years of evidence)
2. News topics explain WHY some stocks benefit more overnight
3. Out-of-sample forecasting works — alpha is real
4. Continuation and reversal patterns are news-driven
5. For us: don't build pure intraday strategies; instead optimize overnight exposure
6. Our Tiingo news scoring in v2 is on the right track — but could be more sophisticated
