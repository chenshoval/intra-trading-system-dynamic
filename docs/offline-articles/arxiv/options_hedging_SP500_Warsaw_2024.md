# Construction and Hedging of Equity Index Options Portfolios

**Authors:** Maciej Wysocki, Robert Ślepaczuk (Warsaw — same group as our MA crossover paper)
**Published:** 2024-07-18
**Categories:** q-fin.PM, q-fin.RM, q-fin.TR
**URL:** http://arxiv.org/abs/2407.13908v1
**PDF:** https://arxiv.org/pdf/2407.13908v1

## Abstract
Systematic S&P 500 index option-writing strategies (2018-2023). Compares BSM and Variance-Gamma hedging models. Finds systematic option-writing can yield superior returns vs buy-and-hold. BSM hedging outperformed VG. Includes realistic transaction costs. 130-minute rehedging intervals provided best balance of protection and returns.

## Relevance to Our System
- **RELEVANT for $100K+ hedging** (our progress.md mentions adding put hedge at scale)
- Same Warsaw research group (Ślepaczuk) — consistent quality
- Shows option-writing (selling puts/calls) generates income in bear markets
- Option BUYING (protective puts) is the simpler version — costs premium but limits downside
- At $100K+: buying SPY puts during bear regime could cap max drawdown at ~15% vs current 27-36%
- At $500-$20K: too expensive, not practical
- Key finding: BSM hedging is sufficient (don't need fancy models)

## Key Takeaways
1. Systematic option strategies work with realistic costs (finally, a paper that tests fees!)
2. Option-writing (selling premium) generates income but has tail risk
3. Option-buying (puts for protection) costs premium but limits downside
4. 130-minute rehedging is enough (don't need continuous)
5. BSM > VG for hedging (simpler model works better)
6. Same Warsaw group — trustworthy methodology
