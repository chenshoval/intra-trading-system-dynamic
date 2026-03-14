# The Case for Long-Only Agnostic Allocation Portfolios (AAP)

**Authors:** Pierre-Alain Reigneron, Vincent Nguyen, Stefano Ciliberti, Philip Seager, Jean-Philippe Bouchaud
**Published:** 2019-06-12
**Categories:** q-fin.PM
**URL:** http://arxiv.org/abs/1906.05187v1
**PDF:** https://arxiv.org/pdf/1906.05187v1
**NOTE:** Authors from Capital Fund Management (CFM) — top quant hedge fund

## Abstract
Advocates "Agnostic Allocation" for long-only stock portfolios. AAPs mitigate excess concentration, high turnover, and strong low-risk factor exposure of classical methods, while achieving similar performance. Works with or without an active trading signal. A risk-based portfolio construction method.

## Relevance to Our System
- **RELEVANT for portfolio construction improvement**
- From Capital Fund Management (CFM) — one of the world's top quant firms
- Equal-weight is our current approach (close to "agnostic" allocation)
- AAP is a formalized version that handles concentration/turnover better
- Could replace our simple "equal weight top 15" with a more sophisticated allocation
- Works WITHOUT active signals (pure risk-based) → could be used as baseline
- Works WITH active signals → could enhance our momentum scoring output
- Low turnover property is important for our monthly rebalance

## Key Takeaways
1. "Agnostic" = don't try to predict returns, just manage risk
2. Equal-weight is close to agnostic but AAP handles edge cases better
3. Reduces concentration risk (our v2 has 15 stocks, each ~6.7%)
4. Reduces turnover (we already have monthly = low turnover)
5. From top quant firm → credible, practical advice
6. Could be an improvement over equal-weight for our top-15 allocation
