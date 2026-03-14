# Empirical Investigation of State-of-the-Art Mean Reversion Strategies for Equity Markets

**Authors:** Seung-Hyun Moon, Yong-Hyuk Kim, Byung-Ro Moon
**Published:** 2019-09-10
**Categories:** q-fin.PM
**URL:** http://arxiv.org/abs/1909.04327v1
**PDF:** https://arxiv.org/pdf/1909.04327v1

## Abstract
Investigates performance of state-of-the-art mean reversion strategies (PAMR, OLMAR, TCO) on real market data. Findings: well-known benchmark datasets favor mean reversion strategies, but mean reversion strategies may fail even in favorable market conditions, especially when there exist explicit or implicit transaction costs. Tested on S&P 500 constituent stocks from 2000 to 2017.

## Relevance to Our System
- **CRITICAL WARNING:** Mean reversion strategies look great on benchmark datasets but fail in practice with transaction costs.
- This validates our decision to focus on momentum (trend-following) rather than mean reversion.
- If we ever consider a mean reversion component (pairs trading, etc.), this paper warns that transaction costs are the killer.
- PAMR, OLMAR, TCO all fail on recent S&P 500 data with realistic costs.

## Key Takeaways
1. Mean reversion strategies are highly sensitive to transaction costs
2. Benchmark datasets are misleading — favorable conditions for mean reversion
3. On real S&P 500 data (2000-2017), mean reversion often fails
4. Momentum/trend-following is more robust to costs (our v2 approach validated)
5. If pursuing pairs trading, must carefully account for all costs
