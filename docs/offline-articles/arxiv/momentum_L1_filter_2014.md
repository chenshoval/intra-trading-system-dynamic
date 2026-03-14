# Momentum Strategies with L1 Filter

**Authors:** Tung-Lam Dao
**Published:** 2014-03-17
**Categories:** q-fin.PM
**URL:** http://arxiv.org/abs/1403.4069v1
**PDF:** https://arxiv.org/pdf/1403.4069v1

## Abstract
Discusses L1 filtering to detect trend properties in noisy price signals. L1 penalty produces filtered signals composed of straight trends or steps. Financial time series have a global trend and local trends. Combining these two time scales forms a model of global trend with mean-reverting properties. Applications to momentum strategies discussed in detail with trend configurations.

## Relevance to Our System
- **MODERATE** — Alternative trend detection method to our MA crossovers
- L1 filter separates global trend from local noise — could improve our SPY trend gate
- The global + local trend decomposition maps to our multi-timeframe momentum scoring
- Cross-validation for the regularization parameter = principled way to set our trend thresholds
- Could replace our ad-hoc MA 10/50 crossover with a more robust trend detection
- Practical concern: QC may not have L1 filter libraries readily available

## Key Takeaways
1. L1 filter provides cleaner trend detection than moving averages
2. Separates global trend from local mean-reversion
3. Regularization parameter controls sensitivity (analogous to our MA periods)
4. Cross-validated parameter selection avoids overfitting
5. Applicable to momentum strategies at multiple timeframes
