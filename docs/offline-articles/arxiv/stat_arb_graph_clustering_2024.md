# Statistical Arbitrage in Multi-Pair Trading Strategy Based on Graph Clustering Algorithms in US Equities Market

**Authors:** Adam Korniejczuk, Robert Ślepaczuk
**Published:** 2024-06-15
**Categories:** q-fin.PM, q-fin.TR, stat.ML
**URL:** http://arxiv.org/abs/2406.10695v1
**PDF:** https://arxiv.org/pdf/2406.10695v1

## Abstract
Develops statistical arbitrage framework using graph clustering algorithms for pair selection. Combines Kelly criterion with ensemble ML classifiers for signal detection and risk management. Proposes innovative take-profit and stop-loss optimization for daily frequency trading. All tested approaches outperformed benchmarks under realistic transaction costs. NOTE: Same lead researcher (Ślepaczuk) as our Warsaw paper already in docs/.

## Relevance to Our System
- **HIGHLY RELEVANT** — Same research group as our existing Warsaw MA crossover paper
- Daily frequency trading with realistic transaction costs (our weak spot)
- Graph clustering for pair selection is more scalable than brute-force
- Kelly criterion for position sizing (aligns with our "position sizing is where risk lives" philosophy)
- Ensemble ML classifiers for signal quality (better than single model)
- Take-profit/stop-loss optimization — we currently don't use these in v2
- Daily frequency → could be a second strategy running alongside monthly v2

## Key Takeaways
1. Graph clustering finds better pairs than exhaustive search
2. Kelly criterion for sizing improves risk-adjusted returns
3. ML ensemble for signal detection outperforms single classifier
4. Take-profit/stop-loss optimization matters for daily strategies
5. Realistic transaction costs included (honest results)
6. Same Warsaw research group — consistent quality
