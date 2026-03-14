# DeepAries: Adaptive Rebalancing Interval Selection for Enhanced Portfolio Selection

**Authors:** Jinkyu Kim, Hyunjung Yi, Mogan Gim, Donghee Choi, Jaewoo Kang
**Published:** 2025-09-11
**Categories:** q-fin.PM, cs.AI, cs.CE
**URL:** http://arxiv.org/abs/2510.14985v1
**PDF:** https://arxiv.org/pdf/2510.14985v1
**Code:** https://github.com/dmis-lab/DeepAries
**Demo:** https://deep-aries.github.io/

## Abstract
DeepAries is a novel deep reinforcement learning framework for dynamic portfolio management that jointly optimizes the timing and allocation of rebalancing decisions. Unlike prior RL methods that employ fixed rebalancing intervals regardless of market conditions, DeepAries adaptively selects optimal rebalancing intervals along with portfolio weights to reduce unnecessary transaction costs and maximize risk-adjusted returns. The framework integrates a Transformer-based state encoder with PPO to generate simultaneous discrete (rebalancing intervals) and continuous (asset allocations) actions. Experiments on multiple real-world financial markets demonstrate it significantly outperforms traditional fixed-frequency and full-rebalancing strategies in terms of risk-adjusted returns, transaction costs, and drawdowns.

## Relevance to Our System
- **HIGHLY RELEVANT:** Our v2 uses fixed monthly rebalance + Wednesday emergency check. This paper shows adaptive rebalancing (skip when market is stable, rebalance more when volatile) can beat fixed schedules.
- Could inform whether bi-weekly rebalance (already on roadmap) is the right move, or whether adaptive timing is better.
- The Transformer state encoder for regime detection could replace our simple SPY MA 10/50 trend gate.
- Key insight: don't rebalance on a calendar — rebalance when market conditions warrant it.

## Key Takeaways
1. Fixed rebalancing intervals waste money on transaction costs during stable markets
2. Adaptive intervals + allocation in a unified framework beats both
3. Transformer captures long-term market dependencies for regime detection
4. Interpretable decisions aligned with market regime shifts
