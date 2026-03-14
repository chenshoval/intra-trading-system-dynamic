# A Framework for Predictive Directional Trading Based on Volatility and Causal Inference

**Source:** Semantic Scholar / arXiv
**Authors:** Ivan Letteri
**Year:** 2025
**URL:** https://www.semanticscholar.org/paper/d9848b0d9211d6b456acbe7ec169449c2eac027f
**DOI:** 10.48550/arXiv.2507.09347

## Abstract (partial)
Introduces a framework for identifying and exploiting predictive lead-lag relationships in financial markets. Integrates Gaussian Mixture Model (GMM) clustering with machine learning for identifying lead-lag relationships between equities.

## Relevance to Our System
- **INTERESTING** — Lead-lag relationships could improve our cross-stock scoring
- GMM clustering identifies which stocks lead others — useful for our 50-stock universe
- If MSFT leads AAPL by a day, we can use MSFT's signal to score AAPL
- This is the "cross-stock features" idea from our dual-stream paper hypothesis
- Causal inference framework (not just correlation) → more robust
- Could enhance Hypothesis 2 (Global Directional Classifier) with lead-lag features

## Key Takeaways
1. Lead-lag relationships between stocks are predictable and tradeable
2. GMM clustering identifies groups with similar behavior
3. Causal inference > simple correlation for finding lead-lag
4. Could add cross-stock features to our scoring model
