# Trading Concepts Summary — TSLA vs SPY Discussion

## Context
Started from a Reddit r/algotrading post where someone bought TSLA and noticed it underperforming SPY, questioning whether stock picking beats index investing.

---

## Key Concepts

### Beta
- Measures how much a stock moves **relative to the market (SPY)**
- Beta = 1.0 → moves with market
- Beta = 2.0 → moves 2x the market (TSLA is roughly here)
- If beta = 1 and stock matches SPY return → you got nothing extra for the added single-stock risk

### Alpha
- Return **after accounting for risk taken**
- If TSLA has beta=2 and SPY returned 10%, you'd *expect* ~20% from TSLA
- If TSLA only returned 15% → **negative alpha** (-5%)
- Alpha is a **scorecard**, not a strategy

### Alpha/Beta Are Risk Evaluation Tools
- They don't tell you what to buy
- They tell you **whether what you bought was worth the risk**, in hindsight
- Most professional fund managers fail to generate consistent alpha over 10+ years

---

## Cointegration & Mean Reversion

### Mean Reversion
- Some things that drift apart tend to come back together
- Analogy: dog on a leash — wanders but can't go too far

### Cointegration (TSLA vs SPY)
- Two assets share a long-term economic relationship
- TSLA is part of S&P 500, both driven by macro factors
- The ratio between them should mean-revert, not drift forever

### The Pair Trade
- Instead of "will TSLA go up?" → ask "is TSLA cheap/expensive *relative to SPY* right now?"
- Ratio too low → Buy TSLA, Short SPY
- Ratio too high → Short TSLA, Buy SPY
- **Market neutral** — doesn't matter if market crashes or rallies, you bet on the spread closing

### Risk
- Cointegration can break down (leash snaps)
- If TSLA fundamentally changes, historical relationship with SPY may no longer hold
- The downtrending ratio in the post suggests possible breakdown

---

## Screening for Alpha — Practical Approach

### No-Code Option
- **portfoliovisualizer.com** → Factor Analysis → plug in ticker → get alpha, beta, R²

### Python Code (Core Logic)
```python
import numpy as np

# Daily returns arrays
X = np.array(returns_spy)
Y = np.array(returns_stock)

X_with_const = np.column_stack([np.ones(len(X)), X])
coefficients = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]

alpha = coefficients[0]       # Daily alpha
beta = coefficients[1]        # Beta
annual_alpha = alpha * 252    # Annualized

print(f"Beta: {beta:.2f}")
print(f"Annual Alpha: {annual_alpha:.2%}")
```

### What to Look For
| Metric | Good Sign |
|--------|-----------|
| Alpha | Positive and consistent across different time windows |
| Beta | Understand the risk you're taking on |
| R² | How much movement is explained by SPY |

---

## Factor Investing (More Robust Than Stock Picking)

Instead of individual stocks, screen by **factors** with historically persistent alpha:

| Factor | Meaning | ETF |
|--------|---------|-----|
| Momentum | Trending stocks keep trending | MTUM |
| Quality | High margins, low debt | QUAL |
| Value | Cheap vs expensive by P/E | VLUE |
| Low Volatility | Less volatile stocks surprisingly outperform | USMV |

---

## Next Step
Build a QuantConnect algorithm that screens for momentum alpha stocks automatically.
