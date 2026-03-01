# Support Zone Bounce Strategy — Brief for Implementation

## Strategy Overview

A **range-trading / demand-zone bounce** strategy for EUR/USD on the 15-minute and 30-minute timeframes. The core idea: identify horizontal zones where price has repeatedly bounced (clustered swing lows), confirm the market is ranging (not trending) on the 1-hour timeframe, and enter long when price wicks into the zone and shows a bullish rejection candle.

---

## Strategy Components

### 1. Demand Zone Detection (15m/30m)

**Goal:** Find a horizontal price band where multiple swing lows cluster.

- **Swing Low Definition:** A bar whose low is lower than the N bars on either side (default N=5).
- **Lookback:** Scan the most recent 200 bars on the 15m timeframe.
- **Clustering:** Group swing lows that fall within 15 pips of each other. The largest cluster becomes the demand zone.
- **Zone Boundaries:** Bottom = lowest wick in the cluster. Top = highest close in the cluster + small buffer (5 pips).
- **Minimum Touches:** At least 2 swing lows must form the cluster for it to be valid.
- **Update Frequency:** Recalculate every ~20 bars (5 hours on 15m).

### 2. Range Context Filter (1H)

**Goal:** Only trade when the market is ranging, not trending.

- Look back 50 bars on the 1H timeframe.
- Calculate `range_ratio = abs(close_now - close_50_bars_ago) / (highest_high - lowest_low)`.
- If `range_ratio < 0.35` → market is **ranging** → strategy is active.
- If `range_ratio >= 0.35` → market is **trending** → no trades.

### 3. Entry Signal (15m)

All four conditions must be true on a single 15m candle:

1. **Wick into zone:** The candle's low penetrates into the demand zone (low ≤ zone_top).
2. **No blowthrough:** The candle's low does not fall more than 15 pips below zone_bottom.
3. **Close above zone:** The candle closes above zone_top (rejection confirmed).
4. **Bullish rejection wick:** The lower wick is ≥ 60% of the candle's total range (high - low). This means `(min(open, close) - low) / (high - low) >= 0.6`.

### 4. Risk Management

| Parameter | Value |
|-----------|-------|
| Stop Loss | 15 pips below zone bottom |
| Take Profit | 30 pips above entry price |
| Risk:Reward | ~1:2 |
| Risk per trade | 1% of equity |
| Max concurrent positions | 1 |
| Position sizing | `quantity = (equity × 0.01) / (entry_price - stop_price)` |

---

## Parameters Summary (All Tunable)

```python
# Zone Detection
swing_lookback = 5           # Bars on each side to qualify as swing low
zone_lookback = 200          # 15m bars to scan for swing lows
zone_cluster_pips = 15       # Max pip distance to cluster swing lows
min_touches = 2              # Min swing lows to form a valid zone

# Range Filter
range_lookback = 50          # 1H bars for range assessment
range_ratio_threshold = 0.35 # Below this = ranging

# Entry
rejection_wick_ratio = 0.6   # Lower wick must be ≥ 60% of candle range

# Risk
stop_loss_pips = 15
take_profit_pips = 30
risk_per_trade = 0.01        # 1% of equity
```

---

## Implementation Notes for QuantConnect

- **Pair:** EURUSD via Oanda brokerage model.
- **Resolution:** Use `Resolution.Minute` with consolidators for 15m and 1H bars.
- **Consolidators:** `QuoteBarConsolidator(timedelta(minutes=15))` for entry TF, `QuoteBarConsolidator(timedelta(hours=1))` for context TF.
- **Warmup:** `self.SetWarmUp(timedelta(days=10))` to fill rolling windows before trading.
- **Data structures:** Use `collections.deque(maxlen=...)` for rolling candle storage.
- **Stop/TP management:** Check in `OnData()` against live price since QuantConnect Forex doesn't always support bracket orders natively.

---

## Possible Enhancements

1. **Session filter** — Only trade during London/NY overlap (07:00–12:00 UTC) when EUR/USD liquidity is highest.
2. **Moving average filter** — Require price to be above a 50 or 100 SMA on the 15m (the original charts show an MA line).
3. **Partial profit-taking** — Close 50% at 1:1 R:R, trail the rest.
4. **Cooldown** — Skip trades for N bars after a losing trade.
5. **Supply zone shorts** — Mirror the logic for sell setups at clustered swing highs.
6. **Zone freshness** — Prefer zones that haven't been tested more than 2-3 times (zones weaken with repeated touches).

---

## Reference Resources

### Understanding Supply & Demand Zones
- **TrendSpider guide:** https://trendspider.com/learning-center/what-are-supply-and-demand-zones/
- **Dukascopy overview:** https://www.dukascopy.com/swiss/english/marketwatch/articles/supply-and-demand-trading/
- **FXOpen patterns guide:** https://fxopen.com/blog/en/supply-and-demand-trading-patterns-and-strategies/
- **LuxAlgo simple guide:** https://www.luxalgo.com/blog/supply-and-demand-zones-a-simple-guide/

### QuantConnect Implementation
- **Strategy Library:** https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library
- **Writing Algorithms docs:** https://www.quantconnect.com/docs/v2/writing-algorithms
- **Python algorithm examples on GitHub:** https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python

### TradingView Indicators (for visual validation)
- **Supply & Demand scripts:** https://www.tradingview.com/scripts/supplyanddemand/
- These can help you visually validate that your algo is detecting the same zones a manual trader would draw.
