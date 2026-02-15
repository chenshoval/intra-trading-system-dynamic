# Paper Trading Monitoring Plan

## What's Running
| Strategy | Node | Backtest Return | Status |
|---|---|---|---|
| Congressional Copy-Trading | L-MICRO #1 | 66% / 5yr | LIVE on paper |
| Sentiment + Price (50 tickers) | L-MICRO #2 | 93% / 5yr | Deploying now |

## Cost
- QC Researcher: $60/mo
- Second node: $20/mo
- Congress data: $5/mo
- Tiingo News: Free
- **Total: $85/mo**

## Monitoring Schedule

### Daily (2 min check)
- Log into QC → check both live deployments
- Look for: errors, stopped algorithms, unusual positions
- No action needed unless something is broken

### Weekly (Sunday, 15 min)
- **Approve 2FA** — IBKR requires weekly re-auth via IB Key app
- Check P&L for both strategies
- Compare to SPY performance that week
- Note: are trades executing? How many?

### After 2 Weeks (First Review)
- **Key questions:**
  - Are trades actually executing? (not just backtesting)
  - Does live performance match backtest expectations?
  - Any errors or unexpected behavior?
  - What's the P&L vs SPY?
- **If trades are executing and no errors** → continue for 2 more weeks
- **If no trades at all** → investigate (data issue? market conditions?)
- **If losing badly (>10% drawdown)** → review but don't panic (2 weeks is too short to judge)

### After 1 Month (Serious Review)
- **Compare live vs backtest:**
  - If live return is within 50% of backtest pace → good sign
  - If live is flat or negative while backtest showed gains → investigate
- **Check these metrics:**
  - Win rate (should be >45%)
  - Average trade P&L
  - Max drawdown
  - Number of trades (is it as active as backtest suggested?)
- **Parameter tuning candidates:**
  - Holding period (try 5, 10, 15, 30 days)
  - Sentiment threshold (try 0.1, 0.2, 0.3)
  - Position size (try 3%, 5%, 8%)
  - Max positions (try 10, 15, 20, 25)
  - Top performers filter (congressional: try filtering to top 10 politicians)

### After 2 Months (Go/No-Go Decision)
- **If Sharpe > 0.5 and positive return** → consider deploying with real money ($132)
- **If flat or slightly negative** → tune parameters, backtest again, redeploy
- **If consistently losing** → kill the strategy, move to next hypothesis

## Parameter Tuning Process

When you want to change parameters:
1. **Change parameter in the code** (in this repo)
2. **Backtest in QC** with new parameters (5 years, same period)
3. **Compare to previous backtest** — is it better?
4. **If better** → redeploy to live
5. **If worse** → try different values
6. **Never change more than 1-2 parameters at a time** — otherwise you don't know what helped

## What to Log

Keep a simple journal (even just a text file):
```
Date       | Strategy      | Action          | Notes
-----------|---------------|-----------------|---------------------------
2026-02-15 | Congressional | Deployed paper  | 66% backtest, all politicians
2026-02-15 | Sentiment     | Deployed paper  | 93% backtest, 50 tickers
2026-03-01 | Both          | 2-week review   | Congressional: X trades, $Y P&L
                                              | Sentiment: X trades, $Y P&L
```

## Red Flags (Stop and Investigate)
- Algorithm stops running (check QC dashboard)
- >15% drawdown in a week
- Zero trades for 5+ consecutive trading days (sentiment should trade almost daily)
- IBKR connection errors
- Unexpected large positions

## When to Go Live with Real Money
1. ✅ Paper trading for 1-2 months
2. ✅ Live performance roughly matches backtest
3. ✅ You understand the trades it's making (not a black box)
4. ✅ Max drawdown in paper is acceptable to you
5. ✅ You're comfortable losing the initial amount ($132)
6. → Deploy with $132, run for 1 month
7. → If profitable, scale to $500, then $1,000, etc.
