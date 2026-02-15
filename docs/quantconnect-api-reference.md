# QuantConnect API Reference — Lessons Learned

## How We Found These

Installed `QuantConnect-stubs` locally (`pip install QuantConnect-stubs`) and grepped the `.pyi` stub files for correct class names, method signatures, and enum values. The stubs live at:
```
pip show QuantConnect-stubs  # shows Location
```

Then search with:
```bash
grep -r "Congress" <stubs_path>/QuantConnect/DataSource/__init__.pyi
grep -r "def add_universe" <stubs_path>/QuantConnect/Algorithm/__init__.pyi
```

---

## Quiver Congressional Trading Dataset

### Correct Classes
| Class | Type | Purpose |
|---|---|---|
| `QuiverCongress` | `BaseDataCollection` | Universe-level collection of congressional trades |
| `QuiverCongressDataPoint` | `BaseData` | Single congressional trade data point |
| `QuiverQuantCongressUniverse` | extends `QuiverCongress` | Universe selection helper |

### Correct Import
```python
from AlgorithmImports import *
from QuantConnect.DataSource import *
```
**DO NOT** use `from QuantConnect.Data.Custom.Tiingo import *` or similar — Quiver is in `DataSource`.

### How to Add the Universe
```python
# CORRECT — pass type, name string, and filter callback
self.add_universe(QuiverCongress, "QuiverCongress", self.congress_filter)

# WRONG — QuiverQuantCongressUniverse is NOT an IUniverseSelectionModel
self.add_universe_selection(QuiverQuantCongressUniverse())

# WRONG — add_universe doesn't take (Type, callback) without a name
self.add_universe(QuiverQuantCongressUniverse, self.congress_filter)
```

### `add_universe` Signatures (from stubs)
```python
def add_universe(self, t: Type, name: str, selector: Callable) -> Universe
def add_universe(self, t: Type, name: str, resolution: Resolution, selector: Callable) -> Universe
def add_universe(self, t: Type, name: str, resolution: Resolution, universe_settings: UniverseSettings, selector: Callable) -> Universe
```

### Filter Callback
The filter receives the data collection and returns a list of symbols:
```python
def congress_filter(self, data):
    # data is iterable of QuiverCongressDataPoint
    symbols = []
    for point in data:
        # point.symbol        -> Symbol
        # point.representative -> str (e.g., "Nancy Pelosi")
        # point.transaction   -> OrderDirection enum
        # point.amount        -> Optional[float] (USD, can be None)
        # point.maximum_amount -> Optional[float]
        # point.report_date   -> Optional[datetime]
        # point.transaction_date -> datetime
        # point.record_date   -> datetime
        # point.house         -> Congress enum (SENATE=0, REPRESENTATIVES=1)
        # point.party         -> Party enum (INDEPENDENT=0, REPUBLICAN=1, DEMOCRATIC=2)
        # point.district      -> str (empty for Senators)
        # point.state         -> str
        symbols.append(point.symbol)
    return symbols
```

### QuiverCongressDataPoint Fields
| Field | Type | Description |
|---|---|---|
| `symbol` | `Symbol` | The stock ticker |
| `representative` | `str` | Politician name |
| `transaction` | `OrderDirection` | BUY, SELL, or HOLD |
| `amount` | `Optional[float]` | Trade amount in USD (lower bound of range) |
| `maximum_amount` | `Optional[float]` | Upper bound of trade amount range |
| `report_date` | `Optional[datetime]` | When the trade was disclosed |
| `transaction_date` | `datetime` | When the trade actually happened |
| `record_date` | `datetime` | When QuiverQuant recorded it |
| `house` | `Congress` | SENATE=0, REPRESENTATIVES=1 |
| `party` | `Party` | INDEPENDENT=0, REPUBLICAN=1, DEMOCRATIC=2 |
| `district` | `str` | Congressional district (empty for Senators) |
| `state` | `str` | State |

---

## OrderDirection Enum

### CORRECT — uppercase
```python
OrderDirection.BUY    # not .Buy
OrderDirection.SELL   # not .Sell
OrderDirection.HOLD   # not .Hold
```

### WRONG
```python
OrderDirection.Buy     # AttributeError
OrderDirection.Sell     # AttributeError
"Purchase"             # it's an enum, not a string
```

---

## Common QC API Gotchas

### 1. `timedelta` is available
`from AlgorithmImports import *` includes `timedelta`, `datetime`, etc.
Your local IDE may show a warning but it works in QC.

### 2. `self.log()` vs `print()`
- `self.log()` goes to the **Logs** tab (may not show in Cloud Terminal)
- `print()` goes to **Cloud Terminal** (visible during backtest)
- Use `print()` for debugging, `self.log()` for production logging

### 3. `self.get_parameter()`
Only works with string values in QC. For numeric defaults, assign directly:
```python
# CORRECT
self.holding_period = 30

# MAY FAIL in some QC versions
self.holding_period = self.get_parameter("holding_period", 30)
```

### 4. Adding Libraries
The "Choose a library..." dropdown in QC IDE is for **your own custom libraries** (other QC projects), NOT for pip packages. `QuantConnect.DataSource` is always available via import.

### 5. f-strings with `$` and `,`
QC's Python supports f-strings. These work:
```python
print(f"${amount:,.0f}")   # $50,000
print(f"{pct:.2%}")        # 5.00%
```

---

## Tiingo News Dataset (for Sentiment Strategy)

### Import
```python
from QuantConnect.Data.Custom.Tiingo import TiingoNews
```
**TODO**: Verify this import when we test the sentiment strategy. May need `from QuantConnect.DataSource import *` instead.

### Adding Data
```python
# Per-ticker news
news = self.add_data(TiingoNews, ticker)
```

---

## Debugging Checklist

When a QC backtest runs but produces no trades:

1. **Add `print()` at every method entry** — confirm methods are being called
2. **Log the first N data points** — see actual field values and types
3. **Check enum values** — QC uses UPPERCASE (BUY not Buy)
4. **Check if data is None/empty** — `amount` can be `None`
5. **Check `self.securities[symbol]`** — symbol might not be in securities yet
6. **Check universe filter return** — must return `list[Symbol]`
7. **Verify data is being consumed** — flat equity curve = no trades executed
