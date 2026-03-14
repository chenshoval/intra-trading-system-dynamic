"""
============================================================
  CHART READING 101 - Everything Micha Charges 6K NIS For
============================================================

This script generates fake stock charts with annotations
teaching you how to read every pattern, indicator, and signal.

Run this to generate visual lessons as PNG files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrowPatch
import os

# Create output directory
os.makedirs("notebooks/05_chart_reading/output", exist_ok=True)

# Style
plt.style.use('dark_background')
COLORS = {
    'green': '#00ff88',
    'red': '#ff4444',
    'blue': '#4488ff',
    'yellow': '#ffcc00',
    'white': '#ffffff',
    'gray': '#666666',
    'orange': '#ff8800',
    'purple': '#aa66ff',
    'cyan': '#00cccc',
}

def generate_price_data(n=200, start=100, volatility=2, trend=0.05, seed=42):
    """Generate realistic fake OHLCV data"""
    np.random.seed(seed)
    closes = [start]
    for i in range(1, n):
        change = np.random.normal(trend, volatility)
        closes.append(max(closes[-1] + change, 10))

    opens, highs, lows, volumes = [], [], [], []
    for c in closes:
        spread = abs(np.random.normal(0, volatility * 0.5))
        o = c + np.random.normal(0, spread)
        h = max(c, o) + abs(np.random.normal(0, volatility * 0.3))
        l = min(c, o) - abs(np.random.normal(0, volatility * 0.3))
        v = int(np.random.lognormal(15, 0.5))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        volumes.append(v)

    return np.array(opens), np.array(highs), np.array(lows), np.array(closes), np.array(volumes)


def draw_candlestick(ax, opens, highs, lows, closes, offset=0, width=0.6):
    """Draw candlestick chart"""
    for i in range(len(closes)):
        x = i + offset
        if closes[i] >= opens[i]:
            color = COLORS['green']
            body_bottom = opens[i]
            body_height = closes[i] - opens[i]
        else:
            color = COLORS['red']
            body_bottom = closes[i]
            body_height = opens[i] - closes[i]

        # Wick
        ax.plot([x, x], [lows[i], highs[i]], color=color, linewidth=0.5)
        # Body
        ax.add_patch(patches.Rectangle(
            (x - width/2, body_bottom), width, max(body_height, 0.1),
            facecolor=color, edgecolor=color, linewidth=0.5
        ))


# ============================================================
# LESSON 1: CANDLESTICK BASICS - What is a candle?
# ============================================================
print("Generating Lesson 1: Candlestick Basics...")

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
fig.suptitle("LESSON 1: CANDLESTICK BASICS\nWhat Each Candle Tells You",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# Single bullish candle explained
ax = axes[0]
ax.set_title("BULLISH CANDLE (Green)\nPrice went UP", fontsize=14, color=COLORS['green'])
ax.set_xlim(-2, 4)
ax.set_ylim(90, 115)

# Draw a big candle
ax.add_patch(patches.Rectangle((0.5, 95), 1, 10, facecolor=COLORS['green'], edgecolor='white', linewidth=2))
ax.plot([1, 1], [92, 95], color=COLORS['green'], linewidth=3)  # lower wick
ax.plot([1, 1], [105, 112], color=COLORS['green'], linewidth=3)  # upper wick

# Labels
ax.annotate('HIGH = $112\n(highest price reached)', xy=(1, 112), xytext=(2.5, 112),
           fontsize=11, color='white', arrowprops=dict(arrowstyle='->', color='white'),
           bbox=dict(boxstyle='round', facecolor='#333333'))
ax.annotate('CLOSE = $105\n(price at end)', xy=(1.5, 105), xytext=(2.5, 106),
           fontsize=11, color=COLORS['green'], arrowprops=dict(arrowstyle='->', color=COLORS['green']),
           bbox=dict(boxstyle='round', facecolor='#1a3320'))
ax.annotate('OPEN = $95\n(price at start)', xy=(0.5, 95), xytext=(-1.5, 96),
           fontsize=11, color=COLORS['green'], arrowprops=dict(arrowstyle='->', color=COLORS['green']),
           bbox=dict(boxstyle='round', facecolor='#1a3320'))
ax.annotate('LOW = $92\n(lowest price reached)', xy=(1, 92), xytext=(2.5, 93),
           fontsize=11, color='white', arrowprops=dict(arrowstyle='->', color='white'),
           bbox=dict(boxstyle='round', facecolor='#333333'))

ax.annotate('', xy=(2, 95), xytext=(2, 105),
           arrowprops=dict(arrowstyle='<->', color=COLORS['yellow'], lw=2))
ax.text(2.1, 100, 'BODY\n(Open to Close)', fontsize=10, color=COLORS['yellow'])
ax.set_axis_off()

# Single bearish candle explained
ax = axes[1]
ax.set_title("BEARISH CANDLE (Red)\nPrice went DOWN", fontsize=14, color=COLORS['red'])
ax.set_xlim(-2, 4)
ax.set_ylim(90, 115)

ax.add_patch(patches.Rectangle((0.5, 95), 1, 10, facecolor=COLORS['red'], edgecolor='white', linewidth=2))
ax.plot([1, 1], [92, 95], color=COLORS['red'], linewidth=3)
ax.plot([1, 1], [105, 112], color=COLORS['red'], linewidth=3)

ax.annotate('HIGH = $112', xy=(1, 112), xytext=(2.5, 112),
           fontsize=11, color='white', arrowprops=dict(arrowstyle='->', color='white'),
           bbox=dict(boxstyle='round', facecolor='#333333'))
ax.annotate('OPEN = $105\n(price at start - HIGH)', xy=(1.5, 105), xytext=(2.5, 106),
           fontsize=11, color=COLORS['red'], arrowprops=dict(arrowstyle='->', color=COLORS['red']),
           bbox=dict(boxstyle='round', facecolor='#331a1a'))
ax.annotate('CLOSE = $95\n(price at end - LOW)', xy=(0.5, 95), xytext=(-1.5, 96),
           fontsize=11, color=COLORS['red'], arrowprops=dict(arrowstyle='->', color=COLORS['red']),
           bbox=dict(boxstyle='round', facecolor='#331a1a'))
ax.annotate('LOW = $92', xy=(1, 92), xytext=(2.5, 93),
           fontsize=11, color='white', arrowprops=dict(arrowstyle='->', color='white'),
           bbox=dict(boxstyle='round', facecolor='#333333'))
ax.set_axis_off()

# Key candlestick patterns
ax = axes[2]
ax.set_title("KEY SINGLE-CANDLE PATTERNS", fontsize=14, color=COLORS['yellow'])
ax.set_xlim(-1, 12)
ax.set_ylim(85, 120)

# Doji
ax.plot([1, 1], [90, 110], color='white', linewidth=2)
ax.add_patch(patches.Rectangle((0.6, 99.5), 0.8, 1, facecolor='white', edgecolor='white'))
ax.text(1, 113, 'DOJI', fontsize=10, color=COLORS['yellow'], ha='center', fontweight='bold')
ax.text(1, 86, 'Indecision\nOpen=Close', fontsize=8, color='#aaaaaa', ha='center')

# Hammer
ax.plot([4, 4], [90, 105], color=COLORS['green'], linewidth=2)
ax.add_patch(patches.Rectangle((3.6, 102), 0.8, 3, facecolor=COLORS['green'], edgecolor=COLORS['green']))
ax.text(4, 113, 'HAMMER', fontsize=10, color=COLORS['yellow'], ha='center', fontweight='bold')
ax.text(4, 86, 'Bullish\nReversal', fontsize=8, color='#aaaaaa', ha='center')

# Shooting Star
ax.plot([7, 7], [95, 110], color=COLORS['red'], linewidth=2)
ax.add_patch(patches.Rectangle((6.6, 95), 0.8, 3, facecolor=COLORS['red'], edgecolor=COLORS['red']))
ax.text(7, 113, 'SHOOTING\nSTAR', fontsize=10, color=COLORS['yellow'], ha='center', fontweight='bold')
ax.text(7, 86, 'Bearish\nReversal', fontsize=8, color='#aaaaaa', ha='center')

# Engulfing
ax.add_patch(patches.Rectangle((9.7, 98), 0.6, 4, facecolor=COLORS['red'], edgecolor=COLORS['red']))
ax.plot([10, 10], [96, 104], color=COLORS['red'], linewidth=1)
ax.add_patch(patches.Rectangle((10.4, 96), 0.8, 8, facecolor=COLORS['green'], edgecolor=COLORS['green']))
ax.plot([10.8, 10.8], [94, 106], color=COLORS['green'], linewidth=1)
ax.text(10.4, 113, 'BULLISH\nENGULFING', fontsize=10, color=COLORS['yellow'], ha='center', fontweight='bold')
ax.text(10.4, 86, 'Strong\nReversal', fontsize=8, color='#aaaaaa', ha='center')

ax.set_axis_off()

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson1_candlestick_basics.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# LESSON 2: SUPPORT, RESISTANCE & TRENDLINES
# ============================================================
print("Generating Lesson 2: Support & Resistance...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("LESSON 2: SUPPORT, RESISTANCE & TRENDLINES\nThe Most Important Lines on Any Chart",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# --- Support & Resistance ---
ax = axes[0, 0]
ax.set_title("SUPPORT & RESISTANCE", fontsize=14, color=COLORS['cyan'])

np.random.seed(10)
# Create price that bounces between support and resistance
prices = []
p = 50
for i in range(100):
    if p < 42:
        p += abs(np.random.normal(0.5, 1))
    elif p > 58:
        p -= abs(np.random.normal(0.5, 1))
    else:
        p += np.random.normal(0, 1.2)
    prices.append(p)

ax.plot(prices, color=COLORS['blue'], linewidth=1.5)
ax.axhline(y=58, color=COLORS['red'], linewidth=2, linestyle='--', label='RESISTANCE ($58)')
ax.axhline(y=42, color=COLORS['green'], linewidth=2, linestyle='--', label='SUPPORT ($42)')
ax.fill_between(range(100), 42, 58, alpha=0.05, color='white')

# Arrows showing bounces
for i, p_val in enumerate(prices):
    if i > 0 and prices[i-1] < 43 and prices[i] > 43:
        ax.annotate('BOUNCE!', xy=(i, prices[i]), xytext=(i, 38),
                   fontsize=8, color=COLORS['green'], ha='center',
                   arrowprops=dict(arrowstyle='->', color=COLORS['green']))
        break

for i, p_val in enumerate(prices):
    if i > 0 and prices[i-1] > 57 and prices[i] < 57:
        ax.annotate('REJECTED!', xy=(i, prices[i]), xytext=(i, 62),
                   fontsize=8, color=COLORS['red'], ha='center',
                   arrowprops=dict(arrowstyle='->', color=COLORS['red']))
        break

ax.legend(fontsize=10, loc='upper left')
ax.text(50, 50, 'TRADING\nRANGE', fontsize=16, color='#333333', ha='center', va='center', fontweight='bold')
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- Breakout ---
ax = axes[0, 1]
ax.set_title("BREAKOUT (What Micha Looks For!)", fontsize=14, color=COLORS['orange'])

np.random.seed(15)
prices = []
p = 50
for i in range(120):
    if i < 80:
        if p < 45: p += abs(np.random.normal(0.3, 0.8))
        elif p > 55: p -= abs(np.random.normal(0.3, 0.8))
        else: p += np.random.normal(0, 1)
    else:
        p += np.random.normal(0.8, 1.5)  # breakout!
    prices.append(p)

volumes = [int(np.random.lognormal(10, 0.3)) for _ in range(120)]
# Spike volume at breakout
for i in range(78, 90):
    volumes[i] = int(volumes[i] * 3)

ax.plot(prices, color=COLORS['blue'], linewidth=1.5)
ax.axhline(y=55, color=COLORS['red'], linewidth=2, linestyle='--')
ax.axhline(y=45, color=COLORS['green'], linewidth=2, linestyle='--')

# Breakout annotation
ax.annotate('BREAKOUT!\nPrice breaks above\nresistance with HIGH VOLUME',
           xy=(82, 56), xytext=(40, 70),
           fontsize=11, color=COLORS['orange'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#332200', edgecolor=COLORS['orange']))

ax.annotate('Old RESISTANCE\nbecomes new SUPPORT!',
           xy=(100, 55), xytext=(95, 48),
           fontsize=10, color=COLORS['yellow'],
           arrowprops=dict(arrowstyle='->', color=COLORS['yellow']),
           bbox=dict(boxstyle='round', facecolor='#333300'))

# Volume subplot
ax2 = ax.twinx()
ax2.bar(range(120), volumes, alpha=0.3, color=COLORS['orange'], width=1)
ax2.set_ylabel('Volume', color=COLORS['orange'])
ax2.tick_params(axis='y', labelcolor=COLORS['orange'])

ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- Uptrend ---
ax = axes[1, 0]
ax.set_title("UPTREND: Higher Highs + Higher Lows", fontsize=14, color=COLORS['green'])

np.random.seed(20)
prices = []
highs_pts = []
lows_pts = []
p = 40
for i in range(100):
    wave = 5 * np.sin(i * 0.15)
    p = 40 + i * 0.25 + wave + np.random.normal(0, 0.5)
    prices.append(p)

ax.plot(prices, color=COLORS['blue'], linewidth=1.5)

# Mark higher highs and higher lows
hh_x = [15, 42, 70, 95]
hh_y = [prices[i] for i in hh_x]
ll_x = [5, 30, 55, 82]
ll_y = [prices[i] for i in ll_x]

ax.plot(hh_x, hh_y, 'v', color=COLORS['green'], markersize=12, label='Higher Highs')
ax.plot(ll_x, ll_y, '^', color=COLORS['cyan'], markersize=12, label='Higher Lows')

# Trendline
ax.plot(ll_x, ll_y, color=COLORS['green'], linewidth=2, linestyle='--', alpha=0.7)

ax.annotate('HH', xy=(15, hh_y[0]), xytext=(15, hh_y[0]+3), fontsize=10, color=COLORS['green'], ha='center')
ax.annotate('HH', xy=(42, hh_y[1]), xytext=(42, hh_y[1]+3), fontsize=10, color=COLORS['green'], ha='center')
ax.annotate('HH', xy=(70, hh_y[2]), xytext=(70, hh_y[2]+3), fontsize=10, color=COLORS['green'], ha='center')
ax.annotate('HL', xy=(30, ll_y[1]), xytext=(30, ll_y[1]-4), fontsize=10, color=COLORS['cyan'], ha='center')
ax.annotate('HL', xy=(55, ll_y[2]), xytext=(55, ll_y[2]-4), fontsize=10, color=COLORS['cyan'], ha='center')

ax.text(50, 62, 'UPTREND\nBUY THE DIPS', fontsize=14, color=COLORS['green'],
        ha='center', fontweight='bold', alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- Downtrend ---
ax = axes[1, 1]
ax.set_title("DOWNTREND: Lower Highs + Lower Lows", fontsize=14, color=COLORS['red'])

np.random.seed(25)
prices = []
p = 80
for i in range(100):
    wave = 5 * np.sin(i * 0.15)
    p = 80 - i * 0.25 + wave + np.random.normal(0, 0.5)
    prices.append(p)

ax.plot(prices, color=COLORS['blue'], linewidth=1.5)

lh_x = [15, 42, 70, 95]
lh_y = [prices[i] for i in lh_x]
ll_x = [5, 30, 55, 82]
ll_y = [prices[i] for i in ll_x]

ax.plot(lh_x, lh_y, 'v', color=COLORS['red'], markersize=12, label='Lower Highs')
ax.plot(ll_x, ll_y, '^', color=COLORS['orange'], markersize=12, label='Lower Lows')

ax.plot(lh_x, lh_y, color=COLORS['red'], linewidth=2, linestyle='--', alpha=0.7)

ax.annotate('LH', xy=(15, lh_y[0]), xytext=(15, lh_y[0]+3), fontsize=10, color=COLORS['red'], ha='center')
ax.annotate('LH', xy=(42, lh_y[1]), xytext=(42, lh_y[1]+3), fontsize=10, color=COLORS['red'], ha='center')
ax.annotate('LH', xy=(70, lh_y[2]), xytext=(70, lh_y[2]+3), fontsize=10, color=COLORS['red'], ha='center')
ax.annotate('LL', xy=(30, ll_y[1]), xytext=(30, ll_y[1]-4), fontsize=10, color=COLORS['orange'], ha='center')
ax.annotate('LL', xy=(55, ll_y[2]), xytext=(55, ll_y[2]-4), fontsize=10, color=COLORS['orange'], ha='center')

ax.text(50, 62, 'DOWNTREND\nDON\'T CATCH\nFALLING KNIVES', fontsize=14, color=COLORS['red'],
        ha='center', fontweight='bold', alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson2_support_resistance.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# LESSON 3: CHART PATTERNS - The Big Money Patterns
# ============================================================
print("Generating Lesson 3: Chart Patterns...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle("LESSON 3: CHART PATTERNS\nThese Patterns Predict What Happens Next",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# --- Inverse Head & Shoulders (Micha's favorite!) ---
ax = axes[0, 0]
ax.set_title("INVERSE HEAD & SHOULDERS\n(Micha uses this a lot!)", fontsize=13, color=COLORS['green'])

# Create the pattern
x = np.arange(0, 100)
left_shoulder = -5 * np.exp(-((x-20)**2)/30)
head = -10 * np.exp(-((x-50)**2)/40)
right_shoulder = -5 * np.exp(-((x-80)**2)/30)
price = 60 + left_shoulder + head + right_shoulder + np.random.normal(0, 0.3, 100)
# Add breakout
price_full = np.concatenate([price, 60 + np.cumsum(np.random.normal(0.15, 0.3, 30))])

ax.plot(price_full, color=COLORS['blue'], linewidth=2)
ax.axhline(y=60, color=COLORS['yellow'], linewidth=2, linestyle='--', label='NECKLINE')

ax.annotate('Left\nShoulder', xy=(20, 55), fontsize=10, color=COLORS['cyan'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#002233'))
ax.annotate('HEAD\n(Lowest)', xy=(50, 50), fontsize=11, color=COLORS['orange'], ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#332200'))
ax.annotate('Right\nShoulder', xy=(80, 55), fontsize=10, color=COLORS['cyan'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#002233'))
ax.annotate('BREAKOUT!\nBUY HERE', xy=(95, 61), xytext=(105, 67),
           fontsize=12, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green']))

# Target
depth = 60 - 50  # neckline - head = 10
ax.annotate('', xy=(115, 60), xytext=(115, 70),
           arrowprops=dict(arrowstyle='<->', color=COLORS['yellow'], lw=2))
ax.text(118, 65, f'TARGET\n= depth\n= ${int(depth)}', fontsize=9, color=COLORS['yellow'])

ax.legend(fontsize=10)
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- Double Bottom (W pattern) ---
ax = axes[0, 1]
ax.set_title("DOUBLE BOTTOM (W Pattern)\nBullish Reversal", fontsize=13, color=COLORS['green'])

x = np.arange(0, 100)
bottom1 = -8 * np.exp(-((x-30)**2)/50)
bottom2 = -8 * np.exp(-((x-70)**2)/50)
price = 55 + bottom1 + bottom2 + np.random.normal(0, 0.3, 100)
price_full = np.concatenate([price, 55 + np.cumsum(np.random.normal(0.12, 0.3, 25))])

ax.plot(price_full, color=COLORS['blue'], linewidth=2)
ax.axhline(y=55, color=COLORS['yellow'], linewidth=2, linestyle='--', label='NECKLINE')
ax.axhline(y=47, color=COLORS['red'], linewidth=1, linestyle=':', label='SUPPORT')

ax.annotate('Bottom 1', xy=(30, 47.5), fontsize=11, color=COLORS['orange'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#332200'))
ax.annotate('Bottom 2', xy=(70, 47.5), fontsize=11, color=COLORS['orange'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#332200'))
ax.annotate('If both bottoms\nhold same level\n= STRONG support', xy=(50, 44), fontsize=9, color='#aaaaaa', ha='center')

ax.legend(fontsize=10)
ax.set_xlabel('Days')

# --- Double Top (M pattern) ---
ax = axes[0, 2]
ax.set_title("DOUBLE TOP (M Pattern)\nBearish Reversal", fontsize=13, color=COLORS['red'])

x = np.arange(0, 100)
top1 = 8 * np.exp(-((x-30)**2)/50)
top2 = 8 * np.exp(-((x-70)**2)/50)
price = 55 + top1 + top2 + np.random.normal(0, 0.3, 100)
price_full = np.concatenate([price, 55 - np.cumsum(np.abs(np.random.normal(0.12, 0.3, 25)))])

ax.plot(price_full, color=COLORS['blue'], linewidth=2)
ax.axhline(y=55, color=COLORS['yellow'], linewidth=2, linestyle='--', label='NECKLINE')
ax.axhline(y=63, color=COLORS['red'], linewidth=1, linestyle=':', label='RESISTANCE')

ax.annotate('Top 1', xy=(30, 63.5), fontsize=11, color=COLORS['red'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#331a1a'))
ax.annotate('Top 2', xy=(70, 63.5), fontsize=11, color=COLORS['red'], ha='center',
           bbox=dict(boxstyle='round', facecolor='#331a1a'))
ax.annotate('BREAKDOWN!\nSELL / SHORT', xy=(95, 53), xytext=(80, 45),
           fontsize=11, color=COLORS['red'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#331a1a', edgecolor=COLORS['red']))

ax.legend(fontsize=10)
ax.set_xlabel('Days')

# --- Rounding Bottom ---
ax = axes[1, 0]
ax.set_title("ROUNDING BOTTOM\n(Micha spotted this on fertilizer stocks)", fontsize=13, color=COLORS['green'])

x = np.arange(0, 120)
curve = 0.003 * (x - 60)**2
price = 40 + curve + np.random.normal(0, 0.4, 120)
price_full = np.concatenate([price, price[-1] + np.cumsum(np.random.normal(0.2, 0.3, 20))])

ax.plot(price_full, color=COLORS['blue'], linewidth=2)
# Draw the curve
ax.plot(x, 40 + 0.003 * (x-60)**2, color=COLORS['yellow'], linewidth=2, linestyle='--', alpha=0.5)

ax.annotate('Slow, gradual\ncurve = accumulation\n(smart money buying)', xy=(60, 40), xytext=(30, 35),
           fontsize=10, color=COLORS['cyan'],
           bbox=dict(boxstyle='round', facecolor='#002233'))
ax.annotate('BREAKOUT\nwhen curve\ncompletes', xy=(115, 52), xytext=(100, 57),
           fontsize=11, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#003311'))

ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- Bull Flag ---
ax = axes[1, 1]
ax.set_title("BULL FLAG\nContinuation Pattern (Short-term trade)", fontsize=13, color=COLORS['green'])

# Strong move up (pole)
np.random.seed(33)
pole = np.cumsum(np.random.normal(0.8, 0.5, 20)) + 50
# Flag (slight pullback)
flag = pole[-1] - np.cumsum(np.abs(np.random.normal(0.1, 0.2, 30))) + np.random.normal(0, 0.3, 30)
# Continuation
cont = flag[-1] + np.cumsum(np.random.normal(0.6, 0.5, 20))
price = np.concatenate([pole, flag, cont])

ax.plot(price, color=COLORS['blue'], linewidth=2)

# Draw flag channel
ax.plot([20, 50], [pole[-1], flag[-1]], color=COLORS['red'], linewidth=2, linestyle='--')
ax.plot([20, 50], [pole[-1]-2, flag[-1]-2], color=COLORS['red'], linewidth=2, linestyle='--')

ax.annotate('POLE\n(strong move up)', xy=(10, 58), fontsize=10, color=COLORS['green'],
           bbox=dict(boxstyle='round', facecolor='#003311'))
ax.annotate('FLAG\n(small pullback\non LOW volume)', xy=(35, 63), fontsize=10, color=COLORS['orange'],
           bbox=dict(boxstyle='round', facecolor='#332200'))
ax.annotate('CONTINUATION\n(resumes uptrend)', xy=(55, 70), fontsize=10, color=COLORS['green'],
           bbox=dict(boxstyle='round', facecolor='#003311'))

ax.set_xlabel('Days')

# --- Cup & Handle ---
ax = axes[1, 2]
ax.set_title("CUP & HANDLE\nClassic Bullish Pattern", fontsize=13, color=COLORS['green'])

x = np.arange(0, 100)
cup = np.where(x < 70, 0.005 * (x-35)**2, 0.005 * 35**2)
price = 50 + cup + np.random.normal(0, 0.3, 100)
# Handle (small dip)
handle = price[-1] - 2 * np.exp(-((np.arange(20)-10)**2)/20) + np.random.normal(0, 0.2, 20)
# Breakout
breakout = handle[-1] + np.cumsum(np.random.normal(0.3, 0.3, 15))
price_full = np.concatenate([price, handle, breakout])

ax.plot(price_full, color=COLORS['blue'], linewidth=2)
ax.axhline(y=56, color=COLORS['yellow'], linewidth=2, linestyle='--', label='RIM LINE')

ax.annotate('CUP', xy=(35, 48), fontsize=14, color=COLORS['cyan'], ha='center', fontweight='bold')
ax.annotate('HANDLE', xy=(108, 54), fontsize=10, color=COLORS['orange'],
           bbox=dict(boxstyle='round', facecolor='#332200'))
ax.annotate('BUY!', xy=(120, 57), xytext=(125, 62),
           fontsize=12, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#003311'))

ax.legend(fontsize=10)
ax.set_xlabel('Days')

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson3_chart_patterns.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# LESSON 4: MOVING AVERAGES - The Most Used Indicator
# ============================================================
print("Generating Lesson 4: Moving Averages...")

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle("LESSON 4: MOVING AVERAGES\nMA20, MA50, MA150, MA200 — What Micha & Every Trader Uses",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# --- MA Crossovers ---
ax = axes[0]
ax.set_title("GOLDEN CROSS & DEATH CROSS", fontsize=14, color=COLORS['cyan'])

np.random.seed(50)
n = 250
trend = np.concatenate([
    np.linspace(50, 40, 80),   # downtrend
    np.linspace(40, 70, 100),  # uptrend
    np.linspace(70, 65, 70),   # consolidation
])
noise = np.random.normal(0, 1.5, n)
prices = trend + noise

ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
ma50 = np.convolve(prices, np.ones(50)/50, mode='valid')
ma150 = np.convolve(prices, np.ones(150)/150, mode='valid')

ax.plot(prices, color='#555555', linewidth=0.8, label='Price', alpha=0.7)
offset_20 = n - len(ma20)
offset_50 = n - len(ma50)
offset_150 = n - len(ma150)
ax.plot(range(offset_20, n), ma20, color=COLORS['green'], linewidth=2, label='MA 20 (fast)')
ax.plot(range(offset_50, n), ma50, color=COLORS['orange'], linewidth=2, label='MA 50 (medium)')
ax.plot(range(offset_150, n), ma150, color=COLORS['red'], linewidth=2, label='MA 150 (slow)')

# Find golden cross area
ax.annotate('GOLDEN CROSS\nMA20 crosses ABOVE MA50\n= BUY SIGNAL',
           xy=(115, 43), xytext=(30, 55),
           fontsize=11, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green']))

ax.annotate('DEATH CROSS\nMA20 crosses BELOW MA50\n= SELL SIGNAL',
           xy=(55, 47), xytext=(5, 35),
           fontsize=11, color=COLORS['red'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#331a1a', edgecolor=COLORS['red']))

ax.annotate('MA150: Micha uses this\nas THE KEY level.\nAbove = bullish\nBelow = bearish',
           xy=(200, ma150[-50]), xytext=(170, 73),
           fontsize=10, color=COLORS['yellow'],
           arrowprops=dict(arrowstyle='->', color=COLORS['yellow']),
           bbox=dict(boxstyle='round', facecolor='#333300', edgecolor=COLORS['yellow']))

ax.legend(fontsize=11, loc='lower right')
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

# --- MA as Support ---
ax = axes[1]
ax.set_title("MOVING AVERAGE AS SUPPORT\n(Price 'bounces' off the MA)", fontsize=14, color=COLORS['cyan'])

np.random.seed(55)
n = 150
prices = []
p = 50
for i in range(n):
    p += np.random.normal(0.15, 1.2)
    if i > 20:
        ma_val = np.mean(prices[max(0,i-20):i])
        if p < ma_val - 0.5:
            p += 1.5  # bounce off MA
    prices.append(p)

prices = np.array(prices)
ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
offset = n - len(ma20)

ax.plot(prices, color=COLORS['blue'], linewidth=1.5, label='Price')
ax.plot(range(offset, n), ma20, color=COLORS['orange'], linewidth=2.5, label='MA 20')

# Find bounces
for i in range(offset + 5, n - 5):
    ma_i = ma20[i - offset]
    if abs(prices[i] - ma_i) < 1 and prices[i+1] > prices[i] and prices[i] < prices[i-1]:
        ax.annotate('', xy=(i, prices[i]), xytext=(i, prices[i]-3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))

ax.text(75, 55, 'In an uptrend, price\n"bounces" off the MA\nlike a trampoline.\n\nThis is where\nyou BUY.',
       fontsize=12, color=COLORS['green'],
       bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green'], alpha=0.9))

ax.legend(fontsize=11)
ax.set_xlabel('Days')
ax.set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson4_moving_averages.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# LESSON 5: VOLUME - The Truth Detector
# ============================================================
print("Generating Lesson 5: Volume Analysis...")

fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("LESSON 5: VOLUME — THE TRUTH DETECTOR\nVolume Confirms Whether a Move is Real or Fake",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

ax_price = axes[0]
ax_vol = axes[1]

np.random.seed(60)
n = 150
prices = []
volumes = []
p = 50
for i in range(n):
    if 40 < i < 50:
        p += np.random.normal(-0.5, 1)  # fake breakdown
        v = int(np.random.lognormal(10, 0.3))  # low volume
    elif 70 < i < 85:
        p += np.random.normal(1.2, 1)  # real breakout
        v = int(np.random.lognormal(10, 0.3) * 4)  # high volume!
    else:
        p += np.random.normal(0.05, 0.8)
        v = int(np.random.lognormal(10, 0.3))
    prices.append(p)
    volumes.append(v)

prices = np.array(prices)
volumes = np.array(volumes)

ax_price.plot(prices, color=COLORS['blue'], linewidth=1.5)
ax_price.axhline(y=50, color=COLORS['yellow'], linewidth=1, linestyle='--', alpha=0.5)

# Fake breakdown
ax_price.annotate('FAKE BREAKDOWN\nPrice drops but\nVOLUME IS LOW\n= nobody selling\n= trap!',
                 xy=(45, 46), xytext=(10, 38),
                 fontsize=11, color=COLORS['red'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
                 bbox=dict(boxstyle='round', facecolor='#331a1a', edgecolor=COLORS['red']))

# Real breakout
ax_price.annotate('REAL BREAKOUT\nPrice rises AND\nVOLUME IS HIGH\n= everyone buying\n= follow it!',
                 xy=(78, 62), xytext=(95, 70),
                 fontsize=11, color=COLORS['green'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
                 bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green']))

ax_price.set_ylabel('Price ($)')

# Volume bars
colors_vol = [COLORS['green'] if prices[i] >= prices[max(0,i-1)] else COLORS['red'] for i in range(n)]
ax_vol.bar(range(n), volumes, color=colors_vol, alpha=0.7, width=1)
avg_vol = np.mean(volumes)
ax_vol.axhline(y=avg_vol, color=COLORS['yellow'], linewidth=1, linestyle='--', label='Avg Volume')

ax_vol.annotate('LOW volume\n= fake move', xy=(45, volumes[45]), xytext=(20, max(volumes)*0.7),
               fontsize=10, color=COLORS['red'],
               arrowprops=dict(arrowstyle='->', color=COLORS['red']),
               bbox=dict(boxstyle='round', facecolor='#331a1a'))
ax_vol.annotate('HIGH volume\n= real move', xy=(78, volumes[78]), xytext=(100, max(volumes)*0.8),
               fontsize=10, color=COLORS['green'],
               arrowprops=dict(arrowstyle='->', color=COLORS['green']),
               bbox=dict(boxstyle='round', facecolor='#003311'))

ax_vol.set_xlabel('Days')
ax_vol.set_ylabel('Volume')
ax_vol.legend(fontsize=10)

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson5_volume.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# LESSON 6: PUTTING IT ALL TOGETHER - Full Trade Setup
# ============================================================
print("Generating Lesson 6: Full Trade Setup...")

fig, axes = plt.subplots(2, 1, figsize=(22, 14), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("LESSON 6: COMPLETE TRADE SETUP\n\"FAKE STOCK INC\" ($FAKE) — Can You Spot the Trade?",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

ax = axes[0]
ax_vol = axes[1]

np.random.seed(77)
n = 200

# Create a realistic-looking chart with a setup
prices = []
volumes = []
p = 100

for i in range(n):
    if i < 50:
        p += np.random.normal(-0.3, 1.5)  # downtrend
        v = int(np.random.lognormal(12, 0.4))
    elif i < 80:
        p += np.random.normal(0, 1.0)  # base building
        if p < 82: p += 1
        if p > 92: p -= 1
        v = int(np.random.lognormal(11.5, 0.3))  # lower volume
    elif i < 120:
        p += np.random.normal(0, 1.0)  # more base
        if p < 84: p += 1
        if p > 92: p -= 1
        v = int(np.random.lognormal(11.5, 0.3))
    elif 120 <= i < 130:
        p += np.random.normal(0.8, 1.0)  # breakout
        v = int(np.random.lognormal(12, 0.4) * 3)  # volume surge
    elif 130 <= i < 145:
        p += np.random.normal(-0.2, 0.8)  # pullback to test
        v = int(np.random.lognormal(11, 0.3))  # low volume pullback
    else:
        p += np.random.normal(0.4, 1.0)  # continuation
        v = int(np.random.lognormal(12, 0.3))

    prices.append(p)
    volumes.append(v)

prices = np.array(prices)
volumes = np.array(volumes)

# Calculate MAs
ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
ma50 = np.convolve(prices, np.ones(50)/50, mode='valid')

# Plot price
ax.plot(prices, color=COLORS['blue'], linewidth=1.5, label='$FAKE Price')
ax.plot(range(n-len(ma20), n), ma20, color=COLORS['green'], linewidth=1.5, label='MA 20', alpha=0.8)
ax.plot(range(n-len(ma50), n), ma50, color=COLORS['orange'], linewidth=1.5, label='MA 50', alpha=0.8)

# Support & Resistance
ax.axhline(y=92, color=COLORS['red'], linewidth=2, linestyle='--', alpha=0.7, label='Resistance $92')
ax.fill_between(range(50, 125), 82, 92, alpha=0.05, color='white')

# Annotations - telling the story
ax.annotate('1. DOWNTREND\nPrice falling, stay away', xy=(25, 95), fontsize=10, color=COLORS['red'],
           bbox=dict(boxstyle='round', facecolor='#331a1a'))

ax.annotate('2. BASE BUILDING\nPrice stops falling,\nmoves sideways.\nSmart money accumulating.',
           xy=(85, 78), fontsize=10, color=COLORS['cyan'],
           bbox=dict(boxstyle='round', facecolor='#002233'))

ax.annotate('3. BREAKOUT!\nPrice breaks $92\nwith HUGE volume\n+ MA20 crosses MA50',
           xy=(125, 96), xytext=(135, 107),
           fontsize=11, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
           bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green']))

ax.annotate('4. PULLBACK TEST\nPrice comes back to $92\n(old resistance = new support)\nLow volume = healthy',
           xy=(138, 92), xytext=(150, 82),
           fontsize=10, color=COLORS['yellow'],
           arrowprops=dict(arrowstyle='->', color=COLORS['yellow']),
           bbox=dict(boxstyle='round', facecolor='#333300'))

ax.annotate('5. BUY HERE!\nConfirmed breakout\n+ pullback held\n+ volume returning',
           xy=(145, 93), xytext=(160, 103),
           fontsize=12, color=COLORS['green'], fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=3),
           bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green'], linewidth=2))

ax.set_ylabel('Price ($)', fontsize=12)
ax.legend(fontsize=10, loc='upper right')

# Volume
colors_vol = [COLORS['green'] if prices[i] >= prices[max(0,i-1)] else COLORS['red'] for i in range(n)]
ax_vol.bar(range(n), volumes, color=colors_vol, alpha=0.7, width=1)
avg_vol = np.mean(volumes)
ax_vol.axhline(y=avg_vol, color=COLORS['yellow'], linewidth=1, linestyle='--')
ax_vol.set_ylabel('Volume', fontsize=12)
ax_vol.set_xlabel('Days', fontsize=12)

ax_vol.annotate('Volume SURGE\non breakout = REAL', xy=(125, volumes[125]),
               fontsize=10, color=COLORS['green'],
               bbox=dict(boxstyle='round', facecolor='#003311'))

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/lesson6_full_trade_setup.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# CHEAT SHEET
# ============================================================
print("Generating Cheat Sheet...")

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

fig.suptitle("CHART READING CHEAT SHEET\nPrint This & Keep It Next to Your Screen",
             fontsize=24, fontweight='bold', color=COLORS['yellow'])

y = 92
sections = [
    ("BULLISH SIGNALS (BUY)", COLORS['green'], [
        "Price breaks ABOVE resistance with HIGH volume",
        "Golden Cross: MA20 crosses above MA50",
        "Price bouncing off MA20/MA50 in uptrend (buy the dip)",
        "Inverse Head & Shoulders completing",
        "Double Bottom (W) pattern at support",
        "Rounding Bottom completing",
        "Insiders buying their own stock",
        "Higher Highs + Higher Lows = uptrend intact",
    ]),
    ("BEARISH SIGNALS (SELL / AVOID)", COLORS['red'], [
        "Price breaks BELOW support with high volume",
        "Death Cross: MA20 crosses below MA50",
        "Price rejected at resistance multiple times",
        "Double Top (M) pattern at resistance",
        "Head & Shoulders completing = reversal down",
        "Lower Highs + Lower Lows = downtrend",
        "Breakout on LOW volume = probably fake",
        "Price below MA150 = bearish territory",
    ]),
    ("VOLUME RULES", COLORS['orange'], [
        "Breakout + HIGH volume = REAL move, follow it",
        "Breakout + LOW volume = FAKE move, ignore it",
        "Pullback on LOW volume = healthy, look to buy",
        "Pullback on HIGH volume = more selling coming",
    ]),
    ("RISK MANAGEMENT", COLORS['cyan'], [
        "Always set a STOP LOSS below support",
        "Never risk more than 1-2% per trade",
        "Take partial profits at target, let rest run",
        "If wrong, get out FAST. Don't hope.",
    ]),
]

for title, color, items in sections:
    ax.text(2, y, title, fontsize=16, color=color, fontweight='bold')
    y -= 3
    for item in items:
        ax.text(5, y, f"  {item}", fontsize=11, color='white')
        y -= 2.8
    y -= 2

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/cheat_sheet.png", dpi=150, bbox_inches='tight')
plt.close()


print("\n" + "="*60)
print("  ALL LESSONS GENERATED!")
print("="*60)
print("\nFiles saved in: notebooks/05_chart_reading/output/")
print("\n  1. lesson1_candlestick_basics.png")
print("  2. lesson2_support_resistance.png")
print("  3. lesson3_chart_patterns.png")
print("  4. lesson4_moving_averages.png")
print("  5. lesson5_volume.png")
print("  6. lesson6_full_trade_setup.png")
print("  7. cheat_sheet.png")
print("\nThis is what people charge 6K NIS for. You're welcome.")
