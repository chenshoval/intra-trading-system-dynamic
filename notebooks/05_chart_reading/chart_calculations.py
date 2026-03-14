"""
============================================================
  HOW SUPPORT, RESISTANCE, SWING HIGHS/LOWS ARE CALCULATED

  The actual math — no black magic, no 6K NIS needed.
============================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs("notebooks/05_chart_reading/output", exist_ok=True)
plt.style.use('dark_background')

COLORS = {
    'green': '#00ff88', 'red': '#ff4444', 'blue': '#4488ff',
    'yellow': '#ffcc00', 'white': '#ffffff', 'gray': '#666666',
    'orange': '#ff8800', 'purple': '#aa66ff', 'cyan': '#00cccc',
}


# ============================================================
# Generate realistic OHLCV data
# ============================================================
def generate_ohlcv(n=200, seed=42):
    np.random.seed(seed)
    closes = [100]
    for i in range(1, n):
        # Create a pattern: down, base, up, pullback, up
        if i < 40:
            drift = -0.3
        elif i < 80:
            drift = 0.05
        elif i < 130:
            drift = 0.25
        elif i < 155:
            drift = -0.15
        else:
            drift = 0.2
        closes.append(closes[-1] + np.random.normal(drift, 1.2))

    opens, highs, lows, volumes = [], [], [], []
    for c in closes:
        spread = abs(np.random.normal(0, 0.8))
        o = c + np.random.normal(0, spread)
        h = max(c, o) + abs(np.random.normal(0, 0.8))
        l = min(c, o) - abs(np.random.normal(0, 0.8))
        v = int(np.random.lognormal(12, 0.4))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        volumes.append(v)

    return (np.array(opens), np.array(highs), np.array(lows),
            np.array(closes), np.array(volumes))


# ============================================================
# SWING POINT DETECTION - The core algorithm
# ============================================================
def find_swing_highs(highs, n=3):
    """
    A swing high is a point where the HIGH is higher than
    the N bars before it AND the N bars after it.

    Think of it as a local peak / mountain top.

    Parameters:
        highs: array of high prices
        n: how many bars on each side to check (2-5 typical)

    Returns:
        list of (index, price) tuples
    """
    swings = []
    for i in range(n, len(highs) - n):
        is_highest = True
        for j in range(1, n + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_highest = False
                break
        if is_highest:
            swings.append((i, highs[i]))
    return swings


def find_swing_lows(lows, n=3):
    """
    A swing low is a point where the LOW is lower than
    the N bars before it AND the N bars after it.

    Think of it as a local valley / bottom.
    """
    swings = []
    for i in range(n, len(lows) - n):
        is_lowest = True
        for j in range(1, n + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_lowest = False
                break
        if is_lowest:
            swings.append((i, lows[i]))
    return swings


# ============================================================
# SUPPORT & RESISTANCE - Multiple methods
# ============================================================
def find_support_resistance_by_clustering(closes, n_levels=4, tolerance=1.5):
    """
    METHOD 1: Price Clustering

    Idea: Find price levels where price spent a lot of time.
    These are natural support/resistance because many people
    bought/sold there (anchoring bias).

    Algorithm:
        1. Look at all closing prices
        2. Group nearby prices into clusters
        3. The cluster centers = support/resistance levels

    Parameters:
        closes: array of closing prices
        n_levels: how many S/R levels to find
        tolerance: how close prices need to be to cluster together
    """
    from collections import Counter

    # Round prices to nearest 'tolerance' to create bins
    rounded = np.round(closes / tolerance) * tolerance

    # Count how often price visited each level
    price_counts = Counter(rounded)

    # Top N most-visited levels = support/resistance
    levels = [level for level, count in price_counts.most_common(n_levels)]
    levels.sort()

    return levels


def find_support_resistance_by_swings(swing_highs, swing_lows, tolerance=2.0):
    """
    METHOD 2: Swing Point Clustering

    Idea: If multiple swing highs OR swing lows occur at
    similar price levels, that's a strong S/R level.

    A level is stronger when price has bounced off it
    more times (more "touches").
    """
    all_levels = []

    # Collect all swing prices
    for idx, price in swing_highs:
        all_levels.append(('high', price))
    for idx, price in swing_lows:
        all_levels.append(('low', price))

    # Cluster nearby levels
    if not all_levels:
        return []

    all_levels.sort(key=lambda x: x[1])
    clusters = []
    current_cluster = [all_levels[0]]

    for i in range(1, len(all_levels)):
        if all_levels[i][1] - current_cluster[-1][1] <= tolerance:
            current_cluster.append(all_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [all_levels[i]]
    clusters.append(current_cluster)

    # Return clusters with 2+ touches (stronger levels)
    sr_levels = []
    for cluster in clusters:
        if len(cluster) >= 2:  # at least 2 touches = real level
            avg_price = np.mean([p for _, p in cluster])
            touches = len(cluster)
            sr_levels.append((avg_price, touches))

    return sr_levels


def classify_trend(swing_highs, swing_lows):
    """
    TREND DETECTION using swing points.

    Uptrend:   Higher Highs AND Higher Lows
    Downtrend: Lower Highs AND Lower Lows
    Sideways:  Mixed signals
    """
    results = []

    # Check swing highs sequence
    for i in range(1, len(swing_highs)):
        prev_idx, prev_price = swing_highs[i-1]
        curr_idx, curr_price = swing_highs[i]
        if curr_price > prev_price:
            results.append(('HH', curr_idx, curr_price, prev_price))  # Higher High
        else:
            results.append(('LH', curr_idx, curr_price, prev_price))  # Lower High

    # Check swing lows sequence
    for i in range(1, len(swing_lows)):
        prev_idx, prev_price = swing_lows[i-1]
        curr_idx, curr_price = swing_lows[i]
        if curr_price > prev_price:
            results.append(('HL', curr_idx, curr_price, prev_price))  # Higher Low
        else:
            results.append(('LL', curr_idx, curr_price, prev_price))  # Lower Low

    return sorted(results, key=lambda x: x[1])


# ============================================================
# VISUALIZE EVERYTHING
# ============================================================
print("Generating charts...")

opens, highs, lows, closes, volumes = generate_ohlcv(200, seed=42)

# Find swing points
sh = find_swing_highs(highs, n=3)
sl = find_swing_lows(lows, n=3)

# Find S/R levels
sr_swings = find_support_resistance_by_swings(sh, sl, tolerance=2.0)
sr_clusters = find_support_resistance_by_clustering(closes, n_levels=5)

# Classify trend
trend_signals = classify_trend(sh, sl)


# ============================================================
# CHART 1: Swing Point Detection Explained
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(22, 14), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("HOW SWING HIGHS & LOWS ARE DETECTED\nThe Algorithm Behind 'Higher Highs / Lower Lows'",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

ax = axes[0]
ax.plot(closes, color=COLORS['blue'], linewidth=1.5, label='Close Price', alpha=0.8)
ax.fill_between(range(len(closes)), lows, highs, alpha=0.1, color=COLORS['blue'])

# Plot swing highs
sh_x = [s[0] for s in sh]
sh_y = [s[1] for s in sh]
ax.scatter(sh_x, sh_y, color=COLORS['red'], s=100, zorder=5, marker='v', label='Swing Highs (peaks)')

# Plot swing lows
sl_x = [s[0] for s in sl]
sl_y = [s[1] for s in sl]
ax.scatter(sl_x, sl_y, color=COLORS['green'], s=100, zorder=5, marker='^', label='Swing Lows (valleys)')

# Connect swing highs
ax.plot(sh_x, sh_y, color=COLORS['red'], linewidth=1.5, linestyle='--', alpha=0.5)
# Connect swing lows
ax.plot(sl_x, sl_y, color=COLORS['green'], linewidth=1.5, linestyle='--', alpha=0.5)

# Annotate HH/HL/LH/LL
for signal_type, idx, price, prev_price in trend_signals:
    if signal_type == 'HH':
        color = COLORS['green']
        label = f'HH\n({price:.0f}>{prev_price:.0f})'
        offset = 3
    elif 'HL' == signal_type:
        color = COLORS['cyan']
        label = f'HL\n({price:.0f}>{prev_price:.0f})'
        offset = -5
    elif signal_type == 'LH':
        color = COLORS['orange']
        label = f'LH\n({price:.0f}<{prev_price:.0f})'
        offset = 3
    else:  # LL
        color = COLORS['red']
        label = f'LL\n({price:.0f}<{prev_price:.0f})'
        offset = -5

    ax.annotate(label, xy=(idx, price), xytext=(idx, price + offset),
               fontsize=8, color=color, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor=color, alpha=0.8))

# Explain the algorithm
ax.text(0.02, 0.98,
        "ALGORITHM (N=3):\n"
        "Swing High at bar i if:\n"
        "  high[i] > high[i-1]\n"
        "  high[i] > high[i-2]\n"
        "  high[i] > high[i-3]\n"
        "  high[i] > high[i+1]\n"
        "  high[i] > high[i+2]\n"
        "  high[i] > high[i+3]\n"
        "\n"
        "Then compare consecutive\n"
        "swing highs/lows to get\n"
        "HH, HL, LH, LL signals.",
        transform=ax.transAxes, fontsize=10, color=COLORS['yellow'],
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#222200', edgecolor=COLORS['yellow'], alpha=0.9))

ax.legend(fontsize=11, loc='upper right')
ax.set_ylabel('Price ($)')
ax.grid(True, alpha=0.2)

# Volume subplot
ax2 = axes[1]
colors_v = [COLORS['green'] if closes[i] >= closes[max(0,i-1)] else COLORS['red'] for i in range(len(closes))]
ax2.bar(range(len(volumes)), volumes, color=colors_v, alpha=0.6, width=1)
ax2.set_ylabel('Volume')
ax2.set_xlabel('Days')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/calc_swing_points.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# CHART 2: Support & Resistance Detection
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle("HOW SUPPORT & RESISTANCE LEVELS ARE CALCULATED\nTwo Methods — Both Work",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# Method 1: Swing-based S/R
ax = axes[0]
ax.set_title("METHOD 1: Swing Point Clustering\n'Where did price bounce multiple times?'",
             fontsize=13, color=COLORS['cyan'])

ax.plot(closes, color=COLORS['blue'], linewidth=1.5, alpha=0.8)
ax.fill_between(range(len(closes)), lows, highs, alpha=0.08, color=COLORS['blue'])

# Draw S/R levels from swing clustering
for level_price, touches in sr_swings:
    width = min(touches, 5)  # thicker = more touches
    alpha = min(0.3 + touches * 0.1, 0.9)
    ax.axhline(y=level_price, color=COLORS['yellow'], linewidth=width,
              linestyle='--', alpha=alpha)
    ax.text(len(closes) + 2, level_price,
           f'${level_price:.1f}\n({touches} touches)',
           fontsize=9, color=COLORS['yellow'], va='center',
           bbox=dict(boxstyle='round', facecolor='#333300', alpha=0.8))

# Show swing points
ax.scatter([s[0] for s in sh], [s[1] for s in sh],
          color=COLORS['red'], s=60, zorder=5, marker='v', alpha=0.7)
ax.scatter([s[0] for s in sl], [s[1] for s in sl],
          color=COLORS['green'], s=60, zorder=5, marker='^', alpha=0.7)

ax.text(0.02, 0.02,
        "ALGORITHM:\n"
        "1. Find all swing highs & lows\n"
        "2. Group nearby ones (within $2)\n"
        "3. Average price of each group\n"
        "4. More touches = STRONGER level\n"
        "   (thicker line = more touches)",
        transform=ax.transAxes, fontsize=10, color=COLORS['cyan'],
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#002233', edgecolor=COLORS['cyan'], alpha=0.9))

ax.set_ylabel('Price ($)')
ax.set_xlabel('Days')
ax.grid(True, alpha=0.2)

# Method 2: Price clustering / histogram
ax = axes[1]
ax.set_title("METHOD 2: Price Histogram (Volume Profile)\n'Where did price spend the most TIME?'",
             fontsize=13, color=COLORS['orange'])

# Horizontal histogram of prices
ax_hist = ax.twiny()
counts, bins, _ = ax_hist.hist(closes, bins=30, orientation='horizontal',
                                alpha=0.3, color=COLORS['orange'], edgecolor=COLORS['orange'])
ax_hist.set_xlabel('Time Spent at Price Level', color=COLORS['orange'])

# Main price plot
ax.plot(range(len(closes)), closes, color=COLORS['blue'], linewidth=1.5, alpha=0.8)

# Mark the high-density zones
top_bins = np.argsort(counts)[-3:]  # top 3 most visited price zones
for b in top_bins:
    mid = (bins[b] + bins[b+1]) / 2
    ax.axhline(y=mid, color=COLORS['orange'], linewidth=2, linestyle='--', alpha=0.7)
    ax.text(len(closes) + 2, mid,
           f'${mid:.1f}\n(high density)',
           fontsize=9, color=COLORS['orange'], va='center',
           bbox=dict(boxstyle='round', facecolor='#332200', alpha=0.8))

ax.text(0.02, 0.02,
        "ALGORITHM:\n"
        "1. Take all closing prices\n"
        "2. Create histogram (bins)\n"
        "3. Tallest bins = price spent\n"
        "   most time there\n"
        "4. These are natural S/R\n"
        "   because many buyers/sellers\n"
        "   have positions at these levels",
        transform=ax.transAxes, fontsize=10, color=COLORS['orange'],
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#332200', edgecolor=COLORS['orange'], alpha=0.9))

ax.set_ylabel('Price ($)')
ax.set_xlabel('Days')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/calc_support_resistance.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# CHART 3: Moving Averages as Dynamic S/R
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(22, 10))
fig.suptitle("MOVING AVERAGES — THE CALCULATION\nSimple Moving Average (SMA) = Average of last N closing prices",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

ax.plot(closes, color=COLORS['blue'], linewidth=1, alpha=0.6, label='Close Price')

# Calculate and plot MAs
for period, color, name in [(20, COLORS['green'], 'MA20'),
                              (50, COLORS['orange'], 'MA50'),
                              (150, COLORS['red'], 'MA150')]:
    if len(closes) >= period:
        ma = np.convolve(closes, np.ones(period)/period, mode='valid')
        offset = len(closes) - len(ma)
        ax.plot(range(offset, len(closes)), ma, color=color, linewidth=2.5,
               label=f'{name} = avg of last {period} closes')

# Explain the formula
ax.text(0.02, 0.98,
        "THE FORMULA:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "SMA(N) = (C[t] + C[t-1] + ... + C[t-N+1]) / N\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "Example MA20 today:\n"
        "  = (today + yesterday + ... + 20 days ago) / 20\n"
        "\n"
        "Tomorrow: drop oldest day, add new close\n"
        "  That's why it 'moves' - it's a sliding window!\n"
        "\n"
        "GOLDEN CROSS: MA20 crosses ABOVE MA50\n"
        "  = short-term trend now faster than medium\n"
        "  = momentum shifting bullish\n"
        "\n"
        "DEATH CROSS: MA20 crosses BELOW MA50\n"
        "  = short-term slowing down\n"
        "  = momentum shifting bearish\n"
        "\n"
        "WHY MA150/200 MATTERS:\n"
        "  = represents ~6-10 months of data\n"
        "  = institutional investors watch these\n"
        "  = self-fulfilling prophecy",
        transform=ax.transAxes, fontsize=10, color=COLORS['yellow'],
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#222200', edgecolor=COLORS['yellow'], alpha=0.9))

# Find and mark golden cross
ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
ma50 = np.convolve(closes, np.ones(50)/50, mode='valid')
offset_20 = len(closes) - len(ma20)
offset_50 = len(closes) - len(ma50)

# Align them
start = max(offset_20, offset_50)
ma20_aligned = ma20[start - offset_20:]
ma50_aligned = ma50[start - offset_50:]

for i in range(1, min(len(ma20_aligned), len(ma50_aligned))):
    if ma20_aligned[i-1] <= ma50_aligned[i-1] and ma20_aligned[i] > ma50_aligned[i]:
        cross_x = start + i
        cross_y = ma20_aligned[i]
        ax.annotate('GOLDEN CROSS!\nMA20 > MA50\n= BUY signal',
                   xy=(cross_x, cross_y), xytext=(cross_x + 15, cross_y + 8),
                   fontsize=11, color=COLORS['green'], fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
                   bbox=dict(boxstyle='round', facecolor='#003311', edgecolor=COLORS['green']))
        break

for i in range(1, min(len(ma20_aligned), len(ma50_aligned))):
    if ma20_aligned[i-1] >= ma50_aligned[i-1] and ma20_aligned[i] < ma50_aligned[i]:
        cross_x = start + i
        cross_y = ma20_aligned[i]
        ax.annotate('DEATH CROSS!\nMA20 < MA50\n= SELL signal',
                   xy=(cross_x, cross_y), xytext=(cross_x - 25, cross_y - 8),
                   fontsize=11, color=COLORS['red'], fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
                   bbox=dict(boxstyle='round', facecolor='#331a1a', edgecolor=COLORS['red']))
        break

ax.legend(fontsize=12, loc='lower right')
ax.set_ylabel('Price ($)')
ax.set_xlabel('Days')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/calc_moving_averages.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# CHART 4: All-in-one Summary
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(24, 12))
fig.suptitle("EVERYTHING TOGETHER: Swing Points + S/R + MAs + Trend\nThis is how pros read a chart",
             fontsize=20, fontweight='bold', color=COLORS['yellow'])

# Price with range
ax.fill_between(range(len(closes)), lows, highs, alpha=0.08, color=COLORS['blue'])
ax.plot(closes, color=COLORS['blue'], linewidth=1.5, label='Close Price')

# MAs
for period, color, name in [(20, COLORS['green'], 'MA20'),
                              (50, COLORS['orange'], 'MA50')]:
    ma = np.convolve(closes, np.ones(period)/period, mode='valid')
    offset = len(closes) - len(ma)
    ax.plot(range(offset, len(closes)), ma, color=color, linewidth=2,
           label=name, alpha=0.7)

# Swing points
ax.scatter([s[0] for s in sh], [s[1] for s in sh],
          color=COLORS['red'], s=80, zorder=5, marker='v', label='Swing Highs')
ax.scatter([s[0] for s in sl], [s[1] for s in sl],
          color=COLORS['green'], s=80, zorder=5, marker='^', label='Swing Lows')

# S/R levels (top 3 strongest)
sr_sorted = sorted(sr_swings, key=lambda x: x[1], reverse=True)[:3]
for level_price, touches in sr_sorted:
    ax.axhline(y=level_price, color=COLORS['yellow'], linewidth=touches * 0.8,
              linestyle='--', alpha=0.5)

# Trend labels
for signal_type, idx, price, prev_price in trend_signals:
    colors_map = {'HH': COLORS['green'], 'HL': COLORS['cyan'],
                  'LH': COLORS['orange'], 'LL': COLORS['red']}
    offset = 3 if signal_type in ('HH', 'LH') else -4
    ax.annotate(signal_type, xy=(idx, price), xytext=(idx, price + offset),
               fontsize=9, color=colors_map[signal_type], ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                        edgecolor=colors_map[signal_type], alpha=0.8))

# Legend box
ax.text(0.72, 0.02,
        "WHAT TO LOOK FOR:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. What's the TREND?\n"
        "   HH+HL = up  |  LH+LL = down\n"
        "\n"
        "2. Where is SUPPORT/RESISTANCE?\n"
        "   Yellow dashed = key levels\n"
        "\n"
        "3. Where is price vs MAs?\n"
        "   Above MA50 = bullish bias\n"
        "   Below MA50 = bearish bias\n"
        "\n"
        "4. Is MA20 above MA50?\n"
        "   Yes = momentum is bullish\n"
        "\n"
        "5. Is there a PATTERN forming?\n"
        "   (H&S, double bottom, etc.)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Check ALL of these before trading!",
        transform=ax.transAxes, fontsize=10, color=COLORS['white'],
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#222222', edgecolor=COLORS['white'], alpha=0.9))

ax.legend(fontsize=11, loc='upper right')
ax.set_ylabel('Price ($)')
ax.set_xlabel('Days')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("notebooks/05_chart_reading/output/calc_everything_together.png", dpi=150, bbox_inches='tight')
plt.close()


print("\n" + "="*60)
print("  CALCULATION CHARTS GENERATED!")
print("="*60)
print("\nFiles saved in: notebooks/05_chart_reading/output/")
print("\n  calc_swing_points.png        - How HH/HL/LH/LL are found")
print("  calc_support_resistance.png  - How S/R levels are calculated")
print("  calc_moving_averages.png     - MA formula & crossovers")
print("  calc_everything_together.png - All signals on one chart")
