"""
QC Backtest Result Organizer

Usage:
    python organize_results.py <strategy_name> <files...>

Example:
    python organize_results.py v2charttest "Dancing Blue Ant.json" "Dancing Blue Ant.png" "Dancing Blue Ant_orders.csv" "Dancing Blue Ant_trades.csv"

What it does:
    1. Reads the JSON to find the backtest start/end dates
    2. Creates folder: results_from_quant_connect/<strategy_name>/<start_year>-<end_year>/
    3. Moves all files there

Or just drag files into the results_from_quant_connect/ folder and run:
    python organize_results.py --scan

This scans for any unorganized files (JSON/PNG/CSV in strategy folders without a year subfolder)
and auto-sorts them by reading dates from the JSON.
"""

import json
import os
import sys
import shutil
import glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_from_quant_connect")


def get_period_from_json(json_path):
    """Read a QC backtest JSON and extract the start-end year period."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Try totalPerformance first
        tp = data.get('totalPerformance', {})
        ps = tp.get('portfolioStatistics', {})

        # Try to find dates from rolling window keys (M1_YYYYMMDD format)
        rw = data.get('rollingWindow', {})
        if rw:
            keys = sorted(rw.keys())
            # First and last keys like "M1_20160131"
            first_key = keys[0] if keys else None
            last_key = keys[-1] if keys else None

            if first_key and last_key:
                # Extract year from M1_YYYYMMDD or similar
                first_year = None
                last_year = None
                for k in keys:
                    parts = k.split('_')
                    if len(parts) >= 2 and len(parts[-1]) == 8:
                        year = int(parts[-1][:4])
                        if first_year is None:
                            first_year = year
                        last_year = year

                if first_year and last_year:
                    return f"{first_year}-{last_year}"

        # Fallback: try charts data for date range
        charts = data.get('charts', {})
        if 'Strategy Equity' in charts:
            series = charts['Strategy Equity'].get('series', {})
            if 'Equity' in series:
                values = series['Equity'].get('values', [])
                if values:
                    first_ts = values[0].get('x', 0)
                    last_ts = values[-1].get('x', 0)
                    if first_ts and last_ts:
                        from datetime import datetime
                        first_year = datetime.fromtimestamp(first_ts).year
                        last_year = datetime.fromtimestamp(last_ts).year
                        return f"{first_year}-{last_year}"

        return None
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        return None


def organize_files(strategy_name, file_paths):
    """Move files into the correct strategy/period/ folder."""
    # Find the JSON file to get period
    json_file = None
    for f in file_paths:
        if f.endswith('.json'):
            json_file = f
            break

    if not json_file:
        print(f"ERROR: No .json file found in {file_paths}")
        return False

    period = get_period_from_json(json_file)
    if not period:
        print(f"ERROR: Could not determine period from {json_file}")
        return False

    # Create target directory
    target_dir = os.path.join(RESULTS_DIR, strategy_name, period)
    os.makedirs(target_dir, exist_ok=True)

    # Move files
    for f in file_paths:
        if os.path.exists(f):
            dest = os.path.join(target_dir, os.path.basename(f))
            shutil.move(f, dest)
            print(f"  Moved: {os.path.basename(f)} -> {strategy_name}/{period}/")
        else:
            print(f"  SKIP: {f} not found")

    print(f"  Done! {len(file_paths)} files -> {target_dir}")
    return True


def scan_and_organize():
    """Scan results_from_quant_connect/ for unorganized files and sort them."""
    print(f"Scanning {RESULTS_DIR} for unorganized files...\n")

    for strategy_dir in sorted(os.listdir(RESULTS_DIR)):
        strategy_path = os.path.join(RESULTS_DIR, strategy_dir)
        if not os.path.isdir(strategy_path):
            continue

        # Look for JSON files directly in strategy folder (not in year subfolders)
        jsons = glob.glob(os.path.join(strategy_path, "*.json"))

        # Also look in experiment_* subfolders that have year subfolders already
        # (for migrating old structure: strategy/experiment_timestamp/year/ -> strategy/year/)
        for exp_dir in glob.glob(os.path.join(strategy_path, "experiment_*")):
            if not os.path.isdir(exp_dir):
                continue
            for year_dir in os.listdir(exp_dir):
                year_path = os.path.join(exp_dir, year_dir)
                if os.path.isdir(year_path):
                    # Already organized in year folder inside experiment — skip
                    pass

        if not jsons:
            continue

        print(f"Strategy: {strategy_dir}")
        # Group files by base name (before .json, _orders.csv, etc.)
        groups = {}
        all_files = (
            glob.glob(os.path.join(strategy_path, "*.json")) +
            glob.glob(os.path.join(strategy_path, "*.png")) +
            glob.glob(os.path.join(strategy_path, "*.csv"))
        )

        for f in all_files:
            basename = os.path.basename(f)
            # Strip suffixes to find the group name
            name = basename
            for suffix in ['_orders.csv', '_trades.csv', '.json', '.png', '.csv']:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name not in groups:
                groups[name] = []
            groups[name].append(f)

        for group_name, files in groups.items():
            json_file = [f for f in files if f.endswith('.json')]
            if not json_file:
                continue

            period = get_period_from_json(json_file[0])
            if not period:
                print(f"  SKIP {group_name}: could not determine period")
                continue

            target_dir = os.path.join(strategy_path, period)
            os.makedirs(target_dir, exist_ok=True)

            for f in files:
                dest = os.path.join(target_dir, os.path.basename(f))
                if os.path.abspath(f) != os.path.abspath(dest):
                    shutil.move(f, dest)

            print(f"  {group_name} -> {period}/")

    print("\nDone!")


def print_usage():
    print("QC Backtest Result Organizer")
    print("=" * 40)
    print()
    print("Usage:")
    print("  python organize_results.py <strategy_name> <file1> <file2> ...")
    print("  python organize_results.py --scan")
    print()
    print("Examples:")
    print('  python organize_results.py v2charttest results.json results.png results_orders.csv results_trades.csv')
    print('  python organize_results.py --scan    # auto-organize any loose files')
    print()
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    if sys.argv[1] == "--scan":
        scan_and_organize()
    else:
        strategy_name = sys.argv[1]
        files = sys.argv[2:]
        if not files:
            print(f"ERROR: No files specified for strategy '{strategy_name}'")
            print_usage()
            sys.exit(1)
        organize_files(strategy_name, files)
