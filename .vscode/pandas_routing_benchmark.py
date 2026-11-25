# ruff: noqa: D103, PLC0415, PLR0402, T201
"""Benchmark how pandas DataFrames reach LightGBM's C++ side: NumPy (dense copy) vs. Arrow C stream.

Why
---
``Dataset`` and ``predict`` route a pandas DataFrame through ``_pandas_df_is_zero_copy_or_unsupported_by_arrow()``:
NumPy for single-float-dtype or sparse frames, the Arrow C stream otherwise (if pyarrow>=14 is installed).
This script measures peak memory and wall time of ``Dataset.construct()`` and ``Booster.predict()`` for each
path on identical data, so the routing rule can be checked with numbers instead of assumptions.

How
---
    python .vscode/pandas_routing_benchmark.py                       # default grid, 5 repeats
    python .vscode/pandas_routing_benchmark.py --rows 5000000 --repeats 9 --csv out.csv
    python .vscode/pandas_routing_benchmark.py --report a.csv b.csv=pandas-2.0   # markdown tables from CSVs

Every measurement runs in its own fresh interpreter (``--child`` mode), so allocator caches, pandas' internal
caches and earlier peaks never leak between runs. Runs of all paths are interleaved in random order, so
thermal throttling or background load hits every path equally. Report: median, plus min-max spread.

Fairness rules this script follows
-----------------------------------
* Same data: generated in the child from a fixed seed; identical bytes for every path.
* Same code: the path is forced by patching ``_pandas_df_is_zero_copy_or_unsupported_by_arrow`` only; everything else is shared.
* Same work: ``force_col_wise=True``, ``deterministic=True`` and fixed ``num_threads`` remove the row/col-wise auto-timing and thread-order effects on float sums,
  which otherwise adds a variable extra pass.
* Warm-up: a tiny Dataset is constructed first, so library loading / OpenMP pool start is not measured.
* Baseline: RSS is taken after data generation and ``gc.collect()``; the reported number is peak - baseline.
* Correctness: each child trains 5 rounds after measuring and prints a model hash; the parent fails
  if paths disagree, so a "faster" path that computes something else is caught.

Limitations
-----------
RSS is sampled every 1 ms: very short spikes can be missed. For exact peaks (including C++ allocations),
run a single child under ``memray run --native`` or ``/usr/bin/time -v`` (Linux) / ``-l`` (macOS).
On macOS the memory compressor can shrink RSS mid-run; prefer Linux for final numbers.
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys

PATHS = ["numpy", "arrow"]
SCENARIOS = {
    # name: (n float64 cols, n categorical cols, n int64 cols)
    "float_only": (20, 0, 0),
    "float+5cat": (20, 5, 0),
    "float+5cat+5int": (20, 5, 5),
    "mostly_cat": (2, 20, 0),
}


def child(path: str, scenario: str, rows: int, threads: int) -> None:
    import gc
    import threading
    import time

    import numpy as np
    import pandas as pd
    import psutil

    import lightgbm as lgb
    import lightgbm.basic as basic

    n_float, n_cat, n_int = SCENARIOS[scenario]
    rng = np.random.default_rng(42)
    cols = {f"f{i}": rng.normal(size=rows) for i in range(n_float)}
    for i in range(n_cat):
        cols[f"c{i}"] = pd.Categorical.from_codes(rng.integers(0, 50, rows), [f"v{j}" for j in range(50)])
    for i in range(n_int):
        cols[f"i{i}"] = rng.integers(0, 1000, rows)
    df = pd.DataFrame(cols)
    del cols
    y = rng.normal(size=rows)

    basic._pandas_df_is_zero_copy_or_unsupported_by_arrow = lambda _data: path == "numpy"
    params = {
        "verbose": -1,
        "max_bin": 63,
        "num_threads": threads,
        "force_col_wise": True,
        "seed": 0,
        "deterministic": True,
    }
    lgb.Dataset(df.head(1000), y[:1000], params=params).construct()  # warm-up

    proc = psutil.Process()

    def measure(fn):
        gc.collect()
        base = proc.memory_info().rss
        peak, stop = base, threading.Event()

        def sample():
            nonlocal peak
            while not stop.is_set():
                peak = max(peak, proc.memory_info().rss)
                time.sleep(0.001)

        sampler = threading.Thread(target=sample)
        sampler.start()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        stop.set()
        sampler.join()
        return result, (max(peak, proc.memory_info().rss) - base) / 2**20, elapsed

    ds, construct_mb, construct_s = measure(lambda: lgb.Dataset(df, y, params=params, free_raw_data=False).construct())
    bst = lgb.train(params, ds, num_boost_round=5)
    _, predict_mb, predict_s = measure(lambda: bst.predict(df))
    model_hash = hashlib.sha1(bst.model_to_string().split("parameters:")[0].encode()).hexdigest()[:12]
    print(
        json.dumps(
            {
                "construct_mb": construct_mb,
                "construct_s": construct_s,
                "predict_mb": predict_mb,
                "predict_s": predict_s,
                "model_hash": model_hash,
                "pandas": pd.__version__,
            }
        )
    )


METRICS = [
    ("construct_mb", "`Dataset.construct()` peak memory", "MB", 0),
    ("predict_mb", "`Booster.predict()` peak memory", "MB", 0),
    ("construct_s", "`Dataset.construct()` time", "s", 2),
    ("predict_s", "`Booster.predict()` time", "s", 2),
]


def markdown_report(runs_by_label: dict) -> str:
    """Markdown tables (one per metric) comparing NumPy vs. Arrow medians, one column per run set.

    A change is marked with ``~`` when the min-max ranges of both paths overlap,
    i.e. it cannot be told apart from run-to-run noise.
    """
    seen = {r["scenario"] for runs in runs_by_label.values() for r in runs}
    scenarios = [s for s in SCENARIOS if s in seen]
    labels = list(runs_by_label)
    lines = []
    for key, title, unit, decimals in METRICS:
        lines += [f"**{title}**, median NumPy → Arrow (Δ%)", ""]
        lines += ["| scenario | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for scenario in scenarios:
            cells = []
            for label in labels:
                values = {
                    path: [
                        float(r[key]) for r in runs_by_label[label] if r["scenario"] == scenario and r["path"] == path
                    ]
                    for path in PATHS
                }
                if not all(values.values()):
                    cells.append("")
                    continue
                med = {path: statistics.median(v) for path, v in values.items()}
                overlap = min(values["numpy"]) <= max(values["arrow"]) and min(values["arrow"]) <= max(values["numpy"])
                # a % change from a (near) zero baseline is meaningless
                if med["numpy"] >= 1 or unit == "s":
                    delta = f" ({'~' if overlap else ''}{(med['arrow'] - med['numpy']) / med['numpy'] * 100:+.0f}%)"
                else:
                    delta = ""
                cells.append(f"{med['numpy']:.{decimals}f} → {med['arrow']:.{decimals}f} {unit}{delta}")
            name = f"{scenario} (control)" if scenario == "float_only" else scenario
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("`~`: min-max ranges of both paths overlap, so the change is within run-to-run noise.")
    lines.append("control: a single float block, which goes through NumPy in production; Arrow is forced here.")
    return "\n".join(lines)


def read_report_args(specs: list) -> dict:
    import csv

    runs_by_label = {}
    for spec in specs:
        path, _, label = spec.partition("=")
        with open(path, newline="") as f:
            runs = list(csv.DictReader(f))
        label = label or (f"pandas {runs[0]['pandas']}" if runs and runs[0].get("pandas") else path)
        runs_by_label[label] = runs
    return runs_by_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--csv", help="also write raw runs to this CSV file")
    parser.add_argument("--report", nargs="+", metavar="CSV[=LABEL]", help="print markdown tables from CSV files")
    parser.add_argument("--child", nargs=2, metavar=("PATH", "SCENARIO"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        child(args.child[0], args.child[1], args.rows, args.threads)
        return
    if args.report:
        print(markdown_report(read_report_args(args.report)))
        return

    jobs = [(p, s) for s in args.scenarios for p in PATHS for _ in range(args.repeats)]
    random.Random(0).shuffle(jobs)
    runs = []
    for i, (path, scenario) in enumerate(jobs, 1):
        out = subprocess.run(
            [
                sys.executable,
                "-I",
                __file__,
                "--child",
                path,
                scenario,
                "--rows",
                str(args.rows),
                "--threads",
                str(args.threads),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append({"path": path, "scenario": scenario, **json.loads(out.stdout.strip().splitlines()[-1])})
        print(f"\r{i}/{len(jobs)} runs", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)

    def fmt(values, unit):
        return f"{statistics.median(values):7.1f}{unit} [{min(values):.1f}-{max(values):.1f}]"

    print(f"rows={args.rows:,} repeats={args.repeats} threads={args.threads}  (median [min-max])")
    print(
        f"{'scenario':17s} {'path':6s} {'construct peak':>24s} {'construct time':>22s} {'predict peak':>24s} {'predict time':>22s}"
    )
    for scenario in args.scenarios:
        hashes = {r["model_hash"] for r in runs if r["scenario"] == scenario}
        if len(hashes) != 1:
            sys.exit(f"{scenario}: paths produced different models {hashes}, benchmark is not comparing equal work")
        for path in PATHS:
            rs = [r for r in runs if r["scenario"] == scenario and r["path"] == path]
            print(
                f"{scenario:17s} {path:6s} "
                f"{fmt([r['construct_mb'] for r in rs], 'MB'):>24s} {fmt([r['construct_s'] for r in rs], 's'):>22s} "
                f"{fmt([r['predict_mb'] for r in rs], 'MB'):>24s} {fmt([r['predict_s'] for r in rs], 's'):>22s}"
            )


if __name__ == "__main__":
    main()
