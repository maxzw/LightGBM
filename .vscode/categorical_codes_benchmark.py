# ruff: noqa: D103, PLC0415, PLR0402, T201
"""Benchmark the categorical code dtype LightGBM uses for pandas / polars / pyarrow input.

``_data_from_narwhals`` replaces each categorical by integer codes. This compares 8-byte codes
(``int64_codes``, the memory of the previous float64/NaN encoding) with the smallest integer type that fits
(``small_codes``: Int8/Int16/Int32). Both use the same code path; only ``_categorical_code_dtype`` is patched.

    python .vscode/categorical_codes_benchmark.py --rows 2000000 --repeats 9 --csv out.csv
    python .vscode/categorical_codes_benchmark.py --report a.csv b.csv=pyarrow   # markdown tables from CSVs

Every measurement runs in a fresh interpreter, runs are interleaved in random order, the peak RSS is sampled every 1 ms, and each child prints a model hash: the parent fails if the two
variants train different models.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys

PATHS = ["int64_codes", "small_codes"]
# Scenario names list the column counts: f=float64, i=int64, c=unordered categorical, o=ordered categorical,
# with an optional _k<N> suffix when the categoricals have N categories instead of the default 50.
# (n float64 cols, n int64 cols, n unordered cat cols, n ordered cat cols, n categories)
_SPECS = [
    (20, 0, 0, 0, 50),
    (20, 0, 5, 0, 50),
    (20, 0, 0, 5, 50),
    (20, 5, 5, 0, 50),
    (20, 5, 0, 5, 50),
    (2, 0, 20, 0, 50),
    (2, 0, 0, 20, 50),
    (0, 0, 20, 0, 50),
    (0, 0, 0, 20, 50),
    (20, 0, 3, 2, 50),
    (20, 0, 5, 0, 300),
]
NULL_FRACTION = 0.05  # share of missing values in the categorical columns


def _scenario_name(n_float, n_int, n_cat, n_ord, n_categories):
    parts = [f"{prefix}{n}" for prefix, n in (("f", n_float), ("i", n_int), ("c", n_cat), ("o", n_ord)) if n]
    return "_".join(parts) + (f"_k{n_categories}" if n_categories != 50 else "")


SCENARIOS = {_scenario_name(*spec): spec for spec in _SPECS}


def child(path: str, scenario: str, rows: int, threads: int, input_type: str) -> None:
    import gc
    import threading
    import time

    import narwhals.stable.v2 as nw
    import numpy as np
    import pandas as pd
    import psutil

    import lightgbm as lgb
    import lightgbm.basic as basic

    n_float, n_int, n_cat, n_ord, n_categories = SCENARIOS[scenario]
    rng = np.random.default_rng(42)
    cols = {f"f{i}": rng.normal(size=rows) for i in range(n_float)}
    categories = [f"v{j}" for j in range(n_categories)]

    def categorical(ordered):
        codes = rng.integers(0, n_categories, rows)
        codes[rng.random(rows) < NULL_FRACTION] = -1
        return pd.Categorical.from_codes(codes, categories, ordered=ordered)

    for i in range(n_cat):
        cols[f"c{i}"] = categorical(ordered=False)
    for i in range(n_ord):
        cols[f"o{i}"] = categorical(ordered=True)
    for i in range(n_int):
        cols[f"i{i}"] = rng.integers(0, 1000, rows)
    df = pd.DataFrame(cols)
    if input_type == "pandas":
        pass
    elif input_type == "polars":
        import polars as pl

        df = pl.from_pandas(df)
        # polars would otherwise read ordered pandas categoricals as unordered
        df = df.with_columns(pl.col(f"o{i}").cast(pl.Enum(categories)) for i in range(n_ord))
    else:
        import pyarrow as pa

        df = pa.Table.from_pandas(df, preserve_index=False)
    del cols
    y = rng.normal(size=rows)

    if path == "int64_codes":
        basic._categorical_code_dtype = lambda _num_categories: nw.Int64
    params = {
        "verbose": -1,
        "max_bin": 63,
        "num_threads": threads,
        "force_col_wise": True,
        "seed": 0,
        "deterministic": True,
    }
    lgb.Dataset(df.iloc[:1000] if input_type == "pandas" else df.slice(0, 1000), y[:1000], params=params).construct()  # warm-up

    proc = psutil.Process()

    def measure(fn):
        gc.collect()
        peak, stop = proc.memory_info().rss, threading.Event()

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
        peak = max(peak, proc.memory_info().rss)
        return result, peak / 2**20, elapsed

    ds, construct_rss_mb, construct_s = measure(lambda: lgb.Dataset(df, y, params=params, free_raw_data=False).construct())
    bst = lgb.train(params, ds, num_boost_round=5)
    _, predict_rss_mb, predict_s = measure(lambda: bst.predict(df))
    model_hash = hashlib.sha1(bst.model_to_string().split("parameters:")[0].encode()).hexdigest()[:12]
    print(
        json.dumps(
            {
                "construct_rss_mb": construct_rss_mb,
                "construct_s": construct_s,
                "predict_rss_mb": predict_rss_mb,
                "predict_s": predict_s,
                "model_hash": model_hash,
                "input": input_type,
            }
        )
    )


METRICS = [
    ("construct_rss_mb", "`Dataset.construct()` peak total RSS", "MB", 0),
    ("predict_rss_mb", "`Booster.predict()` peak total RSS", "MB", 0),
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
        lines += [f"**{title}**, median int64 codes → small codes (Δ%)", ""]
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
                overlap = min(values[PATHS[0]]) <= max(values[PATHS[1]]) and min(values[PATHS[1]]) <= max(
                    values[PATHS[0]]
                )
                # a % change from a (near) zero baseline is meaningless
                if med[PATHS[0]] >= 1 or unit == "s":
                    delta = f" ({'~' if overlap else ''}{(med[PATHS[1]] - med[PATHS[0]]) / med[PATHS[0]] * 100:+.0f}%)"
                else:
                    delta = ""
                cells.append(f"{med[PATHS[0]]:.{decimals}f} → {med[PATHS[1]]:.{decimals}f} {unit}{delta}")
            name = scenario
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("`~`: min-max ranges of both paths overlap, so the change is within run-to-run noise.")
    return "\n".join(lines)


def read_report_args(specs: list) -> dict:
    import csv

    runs_by_label = {}
    for spec in specs:
        path, _, label = spec.partition("=")
        with open(path, newline="") as f:
            runs = list(csv.DictReader(f))
        label = label or (runs[0]["input"] if runs and runs[0].get("input") else path)
        runs_by_label[label] = runs
    return runs_by_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input", choices=["pandas", "polars", "pyarrow"], default="polars")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--csv", help="also write raw runs to this CSV file")
    parser.add_argument("--report", nargs="+", metavar="CSV[=LABEL]", help="print markdown tables from CSV files")
    parser.add_argument("--child", nargs=2, metavar=("PATH", "SCENARIO"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        child(args.child[0], args.child[1], args.rows, args.threads, args.input)
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
                "--input",
                args.input,
            ],
            capture_output=True,
            text=True,
            check=True,
            # make polars' jemalloc return freed memory right away, otherwise freed setup buffers are reused
            # and hide allocations from the RSS deltas (more so for scenarios that free more during setup)
            env={**os.environ, "_RJEM_MALLOC_CONF": "dirty_decay_ms:0,muzzy_decay_ms:0"},
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

    print(f"input={args.input} rows={args.rows:,} repeats={args.repeats} threads={args.threads}  (median [min-max])")
    print(
        f"{'scenario':17s} {'path':12s} {'construct RSS':>24s} {'construct time':>22s} "
        f"{'predict RSS':>24s} {'predict time':>22s}"
    )
    for scenario in args.scenarios:
        hashes = {r["model_hash"] for r in runs if r["scenario"] == scenario}
        if len(hashes) != 1:
            sys.exit(f"{scenario}: paths produced different models {hashes}, benchmark is not comparing equal work")
        for path in PATHS:
            rs = [r for r in runs if r["scenario"] == scenario and r["path"] == path]
            print(
                f"{scenario:17s} {path:12s} "
                f"{fmt([r['construct_rss_mb'] for r in rs], 'MB'):>24s} {fmt([r['construct_s'] for r in rs], 's'):>22s} "
                f"{fmt([r['predict_rss_mb'] for r in rs], 'MB'):>24s} {fmt([r['predict_s'] for r in rs], 's'):>22s}"
            )


if __name__ == "__main__":
    main()
