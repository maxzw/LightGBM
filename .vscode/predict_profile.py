# ruff: noqa: D103, T201
"""Profile Booster.predict() on a pandas frame with categoricals: NumPy vs. Arrow path, per step."""

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd
import pyarrow as pa

import lightgbm as lgb
from lightgbm import basic

rng = np.random.default_rng(42)
n = 2_000_000
cols = {f"f{i}": rng.normal(size=n) for i in range(20)}
for i in range(5):
    cols[f"c{i}"] = pd.Categorical.from_codes(rng.integers(0, 50, n), [f"v{j}" for j in range(50)])
df = pd.DataFrame(cols)
y = rng.normal(size=n)
params = {"verbose": -1, "max_bin": 63, "num_threads": 2, "force_col_wise": True, "seed": 0, "deterministic": True}
bst = lgb.train(params, lgb.Dataset(df, y), num_boost_round=5)
print(f"pandas {pd.__version__}  pyarrow {pa.__version__}  numpy {np.__version__}")


def best_of(fn, reps=5):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


encoded = basic._data_from_narwhals(df, "auto", "auto", bst.pandas_categorical)[0]
print(
    f"_data_from_narwhals       {best_of(lambda: basic._data_from_narwhals(df, 'auto', 'auto', bst.pandas_categorical)):.3f}s"
)
print(f"to_numpy                  {best_of(lambda: basic._pandas_df_to_numpy(encoded)):.3f}s")
print(f"pa.Table.from_pandas      {best_of(lambda: pa.Table.from_pandas(encoded, preserve_index=False)):.3f}s")
for path in ("numpy", "arrow"):
    basic._pandas_df_is_zero_copy_or_unsupported_by_arrow = lambda _data, p=path: p == "numpy"
    print(f"predict [{path}]           {best_of(lambda: bst.predict(df)):.3f}s")
    profiler = cProfile.Profile()
    profiler.enable()
    bst.predict(df)
    profiler.disable()
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(8)
    print("\n".join(line for line in out.getvalue().splitlines() if line.strip())[:3000])
