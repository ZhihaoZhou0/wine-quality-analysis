import time
from pathlib import Path

import pandas as pd
import polars as pl


DEFAULT_DATA_PATH = Path("wine_quality_merged.csv")


def compare_pandas_polars(file_path=DEFAULT_DATA_PATH, runs=5):
    """Compare Pandas and Polars performance over multiple runs."""
    print("\n=== Pandas vs. Polars Performance ===")

    pandas_times = []
    polars_times = []

    for _ in range(runs):
        pandas_start = time.perf_counter()

        pandas_df = pd.read_csv(file_path)
        pandas_result = pandas_df.groupby("type")["alcohol"].mean()

        pandas_time = time.perf_counter() - pandas_start
        pandas_times.append(pandas_time)

        polars_start = time.perf_counter()

        polars_df = pl.read_csv(file_path)
        polars_result = polars_df.group_by("type").agg(
            pl.col("alcohol").mean()
        )

        polars_time = time.perf_counter() - polars_start
        polars_times.append(polars_time)

    pandas_average = sum(pandas_times) / runs
    polars_average = sum(polars_times) / runs

    print("\nPandas Result:")
    print(pandas_result)

    print("\nPolars Result:")
    print(polars_result)

    print(f"\nAverage Execution Time ({runs} runs):")
    print(f"Pandas: {pandas_average:.6f} seconds")
    print(f"Polars: {polars_average:.6f} seconds")

    return (
        pandas_result,
        polars_result,
        pandas_average,
        polars_average,
    )