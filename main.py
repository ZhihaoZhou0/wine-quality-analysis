import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_data():
    """Load the merged red and white wine quality dataset."""
    df = pd.read_csv("wine_quality_merged.csv")
    return df


def check_data_quality(df):
    """Check the dataset for missing values and duplicate rows."""
    print("\n=== Data Quality Check ===")

    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    print("\nMissing Values:")
    print(missing_values)

    print("\nNumber of Duplicate Rows:")
    print(duplicate_count)

    return missing_values, duplicate_count


def analyze_wine_data(df):
    """Filter high-quality wines and compare summary statistics by wine type."""
    print("\n=== Filtering and Grouping ===")

    # Filter wines with a quality score of 7 or higher
    high_quality = df[df["quality"] >= 7]

    print("\nNumber of High-Quality Wines (Quality >= 7):")
    print(len(high_quality))

    # Compare selected characteristics between red and white wines
    type_summary = df.groupby("type")[
        ["quality", "alcohol", "pH", "residual sugar", "volatile acidity"]
    ].mean()

    print("\nAverage Characteristics by Wine Type:")
    print(type_summary)

    return high_quality, type_summary


def create_visualization(df):
    """Visualize the distribution of alcohol content by wine quality."""
    print("\n=== Creating Visualization ===")

    df.boxplot(column="alcohol", by="quality")

    plt.xlabel("Wine Quality Score")
    plt.ylabel("Alcohol Content (%)")
    plt.title("Alcohol Content by Wine Quality")
    plt.suptitle("")

    plt.tight_layout()

    output_file = "alcohol_by_quality.png"
    plt.savefig(output_file)
    plt.show()

    return output_file


def train_model(df):
    """Train a Random Forest model to predict wine quality."""
    print("\n=== Machine Learning: Random Forest Regression ===")

    features = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    X = df[features]
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R-squared: {r2:.3f}")

    feature_importance = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return model, mae, r2, feature_importance


def compare_pandas_polars():
    """Compare Pandas and Polars performance over multiple runs."""
    print("\n=== Pandas vs. Polars Performance ===")

    runs = 5
    pandas_times = []
    polars_times = []

    for _ in range(runs):
        # Pandas
        pandas_start = time.perf_counter()

        pandas_df = pd.read_csv("wine_quality_merged.csv")
        pandas_result = pandas_df.groupby("type")["alcohol"].mean()

        pandas_time = time.perf_counter() - pandas_start
        pandas_times.append(pandas_time)

        # Polars
        polars_start = time.perf_counter()

        polars_df = pl.read_csv("wine_quality_merged.csv")
        polars_result = polars_df.group_by("type").agg(
            pl.col("alcohol").mean()
        )

        polars_time = time.perf_counter() - polars_start
        polars_times.append(polars_time)

    # Calculate average execution time
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


def main():
    """Run the complete wine quality analysis workflow."""
    df = load_data()

    print("=== First 5 Rows ===")
    print(df.head())

    print("\n=== Dataset Information ===")
    df.info()

    print("\n=== Summary Statistics ===")
    print(df.describe())

    check_data_quality(df)
    analyze_wine_data(df)
    create_visualization(df)
    train_model(df)
    compare_pandas_polars()


if __name__ == "__main__":
    main()