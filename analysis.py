from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATA_PATH = Path("wine_quality_merged.csv")
DEFAULT_PLOT_PATH = Path("alcohol_by_quality.png")


def load_data(file_path=DEFAULT_DATA_PATH):
    """Load the merged red and white wine quality dataset."""
    return pd.read_csv(file_path)


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


def detect_outliers(df):
    """Identify potential outliers in numeric features using the IQR rule."""
    print("\n=== Potential Outliers (IQR Method) ===")

    numeric_columns = df.select_dtypes(include="number").columns.drop("quality")
    outlier_counts = {}

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        outlier_counts[column] = len(outliers)

    outlier_counts = pd.Series(outlier_counts).sort_values(ascending=False)

    print(outlier_counts)
    print(
        "\nPotential outliers are retained because extreme chemical "
        "measurements may represent valid wines rather than data errors."
    )

    return outlier_counts


def analyze_wine_data(df):
    """Filter high-quality wines and compare summary statistics by wine type."""
    print("\n=== Filtering and Grouping ===")

    high_quality = df[df["quality"] >= 7]

    print("\nNumber of High-Quality Wines (Quality >= 7):")
    print(len(high_quality))

    type_summary = df.groupby("type")[
        ["quality", "alcohol", "pH", "residual sugar", "volatile acidity"]
    ].mean()

    print("\nAverage Characteristics by Wine Type:")
    print(type_summary)

    return high_quality, type_summary


def create_visualization(
    df,
    output_file=DEFAULT_PLOT_PATH,
    show_plot=True,
):
    """Visualize the distribution of alcohol content by wine quality."""
    print("\n=== Creating Visualization ===")

    df.boxplot(column="alcohol", by="quality")

    plt.xlabel("Wine Quality Score")
    plt.ylabel("Alcohol Content (%)")
    plt.title("Alcohol Content by Wine Quality")
    plt.suptitle("")
    plt.tight_layout()

    plt.savefig(output_file)

    if show_plot:
        plt.show()

    plt.close()

    return str(output_file)
