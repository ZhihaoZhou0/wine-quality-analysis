from analysis import (
    analyze_wine_data,
    check_data_quality,
    create_visualization,
    load_data,
)
from benchmark import compare_pandas_polars
from modeling import train_model


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