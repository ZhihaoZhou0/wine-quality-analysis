from analysis import (
    analyze_wine_data,
    check_data_quality,
    create_visualization,
    detect_outliers,
    load_data,
)
from benchmark import compare_pandas_polars
from modeling import create_feature_importance_plot, train_model


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
    detect_outliers(df)
    analyze_wine_data(df)
    create_visualization(df)

    _, _, _, feature_importance = train_model(df)
    create_feature_importance_plot(feature_importance)

    compare_pandas_polars()


if __name__ == "__main__":
    main()
