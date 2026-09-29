import os

import pandas as pd

from analysis import (
    analyze_wine_data,
    check_data_quality,
    create_visualization,
    detect_outliers,
    load_data,
)
from modeling import create_feature_importance_plot, train_model


def test_load_data():
    """Test that the wine dataset loads correctly."""
    df = load_data()

    assert df is not None
    assert not df.empty
    assert len(df) == 6497
    assert "quality" in df.columns
    assert "type" in df.columns


def test_data_quality():
    """Test missing-value and duplicate detection."""
    df = load_data()

    missing_values, duplicate_count = check_data_quality(df)

    assert missing_values.sum() == 0
    assert duplicate_count == 1177


def test_detect_outliers():
    """Test that the IQR method detects an obvious extreme value."""
    sample_df = pd.DataFrame(
        {
            "alcohol": [10, 10, 10, 10, 50],
            "pH": [3.2, 3.2, 3.2, 3.2, 3.2],
            "quality": [5, 5, 6, 6, 7],
        }
    )

    outlier_counts = detect_outliers(sample_df)

    assert outlier_counts["alcohol"] == 1
    assert outlier_counts["pH"] == 0
    assert "quality" not in outlier_counts.index


def test_analyze_wine_data():
    """Test filtering and grouping operations."""
    df = load_data()

    high_quality, type_summary = analyze_wine_data(df)

    assert len(high_quality) == 1277
    assert (high_quality["quality"] >= 7).all()

    assert "red" in type_summary.index
    assert "white" in type_summary.index

    assert type_summary.loc["white", "quality"] > type_summary.loc["red", "quality"]


def test_train_model():
    """Test Random Forest training and evaluation outputs."""
    df = load_data()

    model, mae, r2, feature_importance = train_model(df)

    assert model is not None
    assert mae >= 0
    assert 0 <= r2 <= 1
    assert len(feature_importance) == 11
    assert "alcohol" in feature_importance.index


def test_create_visualization():
    """Test that the alcohol-by-quality visualization is created."""
    df = load_data()

    output_file = create_visualization(df)

    assert output_file == "alcohol_by_quality.png"
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_create_feature_importance_plot():
    """Test that the feature-importance visualization is created."""
    df = load_data()

    _, _, _, feature_importance = train_model(df)

    output_file = create_feature_importance_plot(feature_importance)

    assert output_file == "feature_importance.png"
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0
