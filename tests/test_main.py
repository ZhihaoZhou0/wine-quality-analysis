import os

from main import (
    load_data,
    check_data_quality,
    analyze_wine_data,
    create_visualization,
    train_model,
)


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


def test_analyze_wine_data():
    """Test filtering and grouping operations."""
    df = load_data()

    high_quality, type_summary = analyze_wine_data(df)

    # Every filtered wine should have quality >= 7
    assert len(high_quality) == 1277
    assert (high_quality["quality"] >= 7).all()

    # Grouped results should contain both wine types
    assert "red" in type_summary.index
    assert "white" in type_summary.index

    # Check an expected summary statistic
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
    """Test that the visualization is successfully created."""
    df = load_data()

    output_file = create_visualization(df)

    assert output_file == "alcohol_by_quality.png"
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0