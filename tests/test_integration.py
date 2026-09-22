from main import (
    load_data,
    check_data_quality,
    analyze_wine_data,
    train_model,
)


def test_analysis_workflow():
    """Test the core wine analysis workflow from data loading to model training."""

    # Step 1: Load data
    df = load_data()
    assert not df.empty

    # Step 2: Check data quality
    missing_values, duplicate_count = check_data_quality(df)
    assert missing_values.sum() == 0
    assert duplicate_count >= 0

    # Step 3: Analyze and transform data
    high_quality, type_summary = analyze_wine_data(df)
    assert not high_quality.empty
    assert len(type_summary) == 2

    # Step 4: Train and evaluate the model
    model, mae, r2, feature_importance = train_model(df)

    assert model is not None
    assert mae >= 0
    assert 0 <= r2 <= 1
    assert not feature_importance.empty