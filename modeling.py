import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
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


def train_model(df):
    """Train a Random Forest model to predict wine quality."""
    print("\n=== Machine Learning: Random Forest Regression ===")

    X = df[FEATURES]
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
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return model, mae, r2, feature_importance