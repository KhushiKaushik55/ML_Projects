import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "focusflow_demo.csv"
)

# Create engineered features
df["difficulty_experience_ratio"] = (
    df["difficulty"] / df["experience_level"]
)

df["planned_difficulty"] = (
    df["planned_minutes"] * df["difficulty"]
)

original_features = [
    "task_type",
    "planned_minutes",
    "difficulty",
    "sleep_hours",
    "experience_level",
    "time_of_day"
]

engineered_features = original_features + [
    "difficulty_experience_ratio",
    "planned_difficulty"
]

categorical_features = [
    "task_type",
    "time_of_day"
]

# Keep the same training subset for both experiments
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=21
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def evaluate_features(feature_names):

    numeric_features = [
        name for name in feature_names
        if name not in categorical_features
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LinearRegression())
    ])

    scores = -cross_val_score(
        pipeline,
        train_df[feature_names],
        train_df["actual_minutes"],
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    return scores

original_scores = evaluate_features(
    original_features
)

engineered_scores = evaluate_features(
    engineered_features
)

print("\n===== FEATURE ENGINEERING RESULTS =====")

print(
    f"Original features: "
    f"{original_scores.mean():.3f} "
    f"+/- {original_scores.std():.3f} min"
)

print(
    f"Engineered features: "
    f"{engineered_scores.mean():.3f} "
    f"+/- {engineered_scores.std():.3f} min"
)

improvement = (
    original_scores.mean()
    - engineered_scores.mean()
)

print(
    f"\nMAE improvement: {improvement:.3f} min"
)

if improvement > 0:
    print("Engineered features performed better.")
elif improvement < 0:
    print("Original features performed better.")
else:
    print("Both feature sets performed equally.")