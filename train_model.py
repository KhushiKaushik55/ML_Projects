import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------
# STEP 1: Load the dataset
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "focusflow_demo.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# ----------------------------------
# STEP 2: Define features and target
# ----------------------------------

features = [
    "task_type",
    "planned_minutes",
    "difficulty",
    "sleep_hours",
    "experience_level",
    "time_of_day"
]

X = df[features]
y = df["actual_minutes"]

categorical_features = [
    "task_type",
    "time_of_day"
]

numerical_features = [
    "planned_minutes",
    "difficulty",
    "sleep_hours",
    "experience_level"
]

# ----------------------------------
# STEP 3: Split the data
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ----------------------------------
# STEP 4: Build preprocessing
# ----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ----------------------------------
# STEP 5: Define models
# ----------------------------------

models = {
    "Dummy Regressor": DummyRegressor(
        strategy="median"
    ),

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ----------------------------------
# STEP 6: Train and evaluate
# ----------------------------------

results = []
predictions_by_model = {}

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    })

    predictions_by_model[name] = predictions

# ----------------------------------
# STEP 7: Compare results
# ----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# ----------------------------------
# STEP 8: Visualize model performance
# ----------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (minutes)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ----------------------------------
# STEP 9: Save comparison results
# ----------------------------------

OUTPUT_PATH = BASE_DIR / "data" / "model_results.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:", OUTPUT_PATH)