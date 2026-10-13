import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)

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

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(
    BASE_DIR / "data" / "focusflow_demo.csv"
)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=21
)

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        ["task_type", "time_of_day"]
    ),
    (
        "numerical",
        "passthrough",
        [
            "planned_minutes",
            "difficulty",
            "sleep_hours",
            "experience_level"
        ]
    )
])

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Dummy": DummyRegressor(strategy="median"),

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
        random_state=42
    )
}

results = []

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])

    scores = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "CV_MAE_Mean": scores.mean(),
        "CV_MAE_Std": scores.std()
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV_MAE_Mean")

print("\n===== CROSS-VALIDATION RESULTS =====")
print(results_df.round(3).to_string(index=False))

best_name = results_df.iloc[0]["Model"]
print("\nBest algorithm:", best_name)

# Define small, controlled parameter searches
param_grids = {
    "Dummy": {
        "model__strategy": ["mean", "median"]
    },

    "Linear Regression": {
        "model__fit_intercept": [True, False]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__min_samples_leaf": [1, 3]
    },

    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }
}

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", models[best_name])
])

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grids[best_name],
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n===== TUNING RESULTS =====")
print("Best parameters:", search.best_params_)
print("Best CV MAE:", round(-search.best_score_, 3))

# Final held-out evaluation
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
r2 = r2_score(y_test, predictions)

print("\n===== TEST RESULTS =====")
print(f"MAE:  {mae:.3f} minutes")
print(f"RMSE: {rmse:.3f} minutes")
print(f"R2:   {r2:.4f}")

# Residuals: actual - predicted
residuals = y_test.to_numpy() - predictions

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(
    x=predictions,
    y=residuals,
    ax=axes[0],
    alpha=0.6
)

axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs Predictions")
axes[0].set_xlabel("Predicted Minutes")
axes[0].set_ylabel("Residual (Minutes)")

sns.histplot(
    residuals,
    bins=25,
    kde=True,
    ax=axes[1]
)

axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual (Minutes)")

plt.tight_layout()
plt.show()

# Save the fitted preprocessing + model pipeline
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "focusflow_model.joblib"
joblib.dump(best_model, model_path)

results_df.to_csv(
    BASE_DIR / "data" / "cv_results.csv",
    index=False
)

print("\nModel saved to:", model_path)
print("Cross-validation results saved successfully!")