import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------
# STEP 1: Load the dataset
# ------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "focusflow_demo.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

# ------------------------------------
# STEP 2: Data quality checks
# ------------------------------------

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include="all"))

# ------------------------------------
# STEP 3: Visualize task duration
# ------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="actual_minutes",
    bins=30,
    kde=True
)

plt.title("Distribution of Actual Task Duration")
plt.xlabel("Actual Duration (minutes)")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()

# ------------------------------------
# STEP 4: Compare task categories
# ------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="task_type",
    y="actual_minutes"
)

plt.title("Task Duration by Task Type")
plt.xlabel("Task Type")
plt.ylabel("Actual Duration (minutes)")
plt.tight_layout()
plt.show()

# ------------------------------------
# STEP 5: Difficulty vs duration
# ------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="difficulty",
    y="actual_minutes"
)

plt.title("Task Duration by Difficulty Level")
plt.xlabel("Difficulty (1-5)")
plt.ylabel("Actual Duration (minutes)")
plt.tight_layout()
plt.show()

# ------------------------------------
# STEP 6: Numerical correlations
# ------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

print("\n===== CORRELATION MATRIX =====")
print(correlation_matrix.round(2))

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------
# STEP 7: Planned vs actual duration
# ------------------------------------

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="planned_minutes",
    y="actual_minutes",
    alpha=0.5
)

plt.title("Planned vs Actual Task Duration")
plt.xlabel("Planned Duration (minutes)")
plt.ylabel("Actual Duration (minutes)")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")

df["time_overrun"] = (
    df["actual_minutes"] - df["planned_minutes"]
)

print("\n===== TIME OVERRUN ANALYSIS =====")

print(
    df.groupby("task_type")["time_overrun"]
    .agg(["mean", "median", "std"])
    .round(2)
)

print(
    "\nPercentage of tasks exceeding planned time:",
    round((df["time_overrun"] > 0).mean() * 100, 2),
    "%"
)