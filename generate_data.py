import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible random number generator
rng = np.random.default_rng(42)

# Number of simulated task sessions
n = 1000

task_types = ["coding", "studying", "writing", "research"]
times_of_day = ["morning", "afternoon", "evening", "night"]

# Generate the input features
df = pd.DataFrame({
    "task_type": rng.choice(task_types, n),
    "planned_minutes": rng.integers(20, 121, n),
    "difficulty": rng.integers(1, 6, n),
    "sleep_hours": rng.uniform(4, 9, n).round(1),
    "experience_level": rng.integers(1, 6, n),
    "time_of_day": rng.choice(times_of_day, n),
})

# Simulated effects for different task types
task_effect = df["task_type"].map({
    "coding": 12,
    "studying": 5,
    "writing": 8,
    "research": 16,
})

# Simulated effects for different times of day
time_effect = df["time_of_day"].map({
    "morning": -5,
    "afternoon": 3,
    "evening": 0,
    "night": 7,
})

# Random variation between sessions
noise = rng.normal(loc=0, scale=12, size=n)

# Simulated actual task duration
df["actual_minutes"] = (
    df["planned_minutes"]
    + task_effect
    + time_effect
    + df["difficulty"] * 6
    - df["experience_level"] * 4
    - (df["sleep_hours"] - 7) * 2
    + noise
).round().clip(lower=5)

# Create the data folder and save CSV
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

file_path = data_dir / "focusflow_demo.csv"
df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print("Saved to:", file_path)
print(df.head())