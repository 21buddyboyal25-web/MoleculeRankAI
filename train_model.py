import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# Load data
X = np.load("MoleculeRankAI/data/fingerprints.npy")
activity = pd.read_csv("data/activity.csv")

# Remove missing values
activity = activity.dropna(subset=["pchembl_value"])

# Use matching rows
X = X[:len(activity)]
y = activity["pchembl_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Faster model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Train
print("Training model...")
model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2:", r2)

# Save
joblib.dump(model, "models/rf_model.pkl")

print("Model saved as models/rf_model.pkl")