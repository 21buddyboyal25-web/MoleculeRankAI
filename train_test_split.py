import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

X = np.load("MoleculeRankAI/data/fingerprints.npy")

activity = pd.read_csv("data/activity.csv")

y = activity["pchembl_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)