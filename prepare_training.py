import numpy as np
import pandas as pd

fingerprints = np.load("MoleculeRankAI/data/fingerprints.npy")
activity = pd.read_csv("data/activity.csv")

X = fingerprints
y = activity["pchembl_value"]

print("X:", X.shape)
print("y:", y.shape)