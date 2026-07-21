import pandas as pd

# Load ADMET results

admet = pd.read_csv(
    "data/admet_predictions.csv"
)

print("========== ADMET Evaluation ==========")

print("Total molecules:", len(admet))

print("\nColumns:")
print(admet.columns.tolist())

print("\nStatistics:")
print(admet.describe())

print("\nFirst 5 molecules:")
print(admet.head())