import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem


# -----------------------------------------
# Load molecules
# -----------------------------------------

admet = pd.read_csv(
    "data/admet_predictions.csv"
)


# -----------------------------------------
# Load trained RF model
# -----------------------------------------

model = joblib.load(
    "models/rf_model.pkl"
)


# -----------------------------------------
# Generate fingerprints
# -----------------------------------------

def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    return np.array(fp)


X = []

valid_smiles = []

for smiles in admet["canonical_smiles"]:

    fp = fingerprint(smiles)

    if fp is not None:

        X.append(fp)
        valid_smiles.append(smiles)


X = np.array(X)


# -----------------------------------------
# Predict activity
# -----------------------------------------

predictions = model.predict(X)


results = pd.DataFrame(
    {
        "canonical_smiles": valid_smiles,
        "predicted_activity": predictions
    }
)


# -----------------------------------------
# Save predictions
# -----------------------------------------

results.to_csv(
    "data/activity_predictions.csv",
    index=False
)


print("Activity prediction complete")
print("Molecules predicted:", len(results))

print(results.head())