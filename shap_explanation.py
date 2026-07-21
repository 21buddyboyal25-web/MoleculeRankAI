import pandas as pd
import numpy as np
import joblib
import shap

from rdkit import Chem
from rdkit.Chem import AllChem


# -----------------------------------------
# Load model
# -----------------------------------------

model = joblib.load(
    "models/rf_model.pkl"
)


# -----------------------------------------
# Load top ranked molecules
# -----------------------------------------

ranked = pd.read_csv(
    "data/ranked_molecules.csv"
)


top_molecule = ranked.iloc[0]["canonical_smiles"]

print("Explaining molecule:")
print(top_molecule)


# -----------------------------------------
# Convert molecule to fingerprint
# -----------------------------------------

def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    return np.array(fp)


X = fingerprint(top_molecule).reshape(1, -1)


# -----------------------------------------
# SHAP explanation
# -----------------------------------------

explainer = shap.TreeExplainer(
    model
)

values = explainer.shap_values(
    X
)


# -----------------------------------------
# Show important features
# -----------------------------------------

importance = pd.DataFrame(
    {
        "feature": range(2048),
        "shap_value": values[0]
    }
)


importance["impact"] = abs(
    importance["shap_value"]
)


importance = importance.sort_values(
    "impact",
    ascending=False
)


print("\nTop features affecting prediction:")

print(
    importance.head(10)
)