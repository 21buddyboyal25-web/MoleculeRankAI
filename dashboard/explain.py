import numpy as np
import joblib
import shap

from rdkit import Chem
from rdkit.Chem import AllChem


model = joblib.load(
    "models/rf_model.pkl"
)


def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    return np.array(fp)


def explain_molecule(smiles):

    X = fingerprint(smiles).reshape(1,-1)

    explainer = shap.TreeExplainer(
        model
    )

    values = explainer.shap_values(
        X
    )

    importance = np.abs(
        values[0]
    )

    top_features = np.argsort(
        importance
    )[-5:][::-1]

    return top_features