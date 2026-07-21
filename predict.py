import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the trained model
model = joblib.load("model.pkl")

# Enter a SMILES string
smiles = input("Enter SMILES: ").strip()

# Convert SMILES to an RDKit molecule
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES string.")
    raise SystemExit

# Generate a Morgan fingerprint (2048 bits, radius 2)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Convert fingerprint to a NumPy array
arr = np.zeros((2048,), dtype=np.int8)
from rdkit.DataStructs import ConvertToNumpyArray
ConvertToNumpyArray(fp, arr)

# Predict pChEMBL value
prediction = model.predict(arr.reshape(1, -1))[0]

print(f"\nPredicted pChEMBL value: {prediction:.3f}")