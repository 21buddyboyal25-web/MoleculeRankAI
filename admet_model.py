import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski


# Load activity data

activity = pd.read_csv(
    "data/activity.csv"
)

activity = activity.dropna(
    subset=["canonical_smiles"]
)

molecules = activity[
    ["canonical_smiles"]
].drop_duplicates()


# Calculate ADMET properties

def calculate_admet(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {

        "Molecular_Weight":
            Descriptors.MolWt(mol),

        "LogP":
            Crippen.MolLogP(mol),

        "HBD":
            Lipinski.NumHDonors(mol),

        "HBA":
            Lipinski.NumHAcceptors(mol),

        "Rotatable_Bonds":
            Lipinski.NumRotatableBonds(mol),

        "TPSA":
            Descriptors.TPSA(mol),

        "Heavy_Atoms":
            Descriptors.HeavyAtomCount(mol)
    }


# Generate predictions

results = []

for smiles in molecules["canonical_smiles"]:

    properties = calculate_admet(smiles)

    if properties:

        properties["canonical_smiles"] = smiles

        results.append(properties)


admet = pd.DataFrame(results)


# Save results

admet.to_csv(
    "data/admet_predictions.csv",
    index=False
)


print("ADMET prediction complete")
print("Molecules processed:", len(admet))
print(admet.head())