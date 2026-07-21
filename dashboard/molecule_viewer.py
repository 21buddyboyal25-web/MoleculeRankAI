from rdkit import Chem
from rdkit.Chem import Draw


def draw_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    img = Draw.MolToImage(
        mol,
        size=(500,500)
    )

    return img