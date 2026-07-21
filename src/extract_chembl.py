import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///data/chembl_37.db"
)

query = """
SELECT
canonical_smiles,
molecule_chembl_id
FROM compound_structures
LIMIT 100000
"""

df = pd.read_sql(query, engine)

df.to_csv(
    "data/molecules.csv",
    index=False
)

print(df.head())
print("Saved molecules.csv")