import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///data/chembl_37.db"
)

query = """
SELECT
cs.canonical_smiles,
md.chembl_id

FROM compound_structures cs

JOIN molecule_dictionary md
ON cs.molregno = md.molregno

WHERE cs.canonical_smiles IS NOT NULL

LIMIT 100000
"""

df = pd.read_sql(query, engine)

print(df.head())

df.to_csv(
    "data/molecules.csv",
    index=False
)

print("Saved molecules.csv")