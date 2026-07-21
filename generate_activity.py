import sqlite3
import pandas as pd

conn = sqlite3.connect(r"data\chembl_37.db")

query = """
SELECT
    cs.canonical_smiles,
    md.chembl_id,
    a.standard_type,
    a.standard_value,
    a.pchembl_value
FROM molecule_dictionary md
JOIN compound_structures cs
    ON md.molregno = cs.molregno
JOIN activities a
    ON md.molregno = a.molregno
WHERE cs.canonical_smiles IS NOT NULL
AND a.pchembl_value IS NOT NULL
LIMIT 100000
"""

activity = pd.read_sql(query, conn)

activity.to_csv(
    "data/activity.csv",
    index=False
)

print(activity.shape)
print(activity.head())