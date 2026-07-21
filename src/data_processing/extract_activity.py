import sqlite3
import pandas as pd

conn = sqlite3.connect("data/chembl_37.db")

query = """
SELECT
    md.chembl_id,
    cs.canonical_smiles,
    act.standard_value,
    act.standard_units,
    act.standard_type
FROM activities act
JOIN molecule_dictionary md
ON act.molregno = md.molregno
JOIN compound_structures cs
ON md.molregno = cs.molregno
WHERE act.standard_value IS NOT NULL
AND cs.canonical_smiles IS NOT NULL
"""

df = pd.read_sql_query(query, conn)

df.to_csv("data/activity_data.csv", index=False)

print(df.shape)
print("Finished.")