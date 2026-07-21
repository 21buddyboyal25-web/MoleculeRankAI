import pandas as pd


# ----------------------------------
# Load predictions
# ----------------------------------

activity = pd.read_csv(
    "data/activity_predictions.csv"
)


ranked = pd.read_csv(
    "data/ranked_molecules.csv"
)


admet = pd.read_csv(
    "data/admet_predictions.csv"
)



# ----------------------------------
# Merge datasets
# ----------------------------------

df = activity.merge(
    admet,
    on="canonical_smiles"
)


df = df.merge(
    ranked,
    on="canonical_smiles"
)



print("Merged molecules:")
print(len(df))



# ----------------------------------
# Normalize scores
# ----------------------------------

def normalize(column):

    return (
        column-column.min()
    ) / (
        column.max()-column.min()
    )



df["activity_score"] = normalize(
    df["predicted_activity"]
)


df["ranking_score"] = normalize(
    df["Final_score"]
)



# ADMET quality score

df["admet_score"] = (

    1 -
    normalize(
        df["Molecular_Weight"]
    )

)



# ----------------------------------
# Final ensemble score
# ----------------------------------

df["MoleculeRankAI_score"] = (

    0.45 *
    df["activity_score"]

    +

    0.35 *
    df["ranking_score"]

    +

    0.20 *
    df["admet_score"]

)



# ----------------------------------
# Sort
# ----------------------------------

final = df.sort_values(
    "MoleculeRankAI_score",
    ascending=False
)



final.to_csv(
    "data/final_rankings.csv",
    index=False
)



print(
    "Final ranking complete"
)


print(
    final[
        [
            "canonical_smiles",
            "MoleculeRankAI_score"
        ]
    ].head(10)
)