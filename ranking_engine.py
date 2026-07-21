import pandas as pd
import numpy as np
import joblib


# -----------------------------------------
# Load ADMET data
# -----------------------------------------

admet = pd.read_csv(
    "data/admet_predictions.csv"
)


# -----------------------------------------
# Load activity model
# -----------------------------------------

model = joblib.load(
    "models/rf_model.pkl"
)


# -----------------------------------------
# Create ADMET score
# -----------------------------------------

def admet_score(row):

    score = 100

    # Lower molecular weight preferred
    if row["Molecular_Weight"] > 500:
        score -= 20

    # LogP range preference
    if row["LogP"] > 5:
        score -= 20

    # Too many hydrogen bond donors
    if row["HBD"] > 5:
        score -= 10

    # Too many acceptors
    if row["HBA"] > 10:
        score -= 10

    # High polarity penalty
    if row["TPSA"] > 140:
        score -= 15

    return max(score, 0)


# -----------------------------------------
# Drug-likeness score
# -----------------------------------------

def drug_likeness(row):

    score = 100

    if row["Molecular_Weight"] > 500:
        score -= 25

    if row["Rotatable_Bonds"] > 10:
        score -= 15

    return max(score, 0)


# -----------------------------------------
# Apply scoring
# -----------------------------------------

admet["ADMET_score"] = admet.apply(
    admet_score,
    axis=1
)


admet["Drug_score"] = admet.apply(
    drug_likeness,
    axis=1
)


# -----------------------------------------
# Simple toxicity proxy
# -----------------------------------------

admet["Toxicity_score"] = (
    100 -
    (admet["LogP"].clip(lower=0) * 10)
).clip(
    lower=0
)


# -----------------------------------------
# Activity placeholder
# -----------------------------------------

# Uses normalized activity estimate placeholder
# Full fingerprint prediction will be connected later

# -----------------------------------------
# Load predicted activity
# -----------------------------------------

activity = pd.read_csv(
    "data/activity_predictions.csv"
)


# Normalize activity prediction to 0-100 score

activity["Activity_score"] = (
    (activity["predicted_activity"] - activity["predicted_activity"].min())
    /
    (activity["predicted_activity"].max() - activity["predicted_activity"].min())
) * 100


# Merge activity score into ADMET table

admet = admet.merge(
    activity[
        [
            "canonical_smiles",
            "Activity_score"
        ]
    ],
    on="canonical_smiles",
    how="inner"
)
# -----------------------------------------
# Synthetic accessibility placeholder
# -----------------------------------------

admet["Synthesis_score"] = 70


# -----------------------------------------
# Final ranking score
# -----------------------------------------

admet["Final_score"] = (

    admet["Activity_score"] * 0.30 +

    admet["ADMET_score"] * 0.25 +

    admet["Toxicity_score"] * 0.20 +

    admet["Drug_score"] * 0.15 +

    admet["Synthesis_score"] * 0.10

)


# -----------------------------------------
# Rank molecules
# -----------------------------------------

ranked = admet.sort_values(
    "Final_score",
    ascending=False
)


ranked.to_csv(
    "data/ranked_molecules.csv",
    index=False
)


print("Ranking complete")
print("Molecules ranked:", len(ranked))

print("\nTop 10 molecules:")
print(
    ranked[
        [
            "canonical_smiles",
            "Final_score"
        ]
    ].head(10)
)