import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem


# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="MoleculeRankAI",
    page_icon="🧬",
    layout="wide"
)



# -----------------------------------------
# Import dashboard modules
# -----------------------------------------

sys.path.append(
    os.path.dirname(__file__)
)


try:

    from explain import explain_molecule
    SHAP_READY = True

except Exception:

    explain_molecule = None
    SHAP_READY = False



try:

    from molecule_viewer import draw_molecule
    DRAW_READY = True

except Exception:

    draw_molecule = None
    DRAW_READY = False



# -----------------------------------------
# Load model and FINAL rankings
# -----------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "models/rf_model.pkl"
    )



@st.cache_data
def load_ranked():

    return pd.read_csv(
        "data/final_rankings.csv"
    )



try:

    model = load_model()

    ranked = load_ranked()


except Exception as e:

    st.error(
        f"Loading error: {e}"
    )

    st.stop()



# -----------------------------------------
# Fingerprint
# -----------------------------------------

def fingerprint(smiles):

    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:

        return None


    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )


    return np.array(fp)



# -----------------------------------------
# ADMET Properties
# -----------------------------------------

def calculate_properties(smiles):

    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:

        return None


    return {

        "Molecular Weight":
            round(
                Descriptors.MolWt(mol),
                3
            ),

        "LogP":
            round(
                Crippen.MolLogP(mol),
                3
            ),

        "Hydrogen Bond Donors":
            Lipinski.NumHDonors(mol),

        "Hydrogen Bond Acceptors":
            Lipinski.NumHAcceptors(mol),

        "TPSA":
            round(
                Descriptors.TPSA(mol),
                3
            )
    }



# -----------------------------------------
# RF Activity Prediction
# -----------------------------------------

def predict_activity(smiles):

    fp = fingerprint(
        smiles
    )


    if fp is None:

        return None


    return model.predict(
        fp.reshape(1,-1)
    )[0]



# -----------------------------------------
# Molecule Viewer
# -----------------------------------------

def show_structure(smiles):

    if DRAW_READY:

        try:

            img = draw_molecule(
                smiles
            )


            if img:

                st.subheader(
                    "2D Molecular Structure"
                )

                st.image(
                    img
                )

        except Exception as e:

            st.warning(
                f"Drawing error: {e}"
            )



# -----------------------------------------
# Dashboard
# -----------------------------------------

st.title(
    "🧬 MoleculeRankAI Drug Discovery Dashboard"
)


st.write(
    "AI-powered molecule ranking using activity prediction, ADMET analysis, SHAP explainability, and ensemble scoring."
)



# =========================================
# Analyze custom molecule
# =========================================

st.header(
    "Analyze Molecule"
)


smiles = st.text_input(
    "Enter SMILES:",
    "CCO"
)



if st.button(
    "Analyze Molecule"
):

    props = calculate_properties(
        smiles
    )


    if props is None:

        st.error(
            "Invalid SMILES"
        )

        st.stop()



    show_structure(
        smiles
    )


    st.subheader(
        "Molecular Properties"
    )


    st.json(
        props
    )



    prediction = predict_activity(
        smiles
    )


    if prediction:

        st.subheader(
            "Activity Prediction"
        )


        st.success(
            f"Predicted pChEMBL: {prediction:.3f}"
        )



    st.subheader(
        "Explainable AI (SHAP)"
    )


    if SHAP_READY:

        try:

            st.write(
                explain_molecule(
                    smiles
                )
            )

        except Exception as e:

            st.warning(
                f"SHAP error: {e}"
            )

    else:

        st.info(
            "SHAP unavailable"
        )



# =========================================
# Final AI Rankings
# =========================================

st.header(
    "Top MoleculeRankAI Candidates"
)



top10 = ranked[
    [
        "canonical_smiles",
        "MoleculeRankAI_score"
    ]
].head(10).copy()



top10.insert(
    0,
    "Rank",
    range(
        1,
        len(top10)+1
    )
)



st.table(
    top10
)



# =========================================
# Molecule Explorer
# =========================================

st.header(
    "Explore Ranked Molecule"
)



selected_rank = st.number_input(

    "Select molecule rank:",

    min_value=1,

    max_value=10,

    value=1

)



selected = top10.iloc[
    selected_rank-1
]



selected_smiles = selected[
    "canonical_smiles"
]


selected_score = selected[
    "MoleculeRankAI_score"
]



st.subheader(
    "Selected Molecule"
)


st.code(
    selected_smiles
)



st.success(
    f"MoleculeRankAI Score: {selected_score:.4f}"
)



show_structure(
    selected_smiles
)



if st.button(
    "Analyze Selected Molecule"
):


    props = calculate_properties(
        selected_smiles
    )


    st.subheader(
        "Properties"
    )


    st.json(
        props
    )



    prediction = predict_activity(
        selected_smiles
    )


    st.success(
        f"Predicted pChEMBL: {prediction:.3f}"
    )



    if SHAP_READY:

        st.subheader(
            "SHAP Explanation"
        )


        st.write(
            explain_molecule(
                selected_smiles
            )
        )