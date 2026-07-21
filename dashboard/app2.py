import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
from molecule_explorer import molecule_explorer
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



# SHAP explanation

try:

    from explain import explain_molecule

    SHAP_READY = True


except Exception:

    explain_molecule = None

    SHAP_READY = False



# Molecule drawing

try:

    from molecule_viewer import draw_molecule

    DRAW_READY = True


except Exception:

    draw_molecule = None

    DRAW_READY = False



# -----------------------------------------
# Load model and rankings
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

    print("APP2 RANKED SIZE:", ranked.shape)
    print(ranked.head(15))

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
# Activity Prediction
# -----------------------------------------

def predict_activity(smiles):

    fp = fingerprint(
        smiles
    )


    if fp is None:

        return None


    prediction = model.predict(

        fp.reshape(1, -1)

    )[0]


    return prediction



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



# =========================================
# Dashboard
# =========================================


st.title(
    "🧬 MoleculeRankAI Drug Discovery Dashboard"
)



st.write(

    "AI-powered molecule ranking, activity prediction, ADMET analysis, SHAP explainability, and molecule exploration."

)



# =========================================
# Custom Molecule Analysis
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
            "Invalid SMILES string"
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



    if prediction is not None:


        st.subheader(
            "Activity Prediction"
        )


        st.success(

            f"Predicted pChEMBL: {prediction:.3f}"

        )



    # -----------------------------
    # SHAP Explanation
    # -----------------------------

    st.subheader(
        "Explainable AI (SHAP)"
    )



    if SHAP_READY:


        try:

            explanation = explain_molecule(
                smiles
            )


            st.write(
                explanation
            )


        except Exception as e:


            st.warning(

                f"SHAP error: {e}"

            )


    else:

        st.info(
            "SHAP explanation unavailable"
        )



# =========================================
# Phase 13
# Full Molecule Search + Ranking Explorer
# =========================================


molecule_explorer(
    ranked,
    show_structure,
    calculate_properties,
    predict_activity,
    explain_molecule,
    SHAP_READY
)
