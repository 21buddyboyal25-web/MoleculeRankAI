import streamlit as st


def molecule_explorer(
    ranked,
    show_structure,
    calculate_properties=None,
    predict_activity=None,
    explain_molecule=None,
    SHAP_READY=False
):

    # =========================================
    # Phase 13
    # Full Molecule Search + Ranking Explorer
    # =========================================


    st.header(
        "🔎 MoleculeRankAI Search Explorer"
    )


    st.write(
        "Browse all ranked molecules, search candidates, filter scores, download results, and inspect structures."
    )


    # Copy all molecules

    explorer = ranked.copy()



    # -----------------------------------------
    # Search
    # -----------------------------------------

    search = st.text_input(
        "Search SMILES:",
        key="molecule_search_box"
)


    if search:

        explorer = explorer[
            explorer["canonical_smiles"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]



    # -----------------------------------------
    # Score filter
    # -----------------------------------------

    if len(explorer) > 0:


        minimum_score = float(
            explorer["MoleculeRankAI_score"].min()
        )


        maximum_score = float(
            explorer["MoleculeRankAI_score"].max()
        )



        score_range = st.slider(

            "MoleculeRankAI Score Range",

            minimum_score,

            maximum_score,

            (
                minimum_score,
                maximum_score
            )

        )



        explorer = explorer[

            (explorer["MoleculeRankAI_score"] >= score_range[0])

            &

            (explorer["MoleculeRankAI_score"] <= score_range[1])

        ]



    # -----------------------------------------
    # Sorting
    # -----------------------------------------

    sort_option = st.selectbox(

        "Sort Molecules:",

        [
            "Highest Score First",
            "Lowest Score First"
        ]

    )



    if sort_option == "Highest Score First":


        explorer = explorer.sort_values(

            "MoleculeRankAI_score",

            ascending=False

        )


    else:


        explorer = explorer.sort_values(

            "MoleculeRankAI_score",

            ascending=True

        )



    # -----------------------------------------
    # Display all molecules
    # -----------------------------------------

    st.subheader(

        f"Available Molecules: {len(explorer)}"

    )



    display_columns = [

        "canonical_smiles",

        "MoleculeRankAI_score",

        "Final_score",

        "Activity_score",

        "ADMET_score"

    ]



    display = explorer[display_columns].copy()



    display.insert(

        0,

        "Rank",

        range(
            1,
            len(display)+1
        )

    )



    # Virtual scrolling table

    st.dataframe(

        display,

        height=700,

        use_container_width=True

    )



    # -----------------------------------------
    # Download
    # -----------------------------------------

    csv = display.to_csv(
        index=False
    )


    st.download_button(

        label="⬇ Download Filtered Results",

        data=csv,

        file_name="MoleculeRankAI_results.csv",

        mime="text/csv"

    )



    # -----------------------------------------
    # Molecule Inspector
    # -----------------------------------------

    st.header(
        "🧬 Molecule Inspector"
    )



    if len(explorer) > 0:


        selected_rank = st.number_input(

            "Select molecule rank:",

            min_value=1,

            max_value=len(explorer),

            value=1,
	    key="molecule_rank_box"

        )



        selected = explorer.iloc[
            selected_rank - 1
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


            if calculate_properties:


                st.subheader(
                    "Molecular Properties"
                )


                st.json(

                    calculate_properties(
                        selected_smiles
                    )

                )



            if predict_activity:


                prediction = predict_activity(
                    selected_smiles
                )


                st.success(

                    f"Predicted pChEMBL: {prediction:.3f}"

                )



            if SHAP_READY and explain_molecule:


                st.subheader(
                    "SHAP Explanation"
                )


                st.write(

                    explain_molecule(
                        selected_smiles
                    )

                )


    else:

        st.warning(
            "No molecules match the current filters."
        )

