# pages/3_Analysis.py
import streamlit as st
import io
import pandas as pd

st.title("3 — Analysis")

if 'processed' not in st.session_state:
    st.info("Process a dataset first (Data Cleaning).")
else:
    res = st.session_state['processed']
    df = res['df']

    st.subheader("Summary per year")
    summary_df = pd.DataFrame({
        "year": res['floods_per_year'].index,
        "floods_per_year": res['floods_per_year'].values,
        "avg_water_per_year": res['avg_water_per_year'].values
    }).reset_index(drop=True)
    st.dataframe(summary_df)

    st.download_button("Download summary CSV", data=summary_df.to_csv(index=False).encode('utf-8'), file_name='summary_per_year.csv')

    st.subheader("Most affected areas (top 10)")
    if res['most_affected'] is not None:
        st.dataframe(res['most_affected'].reset_index().rename(columns={'index':'area','is_flood':'count'}))
    else:
        st.write("No area/most-affected data detected.")

    st.subheader("Total damage per year (if detected)")
    if not res['total_damage_per_year'].empty:
        st.dataframe(res['total_damage_per_year'])
    else:
        st.write("No damage columns detected or damage summary empty.")
