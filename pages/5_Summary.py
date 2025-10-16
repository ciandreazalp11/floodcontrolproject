# pages/5_Summary.py
import streamlit as st
import io
import pandas as pd

st.title("5 — Summary & Download")

if 'processed' not in st.session_state:
    st.info("No processed dataset available. Run Data Cleaning first.")
else:
    res = st.session_state['processed']
    df = res['df']

    st.subheader("Processed dataset (first 200 rows)")
    st.dataframe(df.head(200))

    summary_df = pd.DataFrame({
        "year": res['floods_per_year'].index,
        "floods_per_year": res['floods_per_year'].values,
        "avg_water_per_year": res['avg_water_per_year'].values
    }).reset_index(drop=True)
    st.subheader("Summary per year")
    st.dataframe(summary_df)

    st.download_button("Download processed dataset (CSV)", data=df.to_csv(index=False).encode('utf-8'),
                       file_name='processed_flood_data.csv', mime='text/csv')

    st.download_button("Download yearly summary (CSV)", data=summary_df.to_csv(index=False).encode('utf-8'),
                       file_name='summary_per_year.csv', mime='text/csv')
