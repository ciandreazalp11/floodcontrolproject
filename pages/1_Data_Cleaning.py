# pages/1_Data_Cleaning.py
import streamlit as st
from utils.data_cleaning import load_csv, process_df
import pandas as pd

st.title("1 — Data Cleaning & Upload")

uploaded_file = st.file_uploader("Upload CSV (or leave blank to use sample in /data)", type=["csv","txt","xlsx"])
if uploaded_file is None:
    st.info("No upload detected. If you have created sample_data.csv via the Home page, it will be used.")
    try:
        df_preview = pd.read_csv("data/sample_data.csv")
        st.success("Loaded data/sample_data.csv")
    except Exception:
        df_preview = None
else:
    try:
        df_preview = load_csv(uploaded_file)
        st.success("File loaded.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df_preview = None

if df_preview is not None:
    st.subheader("Preview")
    st.dataframe(df_preview.head(10))
    st.write("Columns:", df_preview.columns.tolist())

st.subheader("Optional: Column overrides")
date_col = st.text_input("Date column name (leave blank to auto-detect)", "")
water_col = st.text_input("Water column name (leave blank to auto-detect)", "")
area_col = st.text_input("Area/Location column name (leave blank to auto-detect)", "")
damage_cols_raw = st.text_input("Damage column names (comma separated, optional)", "")
damage_cols = [s.strip() for s in damage_cols_raw.split(",") if s.strip()]

st.write("Preprocessing settings")
interp_method = st.selectbox("Interpolation method", options=['linear','time','pad','nearest'], index=0)
zscore_outlier_thresh = st.number_input("Outlier z-score threshold", value=3.0, step=0.5)
flood_zscore_thresh = st.number_input("Flood z-score threshold (lower)", value=1.5, step=0.1)
flood_threshold_multiplier = st.slider("Flood threshold multiplier for mean+mult*std", 0.0, 3.0, 1.0, 0.1)

if st.button("Process dataset"):
    if df_preview is None:
        st.error("No dataset to process. Upload or create sample data first.")
    else:
        try:
            res = process_df(
                df_preview,
                date_col=date_col or None,
                water_col=water_col or None,
                area_col=area_col or None,
                damage_cols_candidates=damage_cols or None,
                interp_method=interp_method,
                zscore_outlier_thresh=zscore_outlier_thresh,
                flood_zscore_thresh=flood_zscore_thresh,
                flood_threshold_multiplier=flood_threshold_multiplier
            )
            st.session_state['processed'] = res
            st.success("Processing completed. Go to Visualization or Analysis pages.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
