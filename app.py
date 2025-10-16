# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Flood Forecast System", layout="wide")
st.title("Flood Forecasting & Analysis — Home")

st.markdown(
    """
    This project is a multi-page Streamlit app for flood detection, visualization, and SARIMA forecasting.
    Use the left sidebar or the pages menu (top-left) to navigate:
    - **Data Cleaning** (detect columns, preview)
    - **Visualization** (interactive charts)
    - **Analysis** (summary by year, affected areas, damage)
    - **Forecasting** (SARIMA)
    - **Summary** (download processed outputs)
    """
)

st.header("Quick start")
st.markdown(
    """
    1. Go to **Data Cleaning** and upload your CSV (or use the sample dataset).
    2. Set optional column overrides (date, water level, area).
    3. Run analysis and use the other pages for charts and forecasting.
    """
)

# Show small demo / sample file generator
if st.button("Generate small sample dataset and navigate to Data Cleaning"):
    # create sample dataset in memory and save locally to ./data/sample_data.csv
    Path("data").mkdir(exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=360, freq="D")
    sample = pd.DataFrame({
        "Date": dates,
        "WaterLevel_m": (np.sin(np.arange(len(dates))/30) * 0.6 + 1.5 + np.random.normal(0, 0.15, len(dates))).round(3),
        "Barangay": np.random.choice(['Brgy A','Brgy B','Brgy C','Brgy D'], len(dates)),
        "Estimated_damage": (np.random.rand(len(dates))*1000).round(2)
    })
    sample.to_csv("data/sample_data.csv", index=False)
    st.success("Sample dataset saved to data/sample_data.csv. Open the 'Data Cleaning' page.")
