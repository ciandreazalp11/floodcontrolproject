# app.py
import streamlit as st
from streamlit_option_menu import option_menu

# import pages as modules
from pages import data_cleaning, visualization, analysis, forecasting, summary

st.set_page_config(page_title="Flood Forecast System", layout="wide", page_icon="🌊")

# Sidebar menu
with st.sidebar:
    st.title("Flood System")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Data Cleaning", "Visualization", "Analysis", "Forecasting", "Summary"],
        icons=["house", "cloud-upload", "bar-chart", "clipboard-data", "activity", "download"],
        menu_icon="cast",
        default_index=0,
    )

# Home content
if selected == "Home":
    st.title("🌊 Flood Forecasting & Analysis System")
    st.markdown("""
    **Welcome!** Use the sidebar to navigate:
    - **Data Cleaning** — upload and process CSV
    - **Visualization** — interactive charts
    - **Analysis** — yearly summaries & top areas
    - **Forecasting** — SARIMA forecasting (optional)
    - **Summary** — download processed outputs
    """)
    st.info("Start at **Data Cleaning** to upload and process your dataset.")
    st.write("If you want a sample CSV, generate it on the Data Cleaning page.")

# Delegate to page modules
elif selected == "Data Cleaning":
    data_cleaning.app()
elif selected == "Visualization":
    visualization.app()
elif selected == "Analysis":
    analysis.app()
elif selected == "Forecasting":
    forecasting.app()
elif selected == "Summary":
    summary.app()
