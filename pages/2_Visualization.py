# pages/2_Visualization.py
import streamlit as st
from utils.visualization import plot_time_series, plot_with_flood_markers, plot_bar_yearly
import pandas as pd

st.title("2 — Visualization")

if 'processed' not in st.session_state:
    st.info("No processed data found. First upload & process a dataset in 'Data Cleaning' page.")
else:
    res = st.session_state['processed']
    df = res['df']
    wc = res['water_col']

    st.subheader("Water level time series")
    st.plotly_chart(plot_time_series(df, wc), use_container_width=True)

    st.subheader("Water level with flood markers")
    st.plotly_chart(plot_with_flood_markers(df, wc), use_container_width=True)

    st.subheader("Flood occurrences per year")
    if not res['floods_per_year'].empty:
        st.plotly_chart(plot_bar_yearly(res['floods_per_year'], y_label='Flood count', title='Floods per year'), use_container_width=True)
    else:
        st.write("No flood/year aggregation available")

    st.subheader("Average water level per year")
    if not res['avg_water_per_year'].empty:
        st.plotly_chart(plot_bar_yearly(res['avg_water_per_year'], y_label=f'Average {wc}', title='Average water level per year'), use_container_width=True)
