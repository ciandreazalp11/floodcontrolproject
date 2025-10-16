# streamlit_flood_app.py
"""
Flood Forecasting / Analysis - Streamlit app
Based on the provided Google Colab script, adapted for interactive use.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Flood Forecasting & Analysis")

# ---------------------------
# Helper utilities
# ---------------------------
def find_col_by_keywords(columns, keywords):
    cols = [c.lower() for c in columns]
    for k in keywords:
        for i, c in enumerate(cols):
            if k in c:
                return columns[i]
    return None

@st.cache_data
def load_dataframe(uploaded_file, encoding='latin1'):
    # Accept either uploaded file or path-like (for dev)
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
    except Exception:
        # try with utf-8
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
    return df

@st.cache_data
def process_df(df,
               date_col=None,
               water_col=None,
               area_col=None,
               damage_cols_candidates=None,
               interp_method='linear',
               zscore_outlier_thresh=3.0,
               flood_zscore_thresh=1.5,
               flood_threshold_multiplier=1.0):
    """
    Returns processed dataframe and derived aggregations.
    """
    df = df.copy()
    # Auto-detect columns if not provided
    cols = df.columns.tolist()
    lower_cols = [c.lower() for c in cols]
    if not date_col:
        date_col = find_col_by_keywords(cols, ['date','datetime','time','day'])
    if not water_col:
        water_col = find_col_by_keywords(cols, ['water','level','wl','depth','height'])
    if not area_col:
        area_col = find_col_by_keywords(cols, ['barangay','brgy','area','location','sitio'])

    # Combine Date/Day/Year if present (as in original)
    if date_col and 'Day' in df.columns and 'Year' in df.columns and 'Date' in df.columns:
        df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
        date_col = '__combined_date'
    elif date_col is None:
        df['__date_autogen'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        date_col = '__date_autogen'

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.index = pd.DatetimeIndex(df[date_col])

    # Find water column if still None
    if water_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No water-level or numeric column found. Add a water-level column.")
        water_col = numeric_cols[0]

    # Convert water to numeric and interpolate
    df[water_col] = pd.to_numeric(df[water_col], errors='coerce')
    try:
        df[water_col] = df[water_col].interpolate(method=interp_method, limit_direction='both')
    except Exception:
        df[water_col] = df[water_col].interpolate(method='linear', limit_direction='both')

    # z-score and flood heuristic
    df['zscore_water'] = stats.zscore(df[water_col].fillna(df[water_col].mean()))
    df['is_outlier_water'] = df['zscore_water'].abs() > zscore_outlier_thresh

    occurrence_col = find_col_by_keywords(cols, ['flood','event','is_flood','flooded','occurrence'])
    if occurrence_col:
        try:
            df['is_flood'] = df[occurrence_col].astype(bool)
        except Exception:
            df['is_flood'] = df[occurrence_col].astype(str).str.lower().isin(['1','true','yes','y','t'])
    else:
        threshold = df[water_col].mean() + flood_threshold_multiplier * df[water_col].std()
        df['is_flood'] = (df[water_col] >= threshold) | (df['zscore_water'].abs() > flood_zscore_thresh)

    df['year'] = df.index.year

    # Damage columns handling
    damage_cols = []
    if damage_cols_candidates:
        for c in damage_cols_candidates:
            if c in df.columns:
                damage_cols.append(c)
    else:
        candidates = [find_col_by_keywords(cols, ['infrastruct','infra','building']),
                      find_col_by_keywords(cols, ['agri','agriculture','crop','farm']),
                      find_col_by_keywords(cols, ['damage','loss','estimated_damage','total_damage'])]
        damage_cols = [c for c in candidates if c is not None and c in df.columns]
    damage_cols = list(dict.fromkeys(damage_cols))
    total_damage_per_year = pd.DataFrame()
    if damage_cols:
        for c in damage_cols:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('[^0-9.-]','', regex=True), errors='coerce').fillna(0)
        total_damage_per_year = df.groupby('year')[damage_cols].sum().fillna(0)

    # Aggregations
    floods_per_year = df.groupby('year')['is_flood'].sum().astype(int)
    avg_water_per_year = df.groupby('year')[water_col].mean()

    most_affected = None
    if area_col and area_col in df.columns:
        most_affected = df[df['is_flood']].groupby(area_col)['is_flood'].sum().sort_values(ascending=False).head(10)

    return {
        'df': df,
        'date_col': date_col,
        'water_col': water_col,
        'area_col': area_col,
        'damage_cols': damage_cols,
        'total_damage_per_year': total_damage_per_year,
        'floods_per_year': floods_per_year,
        'avg_water_per_year': avg_water_per_year,
        'most_affected': most_affected
    }

# ---------------------------
# Sidebar (structured + interactive options)
# ---------------------------
st.sidebar.title("Flood Analysis - Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV (FLOOD DATA)", type=["csv", "txt"], accept_multiple_files=False)

# Options: detection overrides
with st.sidebar.expander("Column overrides (optional)"):
    col_date = st.text_input("Date column name (leave blank to auto-detect)", value="")
    col_water = st.text_input("Water level column name (leave blank to auto-detect)", value="")
    col_area = st.text_input("Area column name (leave blank to auto-detect)", value="")
    damage_candidates_raw = st.text_area("Damage column names (comma-separated, optional)", value="")
    damage_candidates = [s.strip() for s in damage_candidates_raw.split(',') if s.strip()]

with st.sidebar.expander("Preprocessing options"):
    interp_method = st.selectbox("Interpolation method for water column", options=['linear','time','pad','nearest','spline'], index=0)
    zscore_outlier_thresh = st.number_input("Z-score threshold for outliers", value=3.0, step=0.5, format="%.1f")
    flood_zscore_thresh = st.number_input("Z-score threshold to flag flood (lower)", value=1.5, step=0.1, format="%.1f")
    flood_threshold_multiplier = st.slider("Flood threshold = mean + multiplier * std", 0.0, 3.0, 1.0, 0.1)

with st.sidebar.expander("SARIMA (optional, expensive)"):
    enable_sarima = st.checkbox("Enable SARIMA forecasting", value=False)
    sarima_auto_search = st.checkbox("Auto hyperparameter search (slow)", value=False)
    sarima_train_ratio = st.slider("Train split ratio (when SARIMA runs)", 0.6, 0.95, 0.8, 0.01)
    user_order = st.text_input("Manual order (p,d,q) - optional (e.g. 1,1,1)", value="")
    user_seasonal = st.text_input("Manual seasonal order (P,D,Q,s) - optional (e.g. 1,1,1,12)", value="")

# Buttons
run_analysis = st.sidebar.button("Run Analysis")

# ---------------------------
# Main layout with tabs (B: structured)
# ---------------------------
tabs = st.tabs(["Upload & Preview", "Data & Diagnostics", "Charts", "SARIMA Forecast", "Summary & Download"])

# Tab: Upload & Preview
with tabs[0]:
    st.header("Upload & Preview")
    st.markdown("Upload your dataset CSV. The app will attempt to auto-detect date, water level, and area columns.")
    if uploaded_file is None:
        st.info("No file uploaded yet. You can use the sample dataset if you'd like.")
        # Provide a tiny sample option
        if st.button("Load sample small demo dataset"):
            # Create tiny demo
            dates = pd.date_range("2020-01-01", periods=200, freq='D')
            sample = pd.DataFrame({
                "Date": dates,
                "WaterLevel_m": (np.sin(np.arange(200)/20)*0.5 + 1.5 + np.random.normal(0,0.1,200)).round(3),
                "Barangay": np.random.choice(['A','B','C','D'], 200),
            })
            df = sample
            st.session_state['_uploaded_demo_df'] = df
            st.success("Demo dataset loaded into session")
    else:
        st.write("Uploaded file:", uploaded_file.name)
    # show dataframe preview if available
    df_to_preview = None
    if uploaded_file:
        try:
            df_to_preview = load_dataframe(uploaded_file)
        except Exception as e:
            st.error(f"Could not load file: {e}")
    elif '_uploaded_demo_df' in st.session_state:
        df_to_preview = st.session_state['_uploaded_demo_df']

    if df_to_preview is not None:
        st.subheader("Preview (first 10 rows)")
        st.dataframe(df_to_preview.head(10))
        st.write("Columns detected:", df_to_preview.columns.tolist())

# Tab: Data & Diagnostics
with tabs[1]:
    st.header("Data Processing & Diagnostics")
    if not run_analysis:
        st.info("Click **Run Analysis** in the sidebar to process the uploaded dataset with the chosen options.")
    else:
        if df_to_preview is None:
            st.warning("No dataset available. Please upload a CSV in 'Upload & Preview' tab.")
        else:
            st.info("Processing dataset...")
            try:
                res = process_df(
                    df=df_to_preview,
                    date_col=col_date if col_date else None,
                    water_col=col_water if col_water else None,
                    area_col=col_area if col_area else None,
                    damage_cols_candidates=damage_candidates,
                    interp_method=interp_method,
                    zscore_outlier_thresh=zscore_outlier_thresh,
                    flood_zscore_thresh=flood_zscore_thresh,
                    flood_threshold_multiplier=flood_threshold_multiplier
                )
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.stop()

            df = res['df']
            st.subheader("Detected columns")
            st.write("Date column used:", res['date_col'])
            st.write("Water column used:", res['water_col'])
            st.write("Area column used:", res['area_col'])
            st.write("Damage columns:", res['damage_cols'] or "None detected")

            st.subheader("Head of processed data")
            st.dataframe(df.head(10))

            st.subheader("Basic stats for water column")
            st.write(df[res['water_col']].describe())

            st.subheader("Outliers and Flood flags (sample rows)")
            sample = df[[res['water_col'], 'zscore_water', 'is_outlier_water', 'is_flood']].head(20)
            st.dataframe(sample)

# Tab: Charts (visualizations)
with tabs[2]:
    st.header("Charts & Visualizations")
    if not run_analysis or df_to_preview is None:
        st.info("Run analysis first to see charts.")
    else:
        # Use res from previous section if present; if app flow didn't run that block due to tabs order, compute again
        try:
            res
        except NameError:
            res = process_df(
                df=df_to_preview,
                date_col=col_date if col_date else None,
                water_col=col_water if col_water else None,
                area_col=col_area if col_area else None,
                damage_cols_candidates=damage_candidates,
                interp_method=interp_method,
                zscore_outlier_thresh=zscore_outlier_thresh,
                flood_zscore_thresh=flood_zscore_thresh,
                flood_threshold_multiplier=flood_threshold_multiplier
            )
        df = res['df']
        wc = res['water_col']

        st.subheader("Water level time series")
        fig = px.line(df, x=df.index, y=wc, labels={ 'x': 'Date', wc: wc })
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Water level with flood markers")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df[wc], mode='lines', name='Water level'))
        flood_idx = df[df['is_flood']].index
        fig2.add_trace(go.Scatter(x=flood_idx, y=df.loc[flood_idx, wc], mode='markers', name='Floods',
                                  marker=dict(size=8, symbol='circle')))
        fig2.update_layout(xaxis_title='Date', yaxis_title=wc)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Flood occurrences per year")
        fp = res['floods_per_year']
        if not fp.empty:
            fig3 = px.bar(x=fp.index, y=fp.values, labels={'x':'Year', 'y':'Flood count'})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No flood-per-year data to show.")

        st.subheader("Average water level per year")
        aw = res['avg_water_per_year']
        if not aw.empty:
            fig4 = px.bar(x=aw.index, y=aw.values, labels={'x':'Year', 'y':f'Average {wc}'})
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.write("Not enough data aggregated by year to show.")

        if res['most_affected'] is not None:
            st.subheader("Top affected areas by flood count")
            ma = res['most_affected']
            fig5 = px.bar(x=ma.index.astype(str), y=ma.values, labels={'x':'Area','y':'Flood count'})
            st.plotly_chart(fig5, use_container_width=True)

        if not res['total_damage_per_year'].empty:
            st.subheader("Total damage per year (by damage column)")
            tdp = res['total_damage_per_year']
            fig6 = go.Figure()
            for c in tdp.columns:
                fig6.add_trace(go.Scatter(x=tdp.index, y=tdp[c], mode='lines+markers', name=c))
            fig6.update_layout(xaxis_title='Year', yaxis_title='Damage (dataset units)')
            st.plotly_chart(fig6, use_container_width=True)

# Tab: SARIMA Forecast
with tabs[3]:
    st.header("SARIMA Forecasting")
    if not run_analysis or df_to_preview is None:
        st.info("Run analysis first and enable SARIMA in sidebar to use forecasting.")
    elif not enable_sarima:
        st.info("Enable SARIMA in the sidebar to run forecasting.")
    else:
        # Prepare series (monthly average)
        try:
            res
        except NameError:
            res = process_df(
                df=df_to_preview,
                date_col=col_date if col_date else None,
                water_col=col_water if col_water else None,
                area_col=col_area if col_area else None,
                damage_cols_candidates=damage_candidates,
                interp_method=interp_method,
                zscore_outlier_thresh=zscore_outlier_thresh,
                flood_zscore_thresh=flood_zscore_thresh,
                flood_threshold_multiplier=flood_threshold_multiplier
            )
        df = res['df']
        wc = res['water_col']
        series = df[wc].resample('M').mean()

        if len(series.dropna()) < 12:
            st.warning("Not enough monthly data for SARIMA (need >= 12 aggregated months).")
        else:
            # Prepare train/test
            series = series.fillna(method='ffill').fillna(0)
            split = int(len(series) * sarima_train_ratio)
            train = series.iloc[:split]
            test = series.iloc[split:]

            st.write(f"Series length: {len(series)} months. Train: {len(train)} months, Test: {len(test)} months.")

            # Determine orders
            manual_order = None
            manual_seasonal = None
            if user_order:
                try:
                    parts = [int(x.strip()) for x in user_order.split(',')]
                    if len(parts) == 3:
                        manual_order = tuple(parts)
                except:
                    st.error("Manual order format invalid. Use p,d,q (e.g. 1,1,1)")

            if user_seasonal:
                try:
                    parts = [int(x.strip()) for x in user_seasonal.split(',')]
                    if len(parts) == 4:
                        manual_seasonal = tuple(parts)
                except:
                    st.error("Manual seasonal format invalid. Use P,D,Q,s (e.g. 1,1,1,12)")

            run_sarima_now = st.button("Run SARIMA now")

            if run_sarima_now:
                with st.spinner("Running SARIMA (this may take a while)..."):
                    best_res = None
                    best_aic = np.inf
                    chosen_order = None
                    if manual_order and manual_seasonal:
                        try:
                            mod = SARIMAX(train, order=manual_order, seasonal_order=manual_seasonal,
                                          enforce_stationarity=False, enforce_invertibility=False)
                            res_fit = mod.fit(disp=False)
                            best_res = res_fit
                            chosen_order = (manual_order, manual_seasonal)
                        except Exception as e:
                            st.error(f"Manual SARIMA fit failed: {e}")
                    else:
                        if sarima_auto_search:
                            # simple small grid search - keep small ranges to avoid extreme slowness
                            p_range = range(0, 2)
                            d_range = range(0, 2)
                            q_range = range(0, 2)
                            P_range = range(0, 2)
                            D_range = range(0, 2)
                            Q_range = range(0, 2)
                            s = 12
                            tried = 0
                            for p in p_range:
                                for d in d_range:
                                    for q in q_range:
                                        for P in P_range:
                                            for D in D_range:
                                                for Q in Q_range:
                                                    try:
                                                        mod = SARIMAX(train, order=(p,d,q),
                                                                      seasonal_order=(P,D,Q,s),
                                                                      enforce_stationarity=False,
                                                                      enforce_invertibility=False)
                                                        res_fit = mod.fit(disp=False)
                                                        tried += 1
                                                        if res_fit.aic < best_aic:
                                                            best_aic = res_fit.aic
                                                            best_res = res_fit
                                                            chosen_order = ((p,d,q),(P,D,Q,s))
                                                    except Exception:
                                                        continue
                            if tried == 0:
                                st.error("Auto-search tried no models; reduce restrictions or provide manual order.")
                        else:
                            # A small default try (0-1)
                            try:
                                mod = SARIMAX(train, order=(1,0,1), seasonal_order=(1,0,1,12),
                                              enforce_stationarity=False, enforce_invertibility=False)
                                res_fit = mod.fit(disp=False)
                                best_res = res_fit
                                chosen_order = ((1,0,1),(1,0,1,12))
                            except Exception as e:
                                st.error(f"Default SARIMA fit failed: {e}")

                    if best_res is not None:
                        # Predict on test
                        pred = best_res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
                        forecast = pred.predicted_mean
                        mae = mean_absolute_error(test, forecast)
                        mse = mean_squared_error(test, forecast)
                        st.success(f"SARIMA completed. Order: {chosen_order}, AIC: {getattr(best_res,'aic',np.nan):.2f}")
                        st.write(f"MAE: {mae:.4f}   MSE: {mse:.4f}")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train'))
                        fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test'))
                        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='SARIMA Forecast'))
                        fig.update_layout(title="SARIMA: Actual vs Forecast", xaxis_title="Date", yaxis_title=wc)
                        st.plotly_chart(fig, use_container_width=True)

                        # Option to forecast N future periods
                        periods = st.number_input("Forecast future months", min_value=1, max_value=60, value=6, step=1)
                        if st.button("Generate future forecast"):
                            future_pred = best_res.get_forecast(steps=int(periods))
                            future_mean = future_pred.predicted_mean
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Historical'))
                            future_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='M')
                            fig2.add_trace(go.Scatter(x=future_index, y=future_mean.values, mode='lines+markers', name='Forecast'))
                            fig2.update_layout(title=f"Forecast next {periods} months", xaxis_title="Date", yaxis_title=wc)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.error("SARIMA model could not be fit with the provided settings.")

# Tab: Summary & Download
with tabs[4]:
    st.header("Summary & Download")
    if not run_analysis or df_to_preview is None:
        st.info("Run analysis first to create summary and outputs.")
    else:
        try:
            res
        except NameError:
            res = process_df(
                df=df_to_preview,
                date_col=col_date if col_date else None,
                water_col=col_water if col_water else None,
                area_col=col_area if col_area else None,
                damage_cols_candidates=damage_candidates,
                interp_method=interp_method,
                zscore_outlier_thresh=zscore_outlier_thresh,
                flood_zscore_thresh=flood_zscore_thresh,
                flood_threshold_multiplier=flood_threshold_multiplier
            )
        df = res['df']
        summary_df = pd.DataFrame({
            "year": res['floods_per_year'].index,
            "floods_per_year": res['floods_per_year'].values,
            "avg_water_per_year": res['avg_water_per_year'].values
        }).reset_index(drop=True)

        st.subheader("Summary per year")
        st.dataframe(summary_df)

        csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download summary CSV", data=csv_bytes, file_name="summary_per_year.csv", mime="text/csv")

        st.subheader("Processed full dataset (first 200 rows)")
        st.dataframe(df.head(200))

        # Allow user to download processed full dataset
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download processed dataset CSV", data=buf.getvalue(), file_name="processed_flood_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: SARIMA can be slow for auto-search. Use manual orders for faster runs. Interpolation and thresholds are adjustable in the sidebar.")
