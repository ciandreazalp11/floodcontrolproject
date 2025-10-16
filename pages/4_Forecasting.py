# pages/4_Forecasting.py
import streamlit as st
import pandas as pd
from utils.forecasting import fit_sarima, evaluate_forecast, simple_grid_search

st.title("4 — Forecasting (SARIMA)")

if 'processed' not in st.session_state:
    st.info("Process data first (Data Cleaning).")
else:
    res = st.session_state['processed']
    df = res['df']
    wc = res['water_col']

    series = df[wc].resample('M').mean().fillna(method='ffill').fillna(0)

    if len(series.dropna()) < 12:
        st.warning("Not enough monthly data for SARIMA (need >= 12 months aggregated).")
    else:
        st.write(f"Monthly series length: {len(series)} months.")
        train_ratio = st.slider("Train split ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.01)
        split = int(len(series)*train_ratio)
        train = series.iloc[:split]
        test = series.iloc[split:]

        st.write(f"Train: {len(train)} months; Test: {len(test)} months")

        manual_order = st.text_input("Manual (p,d,q) (e.g. 1,1,1)", "")
        manual_seasonal = st.text_input("Manual seasonal (P,D,Q,s) (e.g. 1,1,1,12)", "")

        auto_search = st.checkbox("Auto small grid search (may be slow)", value=False)
        max_models = st.number_input("Max models for grid search", min_value=5, max_value=200, value=30, step=5)

        run_button = st.button("Run SARIMA")

        if run_button:
            chosen_res = None
            if manual_order and manual_seasonal:
                try:
                    order = tuple(int(x.strip()) for x in manual_order.split(","))
                    seasonal = tuple(int(x.strip()) for x in manual_seasonal.split(","))
                    st.write(f"Fitting manual order {order} seasonal {seasonal}")
                    chosen_res = fit_sarima(train, order=order, seasonal_order=seasonal)
                except Exception as e:
                    st.error(f"Manual SARIMA fit failed: {e}")
            else:
                if auto_search:
                    st.info("Starting auto grid search (this may take time)...")
                    best_res, best_aic = simple_grid_search(train, p_range=(0,1), d_range=(0,1), q_range=(0,1),
                                                           P_range=(0,1), D_range=(0,1), Q_range=(0,1), s=12, max_models=int(max_models))
                    if best_res is None:
                        st.error("Auto search did not find a model.")
                    else:
                        chosen_res = best_res
                        st.write(f"Auto-search best AIC: {best_aic:.2f}")
                else:
                    # try a simple default
                    try:
                        chosen_res = fit_sarima(train, order=(1,0,1), seasonal_order=(1,0,1,12))
                    except Exception as e:
                        st.error(f"Default SARIMA fit failed: {e}")

            if chosen_res is not None:
                eval_res = evaluate_forecast(chosen_res, test)
                st.success(f"SARIMA done. AIC: {eval_res['aic']:.2f} MAE: {eval_res['mae']:.4f} MSE: {eval_res['mse']:.4f}")
                # plot
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train, name='Train', mode='lines'))
                fig.add_trace(go.Scatter(x=test.index, y=test, name='Test', mode='lines'))
                fig.add_trace(go.Scatter(x=eval_res['forecast'].index, y=eval_res['forecast'].values, name='Forecast', mode='lines'))
                fig.update_layout(title="SARIMA: Actual vs Forecast", xaxis_title="Date")
                st.plotly_chart(fig, use_container_width=True)

                # future forecast
                periods = st.number_input("Forecast N future months", min_value=1, max_value=60, value=6, step=1)
                if st.button("Generate future forecast"):
                    future = chosen_res.get_forecast(steps=int(periods))
                    fut_mean = future.predicted_mean
                    future_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='M')
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=series.index, y=series, name='Historical'))
                    fig2.add_trace(go.Scatter(x=future_index, y=fut_mean.values, name='Forecast'))
                    fig2.update_layout(title=f"Forecast next {periods} months", xaxis_title="Date")
                    st.plotly_chart(fig2, use_container_width=True)
