# utils/forecasting.py
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fit_sarima(train_series, order=(1,0,1), seasonal_order=(1,0,1,12)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def evaluate_forecast(res, test_series):
    pred = res.get_prediction(start=test_series.index[0], end=test_series.index[-1], dynamic=False)
    forecast = pred.predicted_mean
    mae = mean_absolute_error(test_series, forecast)
    mse = mean_squared_error(test_series, forecast)
    return {
        'forecast': forecast,
        'mae': mae,
        'mse': mse,
        'aic': getattr(res, 'aic', np.nan)
    }

def simple_grid_search(train_series, p_range=(0,1), d_range=(0,1), q_range=(0,1),
                       P_range=(0,1), D_range=(0,1), Q_range=(0,1), s=12, max_models=50):
    best_res = None
    best_aic = np.inf
    tried = 0
    for p in range(p_range[0], p_range[1]+1):
        for d in range(d_range[0], d_range[1]+1):
            for q in range(q_range[0], q_range[1]+1):
                for P in range(P_range[0], P_range[1]+1):
                    for D in range(D_range[0], D_range[1]+1):
                        for Q in range(Q_range[0], Q_range[1]+1):
                            if tried >= max_models:
                                return best_res, best_aic
                            try:
                                res = fit_sarima(train_series, order=(p,d,q), seasonal_order=(P,D,Q,s))
                                tried += 1
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_res = res
                            except Exception:
                                continue
    return best_res, best_aic
