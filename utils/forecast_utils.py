# utils/forecast_utils.py
import numpy as np
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
    return {'forecast': forecast, 'mae': mae, 'mse': mse, 'aic': getattr(res, 'aic', np.nan)}
