# utils/visualization.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_time_series(df: pd.DataFrame, water_col: str, title="Water level time series"):
    fig = px.line(df, x=df.index, y=water_col, labels={'x':'Date', water_col: water_col}, title=title)
    return fig

def plot_with_flood_markers(df: pd.DataFrame, water_col: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[water_col], mode='lines', name='Water level'))
    flood_idx = df[df['is_flood']].index
    if len(flood_idx) > 0:
        fig.add_trace(go.Scatter(x=flood_idx, y=df.loc[flood_idx, water_col], mode='markers', name='Floods',
                                 marker=dict(size=8, symbol='circle')))
    fig.update_layout(xaxis_title='Date', yaxis_title=water_col, title='Water level with flood markers')
    return fig

def plot_bar_yearly(series_or_df, x_label='Year', y_label='Value', title=''):
    # series_or_df can be pandas Series or DataFrame
    if isinstance(series_or_df, pd.Series):
        fig = px.bar(x=series_or_df.index, y=series_or_df.values, labels={'x':x_label,'y':y_label}, title=title)
    else:
        fig = go.Figure()
        for c in series_or_df.columns:
            fig.add_trace(go.Bar(x=series_or_df.index, y=series_or_df[c], name=c))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig
