# utils/data_utils.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional

def find_col_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for k in keywords:
        for i, c in enumerate(cols):
            if k in c:
                return columns[i]
    return None

def load_csv(file_like, encoding='latin1'):
    try:
        df = pd.read_csv(file_like, encoding=encoding, low_memory=False)
    except Exception:
        df = pd.read_csv(file_like, encoding='utf-8', low_memory=False)
    return df

def process_df(df: pd.DataFrame,
               date_col: Optional[str]=None,
               water_col: Optional[str]=None,
               area_col: Optional[str]=None,
               damage_cols_candidates: Optional[List[str]]=None,
               interp_method='linear',
               zscore_outlier_thresh=3.0,
               flood_zscore_thresh=1.5,
               flood_threshold_multiplier=1.0):
    df = df.copy()
    cols = df.columns.tolist()

    # Auto-detect
    if not date_col:
        date_col = find_col_by_keywords(cols, ['date','datetime','time','day'])
    if not water_col:
        water_col = find_col_by_keywords(cols, ['water','level','wl','depth','height'])
    if not area_col:
        area_col = find_col_by_keywords(cols, ['barangay','brgy','area','location','sitio'])

    # Combine date columns if present
    if date_col and 'Day' in df.columns and 'Year' in df.columns and 'Date' in df.columns:
        df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
        date_col = '__combined_date'
    elif date_col is None:
        df['__date_autogen'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        date_col = '__date_autogen'

    # Parse date and index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.index = pd.DatetimeIndex(df[date_col])

    if water_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric column found to use as water level.")
        water_col = numeric_cols[0]

    df[water_col] = pd.to_numeric(df[water_col], errors='coerce')
    df[water_col] = df[water_col].interpolate(method=interp_method, limit_direction='both')

    # zscore and heuristics
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

    # damage columns
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
