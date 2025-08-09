# utils.py
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

def load_and_prepare(file):
    """
    Загрузить CSV в pandas.DataFrame и привести имена колонок к верхнему регистру.
    Ожидает колонки X, Y, Z (любые регистры) — после этого они будут UPPERCASE.
    """
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper()
    # проверка необходимых колонок
    if not {'X','Y','Z'}.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: X, Y, Z (case-insensitive).")
    # приводим типы
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    df['Z'] = df['Z'].astype(float)
    return df

def median_nn_distance(df):
    """Вернуть медиану расстояния до ближайшего соседа (используется для подбора R)."""
    coords = df[['X','Y']].values
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # k=2: первый — сам, второй — ближайший сосед
    return float(np.median(dists[:,1]))