# utils.py
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import warnings
from typing import Union, IO

def load_and_prepare(file: Union[str, bytes, IO]):
    """
    Загрузить CSV в pandas.DataFrame и привести имена колонок к верхнему регистру (и убрать пробелы).
    Ожидает колонки X, Y, Z (любые регистры) — после этого они будут UPPERCASE.
    Возвращает DataFrame с как минимум колонками X,Y,Z типа float и сброшенным индексом.
    """
    # pandas.read_csv поддерживает путь или file-like (Streamlit UploadedFile)
    df = pd.read_csv(file)
    # убрать пробелы вокруг имён и привести к верхнему регистру
    df.columns = df.columns.str.strip().str.upper()

    # проверка необходимых колонок
    if not {'X', 'Y', 'Z'}.issubset(set(df.columns)):
        raise ValueError("Загрузите CSV-файл со столбцами X,Y,Z (без учета регистра).")

    # оставим только нужные колонки (но можно расширить при необходимости)
    df = df.loc[:, ['X', 'Y', 'Z']].copy()

    # привести типы и отбросить строки с NaN в координатах
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['X', 'Y', 'Z']).reset_index(drop=True)
    after = len(df)
    if after < before:
        warnings.warn(f"Dropped {before - after} rows with non-numeric or missing X/Y/Z values.")

    if len(df) == 0:
        raise ValueError("После предобработки не осталось строк с корректными X,Y,Z.")

    return df


def median_nn_distance(df: pd.DataFrame) -> float:
    """
    Вернуть медиану расстояния до ближайшего соседа (используется для подбора R).
    Если в df < 2 точек — возвращает небольшое положительное значение (1e-6) и предупреждает.
    """
    coords = df[['X', 'Y']].values
    n = len(coords)
    if n < 2:
        warnings.warn("Need at least 2 points to compute nearest-neighbour distance. Returning small default.")
        return float(1e-6)

    tree = cKDTree(coords)
    # k=2: первый — сам, второй — ближайший сосед
    dists, _ = tree.query(coords, k=2)
    # dists shape (n,2); берем вторую колонку
    nn = dists[:, 1]
    med = float(np.median(nn))

    # если медиана нулевая (например, есть дубли точек), вернём небольшой положительный
    if med <= 0.0:
        warnings.warn("Median nearest-neighbor distance is zero or non-positive; returning small default (1e-6).")
        return float(1e-6)

    return med
