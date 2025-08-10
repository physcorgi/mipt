# features.py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numpy.linalg import lstsq
from math import atan2, pi

EPS = 1e-12


def fit_quadratic_local(x, y, z):
    """
    Аппроксимация квадратичной поверхностью:
      z = A*x^2 + B*y^2 + C*x*y + D*x + E*y + F
    Возвращает вектор коэффициентов [A,B,C,D,E,F]
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    A_mat = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
    coef, *_ = lstsq(A_mat, z, rcond=None)
    return coef  # A,B,C,D,E,F


def surface_curvatures_from_coeffs(A, B, C, D, E, F):
    """
    Вычисляет локальные производные и среднюю/гауссову кривизну
    из коэффициентов квадратичной аппроксимации (в точке 0,0 локальной системы).
    """
    zx = D
    zy = E
    z_xx = 2.0 * A
    z_yy = 2.0 * B
    z_xy = C

    denom_H = 2.0 * (1.0 + zx*zx + zy*zy)**1.5
    denom_K = (1.0 + zx*zx + zy*zy)**2

    # защитимся от деления на ноль
    if abs(denom_H) < EPS:
        H = 0.0
    else:
        H = ((1 + zy*zy) * z_xx - 2*zx*zy*z_xy + (1 + zx*zx) * z_yy) / denom_H

    if abs(denom_K) < EPS:
        K = 0.0
    else:
        K = (z_xx * z_yy - z_xy * z_xy) / denom_K

    return zx, zy, H, K


def find_local_minima(df, radius=None, k=None):
    """
    Возвращает булев массив, указывающий, является ли точка локальным минимумом
    относительно радиуса radius или k ближайших соседей.
    """
    coords = df[['X', 'Y']].values
    zs = df['Z'].values
    tree = cKDTree(coords)
    n = len(df)
    is_min = np.zeros(n, dtype=bool)

    if radius is not None:
        neighbors = tree.query_ball_point(coords, r=radius)
        for i in range(n):
            neigh = [j for j in neighbors[i] if j != i]
            if len(neigh) == 0:
                is_min[i] = False
            else:
                is_min[i] = np.all(zs[i] < zs[neigh])
    elif k is not None:
        kq = int(max(1, min(k + 1, n)))  # +1 т.к. включается сама точка
        dists, idx = tree.query(coords, k=kq)
        # гарантируем, что idx -> 2D массив
        idx = np.atleast_2d(idx)
        for i in range(n):
            neigh = idx[i].ravel().tolist()
            # исключаем саму точку
            neigh = [j for j in neigh if int(j) != i]
            if len(neigh) == 0:
                is_min[i] = False
            else:
                neigh = np.array(neigh, dtype=int)
                is_min[i] = np.all(zs[i] < zs[neigh])
    else:
        raise ValueError("Specify radius or k")
    return is_min


def compute_geometric_features(df, R, R_basin=None, min_neighbors=6,
                               basin_frac_of_depth=0.5, basin_threshold=None, basin_condition='below'):
    """
    Возвращает DataFrame c признаками:
      depth, basin_count, basin_area_est, mean_grad, grad_x, grad_y, mean_curvature, gauss_curvature, orientation
    """
    if R_basin is None:
        R_basin = R

    coords = df[['X', 'Y']].values
    zs = df['Z'].values
    n = len(df)
    tree = cKDTree(coords)

    depth = np.zeros(n, dtype=float)
    basin_count = np.zeros(n, dtype=int)
    basin_area_est = np.zeros(n, dtype=float)
    mean_grad = np.zeros(n, dtype=float)
    grad_x = np.zeros(n, dtype=float)
    grad_y = np.zeros(n, dtype=float)
    mean_curv = np.zeros(n, dtype=float)
    gauss_curv = np.zeros(n, dtype=float)
    orientation = np.zeros(n, dtype=float)

    # Дополнительные многошкальные и морфометрические признаки
    depth_r2 = np.zeros(n, dtype=float)
    mean_grad_r2 = np.zeros(n, dtype=float)
    grad_x_r2 = np.zeros(n, dtype=float)
    grad_y_r2 = np.zeros(n, dtype=float)
    mean_curv_r2 = np.zeros(n, dtype=float)
    gauss_curv_r2 = np.zeros(n, dtype=float)
    k1 = np.zeros(n, dtype=float)
    k2 = np.zeros(n, dtype=float)
    shape_index = np.zeros(n, dtype=float)
    curvedness = np.zeros(n, dtype=float)
    shape_index_r2 = np.zeros(n, dtype=float)
    curvedness_r2 = np.zeros(n, dtype=float)

    neighbors_R = tree.query_ball_point(coords, r=R)
    neighbors_Rb = tree.query_ball_point(coords, r=R_basin)
    area_circle = pi * (R_basin ** 2)

    for i in range(n):
        neigh = [j for j in neighbors_R[i] if j != i]

        # fallback: если мало соседей, возьмём k ближайших (min_neighbors)
        if len(neigh) < max(3, min_neighbors):
            requested_k = max(min_neighbors + 1, len(neigh) + 1)
            requested_k = min(requested_k, n)
            dists_k, idx_k = tree.query(coords[i], k=requested_k)
            idx_k = np.atleast_1d(idx_k).ravel().tolist()
            neigh = [j for j in idx_k if int(j) != i][:max(3, min_neighbors)]

        neigh = np.array(neigh, dtype=int)
        if neigh.size == 0:
            neigh_z = np.array([zs[i]])
        else:
            neigh_z = zs[neigh]

        depth_i = np.mean(neigh_z) - zs[i]
        depth[i] = depth_i if depth_i > 0 else 0.0

        if basin_threshold is None:
            thr = zs[i] + depth_i * basin_frac_of_depth
        else:
            thr = basin_threshold

        idx_rb = [j for j in neighbors_Rb[i] if j != i]
        if basin_condition == 'below':
            basin_idx = [j for j in idx_rb if zs[j] <= thr]
        else:
            basin_idx = [j for j in idx_rb if zs[j] >= thr]

        basin_count[i] = len(basin_idx)
        total_in_circle = len(idx_rb)
        basin_area_est[i] = (basin_count[i] * (area_circle / total_in_circle)) if total_in_circle > 0 else 0.0

        # подготовим точки для локальной аппроксимации (локальная система с центром в i)
        if neigh.size > 0:
            xs = coords[neigh, 0] - coords[i, 0]
            ys = coords[neigh, 1] - coords[i, 1]
            zs_fit = neigh_z.copy()
        else:
            xs = np.array([0.0])
            ys = np.array([0.0])
            zs_fit = np.array([zs[i]])

        # всегда добавляем центральную точку (0,0,z_i) чтобы аппроксимация была стабильнее
        xs_fit = np.append(xs, 0.0)
        ys_fit = np.append(ys, 0.0)
        zs_fit = np.append(zs_fit, zs[i])

        # если точек для квадратичной аппроксимации меньше 6, попробуем линейную модель
        try:
            if xs_fit.size >= 6:
                A_coef, B_coef, C_coef, D_coef, E_coef, F_coef = fit_quadratic_local(xs_fit, ys_fit, zs_fit)
            else:
                # линейная аппроксимация z = a*x + b*y + c
                Xlin = np.column_stack([xs_fit, ys_fit, np.ones_like(xs_fit)])
                a_lin, b_lin, c_lin = lstsq(Xlin, zs_fit, rcond=None)[0]
                A_coef = B_coef = C_coef = 0.0
                D_coef, E_coef, F_coef = a_lin, b_lin, c_lin
        except Exception:
            # в крайнем случае — вернём нули
            A_coef = B_coef = C_coef = D_coef = E_coef = F_coef = 0.0

        zx, zy, H, K = surface_curvatures_from_coeffs(A_coef, B_coef, C_coef, D_coef, E_coef, F_coef)
        grad_x[i] = zx
        grad_y[i] = zy
        mean_grad[i] = np.sqrt(zx*zx + zy*zy)
        mean_curv[i] = H
        gauss_curv[i] = K
        orientation[i] = atan2(zy, zx) if not (np.isnan(zx) or np.isnan(zy)) else 0.0

        # Главные кривизны, индекс формы и кривизна (curvedness)
        disc = H*H - K
        if disc < 0:
            disc = 0.0
        root = np.sqrt(disc)
        k1_i = H + root
        k2_i = H - root
        # Упорядочим так, чтобы |k2| >= |k1| (не строго критично)
        if abs(k1_i) > abs(k2_i):
            k1_i, k2_i = k2_i, k1_i
        k1[i] = k1_i
        k2[i] = k2_i
        denom = (k2_i - k1_i)
        if abs(denom) < EPS:
            shape_index[i] = 0.0
        else:
            shape_index[i] = (2.0 / pi) * np.arctan((k2_i + k1_i) / denom)
        curvedness[i] = np.sqrt(0.5 * (k1_i*k1_i + k2_i*k2_i))

        # Многошкальная оценка (используем R_basin как вторую шкалу)
        idx_rb = [j for j in neighbors_Rb[i] if j != i]
        if len(idx_rb) < max(3, min_neighbors):
            requested_k2 = max(min_neighbors + 1, len(idx_rb) + 1)
            requested_k2 = min(requested_k2, n)
            d2, id2 = tree.query(coords[i], k=requested_k2)
            id2 = np.atleast_1d(id2).ravel().tolist()
            idx_rb = [j for j in id2 if int(j) != i][:max(3, min_neighbors)]
        idx_rb = np.array(idx_rb, dtype=int)
        if idx_rb.size > 0:
            neigh_z_rb = zs[idx_rb]
            depth2 = np.mean(neigh_z_rb) - zs[i]
            depth_r2[i] = depth2 if depth2 > 0 else 0.0
            xs2 = coords[idx_rb, 0] - coords[i, 0]
            ys2 = coords[idx_rb, 1] - coords[i, 1]
            zs2 = neigh_z_rb.copy()
            xs2 = np.append(xs2, 0.0)
            ys2 = np.append(ys2, 0.0)
            zs2 = np.append(zs2, zs[i])
            try:
                if xs2.size >= 6:
                    A2, B2, C2, D2, E2, F2 = fit_quadratic_local(xs2, ys2, zs2)
                else:
                    Xlin2 = np.column_stack([xs2, ys2, np.ones_like(xs2)])
                    a2, b2, c2 = lstsq(Xlin2, zs2, rcond=None)[0]
                    A2 = B2 = C2 = 0.0
                    D2, E2, F2 = a2, b2, c2
            except Exception:
                A2 = B2 = C2 = D2 = E2 = F2 = 0.0
            zx2, zy2, H2, K2 = surface_curvatures_from_coeffs(A2, B2, C2, D2, E2, F2)
            grad_x_r2[i] = zx2
            grad_y_r2[i] = zy2
            mean_grad_r2[i] = np.sqrt(zx2*zx2 + zy2*zy2)
            mean_curv_r2[i] = H2
            gauss_curv_r2[i] = K2
            disc2 = H2*H2 - K2
            if disc2 < 0:
                disc2 = 0.0
            root2 = np.sqrt(disc2)
            k1_2 = H2 + root2
            k2_2 = H2 - root2
            if abs(k1_2) > abs(k2_2):
                k1_2, k2_2 = k2_2, k1_2
            denom2 = (k2_2 - k1_2)
            if abs(denom2) < EPS:
                shape_index_r2[i] = 0.0
                curvedness_r2[i] = np.sqrt(0.5 * (k1_2*k1_2 + k2_2*k2_2))
            else:
                shape_index_r2[i] = (2.0 / pi) * np.arctan((k2_2 + k1_2) / denom2)
                curvedness_r2[i] = np.sqrt(0.5 * (k1_2*k1_2 + k2_2*k2_2))

    out = pd.DataFrame({
        'depth': depth,
        'basin_count': basin_count,
        'basin_area_est': basin_area_est,
        'mean_grad': mean_grad,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'mean_curvature': mean_curv,
        'gauss_curvature': gauss_curv,
        'orientation': orientation,
        # морфометрия и многошкальные признаки
        'k1': k1,
        'k2': k2,
        'shape_index': shape_index,
        'curvedness': curvedness,
        'depth_r2': depth_r2,
        'mean_grad_r2': mean_grad_r2,
        'grad_x_r2': grad_x_r2,
        'grad_y_r2': grad_y_r2,
        'mean_curvature_r2': mean_curv_r2,
        'gauss_curvature_r2': gauss_curv_r2,
        'shape_index_r2': shape_index_r2,
        'curvedness_r2': curvedness_r2
    }, index=df.index)

    # уберём NaN/inf, если что
    out = out.fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    return out



