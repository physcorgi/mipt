# features.py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numpy.linalg import lstsq
from math import atan2, pi

def fit_quadratic_local(x, y, z):
    A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
    coef, *_ = lstsq(A, z, rcond=None)
    return coef  # A,B,C,D,E,F

def surface_curvatures_from_coeffs(A,B,C,D,E,F):
    zx = D; zy = E
    z_xx = 2.0*A; z_yy = 2.0*B; z_xy = C
    denom_H = 2.0 * (1.0 + zx*zx + zy*zy)**1.5
    H = ((1+zy*zy)*z_xx - 2*zx*zy*z_xy + (1+zx*zx)*z_yy) / denom_H if denom_H != 0 else 0.0
    denom_K = (1.0 + zx*zx + zy*zy)**2
    K = (z_xx*z_yy - z_xy*z_xy) / denom_K if denom_K != 0 else 0.0
    return zx, zy, H, K

def find_local_minima(df, radius=None, k=None):
    coords = df[['X','Y']].values
    zs = df['Z'].values
    tree = cKDTree(coords)
    n = len(df)
    is_min = np.zeros(n, dtype=bool)
    if radius is not None:
        neighbors = tree.query_ball_point(coords, r=radius)
        for i in range(n):
            neigh = [j for j in neighbors[i] if j != i]
            is_min[i] = (len(neigh)>0) and np.all(zs[i] < zs[neigh])
    elif k is not None:
        dists, idx = tree.query(coords, k=k+1)
        for i in range(n):
            neigh = idx[i][1:]
            is_min[i] = np.all(zs[i] < zs[neigh])
    else:
        raise ValueError("Specify radius or k")
    return is_min

def compute_geometric_features(df, R, R_basin=None, min_neighbors=6, basin_frac_of_depth=0.5, basin_threshold=None, basin_condition='below'):
    """
    Возвращает DataFrame c признаками:
      depth, basin_count, basin_area_est, mean_grad, grad_x, grad_y, mean_curvature, gauss_curvature, orientation
    """
    if R_basin is None:
        R_basin = R
    coords = df[['X','Y']].values
    zs = df['Z'].values
    n = len(df)
    tree = cKDTree(coords)

    depth = np.zeros(n)
    basin_count = np.zeros(n, dtype=int)
    basin_area_est = np.zeros(n, dtype=float)
    mean_grad = np.zeros(n)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    mean_curv = np.zeros(n)
    gauss_curv = np.zeros(n)
    orientation = np.zeros(n)

    neighbors_R = tree.query_ball_point(coords, r=R)
    neighbors_Rb = tree.query_ball_point(coords, r=R_basin)
    area_circle = pi * (R_basin**2)

    for i in range(n):
        neigh = [j for j in neighbors_R[i] if j != i]
        if len(neigh) < max(3, min_neighbors):
            dists_k, idx_k = tree.query(coords[i], k=max(min_neighbors+1, len(neigh)+1))
            idx_k = np.array(idx_k).tolist() if np.ndim(idx_k)>0 else [idx_k]
            neigh = [j for j in idx_k if j != i][:max(3, min_neighbors)]

        neigh_z = zs[neigh] if len(neigh)>0 else np.array([zs[i]])
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

        xs = coords[neigh,0] - coords[i,0] if len(neigh)>0 else np.array([0.0])
        ys = coords[neigh,1] - coords[i,1] if len(neigh)>0 else np.array([0.0])
        zs_fit = np.append(neigh_z, zs[i])
        xs_fit = np.append(xs, 0.0); ys_fit = np.append(ys, 0.0)
        try:
            A,B,C,D,E,F = fit_quadratic_local(xs_fit, ys_fit, zs_fit)
        except Exception:
            Xlin = np.column_stack([xs_fit, ys_fit, np.ones_like(xs_fit)])
            a_lin, b_lin, c_lin = lstsq(Xlin, zs_fit, rcond=None)[0]
            A=B=C=0.0; D,E,F = a_lin, b_lin, c_lin

        zx, zy, H, K = surface_curvatures_from_coeffs(A,B,C,D,E,F)
        grad_x[i] = zx; grad_y[i] = zy
        mean_grad[i] = np.sqrt(zx*zx + zy*zy)
        mean_curv[i] = H; gauss_curv[i] = K
        orientation[i] = atan2(zy, zx)

    out = pd.DataFrame({
        'depth': depth,
        'basin_count': basin_count,
        'basin_area_est': basin_area_est,
        'mean_grad': mean_grad,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'mean_curvature': mean_curv,
        'gauss_curvature': gauss_curv,
        'orientation': orientation
    }, index=df.index)
    return out
