# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from numpy.linalg import lstsq
from scipy.interpolate import griddata
import math
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import io

sns.set()


# ---- Helper functions ----
def fit_quadratic_local(x, y, z):
    A = np.column_stack([x ** 2, y ** 2, x * y, x, y, np.ones_like(x)])
    coef, *_ = lstsq(A, z, rcond=None)
    return coef


def surface_curvatures_from_coeffs(A, B, C, D, E, F):
    zx = D;
    zy = E
    z_xx = 2.0 * A;
    z_yy = 2.0 * B;
    z_xy = C
    denom_H = 2.0 * (1.0 + zx * zx + zy * zy) ** 1.5
    H = ((1 + zy * zy) * z_xx - 2 * zx * zy * z_xy + (1 + zx * zx) * z_yy) / denom_H if denom_H != 0 else 0.0
    denom_K = (1.0 + zx * zx + zy * zy) ** 2
    K = (z_xx * z_yy - z_xy * z_xy) / denom_K if denom_K != 0 else 0.0
    return zx, zy, H, K


def find_local_minima(df, radius=None, k=None):
    coords = df[['X', 'Y']].values
    zs = df['Z'].values
    n = len(df)
    is_min = np.zeros(n, dtype=bool)

    if radius is not None:
        tree = cKDTree(coords)
        neighbors = tree.query_ball_point(coords, r=radius)
        for i in range(n):
            # Include only neighbors strictly within radius (excluding self)
            neigh = [j for j in neighbors[i] if j != i and j < n]
            if len(neigh) > 0:
                is_min[i] = np.all(zs[i] < zs[neigh])
    else:
        tree = cKDTree(coords)
        dist, idx = tree.query(coords, k=k + 1)
        for i in range(n):
            neigh = idx[i][1:]
            is_min[i] = np.all(zs[i] < zs[neigh])
    return is_min


def compute_geometric_features(df, R, R_basin=None, min_neighbors=6, basin_frac_of_depth=0.5, basin_threshold=None):
    if R_basin is None:
        R_basin = R

    # Ensure we have enough points
    if len(df) < 3:
        raise ValueError("Need at least 3 points for feature computation")

    coords = df[['X', 'Y']].values
    zs = df['Z'].values
    n = len(df)

    # Handle edge cases for R_basin
    if R_basin <= 0:
        R_basin = R

    tree = cKDTree(coords)
    depth = np.zeros(n)
    basin_count = np.zeros(n, dtype=int)
    basin_area_est = np.zeros(n)
    mean_grad = np.zeros(n)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    mean_curv = np.zeros(n)
    gauss_curv = np.zeros(n)
    orientation = np.zeros(n)

    # Precompute neighbors
    neighbors_R = tree.query_ball_point(coords, r=R)
    neighbors_Rb = tree.query_ball_point(coords, r=R_basin)
    area_circle = math.pi * (R_basin ** 2)

    for i in range(n):
        # Get neighbors within R (excluding self)
        neigh = [j for j in neighbors_R[i] if j != i and j < n]

        # Handle insufficient neighbors
        if len(neigh) < max(3, min_neighbors):
            # Get more neighbors if available
            k_needed = max(min_neighbors + 1, 4)
            dists_k, idx_k = tree.query(coords[i], k=min(k_needed, n))
            idx_k = np.array(idx_k) if np.ndim(idx_k) > 0 else np.array([idx_k])
            neigh = [j for j in idx_k if j != i and j < n][:max(3, min_neighbors)]

        # Calculate depth
        if len(neigh) > 0:
            neigh_z = zs[neigh]
            depth_i = np.mean(neigh_z) - zs[i]
            depth[i] = max(0, depth_i)  # Ensure non-negative
        else:
            depth[i] = 0.0

        # Calculate basin properties
        thr = zs[i] + depth[i] * basin_frac_of_depth if basin_threshold is None else basin_threshold
        idx_rb = [j for j in neighbors_Rb[i] if j != i and j < n]
        basin_idx = [j for j in idx_rb if zs[j] <= thr]
        basin_count[i] = len(basin_idx)
        total_in_circle = len(idx_rb)

        if total_in_circle > 0:
            basin_area_est[i] = basin_count[i] * (area_circle / total_in_circle)
        else:
            basin_area_est[i] = 0.0

        # Prepare local coordinates for surface fitting
        if len(neigh) > 0:
            xs = coords[neigh, 0] - coords[i, 0]
            ys = coords[neigh, 1] - coords[i, 1]
            zs_fit = zs[neigh]

            # Add the central point
            xs_fit = np.append(xs, 0.0)
            ys_fit = np.append(ys, 0.0)
            zs_fit = np.append(zs_fit, zs[i])

            try:
                A, B, C, D, E, F = fit_quadratic_local(xs_fit, ys_fit, zs_fit)
            except:
                # Fallback to linear fit
                Xlin = np.column_stack([xs_fit, ys_fit, np.ones_like(xs_fit)])
                a_lin, b_lin, c_lin = lstsq(Xlin, zs_fit, rcond=None)[0]
                A = B = C = 0.0
                D, E, F = a_lin, b_lin, c_lin

            zx, zy, H, K = surface_curvatures_from_coeffs(A, B, C, D, E, F)
            grad_x[i] = zx
            grad_y[i] = zy
            mean_grad[i] = np.sqrt(zx * zx + zy * zy)
            mean_curv[i] = H
            gauss_curv[i] = K
            orientation[i] = math.atan2(zy, zx)
        else:
            # Default values when no neighbors
            grad_x[i] = grad_y[i] = mean_grad[i] = 0.0
            mean_curv[i] = gauss_curv[i] = 0.0
            orientation[i] = 0.0

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


def make_grid_surface(df, grid_res=120):
    xi = np.linspace(df['X'].min(), df['X'].max(), grid_res)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((df['X'], df['Y']), df['Z'], (XI, YI), method='linear')
    return XI, YI, ZI


# ---- Streamlit UI ----
st.title("Terrain minima clustering — demo")
st.markdown("Загрузите CSV с колонками X,Y,Z")

uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Convert columns to uppercase for consistency
    df.columns = [col.upper() for col in df.columns]

    # Check required columns
    if not {'X', 'Y', 'Z'}.issubset(df.columns):
        st.error("CSV must contain columns X, Y, Z (case insensitive)")
        st.stop()

    # Check minimum data points
    if len(df) < 3:
        st.error("Need at least 3 points in the dataset")
        st.stop()

    st.write("Входные данные:", df.shape)
    st.dataframe(df.head())

    # Compute default radius based on nearest neighbors
    coords = df[['X', 'Y']].values
    tree = cKDTree(coords)
    dists_k, _ = tree.query(coords, k=2)
    median_nn = np.median(dists_k[:, 1])

    # Sidebar inputs with validation
    R_default = float(st.sidebar.number_input(
        "R (local radius)",
        value=max(0.001, 3.0 * median_nn),
        min_value=0.001
    ))
    R_basin = float(st.sidebar.number_input(
        "R_basin",
        value=max(0.001, 2.0 * R_default),
        min_value=0.001
    ))
    min_neighbors = int(st.sidebar.number_input(
        "min_neighbors",
        value=8,
        min_value=3
    ))
    basin_frac = float(st.sidebar.number_input(
        "basin_frac_of_depth",
        value=0.5,
        min_value=0.0,
        max_value=1.0
    ))

    if st.sidebar.button("Compute features"):
        with st.spinner("Computing features..."):
            try:
                geo = compute_geometric_features(
                    df,
                    R=R_default,
                    R_basin=R_basin,
                    min_neighbors=min_neighbors,
                    basin_frac_of_depth=basin_frac
                )
                df_geo = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
                st.success("Features computed successfully")
                st.dataframe(df_geo.head())

                # Find local minima
                df_geo['is_local_min'] = find_local_minima(df_geo, radius=R_default)
                mins = df_geo[df_geo['is_local_min']].copy().reset_index(drop=True)
                st.write("Найдено минимумов:", len(mins))

                if len(mins) == 0:
                    st.warning("Нет минимумов — попробуйте увеличить R или поменять критерий")
                else:
                    # Prepare features for clustering
                    mins['ori_sin'] = np.sin(mins['orientation'])
                    mins['ori_cos'] = np.cos(mins['orientation'])
                    feat_cols = [
                        'depth', 'basin_area_est', 'mean_grad',
                        'mean_curvature', 'gauss_curvature',
                        'ori_sin', 'ori_cos'
                    ]

                    # Handle missing values
                    X = mins[feat_cols].fillna(0).values
                    if len(X) == 0:
                        st.error("No valid data points for clustering")
                        st.stop()

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Clustering method selection
                    method = st.sidebar.selectbox(
                        "Clustering method",
                        ["KMeans", "DBSCAN", "Agglomerative", "GMM"]
                    )

                    if method == "KMeans":
                        k = st.sidebar.slider("K", 2, min(8, len(mins) - 1), 3)
                        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                        mins['cluster'] = km.labels_
                        if len(set(km.labels_)) > 1:
                            s = silhouette_score(X_scaled, km.labels_)
                            st.write("Silhouette:", round(s, 3))
                        else:
                            st.write("Silhouette: N/A (only one cluster)")

                    elif method == "DBSCAN":
                        eps = st.sidebar.slider("eps", 0.1, 2.0, 0.7)
                        min_s = st.sidebar.slider("min_samples", 2, 10, 4)
                        db = DBSCAN(eps=eps, min_samples=min_s).fit(X_scaled)
                        mins['cluster'] = db.labels_
                        labels = db.labels_
                        unique_labels = set(labels)
                        st.write(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters")

                        if len(unique_labels - {-1}) > 1:
                            mask = labels != -1
                            s = silhouette_score(X_scaled[mask], labels[mask])
                            st.write("Silhouette (non-noise):", round(s, 3))

                    elif method == "Agglomerative":
                        k = st.sidebar.slider("K", 2, min(8, len(mins) - 1), 3)
                        ag = AgglomerativeClustering(n_clusters=k).fit(X_scaled)
                        mins['cluster'] = ag.labels_

                    else:  # GMM
                        k = st.sidebar.slider("K", 2, min(8, len(mins) - 1), 3)
                        gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
                        mins['cluster'] = gmm.predict(X_scaled)

                    # Display cluster summary
                    st.subheader("Clusters summary")
                    summ = mins.groupby('cluster')[feat_cols].agg(['count', 'mean', 'std'])
                    st.dataframe(summ)

                    # 3D Visualization
                    XI, YI, ZI = make_grid_surface(df, grid_res=160)
                    fig = go.Figure()

                    # Add terrain surface
                    fig.add_trace(go.Surface(
                        x=XI,
                        y=YI,
                        z=ZI,
                        colorscale='Earth',
                        opacity=0.9,
                        showscale=False,
                        name="Terrain"
                    ))

                    # Add minima points
                    palette = px.colors.qualitative.Plotly
                    for cl in sorted(mins['cluster'].unique()):
                        m = mins[mins['cluster'] == cl]
                        color = 'lightgray' if cl == -1 else palette[int(cl) % len(palette)]
                        name = 'noise' if cl == -1 else f'Cluster {cl}'

                        fig.add_trace(go.Scatter3d(
                            x=m['X'],
                            y=m['Y'],
                            z=m['Z'] + 0.02 * (df['Z'].max() - df['Z'].min()),
                            mode='markers',
                            marker=dict(size=4, color=color),
                            name=name
                        ))

                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z',
                            aspectmode='data'
                        ),
                        margin=dict(l=0, r=0, b=0, t=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Download button
                    to_download = mins.copy()
                    csv_bytes = to_download.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download minima with clusters",
                        data=csv_bytes,
                        file_name="minima_clusters.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error during computation: {str(e)}")
                st.exception(e)