# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import json
import warnings

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import load_and_prepare, median_nn_distance
from eda import run_eda
from features import compute_geometric_features, find_local_minima
from clustering import run_kmeans, run_dbscan, run_agglomerative, run_gmm
from visualization import grid_surface, plot_heatmap_matplotlib, plot_3d_plotly, plot_quiver
from report import build_pdf_report







st.set_page_config(layout="wide", page_title="Кластеризация точек на поверхности земли")
st.title("Кластеризация точек на поверхности земли")













#
# Upload
#
uploaded = st.file_uploader("Upload CSV (X,Y,Z)", type=["csv"])
if not uploaded:
    st.info("Upload CSV with columns X,Y,Z (case-insensitive).")
    st.stop()

try:
    df = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

#
# Sidebar: global scale controls (these don't trigger heavy compute by themselves)
#

st.sidebar.header("Scale & features")
median_nn = median_nn_distance(df)
st.sidebar.write("Median nearest-neighbor distance:", round(median_nn, 6))

R_factor = st.sidebar.number_input("R multiplier (determine local neighborhood)", value=3.0, step=0.5)
R = float(R_factor * median_nn)

R_basin_factor = st.sidebar.number_input("R_basin multiplier", value=2.0, step=0.5)
R_basin = float(R_basin_factor * median_nn)

min_neighbors = int(st.sidebar.number_input("min neighbors fallback", value=8, min_value=3))
basin_frac = float(st.sidebar.slider("basin_frac_of_depth (thr=z + depth*frac)", 0.0, 1.0, 0.5))

st.sidebar.markdown("---")

#
# Compute features button (heavy work). store df_geo in session_state
#
if "df_geo" not in st.session_state:
    st.session_state.df_geo = None

compute = st.sidebar.button("Compute features & minima")
if compute:
    with st.spinner("Computing geometric features and local minima..."):
        try:
            geo = compute_geometric_features(df, R=R, R_basin=R_basin, min_neighbors=min_neighbors,
                                             basin_frac_of_depth=basin_frac)
            df_geo = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
            df_geo['IS_LOCAL_MIN'] = find_local_minima(df_geo, radius=R)
            st.session_state.df_geo = df_geo
            st.success("Computed and saved features to session.")
        except Exception as e:
            st.error(f"Error while computing features: {e}")
            st.stop()

# If already computed previously, use it
if st.session_state.df_geo is None:
    st.info("Press 'Compute features & minima' in the sidebar to compute features before clustering.")
    st.stop()

df_geo = st.session_state.df_geo

#
# Show basic table & minima count
#
st.subheader("Input + computed features (head)")
st.dataframe(df_geo.head())
n_min = int(df_geo['IS_LOCAL_MIN'].sum())
st.write("Found local minima:", n_min)
if n_min == 0:
    st.warning("No minima found. Try increasing R multiplier or check the input data.")
    st.stop()

# Prepare minima dataframe for clustering/visualization
mins = df_geo[df_geo['IS_LOCAL_MIN']].reset_index(drop=True)
# ensure orientation-based sin/cos features exist
if 'orientation' in mins.columns:
    mins['ori_sin'] = np.sin(mins['orientation'])
    mins['ori_cos'] = np.cos(mins['orientation'])
else:
    mins['ori_sin'] = 0.0
    mins['ori_cos'] = 0.0

#
# Clustering UI (controls). We will not recompute heavy features here.
#
st.sidebar.header("Clustering")
method = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative", "GMM"])

max_k = min(8, max(2, len(mins)))
default_k = min(3, max_k)
if method in ["KMeans", "Agglomerative", "GMM"]:
    k = st.sidebar.slider("k/components", min_value=2, max_value=max_k, value=default_k)
else:
    eps = st.sidebar.slider("DBSCAN eps", 0.01, 5.0, 0.7, step=0.01)
    min_samp = st.sidebar.slider("DBSCAN min_samples", 2, 30, 4)

default_feats = ['depth', 'basin_area_est', 'mean_grad', 'mean_curvature', 'gauss_curvature', 'ori_sin', 'ori_cos']
available_feats = [f for f in default_feats if f in mins.columns]
if not available_feats:
    st.error("No clustering features available. Check compute step.")
    st.stop()

feats = st.sidebar.multiselect("Features for clustering", options=available_feats,
                               default=available_feats[:min(5, len(available_feats))])
if not feats:
    st.error("Select at least one feature for clustering.")
    st.stop()

# Button to run clustering (so changing sliders doesn't auto-trigger heavy recompute)
if "cluster_result" not in st.session_state:
    st.session_state.cluster_result = None

cluster_run = st.sidebar.button("Run clustering")
if cluster_run:
    with st.spinner("Running clustering..."):
        try:
            X = mins[feats].fillna(0).values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            if method == "KMeans":
                labels, model = run_kmeans(Xs, n_clusters=k)
            elif method == "Agglomerative":
                labels, model = run_agglomerative(Xs, n_clusters=k)
            elif method == "GMM":
                labels, model = run_gmm(Xs, n_components=k)
            else:
                labels, model = run_dbscan(Xs, eps=eps, min_samples=min_samp)

            labels = np.asarray(labels)
            mins['cluster'] = labels
            st.session_state.cluster_result = {"labels": labels, "model": model, "feats": feats, "Xs": Xs}
            st.success("Clustering finished and saved to session.")
        except Exception as e:
            st.error(f"Clustering error: {e}")
            st.session_state.cluster_result = None

# If we have previous clustering result, display it
if st.session_state.cluster_result is None:
    st.info("Press 'Run clustering' to compute clusters with current parameters.")
    st.stop()

res = st.session_state.cluster_result
labels = res["labels"]
mins['cluster'] = labels

st.subheader("Clustering results")
st.write("Cluster counts:")
st.write(pd.Series(labels).value_counts())

# Silhouette (handle DBSCAN noise -1)
sil = None
try:
    unique_labels = set(labels.tolist())
    if len(unique_labels) > 1:
        if -1 in unique_labels:
            mask = (labels != -1)
            if mask.sum() >= 2 and len(set(labels[mask].tolist())) > 1:
                sil = silhouette_score(res["Xs"][mask], labels[mask])
        else:
            sil = silhouette_score(res["Xs"], labels)
except Exception:
    sil = None
st.write("Silhouette:", sil)

#
# Visualizations + optional watershed
#
XI, YI, ZI = grid_surface(df_geo, grid_res=200)

do_watershed = st.sidebar.checkbox("Compute watershed basins (accurate area)", value=False)
basin_areas = None
if do_watershed:
    try:
        from skimage.morphology import watershed
        from scipy import ndimage as ndi

        nan_mask = np.isnan(ZI)
        if nan_mask.any():
            ZI_copy = ZI.copy()
            indices = ndi.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            ZI_copy[nan_mask] = ZI_copy[tuple(indices[:, nan_mask])]
            ZI = ZI_copy

        inv = np.max(ZI) - ZI
        xi = np.linspace(df_geo['X'].min(), df_geo['X'].max(), XI.shape[1])
        yi = np.linspace(df_geo['Y'].min(), df_geo['Y'].max(), XI.shape[0])
        marker_mask = np.zeros_like(ZI, dtype=bool)
        for _, r in mins.iterrows():
            ix = int(np.searchsorted(xi, r['X']))
            iy = int(np.searchsorted(yi, r['Y']))
            ix = np.clip(ix, 0, XI.shape[1] - 1)
            iy = np.clip(iy, 0, XI.shape[0] - 1)
            marker_mask[iy, ix] = True
        markers, _ = ndi.label(marker_mask.astype(int))
        labels_ws = watershed(inv, markers=markers)
        unique, counts = np.unique(labels_ws, return_counts=True)
        dx = xi[1] - xi[0] if len(xi) > 1 else 0
        dy = yi[1] - yi[0] if len(yi) > 1 else 0
        area_pix = dx * dy
        basin_areas = {int(u): float(c * area_pix) for u, c in zip(unique, counts) if int(u) != 0}

        minima_labels = []
        for _, r in mins.iterrows():
            ix = int(np.searchsorted(xi, r['X']))
            iy = int(np.searchsorted(yi, r['Y']))
            ix = np.clip(ix, 0, XI.shape[1] - 1)
            iy = np.clip(iy, 0, XI.shape[0] - 1)
            minima_labels.append(int(labels_ws[iy, ix]))
        mins['basin_label'] = minima_labels
        mins['basin_area_ws'] = mins['basin_label'].map(lambda l: basin_areas.get(l, 0.0))
    except Exception as e:
        st.warning(f"Watershed failed: {e}")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("2D heatmap + clusters")

    # Используем ТУ ЖЕ палитру, что и в 3D
    selected_cmap = st.session_state.get('surface_palette', 'twilight')

    fig2d = plot_heatmap_matplotlib(
        XI, YI, ZI,
        minima_df=mins,
        clusters='cluster',
        title="Heatmap + minima clusters",
        cmap=selected_cmap  # <-- передаём выбранную палитру
    )
    st.pyplot(fig2d)
with col2:
    st.subheader("3D surface (interactive)")

    # График 3D строится выше
    fig3 = plot_3d_plotly(
        XI, YI, ZI,
        minima_df=mins,
        cluster_col='cluster',
        surface_colorscale=st.session_state.get('surface_palette', 'twilight'),
        cluster_palette=st.session_state.get('cluster_palette', 'Dark24')
    )
    st.plotly_chart(fig3, use_container_width=True)

    # === Элементы управления ПОД графиком ===
    with st.container():
        st.markdown("---")
        st.caption("🎨 Adjust color schemes:")

        # Выбор палитры для surface (влияет на 2D и 3D)
        surface_palette = st.selectbox(
            "Surface Color Palette",
            options=[
                'viridis', 'plasma', 'inferno', 'cividis', 'twilight',
                'hot', 'jet', 'rainbow', 'coolwarm', 'terrain', 'ocean'
            ],
            format_func=str.title,
            index=4,  # default: 'twilight'
            key='surface_palette'
        )

        # Выбор палитры для кластеров (только для точек)
        cluster_palette = st.selectbox(
            "Cluster Color Palette",
            options=[
                'Dark24', 'Set1', 'Plotly', 'Bold', 'Safe', 'Vivid',
                'Pastel1', 'Paired', 'Accent', 'Dark2'
            ],
            index=0,
            key='cluster_palette'
        )

# После успешной загрузки df
# st.subheader("Exploratory Data Analysis (EDA)")
# run_eda(df)
# figq = plot_quiver(mins, scale=1.0, nmax=200)
# if figq:
#     st.subheader("Gradient vectors (sample)")
#     st.pyplot(figq)

#
# Outputs: CSV, PDF, ZIP
#
csv_bytes = mins.to_csv(index=False).encode('utf-8')
st.download_button("Download minima CSV", data=csv_bytes, file_name="minima_with_features_clusters.csv", mime="text/csv")

imgs = []
# heatmap image
try:
    with io.BytesIO() as buf:
        fig2d.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_data = buf.getvalue()  # считываем ДО закрытия
        imgs.append(("heatmap.png", img_data))
    # buf автоматически закрывается при выходе из `with`
except Exception as e:
    st.error(f"Error saving heatmap: {e}")


# 3d surface image
try:
    png3 = fig3.to_image(format="png", width=900, height=600, scale=1)
    imgs.append(("3d_surface.png", png3))
except Exception:
    pass

summary_table = mins.groupby('cluster')[['depth', 'mean_grad', 'basin_area_est']].agg(['count', 'mean', 'std']).reset_index()
params = {"R": R, "R_basin": R_basin, "min_neighbors": min_neighbors, "basin_frac": basin_frac, "clustering_method": method}
try:
    pdf_bytes = build_pdf_report("Terrain minima clustering", params, {"Cluster summary": summary_table}, imgs)
    st.download_button("Download PDF report", data=pdf_bytes, file_name="terrain_report.pdf", mime="application/pdf")
except Exception as e:
    st.warning(f"PDF generation failed: {e}")

zipbuf = io.BytesIO()
with zipfile.ZipFile(zipbuf, "w") as zf:
    zf.writestr("minima.csv", csv_bytes)
    for fname, bts in imgs:
        zf.writestr(fname, bts)
    zf.writestr("params.json", json.dumps(params, indent=2))
zipbuf.seek(0)
st.download_button("Download results ZIP", data=zipbuf.getvalue(), file_name="results.zip", mime="application/zip")

st.success("All done. You can download CSV / PDF / ZIP.")
