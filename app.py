# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, json

from sklearn.metrics import silhouette_score

from utils import load_and_prepare, median_nn_distance
from eda import run_eda
from features import compute_geometric_features, find_local_minima
from clustering import run_kmeans, run_dbscan, run_agglomerative, run_gmm
from visualization import grid_surface, plot_heatmap_matplotlib, plot_3d_plotly, plot_quiver
from report import build_pdf_report

st.set_page_config(layout="wide", page_title="Terrain Minima Clustering (modular)")
st.title("Terrain Minima Clustering — Modular Service")

uploaded = st.file_uploader("Upload CSV (X,Y,Z)", type=["csv"])
if not uploaded:
    st.info("Upload CSV with columns X,Y,Z (case-insensitive).")
    st.stop()

try:
    df = load_and_prepare(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls
st.sidebar.header("Scale & features")
median_nn = median_nn_distance(df)
st.sidebar.write("Median nearest-neighbor distance:", round(median_nn,6))
R_factor = st.sidebar.number_input("R multiplier (determine local neighborhood)", value=3.0, step=0.5)
R = float(R_factor * median_nn)
R_basin_factor = st.sidebar.number_input("R_basin multiplier", value=2.0, step=0.5)
R_basin = float(R_basin_factor * median_nn)
min_neighbors = int(st.sidebar.number_input("min neighbors fallback", value=8, min_value=3))
basin_frac = float(st.sidebar.slider("basin_frac_of_depth (thr=z + depth*frac)", 0.0, 1.0, 0.5))
st.sidebar.markdown("---")

compute = st.sidebar.button("Compute features & minima")
if compute:
    with st.spinner("Computing..."):
        geo = compute_geometric_features(df, R=R, R_basin=R_basin, min_neighbors=min_neighbors, basin_frac_of_depth=basin_frac)
        df_geo = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
        df_geo['IS_LOCAL_MIN'] = find_local_minima(df_geo, radius=R)
        st.success("Computed features and local minima.")
        st.write("Full table (first rows):")
        st.dataframe(df_geo.head())

        # summary counts
        n_min = int(df_geo['IS_LOCAL_MIN'].sum())
        st.write("Found local minima:", n_min)
        if n_min == 0:
            st.warning("No minima found. Try increasing R multiplier.")
        else:
            mins = df_geo[df_geo['IS_LOCAL_MIN']].reset_index(drop=True)
            mins['ORI_SIN'] = np.sin(mins['orientation']); mins['ORI_COS'] = np.cos(mins['orientation'])

            st.sidebar.header("Clustering")
            method = st.sidebar.selectbox("Algorithm", ["KMeans","DBSCAN","Agglomerative","GMM"])
            if method in ["KMeans","Agglomerative","GMM"]:
                k = st.sidebar.slider("k/components", 2, min(8, max(2, len(mins))), value=3)
            else:
                eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.7)
                min_samp = st.sidebar.slider("DBSCAN min_samples", 2, 12, 4)

            # feature selection
            default_feats = ['depth','basin_area_est','mean_grad','mean_curvature','gauss_curvature','ori_sin','ori_cos']
            feats = st.sidebar.multiselect("Features for clustering", default_feats, default=default_feats[:5])

            X = mins[feats].fillna(0).values
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(); Xs = scaler.fit_transform(X)

            if method == "KMeans":
                labels, model = run_kmeans(Xs, k=k)
            elif method == "Agglomerative":
                labels, model = run_agglomerative(Xs, k=k)
            elif method == "GMM":
                labels, model = run_gmm(Xs, k=k)
            else:
                labels, model = run_dbscan(Xs, eps=eps, min_samples=min_samp)

            mins['cluster'] = labels
            st.write("Cluster counts:")
            st.write(mins['cluster'].value_counts())

            try:
                sil = silhouette_score(Xs, labels) if len(set(labels))>1 else None
            except:
                sil = None
            st.write("Silhouette:", sil)

            # Surface grid & optional watershed

            XI, YI, ZI = grid_surface(df_geo, grid_res=200)
            do_watershed = st.sidebar.checkbox("Compute watershed basins (accurate area)", value=False)
            basin_areas = None
            if do_watershed:
                # minimal built-in watershed: project minima to grid and use markers (lightweight)
                from skimage.morphology import watershed
                from skimage.feature import peak_local_max
                from scipy import ndimage as ndi

                nan_mask = np.isnan(ZI)
                if nan_mask.any():
                    # fill with nearest
                    ZI_copy = ZI.copy()
                    indices = ndi.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
                    ZI_copy[nan_mask] = ZI_copy[tuple(indices[:, nan_mask])]
                    ZI = ZI_copy
                inv = np.max(ZI) - ZI
                # seeds: project minima onto grid positions
                xi = np.linspace(df_geo['X'].min(), df_geo['X'].max(), XI.shape[1])
                yi = np.linspace(df_geo['Y'].min(), df_geo['Y'].max(), XI.shape[0])
                marker_mask = np.zeros_like(ZI, dtype=bool)
                for _, r in mins.iterrows():
                    ix = int(np.searchsorted(xi, r['X']));
                    iy = int(np.searchsorted(yi, r['Y']))
                    ix = np.clip(ix, 0, XI.shape[1] - 1);
                    iy = np.clip(iy, 0, XI.shape[0] - 1)
                    marker_mask[iy, ix] = True
                markers, _ = ndi.label(marker_mask.astype(int))
                labels_ws = watershed(inv, markers=markers)
                unique, counts = np.unique(labels_ws, return_counts=True)
                # area per pixel
                dx = xi[1] - xi[0];
                dy = yi[1] - yi[0]
                area_pix = dx * dy
                basin_areas = {int(u): int(c * area_pix) for u, c in zip(unique, counts) if int(u) != 0}
                # map minima to labels
                minima_labels = []
                for _, r in mins.iterrows():
                    ix = int(np.searchsorted(xi, r['X']));
                    iy = int(np.searchsorted(yi, r['Y']))
                    ix = np.clip(ix, 0, XI.shape[1] - 1);
                    iy = np.clip(iy, 0, XI.shape[0] - 1)
                    minima_labels.append(int(labels_ws[iy, ix]))
                mins['basin_label'] = minima_labels
                mins['basin_area_ws'] = mins['basin_label'].map(lambda l: basin_areas.get(l, 0.0))

            # Visualizations
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("2D heatmap + clusters")
                fig = plot_heatmap_matplotlib(XI, YI, ZI, minima_df=mins, clusters='cluster',
                                              title="Heatmap + minima clusters")
                st.pyplot(fig)
            with col2:
                st.subheader("3D surface (interactive)")
                fig3 = plot_3d_plotly(XI, YI, ZI, minima_df=mins, cluster_col='cluster')
                st.plotly_chart(fig3, use_container_width=True)

            # quiver
            figq = plot_quiver(mins, scale=1.0, nmax=200)
            if figq:
                st.subheader("Gradient vectors (sample)")
                st.pyplot(figq)

            # outputs: download minima CSV, pdf report, zip
            csv_bytes = mins.to_csv(index=False).encode('utf-8')
            st.download_button("Download minima CSV", data=csv_bytes, file_name="minima_with_features_clusters.csv",
                               mime="text/csv")

            # prepare images for PDF
            import io

            imgs = []
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0);
            imgs.append(("heatmap.png", buf.getvalue()));
            buf.close()
            try:
                png3 = fig3.to_image(format="png", width=900, height=600, scale=1)
                imgs.append(("3d_surface.png", png3))
            except Exception:
                pass

            summary_table = mins.groupby('cluster')[['depth', 'mean_grad', 'basin_area_est']].agg(
                ['count', 'mean', 'std']).reset_index()
            params = {
                "R": R, "R_basin": R_basin, "min_neighbors": min_neighbors,
                "basin_frac": basin_frac, "clustering_method": method
            }
            pdf_bytes = build_pdf_report("Terrain minima clustering", params, {"Cluster summary": summary_table}, imgs)
            st.download_button("Download PDF report", data=pdf_bytes, file_name="terrain_report.pdf",
                               mime="application/pdf")

            # zip all
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w") as zf:
                zf.writestr("minima.csv", csv_bytes)
                for fname, bts in imgs:
                    zf.writestr(fname, bts)
                zf.writestr("params.json", json.dumps(params, indent=2))
            zipbuf.seek(0)
            st.download_button("Download results ZIP", data=zipbuf.getvalue(), file_name="results.zip",
                               mime="application/zip")

            st.success("All done. You can download CSV / PDF / ZIP.")