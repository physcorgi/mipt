

# app.py — расширённый Streamlit сервис для кластеризации минимумов рельефа
# Нужные пакеты в requirements.txt (см. ниже)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from numpy.linalg import lstsq
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from math import atan2, pi
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from fpdf import FPDF
import io, zipfile, os, json, tempfile, base64

sns.set()

st.set_page_config(layout="wide", page_title="Terrain Minima Clustering — Advanced")

# ---------------------------
# Helper math / geometry functions
# ---------------------------
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
    coords = df[['X','Y']].values; zs = df['Z'].values
    tree = cKDTree(coords); n=len(df); is_min=np.zeros(n, dtype=bool)
    if radius is not None:
        neighbors = tree.query_ball_point(coords, r=radius)
        for i in range(n):
            neigh = [j for j in neighbors[i] if j != i]
            is_min[i] = (len(neigh)>0) and np.all(zs[i] < zs[neigh])
    else:
        dist, idx = tree.query(coords, k=k+1)
        for i in range(n):
            neigh = idx[i][1:]
            is_min[i] = np.all(zs[i] < zs[neigh])
    return is_min

def compute_geometric_features(df, R, R_basin=None, min_neighbors=6, basin_frac_of_depth=0.5, basin_threshold=None, basin_condition='below'):
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

        # fit quadratic on neighbors + center
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
        grad_x[i] = zx; grad_y[i] = zy; mean_grad[i] = np.sqrt(zx*zx + zy*zy)
        mean_curv[i] = H; gauss_curv[i] = K; orientation[i] = atan2(zy, zx)

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

def make_grid_surface(df, grid_res=200, method='linear'):
    xi = np.linspace(df['X'].min(), df['X'].max(), grid_res)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((df['X'], df['Y']), df['Z'], (XI, YI), method=method)
    return XI, YI, ZI

def watershed_basins_from_grid(XI, YI, ZI, minima_mask=None, footprint_size=5):
    """
    Compute watershed basins on inverted height map (so minima become basins).
    minima_mask: boolean grid same shape as ZI marking marker seeds (optional).
    Returns labels (ints) of same shape as ZI, and dict area_by_label (in pixel counts).
    """
    # preprocess: fill nan by nearest to avoid holes
    nan_mask = np.isnan(ZI)
    if nan_mask.any():
        ZI_copy = ZI.copy()
        # distance transform to nearest non-nan
        mask_valid = ~nan_mask
        indices = ndi.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        ZI_copy[nan_mask] = ZI_copy[tuple(indices[:, nan_mask])]
        ZI = ZI_copy

    # invert height for watershed (we want catchment basins)
    inv = np.max(ZI) - ZI
    # compute local maxima of inverted (i.e. minima of original) for markers if not provided
    if minima_mask is None:
        # find peaks in inverted image (i.e. minima)
        local_maxi = peak_local_max(inv, indices=False, footprint=np.ones((footprint_size, footprint_size)))
        markers, _ = ndi.label(local_maxi)
    else:
        markers = ndi.label(minima_mask.astype(int))[0]
    labels = watershed(inv, markers=markers)
    # area per label (pixel counts)
    unique, counts = np.unique(labels, return_counts=True)
    area_by_label = {int(u):int(c) for u,c in zip(unique, counts) if int(u) != 0}
    return labels, area_by_label

# convex hull helper
def convex_hull_coords(df_sub):
    if len(df_sub) < 3:
        return None
    try:
        hull = ConvexHull(df_sub[['X','Y']].values)
        return df_sub[['X','Y']].values[hull.vertices]
    except Exception:
        return None

# medoid computation
def compute_medoid(sub):
    coords = sub[['X','Y']].values
    D = distance_matrix(coords, coords)
    idx = D.sum(axis=1).argmin()
    return sub.iloc[idx]

# PDF report builder (FPDF)
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Terrain minima clustering report', ln=1, align='C')
        self.ln(2)

def build_pdf_report(title, parameters, summary_tables, image_files_bytes):
    """
    title: str
    parameters: dict
    summary_tables: dict: {table_name: pandas.DataFrame}
    image_files_bytes: list of tuples (filename, bytes)
    returns PDF bytes
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Title: {title}", ln=1)
    pdf.ln(2)
    pdf.cell(0, 6, "Parameters:", ln=1)
    for k,v in parameters.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=1)
    pdf.ln(4)

    for tname, df_table in summary_tables.items():
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 6, tname, ln=1)
        pdf.set_font("Helvetica", size=9)
        # pretty print table: take head (up to 10 rows)
        pdf.ln(1)
        cols = list(df_table.columns)
        # header row
        for c in cols:
            pdf.cell(40, 6, str(c), border=1)
        pdf.ln()
        for i, row in df_table.head(10).iterrows():
            for c in cols:
                text = str(round(row[c],3)) if isinstance(row[c], (float, np.floating)) else str(row[c])
                pdf.cell(40, 6, text, border=1)
            pdf.ln()
        pdf.ln(4)

    # images
    for fname, bts in image_files_bytes:
        # write image to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1])
        tmp.write(bts)
        tmp.flush()
        tmp.close()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0,6, fname, ln=1)
        pdf.image(tmp.name, x=15, w=180)
        os.unlink(tmp.name)
    return pdf.output(dest='S').encode('latin-1')  # return bytes

# ---------------------------
# Streamlit UI & flow
# ---------------------------
st.title("Terrain Minima Clustering — Advanced (with reports & watershed)")

st.markdown("""
Загрузите CSV с колонками `X,Y,Z` (float).  
Потом:
1. Рассчитать геометрические признаки;  
2. Найти локальные минимумы;  
3. (опционально) посчитать watershed для точной площади впадин;  
4. Кластеризовать по выбранным признакам;  
5. Сгенерировать PDF-отчет и скачать CSV.
""")

uploaded = st.file_uploader("Upload CSV (X,Y,Z)", type=["csv"])
col1, col2 = st.columns([1,2])

if uploaded is None:
    st.info("Загрузите файл, примеры: small CSV с ~600 точками.")
    st.stop()

# read
df = pd.read_csv(uploaded)
df.columns = df.columns.str.upper()
if not set(['X','Y','Z']).issubset(df.columns):
    st.error("CSV должен содержать колонки: X, Y, Z")
    st.stop()

st.sidebar.header("Data & scale")
st.sidebar.write("Rows:", len(df))
st.sidebar.write("X range:", f"{df['X'].min():.3f} — {df['X'].max():.3f}")
st.sidebar.write("Y range:", f"{df['Y'].min():.3f} — {df['Y'].max():.3f}")
st.sidebar.write("Z range:", f"{df['Z'].min():.3f} — {df['Z'].max():.3f}")

# auto R guess
coords = df[['X','Y']].values
tree = cKDTree(coords)
dists_k, idx_k = tree.query(coords, k=2)
median_nn = np.median(dists_k[:,1])
st.sidebar.markdown("### Local scale (auto)")
R_default = st.sidebar.number_input("R factor (multiplier of median NN dist)", value=3.0, min_value=0.5, step=0.5)
R = float(R_default * median_nn)
R_basin = float(st.sidebar.number_input("R_basin (multiplier)", value=2.0) * median_nn)
st.sidebar.write(f"Computed R = {R:.4g}, R_basin = {R_basin:.4g}")

st.sidebar.markdown("### Feature computation")
min_neighbors = st.sidebar.number_input("min neighbors fallback (k)", value=8, min_value=3)
basin_frac = st.sidebar.slider("basin_frac_of_depth (thr = z + depth * frac)", 0.0, 1.0, 0.5)
footprint_size = st.sidebar.slider("watershed footprint (px)", 3, 15, 5)

compute_btn = st.sidebar.button("Compute features & minima")

if compute_btn:
    with st.spinner("Computing features..."):
        geo = compute_geometric_features(df, R=R, R_basin=R_basin, min_neighbors=int(min_neighbors), basin_frac_of_depth=float(basin_frac))
        df_geo = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
        df_geo['is_local_min'] = find_local_minima(df_geo, radius=R)
        st.success("Features computed")
        st.write("Full table (first rows):")
        st.dataframe(df_geo.head())

        # show counts
        n_min = df_geo['is_local_min'].sum()
        st.write("Found local minima:", int(n_min))

        # prepare minima DF
        mins = df_geo[df_geo['is_local_min']].reset_index(drop=True)
        if len(mins)==0:
            st.warning("No minima found — try increasing R multiplier.")
        else:
            # compute sin/cos orientation for clustering
            mins['ori_sin'] = np.sin(mins['orientation']); mins['ori_cos'] = np.cos(mins['orientation'])
            feature_cols = st.sidebar.multiselect("Select features for clustering",
                                                  ['depth','basin_area_est','mean_grad','mean_curvature','gauss_curvature','ori_sin','ori_cos'],
                                                  default=['depth','basin_area_est','mean_grad','ori_sin','ori_cos'])
            st.write("Using features:", feature_cols)

            # grid surface for maps and watershed
            XI, YI, ZI = make_grid_surface(df_geo, grid_res=300, method='linear')

            # optional watershed: markers from minima projected to grid
            do_watershed = st.sidebar.checkbox("Compute watershed basins (accurate basin area)", value=True)
            basin_map = None; basin_areas = None
            if do_watershed:
                # create marker grid: for each minima, find nearest pixel and set marker id
                # we'll create boolean minima mask by projecting minima coordinates onto grid
                marker_mask = np.zeros_like(ZI, dtype=bool)
                # compute nearest pixel index for each minima
                xi = np.linspace(df_geo['X'].min(), df_geo['X'].max(), XI.shape[1])
                yi = np.linspace(df_geo['Y'].min(), df_geo['Y'].max(), XI.shape[0])
                # pixel size
                dx = xi[1] - xi[0]; dy = yi[1] - yi[0]
                for _, r in mins.iterrows():
                    ix = int(np.searchsorted(xi, r['X']))
                    iy = int(np.searchsorted(yi, r['Y']))
                    # bounds check
                    ix = np.clip(ix, 0, XI.shape[1]-1)
                    iy = np.clip(iy, 0, XI.shape[0]-1)
                    marker_mask[iy, ix] = True
                labels_ws, area_by_label = watershed_basins_from_grid(XI, YI, ZI, minima_mask=marker_mask, footprint_size=footprint_size)
                basin_map = labels_ws
                basin_areas = area_by_label  # pixel counts per label
                # convert pixel counts to area in XY units: area_per_pixel = dx*dy
                area_pix = dx * dy
                basin_areas_phys = {lab: cnt * area_pix for lab, cnt in basin_areas.items()}
                st.write("Watershed found basins:", len(basin_areas_phys))
                # map minima to basin label by looking up their pixel
                minima_basins = []
                for _, r in mins.iterrows():
                    ix = int(np.searchsorted(xi, r['X'])); iy = int(np.searchsorted(yi, r['Y']))
                    ix = np.clip(ix, 0, XI.shape[1]-1); iy = np.clip(iy, 0, XI.shape[0]-1)
                    lab = int(labels_ws[iy, ix])
                    minima_basins.append(lab)
                mins['basin_label'] = minima_basins
                mins['basin_area_ws'] = mins['basin_label'].map(lambda l: basin_areas_phys.get(l, 0.0))
                st.write("Sample minima with basin area (first rows):")
                st.dataframe(mins[['X','Y','Z','depth','basin_area_ws']].head())

            # clustering controls
            st.sidebar.markdown("### Clustering")
            method = st.sidebar.selectbox("Algorithm", ["KMeans","DBSCAN","Agglomerative","GMM"])
            if method in ["KMeans","Agglomerative","GMM"]:
                k = st.sidebar.slider("k / n_components", 2, min(8, max(2, len(mins))), value=3)
            else:
                eps = st.sidebar.slider("eps (DBSCAN)", 0.1, 2.0, 0.7)
                min_samp = st.sidebar.slider("min_samples (DBSCAN)", 2, 12, 4)

            # prepare feature matrix and scale
            X = mins[feature_cols].fillna(0).values
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)

            # train chosen algorithm
            if method == "KMeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                mins['cluster'] = model.labels_
                sil = silhouette_score(X_scaled, model.labels_) if len(set(model.labels_))>1 else None
            elif method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=k).fit(X_scaled)
                mins['cluster'] = model.labels_
                sil = silhouette_score(X_scaled, model.labels_) if len(set(model.labels_))>1 else None
            elif method == "GMM":
                model = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
                mins['cluster'] = model.predict(X_scaled)
                sil = silhouette_score(X_scaled, mins['cluster']) if len(set(mins['cluster']))>1 else None
            else:
                model = DBSCAN(eps=eps, min_samples=int(min_samp)).fit(X_scaled)
                mins['cluster'] = model.labels_
                labs_db = model.labels_
                mask = labs_db != -1
                sil = silhouette_score(X_scaled[mask], labs_db[mask]) if mask.sum() > 1 else None

            st.write(f"Clustering done. silhouette = {sil}")
            st.write("Clusters count (including noise -1):")
            st.write(mins['cluster'].value_counts())

            # cluster summaries
            feat_summary = ['depth','basin_area_est','mean_grad','mean_curvature','gauss_curvature']
            if 'basin_area_ws' in mins.columns:
                feat_summary = ['depth','basin_area_ws','mean_grad','mean_curvature','gauss_curvature']
            summary = mins.groupby('cluster')[feat_summary].agg(['count','mean','std'])
            st.subheader("Cluster summary")
            st.dataframe(summary)

            # visualizations (2 columns)
            col_map, col_3d = st.columns([1,1])
            with col_map:
                st.subheader("2D Heatmap + clusters")
                XI_vis, YI_vis, ZI_vis = XI, YI, ZI
                fig, ax = plt.subplots(figsize=(8,6))
                im = ax.pcolormesh(XI_vis, YI_vis, ZI_vis, shading='auto', cmap='terrain')
                plt.colorbar(im, ax=ax, label='Z')
                unique = sorted(mins['cluster'].unique())
                palette = sns.color_palette('tab10', n_colors=max(10, len(unique))).as_hex()
                for lab in unique:
                    sub = mins[mins['cluster']==lab]
                    if lab == -1:
                        ax.scatter(sub['X'], sub['Y'], c='lightgray', s=25, label='noise')
                    else:
                        ax.scatter(sub['X'], sub['Y'], c=palette[int(lab)%len(palette)], s=40, label=f'cl{lab}', edgecolor='k')
                        hull = convex_hull_coords(sub)
                        if hull is not None:
                            ax.plot(np.append(hull[:,0], hull[0,0]), np.append(hull[:,1], hull[0,1]), linestyle='--', color=palette[int(lab)%len(palette)])
                ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
                ax.set_title("Heatmap + cluster minima")
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.axis('equal')
                st.pyplot(fig)

            with col_3d:
                st.subheader("3D Surface + minima (Plotly)")
                fig3 = go.Figure()
                fig3.add_trace(go.Surface(x=XI, y=YI, z=ZI, colorscale='Earth', opacity=0.9, showscale=False))
                unique = sorted(mins['cluster'].unique())
                palette = sns.color_palette('tab10', n_colors=max(10, len(unique))).as_hex()
                for lab in unique:
                    sub = mins[mins['cluster']==lab]
                    color = 'lightgray' if lab == -1 else palette[int(lab)%len(palette)]
                    fig3.add_trace(go.Scatter3d(x=sub['X'], y=sub['Y'], z=sub['Z']+0.2, mode='markers', marker=dict(size=4, color=color), name=f'cl{lab}'))
                fig3.update_layout(height=600, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig3, use_container_width=True)

            # save artifacts into memory for download & report
            # 1) minima CSV
            minima_csv = mins.to_csv(index=False).encode('utf-8')
            st.download_button("Download minima CSV", data=minima_csv, file_name="minima_with_features_clusters.csv", mime="text/csv")

            # 2) images for report: create PNGs in memory (heatmap and 3D as static PNG)
            imgs = []
            # heatmap PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            imgs.append(("heatmap.png", buf.getvalue()))
            buf.close()
            # 3D static: render with plotly to PNG (requires orca / kaleido) -> use kaleido (Plotly built-in)
            try:
                pbytes = fig3.to_image(format="png", width=900, height=600, scale=1)
                imgs.append(("3d_surface.png", pbytes))
            except Exception as e:
                st.warning("Plotly to_image failed (kaleido not installed). Skipping 3D PNG in report.")
            # summary tables for pdf
            summary_tables = {"Cluster summary": summary.reset_index()}
            parameters = {
                "R (units)": R,
                "R_basin (units)": R_basin,
                "min_neighbors": int(min_neighbors),
                "basin_frac_of_depth": float(basin_frac),
                "watershed_computed": bool(do_watershed),
                "clustering_method": method
            }
            # build pdf
            pdf_bytes = build_pdf_report("Terrain minima clustering", parameters, summary_tables, imgs)
            st.download_button("Download PDF report", data=pdf_bytes, file_name="terrain_minima_report.pdf", mime="application/pdf")

            # Optionally: pack everything into zip (CSV + images + JSON params)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode='w') as zf:
                zf.writestr("minima.csv", minima_csv)
                for fname, bts in imgs:
                    zf.writestr(fname, bts)
                zf.writestr("params.json", json.dumps(parameters, indent=2))
            zip_buf.seek(0)
            st.download_button("Download ZIP (CSV + images + params.json)", data=zip_buf.getvalue(), file_name="results_bundle.zip", mime="application/zip")

            # log: show medoids per cluster for quick inspection
            st.subheader("Medoids (representative minima) per cluster")
            medoids = []
            for lab in sorted(mins['cluster'].unique()):
                if lab == -1:
                    continue
                sub = mins[mins['cluster']==lab]
                med = compute_medoid(sub)
                medoids.append(pd.DataFrame(med).T)
            if medoids:
                med_df = pd.concat(medoids, ignore_index=True)
                st.dataframe(med_df[['X','Y','Z','depth','basin_area_est','mean_grad','mean_curvature']])
            else:
                st.write("No medoids (clusters empty or only noise).")

            # show drill-down control
            st.subheader("Drill-down: show neighborhood & profile")
            cl_choice = st.selectbox("Choose cluster to inspect", sorted(list(mins['cluster'].unique())))
            if st.button("Show neighborhood"):
                sel = mins[mins['cluster']==cl_choice]
                if sel.empty:
                    st.warning("Empty cluster")
                else:
                    centroid = sel[['X','Y']].mean().values
                    xi = np.linspace(df['X'].min(), df['X'].max(), XI.shape[1])
                    yi = np.linspace(df['Y'].min(), df['Y'].max(), XI.shape[0])
                    dx = xi[1]-xi[0]
                    R_neigh = float(st.sidebar.number_input("Neighborhood radius (units)", value=max(R, dx*5)))
                    # points in neighborhood
                    tree_all = cKDTree(df[['X','Y']].values)
                    idxs = tree_all.query_ball_point(centroid, r=R_neigh)
                    neigh = df.iloc[idxs]
                    fign, axn = plt.subplots(1,2, figsize=(12,4))
                    axn[0].scatter(df['X'], df['Y'], c='lightgray', s=6)
                    axn[0].scatter(neigh['X'], neigh['Y'], c='orange', s=10, label='neighborhood')
                    axn[0].scatter(sel['X'], sel['Y'], c='red', s=30, label='minima in cluster')
                    axn[0].scatter(centroid[0], centroid[1], c='black', marker='X', s=120, label='centroid')
                    circ = plt.Circle((centroid[0], centroid[1]), R_neigh, facecolor='none', edgecolor='black', linestyle='--')
                    axn[0].add_patch(circ)
                    axn[0].legend(); axn[0].axis('equal'); axn[0].set_title('Neighborhood')
                    dists = np.sqrt((neigh['X']-centroid[0])**2 + (neigh['Y']-centroid[1])**2)
                    axn[1].scatter(dists, neigh['Z'], s=8, alpha=0.6)
                    axn[1].set_xlabel('distance'); axn[1].set_ylabel('Z'); axn[1].set_title('Z profile by distance')
                    st.pyplot(fign)