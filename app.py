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
from clustering import run_kmeans, run_dbscan, run_agglomerative, run_gmm, choose_param_by_silhouette
from visualization import grid_surface, plot_heatmap_matplotlib, plot_3d_plotly, plot_quiver
from report import build_pdf_report







st.set_page_config(layout="wide", page_title="Кластеризация точек на поверхности земли")
st.title("Кластеризация точек на поверхности земли")













#
# Загрузка данных
#
uploaded = st.file_uploader("Загрузите CSV (X,Y,Z)", type=["csv"])
if not uploaded:
    st.info("Загрузите CSV со столбцами X,Y,Z (без учёта регистра).")
    st.stop()

try:
    df = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"Не удалось загрузить данные: {e}")
    st.stop()

#
# Боковая панель: глобальные параметры масштаба (не запускают тяжёлые вычисления сами по себе)
#

st.sidebar.header("Масштаб и признаки")
median_nn = median_nn_distance(df)
st.sidebar.write("Медиана расстояния до ближайшего соседа:", round(median_nn, 6))

R_factor = st.sidebar.number_input("Множитель R (локальное соседство)", value=3.0, step=0.5)
R = float(R_factor * median_nn)

R_basin_factor = st.sidebar.number_input("Множитель R_basin", value=2.0, step=0.5)
R_basin = float(R_basin_factor * median_nn)

min_neighbors = int(st.sidebar.number_input("Минимум соседей (резерв)", value=8, min_value=3))
basin_frac = float(st.sidebar.slider("Доля глубины для бассейна (thr = z + depth * frac)", 0.0, 1.0, 0.5))

st.sidebar.markdown("---")

#
# Кнопка вычисления признаков (тяжёлая операция). df_geo сохраняется в session_state
#
if "df_geo" not in st.session_state:
    st.session_state.df_geo = None

compute = st.sidebar.button("Рассчитать признаки и минимумы")
if compute:
    with st.spinner("Вычисление геометрических признаков и локальных минимумов..."):
        try:
            geo = compute_geometric_features(df, R=R, R_basin=R_basin, min_neighbors=min_neighbors,
                                             basin_frac_of_depth=basin_frac)
            df_geo = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
            df_geo['IS_LOCAL_MIN'] = find_local_minima(df_geo, radius=R)
            st.session_state.df_geo = df_geo
            st.success("Признаки рассчитаны и сохранены в сессии.")
        except Exception as e:
            st.error(f"Ошибка при вычислении признаков: {e}")
            st.stop()

# Если ранее уже считали — используем
if st.session_state.df_geo is None:
    st.info("Нажмите в сайдбаре ‘Рассчитать признаки и минимумы’, чтобы подготовить данные.")
    st.stop()

df_geo = st.session_state.df_geo

#
# Таблица и число минимумов
#
st.subheader("Входные данные + вычисленные признаки (первые строки)")
st.dataframe(df_geo.head())
n_min = int(df_geo['IS_LOCAL_MIN'].sum())
st.write("Найдено локальных минимумов:", n_min)
if n_min == 0:
    st.warning("Минимумы не найдены. Увеличьте множитель R или проверьте входные данные.")
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
st.sidebar.header("Кластеризация")
method = st.sidebar.selectbox("Алгоритм", ["KMeans", "DBSCAN", "Agglomerative", "GMM"])

max_k = min(8, max(2, len(mins)))
default_k = min(3, max_k)
if method in ["KMeans", "Agglomerative", "GMM"]:
    k = st.sidebar.slider("Число кластеров / компонентов (k)", min_value=2, max_value=max_k, value=default_k)
else:
    eps = st.sidebar.slider("DBSCAN eps", 0.01, 5.0, 0.7, step=0.01)
    min_samp = st.sidebar.slider("DBSCAN min_samples", 2, 30, 4)

default_feats = ['depth', 'basin_area_est', 'mean_grad', 'mean_curvature', 'gauss_curvature', 'ori_sin', 'ori_cos']
available_feats = [f for f in default_feats if f in mins.columns]
if not available_feats:
    st.error("No clustering features available. Check compute step.")
    st.stop()

feats = st.sidebar.multiselect("Признаки для кластеризации", options=available_feats,
                               default=available_feats[:min(5, len(available_feats))])
if not feats:
    st.error("Select at least one feature for clustering.")
    st.stop()

# ---
# Автоподбор по силуэту ("киллер-фича"): перебор параметров нескольких алгоритмов
# и выбор лучшего по silhouette score. Сохраняем лидерборд в session_state.
# ---
st.sidebar.markdown("---")
st.sidebar.subheader("Автоподбор")
auto_tune = st.sidebar.checkbox("Автоматически подобрать лучший метод (силуэт)", value=False)

if auto_tune:
    k_max_auto = st.sidebar.slider("Максимальное k для KMeans/Agglomerative/GMM", min_value=2, max_value=max_k, value=default_k)
    eps_min = st.sidebar.number_input("DBSCAN eps min", value=0.2, min_value=0.01, max_value=10.0, step=0.01)
    eps_max = st.sidebar.number_input("DBSCAN eps max", value=1.2, min_value=0.05, max_value=10.0, step=0.05)
    eps_step = st.sidebar.number_input("DBSCAN eps шаг", value=0.1, min_value=0.01, max_value=2.0, step=0.01)
    min_samp_auto = st.sidebar.slider("DBSCAN min_samples (авто)", 2, 30, 4)

    auto_btn = st.sidebar.button("Запустить автоподбор")
    if auto_btn:
        with st.spinner("Подбираю алгоритм и параметры по силуэту..."):
            try:
                # Подготовка признаков и стандартизация
                X = mins[feats].fillna(0).values
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)

                candidates = []

                # Перебор k для нескольких методов
                k_values = list(range(2, int(k_max_auto) + 1))
                res_km = choose_param_by_silhouette(Xs, run_kmeans, 'n_clusters', k_values)
                candidates.append(("KMeans", 'k', res_km))

                res_ag = choose_param_by_silhouette(Xs, run_agglomerative, 'n_clusters', k_values)
                candidates.append(("Agglomerative", 'k', res_ag))

                res_gm = choose_param_by_silhouette(Xs, run_gmm, 'n_components', k_values)
                candidates.append(("GMM", 'k', res_gm))

                 # Перебор eps для DBSCAN (min_samples фиксируем)
                eps_values = np.arange(float(eps_min), float(eps_max) + 1e-12, float(eps_step)).tolist()
                res_db = choose_param_by_silhouette(
                    Xs,
                    lambda X, **p: run_dbscan(X, **p),
                    'eps',
                    eps_values,
                    fixed_params={'min_samples': int(min_samp_auto)}
                )
                candidates.append(("DBSCAN", 'eps', res_db))

                # Выбор лучшего результата
                leaderboard = []
                best = None
                best_algo = None
                best_param_name = None
                for name, pname, resobj in candidates:
                    import numpy as np
                    score = resobj.best_score
                    leaderboard.append({
                        'algo': name,
                        'param': pname,
                        'value': resobj.best_param,
                        'silhouette': None if np.isneginf(score) else float(score)
                    })
                    if best is None or score > best.best_score:
                        best = resobj
                        best_algo = name
                        best_param_name = pname

                # Применяем лучший
                labels = np.asarray(best.best_labels)
                mins['cluster'] = labels
                st.session_state.cluster_result = {
                    "labels": labels,
                    "model": best.best_model,
                    "feats": feats,
                    "Xs": Xs,
                    "algo": best_algo,
                    "param": {best_param_name: best.best_param}
                }
                st.session_state.auto_leaderboard = leaderboard

                best_sil = None if np.isneginf(best.best_score) else round(float(best.best_score), 4)
                st.success(f"Лучший метод: {best_algo} ({best_param_name}={best.best_param}), silhouette={best_sil}")
            except Exception as e:
                st.error(f"Автоподбор не удался: {e}")

# ---
# Кнопка обычного запуска кластеризации (без автоподбора)
# ---
if "cluster_result" not in st.session_state:
    st.session_state.cluster_result = None

cluster_run = st.sidebar.button("Запустить кластеризацию")
if cluster_run:
    with st.spinner("Выполняется кластеризация..."):
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
            st.success("Кластеризация завершена и сохранена в сессии.")
        except Exception as e:
            st.error(f"Ошибка кластеризации: {e}")
            st.session_state.cluster_result = None

# Если результата кластеризации ещё нет — предложим запустить
if st.session_state.cluster_result is None:
    st.info("Нажмите ‘Запустить кластеризацию’, чтобы вычислить кластеры с текущими параметрами.")
    st.stop()

res = st.session_state.cluster_result
labels = res["labels"]
mins['cluster'] = labels

st.subheader("Результаты кластеризации")
st.write("Размеры кластеров:")
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
st.write("Силуэт (silhouette):", sil)

# Если был автоподбор, покажем лидерборд
if 'auto_leaderboard' in st.session_state and st.session_state.auto_leaderboard:
    st.write("Лидерборд автоподбора (выше — лучше):")
    st.dataframe(pd.DataFrame(st.session_state.auto_leaderboard).sort_values('silhouette', ascending=False), use_container_width=True)

#
# Visualizations + optional watershed
#
XI, YI, ZI = grid_surface(df_geo, grid_res=200)

do_watershed = st.sidebar.checkbox("Рассчитать бассейны методом водораздела (точная площадь)", value=False)
basin_areas = None

if do_watershed:
    try:
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi
        import numpy as np

        # Делаем рабочую копию ZI, чтобы не испортить исходный массив
        ZI_work = ZI.copy()

        # Заполняем NaN ближайшими валидными значениями через distance transform
        nan_mask = np.isnan(ZI_work)
        if nan_mask.any():
            # Индексы ближайших валидных пикселей
            indices = ndi.distance_transform_edt(
                nan_mask,
                return_distances=False,
                return_indices=True
            )
            # Заполняем NaN значениями ближайших валидных соседей
            ZI_work[nan_mask] = ZI_work[tuple(indices[:, nan_mask])]

        # Инвертируем поверхность, чтобы минимумы стали "пиками" для водораздела
        inv = np.max(ZI_work) - ZI_work

        # Координатные сетки
        x_min, x_max = df_geo['X'].min(), df_geo['X'].max()
        y_min, y_max = df_geo['Y'].min(), df_geo['Y'].max()
        xi = np.linspace(x_min, x_max, ZI_work.shape[1])
        yi = np.linspace(y_min, y_max, ZI_work.shape[0])

        # Маркеры в позициях локальных минимумов
        marker_mask = np.zeros_like(ZI_work, dtype=bool)
        for _, r in mins.iterrows():
            # Ближайшие индексы сетки
            ix = int(np.searchsorted(xi, r['X']))
            iy = int(np.searchsorted(yi, r['Y']))

            # Ограничиваем индексы валидными границами
            ix = np.clip(ix, 0, ZI_work.shape[1] - 1)
            iy = np.clip(iy, 0, ZI_work.shape[0] - 1)

            marker_mask[iy, ix] = True

        # Нумеруем связные маркеры (на случай дубликатов)
        markers, num_markers = ndi.label(marker_mask.astype(int))
        if num_markers == 0:
            raise ValueError("No valid markers found for watershed.")

        # Запускаем водораздел
        labels_ws = watershed(inv, markers=markers, mask=~nan_mask)  # Respect original NaN regions

        # Площадь пикселя
        dx = np.diff(xi)[0] if len(xi) > 1 else 1.0
        dy = np.diff(yi)[0] if len(yi) > 1 else 1.0
        area_per_pixel = dx * dy

        # Считаем пиксели в каждом бассейне (0 — фон, исключаем)
        unique_labels, counts = np.unique(labels_ws, return_counts=True)
        basin_areas = {
            int(label): float(count * area_per_pixel)
            for label, count in zip(unique_labels, counts)
            if label != 0  # Skip background
        }

        # Сопоставляем минимуму его метку бассейна и площадь
        minima_labels = []
        for _, r in mins.iterrows():
            ix = int(np.searchsorted(xi, r['X']))
            iy = int(np.searchsorted(yi, r['Y']))
            ix = np.clip(ix, 0, ZI_work.shape[1] - 1)
            iy = np.clip(iy, 0, ZI_work.shape[0] - 1)
            label = int(labels_ws[iy, ix])
            minima_labels.append(label)

        mins = mins.copy()  # Ensure we don't modify original DataFrame outside scope
        mins['basin_label'] = minima_labels
        mins['basin_area_ws'] = mins['basin_label'].map(lambda lbl: basin_areas.get(lbl, 0.0))

    except Exception as e:
        st.warning(f"Сегментация водоразделом не удалась: {e}")
        # Optionally log full traceback
        # import traceback; st.code(traceback.format_exc())

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("2D тепловая карта + кластеры")

    # Используем ТУ ЖЕ палитру, что и в 3D
    selected_cmap = st.session_state.get('surface_palette', 'twilight')

    fig2d = plot_heatmap_matplotlib(
        XI, YI, ZI,
        minima_df=mins,
        clusters='cluster',
        title="Тепловая карта + кластеры минимумов",
        cmap=selected_cmap  # <-- передаём выбранную палитру
    )
    st.pyplot(fig2d)
with col2:
    st.subheader("3D поверхность (интерактивно)")

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
        st.caption("🎨 Настройка цветовых схем:")

        # Выбор палитры для surface (влияет на 2D и 3D)
        surface_palette = st.selectbox(
            "Палитра поверхности",
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
            "Палитра кластеров",
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
st.download_button("Скачать CSV минимумов", data=csv_bytes, file_name="minima_with_features_clusters.csv", mime="text/csv")

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
    st.warning(f"Не удалось сформировать PDF: {e}")

zipbuf = io.BytesIO()
with zipfile.ZipFile(zipbuf, "w") as zf:
    zf.writestr("minima.csv", csv_bytes)
    for fname, bts in imgs:
        zf.writestr(fname, bts)
    zf.writestr("params.json", json.dumps(params, indent=2))
zipbuf.seek(0)
st.download_button("Скачать результаты (ZIP)", data=zipbuf.getvalue(), file_name="results.zip", mime="application/zip")

st.success("Готово. Можно скачать CSV / PDF / ZIP.")
