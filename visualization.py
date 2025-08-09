# visualization.py (исправленная версия)
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import numpy as np

# опционально: включает приятный стиль для matplotlib
sns.set()

def grid_surface(df, grid_res=200, method='linear'):
    """
    Возвращает XI, YI, ZI (meshgrid) для поверхностной визуализации.
    df должен содержать колонки 'X','Y','Z'.
    """
    grid_res = int(max(2, grid_res))
    x = np.asarray(df['X'].values)
    y = np.asarray(df['Y'].values)
    z = np.asarray(df['Z'].values)

    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    # griddata принимает (x, y) как кортеж
    ZI = griddata((x, y), z, (XI, YI), method=method)
    return XI, YI, ZI


def plot_heatmap_matplotlib(XI, YI, ZI, minima_df=None, clusters=None, title="Heatmap"):
    """
    Рисует 2D тепловую карту (matplotlib). Поддерживает подсветку минимумов по колонке clusters.
    clusters может быть строкой (имя колонки в minima_df).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # защищаемся от полностью пустой сетки
    if ZI is None or np.all(np.isnan(ZI)):
        ax.text(0.5, 0.5, "No surface data (ZI is empty)", ha='center', va='center')
        ax.set_title(title)
        return fig

    # mask NaNs чтобы pcolormesh не ломался
    ZI_masked = np.ma.masked_invalid(ZI)
    im = ax.pcolormesh(XI, YI, ZI_masked, shading='auto', cmap='terrain')
    fig.colorbar(im, ax=ax, label='Z')

    if minima_df is not None and clusters is not None:
        # clusters может быть строкой (имя колонки) или массивом
        if isinstance(clusters, str):
            if clusters not in minima_df.columns:
                ax.set_title(title + " (clusters column missing)")
                return fig
            labels = minima_df[clusters]
        else:
            labels = clusters

        unique = sorted(pd_unique_safe(labels))
        # build palette with at least as many colors as unique labels (but at least 10)
        ncols = max(10, len(unique))
        palette = sns.color_palette('tab10', n_colors=ncols)
        # plot each label
        for lab in unique:
            sub = minima_df[labels == lab]
            if lab == -1:
                ax.scatter(sub['X'], sub['Y'], c='lightgray', s=25, alpha=0.6, edgecolors='none')
            else:
                color = palette[int(lab) % len(palette)]
                ax.scatter(sub['X'], sub['Y'], c=[color], s=40, edgecolors='k')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    return fig




def plot_3d_plotly(XI, YI, ZI, minima_df=None, cluster_col=None,
                   surface_colorscale='twilight', cluster_palette='Dark24'):
    """
    Интерактивная 3D поверхность + точки минимумов (plotly) с возможностью выбора цветовых палитр.

    Параметры:
    ----------
    XI, YI, ZI : array-like
        Массивы для построения 3D-поверхности.
    minima_df : pd.DataFrame, optional
        DataFrame с координатами минимумов и метками кластеров.
    cluster_col : str, optional
        Имя столбца в minima_df с метками кластеров.
    surface_colorscale : str или list, default='twilight'
        Цветовая палитра для поверхности (поддерживается любая из plotly).
        Примеры: 'Viridis', 'Plasma', 'Cividis', 'Inferno', 'Twilight', 'Jet', 'Hot', и др.
    cluster_palette : str или list, default='Dark24'
        Цветовая палитра для кластеров (из plotly.express.colors.qualitative).
        Примеры: 'Set1', 'Dark24', 'Plotly', 'Bold', 'Safe', 'Vivid', и др.

    Возвращает:
    ----------
    plotly.graph_objects.Figure
    """
    # безопасные приведения
    XI = np.asarray(XI)
    YI = np.asarray(YI)
    ZI = np.asarray(ZI)

    fig = go.Figure()

    # Добавляем 3D-поверхность с выбранной цветовой палитрой
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=surface_colorscale,
        opacity=0.9,
        showscale=False,
        hoverinfo='skip'
    ))

    # Если переданы минимумы и колонка кластеров
    if minima_df is not None and cluster_col is not None:
        if cluster_col not in minima_df.columns:
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
            return fig

        unique_labels = sorted(pd_unique_safe(minima_df[cluster_col]))

        # Получаем палитру кластеров
        if isinstance(cluster_palette, str):
            palette = getattr(px.colors.qualitative, cluster_palette, px.colors.qualitative.Dark24)
        else:
            palette = cluster_palette  # если передан список цветов

        for lab in unique_labels:
            sub = minima_df[minima_df[cluster_col] == lab]
            # Серый цвет для шумов (кластер -1)
            color = 'lightgray' if lab == -1 else palette[int(lab) % len(palette)]
            fig.add_trace(go.Scatter3d(
                x=sub['X'],
                y=sub['Y'],
                z=sub['Z'] + 0.2,  # чуть выше поверхности
                mode='markers',
                marker=dict(size=5, color=color, symbol='circle'),
                name=f'cl{lab}',
                hovertemplate=f'Cluster: {lab}<br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            ))

    # Оформление
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(title="Clusters")
    )

    return fig


def plot_quiver(minima_df, scale=1.0, nmax=200):
    """
    Рисует вектора градиента (matplotlib). scale управляет длиной стрелок.
    """
    if minima_df is None or 'grad_x' not in minima_df.columns or 'grad_y' not in minima_df.columns:
        return None

    nplot = min(nmax, len(minima_df))
    if nplot == 0:
        return None

    sub = minima_df.sample(n=nplot, random_state=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(minima_df['X'], minima_df['Y'], c='lightgray', s=6, zorder=0)

    # matplotlib.quiver: scale - greater -> arrows shorter. Подадим scale_units='xy' и scale=1/scale
    quiv = ax.quiver(sub['X'], sub['Y'], sub['grad_x'], sub['grad_y'],
                     angles='xy', scale_units='xy',
                     scale=(1.0 / max(1e-6, scale)), width=0.003, zorder=1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Gradient vectors (arrows)')
    return fig


# вспомогательная функция для безопасного получения уникальных значений в порядке возрастания
def pd_unique_safe(series_like):
    try:
        arr = np.asarray(series_like)
        # преобразуем к списку питоновских скалярных значений для сортировки
        # если есть смешанные типы, попробуем по строковому представлению
        try:
            unique = np.unique(arr)
            return unique.tolist()
        except Exception:
            return sorted(list({str(x): x for x in arr}.values()))
    except Exception:
        return []
