# visualization.py
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import numpy as np

sns.set()

def grid_surface(df, grid_res=200, method='linear'):
    xi = np.linspace(df['X'].min(), df['X'].max(), grid_res)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((df['X'], df['Y']), df['Z'], (XI, YI), method=method)
    return XI, YI, ZI

def plot_heatmap_matplotlib(XI, YI, ZI, minima_df=None, clusters=None, title="Heatmap"):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.pcolormesh(XI, YI, ZI, shading='auto', cmap='terrain')
    fig.colorbar(im, ax=ax, label='Z')
    if minima_df is not None and clusters is not None:
        unique = sorted(minima_df[clusters].unique())
        palette = sns.color_palette('tab10', n_colors=max(10, len(unique))).as_hex()
        for lab in unique:
            sub = minima_df[minima_df[clusters]==lab]
            if lab == -1:
                ax.scatter(sub['X'], sub['Y'], c='lightgray', s=25, alpha=0.6)
            else:
                ax.scatter(sub['X'], sub['Y'], c=palette[int(lab)%len(palette)], s=40, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.axis('equal')
    return fig

def plot_3d_plotly(XI, YI, ZI, minima_df=None, cluster_col=None):
    fig = go.Figure()
    fig.add_trace(go.Surface(x=XI, y=YI, z=ZI, colorscale='Earth', opacity=0.9, showscale=False))
    if minima_df is not None and cluster_col is not None:
        unique = sorted(minima_df[cluster_col].unique())
        palette = px.colors.qualitative.Dark24
        for lab in unique:
            sub = minima_df[minima_df[cluster_col]==lab]
            color = 'lightgray' if lab == -1 else palette[int(lab) % len(palette)]
            fig.add_trace(go.Scatter3d(x=sub['X'], y=sub['Y'], z=sub['Z']+0.2, mode='markers', marker=dict(size=4, color=color), name=f'cl{lab}'))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=600)
    return fig

def plot_quiver(minima_df, scale=1.0, nmax=200):
    # рисуем векторы градиента grad_x, grad_y (подвыборка для наглядности)
    if minima_df is None or 'grad_x' not in minima_df.columns:
        return None
    sub = minima_df.sample(n=min(nmax, len(minima_df)), random_state=1)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(minima_df['X'], minima_df['Y'], c='lightgray', s=6)
    ax.quiver(sub['X'], sub['Y'], sub['grad_x'], sub['grad_y'], scale=1/scale, angles='xy', scale_units='xy', width=0.003)
    ax.set_aspect('equal', 'box')
    ax.set_title('Gradient vectors (arrows)')
    return fig