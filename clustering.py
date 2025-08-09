# clustering.py
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

def run_kmeans(X, k=3):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.7, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, k=3):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return labels, model

def run_gmm(X, k=3):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def choose_k_by_silhouette(X, method_fn, k_min=2, k_max=6):
    """
    Подбор k по silhouette (пробует k_min..k_max), возвращает (best_k, best_labels, best_model, best_score)
    method_fn — функция которая принимает (X,k) и возвращает (labels, model)
    """
    best = (None, None, None, -1)
    for k in range(k_min, k_max+1):
        labels, model = method_fn(X, k)
        if len(set(labels)) > 1:
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -1
        else:
            score = -1
        if score > best[3]:
            best = (k, labels, model, score)
    return best