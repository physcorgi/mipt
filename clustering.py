from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from collections import namedtuple
import warnings
import numpy as np

ClusteringResult = namedtuple('ClusteringResult', ['best_param', 'best_labels', 'best_model', 'best_score'])


def run_kmeans(X, n_clusters=3, **kwargs):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def run_dbscan(X, eps=0.7, min_samples=5, **kwargs):
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def run_agglomerative(X, n_clusters=3, **kwargs):
    model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def run_gmm(X, n_components=3, **kwargs):
    model = GaussianMixture(n_components=n_components, random_state=42, **kwargs)
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def choose_param_by_silhouette(X, method_fn, param_name, param_values, fixed_params=None):
    """
    Подбирает лучший параметр по silhouette score.

    - X: array-like (will be converted to np.ndarray).
    - method_fn: функция кластеризации, принимает (X, **params) и возвращает (labels, model).
    - param_name: имя перебираемого параметра (str).
    - param_values: iterable значений параметра для перебора.
    - fixed_params: dict других неизменяемых параметров для передачи в функцию.

    Возвращает ClusteringResult с лучшим параметром, метками, моделью и скором.
    """
    if fixed_params is None:
        fixed_params = {}

    X_arr = np.asarray(X)
    best = ClusteringResult(best_param=None, best_labels=None, best_model=None, best_score=float('-inf'))

    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        try:
            labels, model = method_fn(X_arr, **params)
            labels = np.asarray(labels)

            # если есть метка -1 (шум у DBSCAN) — исключаем её для подсчёта силуета
            if -1 in labels:
                mask = labels != -1
                # нужно иметь как минимум 2 кластера в непустой выборке для silhouette
                if mask.sum() < 2:
                    score = float('-inf')
                else:
                    labels_masked = labels[mask]
                    unique_masked = set(labels_masked)
                    if len(unique_masked) < 2:
                        score = float('-inf')
                    else:
                        score = silhouette_score(X_arr[mask], labels_masked)
            else:
                unique_labels = set(labels)
                if len(unique_labels) < 2:
                    score = float('-inf')
                else:
                    score = silhouette_score(X_arr, labels)
        except Exception as e:
            warnings.warn(f"Ошибка при вычислении silhouette для {param_name}={val}: {e}")
            score = float('-inf')

        # выбираем лучший (максимизируем silhouette)
        if score > best.best_score:
            best = ClusteringResult(best_param=val, best_labels=labels, best_model=model, best_score=score)

    return best
