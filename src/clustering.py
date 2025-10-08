import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from .utils import evaluate_clustering




def run_kmeans(gdf: gpd.GeoDataFrame, n_clusters: int = 10, random_state: int = 42):
    coords = gdf[['x', 'y']].values
    model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(coords)
    gdf[f'kmeans_{n_clusters}'] = model.labels_
    metrics = evaluate_clustering(coords, model.labels_)
    metrics.update({'method': 'kmeans', 'params': {'k': n_clusters}})
    return gdf, metrics




def run_dbscan(gdf: gpd.GeoDataFrame, eps: float = 20000, min_samples: int = 5):
    coords = gdf[['x', 'y']].values
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    gdf[f'dbscan_{int(eps/1000)}k_{min_samples}'] = model.labels_
    metrics = evaluate_clustering(coords, model.labels_)
    metrics.update({'method': 'dbscan', 'params': {'eps': eps, 'min_samples': min_samples}})
    return gdf, metrics




def run_hdbscan(gdf: gpd.GeoDataFrame, min_cluster_size: int = 10, min_samples: int = 5):
    coords = gdf[['x', 'y']].values
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(coords)
    gdf[f'hdbscan_{min_cluster_size}'] = labels
    metrics = evaluate_clustering(coords, labels)
    metrics.update({'method': 'hdbscan', 'params': {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}})
    return gdf, metrics




def compare_methods(gdf: gpd.GeoDataFrame, output_csv: str = '../outputs/clustering_metrics.csv'):
#Run all clustering baselines and save results to CSV
    results = []
    gdf, km_metrics = run_kmeans(gdf, 10)
    gdf, db_metrics = run_dbscan(gdf, 20000, 5)
    gdf, hdb_metrics = run_hdbscan(gdf, 10)
    results.extend([km_metrics, db_metrics, hdb_metrics])
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_csv, index=False)
    return gdf, metrics_df