import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score




def load_chipotle_data(path: str) -> gpd.GeoDataFrame:
#Load Chipotle dataset and return projected GeoDataFrame
    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=2163)
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    return gdf




def evaluate_clustering(coords: np.ndarray, labels: np.ndarray) -> dict:
#Compute clustering evaluation metrics
    res = {}
    unique_labels = set(labels)
    n_clusters = len([x for x in unique_labels if x != -1])
    res['n_clusters'] = n_clusters
    res['noise_rate'] = float((labels == -1).sum()) / len(labels)
    if n_clusters > 1:
        res['silhouette'] = silhouette_score(coords, labels)
        res['davies'] = davies_bouldin_score(coords, labels)
        res['calinski'] = calinski_harabasz_score(coords, labels)
    else:
        res['silhouette'] = res['davies'] = res['calinski'] = None
    return res