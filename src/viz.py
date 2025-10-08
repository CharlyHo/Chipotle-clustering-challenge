import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
import folium




def plot_points(gdf: gpd.GeoDataFrame, title: str, savepath: str = None):
    fig, ax = plt.subplots(figsize=(10,6))
    gdf.plot(ax=ax, markersize=5, color='darkred')
    ax.set_title(title)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)




def plot_k_distance(gdf: gpd.GeoDataFrame, k: int = 4, savepath: str = None):
    coords = gdf[['x','y']].values
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    k_dist = np.sort(distances[:, -1])
    plt.figure(figsize=(8,4))
    plt.plot(k_dist)
    plt.ylabel(f'{k}-distance (m)')
    plt.xlabel('Points sorted')
    plt.title(f'K-distance plot (k={k})')
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close()




def plot_kde_heatmap(gdf: gpd.GeoDataFrame, bandwidth: float = 50000, grid_size: int = 200, savepath: str = None):
    xs, ys = gdf['x'].values, gdf['y'].values
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_size), np.linspace(ymin, ymax, grid_size))
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth).fit(np.vstack([xs, ys]).T)
    Z = np.exp(kde.score_samples(grid)).reshape(xx.shape)


    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(np.flipud(Z), extent=(xmin,xmax,ymin,ymax), cmap='Reds')
    gdf.plot(ax=ax, markersize=2, color='black')
    ax.set_title(f'Chipotle Density Heatmap (bandwidth={bandwidth/1000:.0f}km)')
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)




def make_interactive_map(gdf: gpd.GeoDataFrame, candidates: dict = None, savepath: str = '../outputs/maps/chipotle_map.html'):
    gdf_wgs = gdf.to_crs(epsg=4326)
    center = [float(gdf_wgs.geometry.y.mean()), float(gdf_wgs.geometry.x.mean())]
    m = folium.Map(location=center, zoom_start=5)
    for _, row in gdf_wgs.iterrows():
        folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=3, fill=True, opacity=0.6).add_to(m)
        if candidates:
            for name, loc in candidates.items():
                folium.Marker(location=[loc['lat'], loc['lon']], popup=name, tooltip=name).add_to(m)
                m.save(savepath)
                return savepath