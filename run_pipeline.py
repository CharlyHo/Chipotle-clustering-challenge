from src.utils import load_chipotle_data
from src.clustering import compare_methods
from src.viz import plot_points, plot_k_distance, plot_kde_heatmap, make_interactive_map
import json
from pathlib import Path

data_path = Path("data/chipotle_stores.csv")
gdf = load_chipotle_data(data_path)

# Visualizations
plot_points(gdf, "Chipotle Locations", "outputs/figures/all_chipotles_projected.png")
plot_k_distance(gdf, 4, "outputs/figures/k4_distance.png")
plot_kde_heatmap(gdf, 50000, 200, "outputs/figures/kde_heatmap.png")

# Clustering
gdf, metrics_df = compare_methods(gdf, "outputs/clustering_metrics.csv")
print(metrics_df)

# Map
make_interactive_map(gdf, savepath="outputs/maps/chipotle_map.html")

print("Pipeline complete! Check the outputs/ folder.")