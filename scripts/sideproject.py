# poso que s'ha d'instal.lar el package osmnx perque a Google Colab no hi és. En cas de fer-ho amb jupyter s'han d'instal.lar totes

!pip install osmnx --upgrade
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox

## incorporem la xarxa de carrers de Barcelona

# Get the graph from a place
G = ox.graph_from_place("Barcelona, Spain", network_type="drive")

# Plot the original graph
fig, ax = ox.plot_graph(G)

# Convert graph to GeoDataFrames
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

## Computem la node betweenness

import networkx as nx
import osmnx as ox
from collections import defaultdict

G_nx = nx.Graph(G)  # Convert to undirected NetworkX graph

# Initialize dictionaries
betweenness = {node: 0 for node in G_nx.nodes()}
path_density = {node: 0 for node in G_nx.nodes()}

# Compute betweenness centrality using a 5-edge distance cutoff
for source in G_nx.nodes():
    shortest_paths = nx.single_source_shortest_path(G_nx, source, cutoff=5)  # Get all shortest paths up to 5 edges

    for target, path in shortest_paths.items():  # path is already a list of nodes
        if source != target:
            for node in path[1:-1]:  # Exclude source and target
                betweenness[node] += 1

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalize node sizes for better visualization
scaler = MinMaxScaler(feature_range=(5, 50))  # Adjust min/max size as needed
betweenness_values = np.array(list(betweenness.values())).reshape(-1, 1)
scaled_sizes = scaler.fit_transform(betweenness_values).flatten()

# Assign node colors and sizes
node_color = [plt.cm.Reds(betweenness.get(node, 0)) for node in G_nx.nodes()]
node_size = [scaled_sizes[i] for i, node in enumerate(G_nx.nodes())]

# Plot the graph using OSMnx
fig, ax = ox.plot_graph(G, node_color=node_color, node_size=node_size,
                        edge_color="gray", bgcolor="white", show=False)

# Set title and subtitle
plt.suptitle("Node Betweenness Centrality in Road Network", fontsize=14, fontweight="bold")
plt.title("Note: Darker colors indicate higher betweenness values.", fontsize=10, color="black")

# Save the figure as a PNG image
plt.savefig("betweenness_centrality.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

### compute the betweenness for each bicing station

from scipy.spatial import KDTree
import osmnx as ox
import matplotlib.pyplot as plt

# Load Bicing station dataset (update with actual path)
bicing_stations = pd.read_csv("/content/Informacio_Estacions_Bicing_2025.csv")  # Update with actual path

# Get node positions (lat, lon) from graph (assuming G is your graph object with betweenness calculated)
node_positions = {node: (data['y'], data['x']) for node, data in G.nodes(data=True)}
node_coords = np.array(list(node_positions.values()))
node_ids = list(node_positions.keys())

# Build a KDTree for fast nearest neighbor search
tree = KDTree(node_coords)

# Set the radius for the search in degrees (1000 meters in degrees)
radius = 1000 / 111_320   # 1 degree = ~111,320 meters (latitude), so divide by that

# Compute the sum of betweenness for nodes within 1000 meters of each Bicing station
bicing_stations["sum_betweenness"] = np.nan  # Initialize new column

# Assuming `betweenness` is a dictionary containing betweenness values of nodes
for idx, row in bicing_stations.iterrows():
    station_coord = (row["lat"], row["lon"])

    # Find all nodes within 1000 meters (radius)
    nearby_node_indices = tree.query_ball_point(station_coord, r=radius)
    nearby_nodes = [node_ids[i] for i in nearby_node_indices]

    # Compute the sum of betweenness for the nearby nodes
    sum_betweenness = np.sum([betweenness.get(node, 0) for node in nearby_nodes])

    # Store the result in the DataFrame
    bicing_stations.at[idx, "sum_betweenness"] = sum_betweenness

# Save the updated dataset
bicing_stations.to_csv("bicing_with_sum_betweenness.csv", index=False)

# Print sample results
print(bicing_stations.head())

# Optionally, plot Bicing stations on top of the road network
fig, ax = ox.plot_graph(G, show=False, edge_color="gray", bgcolor="white")

sc = ax.scatter(
    bicing_stations["lon"],
    bicing_stations["lat"],
    c=bicing_stations["sum_betweenness"],
    cmap="Reds",
    edgecolors="black",
    alpha=0.8,
    s=40  # Adjust dot size
)

plt.colorbar(sc, label="Sum of Betweenness of Nodes within a 1 Km buffer")
plt.title("BICING Stations' Betweenness")
plt.savefig("bicing_betweenness.png", dpi=300, bbox_inches="tight")
plt.show()


## computem la segona variable independent d'interès el numero d'estacions de metro

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# Load datasets (replace with actual file paths)
points_df = bicing_stations[['station_id', 'lat', 'lon']]
metro_df = pd.read_csv("/content/metro_stops.csv")  # Contains metro stations (lat, lon)

# Rename columns for consistency
points_df.rename(columns={"LATITUD": "lat", "LONGITUD": "lon"}, inplace=True)
metro_df.rename(columns={"LATITUD": "lat", "LONGITUD": "lon"}, inplace=True)

# Convert coordinates to NumPy arrays
points_coords = np.array(points_df[["lat", "lon"]])
metro_coords = np.array(metro_df[["lat", "lon"]])

# Build KDTree for metro station locations
tree = KDTree(metro_coords)

# Set search radius in degrees (~100m in latitude/longitude)
radius = 100 / 111_320  # Approximate conversion from meters to degrees

# Find all metro stations within 100m of each point
points_df["metro_count"] = [len(tree.query_ball_point(coord, r=radius)) for coord in points_coords]

# Save results
points_df.to_csv("points_with_metro_count.csv", index=False)

# Display sample results
print(points_df.head())


########
########
## variable dependent només amb un mes

data_depvar = pd.read_csv('/content/data/2024_05_Maig_BicingNou_ESTACIONS.csv')

grouped_df = data_depvar[['station_id','num_docks_available']].groupby('station_id').mean()
bicing_station = pd.read_csv('/content/bicing_with_sum_betweenness.csv')

# Merge both datasets on 'station_id'
merged_df = pd.merge(bicing_station, grouped_df, on='station_id', how='inner')  
merged_df_2 = pd.merge(merged_df, points_df, on = 'station_id', how = 'inner')

merged_df_2['target'] = merged_df_2['num_docks_available'] / merged_df_2['capacity'] * 100
merged_df_2.info()

###############
###############
## OLS MODEL

# Existing variables
X = np.log1p(merged_df_2['sum_betweenness'])
Y = merged_df_2['target']

# Include the new variable directly in X
X = np.column_stack([X, np.log1p(merged_df_2['metro_count'])])
# Example DataFrame (merged_df_2) - make sure your DataFrame contains these columns: sum_betweenness, metro_count, and any other new variables

# Add the log-transformed metro_count as a new variable
X = np.column_stack([X, merged_df_2['lat_x']])

# If you have more variables to include, follow this pattern
# Example: If 'new_variable' is the additional variable you want to add
X = np.column_stack([X, merged_df_2['lon_y']])  # log-transformed new_variable

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(Y, X).fit()

# Display results
print(model.summary())

