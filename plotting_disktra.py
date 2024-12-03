import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

'''
Built on top of NetworkX and GeoPandas, and interacts with OpenStreetMap APIs, to:

- Download and model street networks or other infrastructure anywhere in the world with a single line of code
- Download geospatial features (e.g., political boundaries, building footprints, grocery stores, transit stops) as a GeoDataFrame
- Query by city name, polygon, bounding box, or point/address + distance
- Model driving, walking, biking, and other travel modes
- Attach node elevations from a local raster file or web service and calculate edge grades
- Impute missing speeds and calculate graph edge travel times
- Simplify and correct the networks topology to clean-up nodes and consolidate complex intersections
- Fast map-matching of points, routes, or trajectories to nearest graph edges or nodes
- Save/load network to/from disk as GraphML, GeoPackage, or OSM XML file
- Conduct topological and spatial analyses to automatically calculate dozens of indicators
- Calculate and visualize street bearings and orientations
- Calculate and visualize shortest-path routes that minimize distance, travel time, elevation, etc
- Explore street networks and geospatial features as a static map or interactive web map
- Visualize travel distance and travel time with isoline and isochrone maps
- Plot figure-ground diagrams of street networks and building footprints

'''
# load in the plot
G = ox.graph_from_place("Houston, Texas, USA", network_type="drive")

# houston texas
# Define start and end coordinates
start_point = (29.7604, -95.3698)  # Downtown Houston
end_point = (29.6297, -95.2735)    # A distant point in Houston

# Find the nearest nodes to the coordinates
start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

# Compute the shortest path using Dijkstra's algorithm
shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight="length")

# Use reverse geocoding to find addresses for the start and end points
geolocator = Nominatim(user_agent="route-annotator")
start_address_full = geolocator.reverse(start_point, language="en").address
end_address_full = geolocator.reverse(end_point, language="en").address

# Extract a shorter version of the address (e.g., street and city only)
def shorten_address(full_address):
    parts = full_address.split(", ")
    if len(parts) > 2:
        return f"{parts[0]}\n{parts[1]}"
    return full_address

start_address = shorten_address(start_address_full)
end_address = shorten_address(end_address_full)

# Plot the graph and the shortest path
fig, ax = ox.plot_graph_route(
    G,
    shortest_path,
    route_linewidth=4,
    node_size=0,
    bgcolor="white",
    route_color="red",
    show=False,
    close=False
)

# Add addresses as text with manual offsets
start_x, start_y = G.nodes[start_node]['x'], G.nodes[start_node]['y']
end_x, end_y = G.nodes[end_node]['x'], G.nodes[end_node]['y']

# Add text below the start node (offset downward)
ax.text(
    start_x, start_y + 0.005, f"Start:\n{start_address}",  # Offset downward
    fontsize=7, color="green", ha="center",
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

# Add text above the end node (offset upward)
ax.text(
    end_x, end_y - 0.04, f"End:\n{end_address}",  # Offset upward
    fontsize=7, color="blue", ha="center",
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

plt.title("Shortest Path with Start and End Addresses")
plt.show()