import matplotlib.pyplot as plt
import osmnx as ox

def plot_solution(G, selected_facilities, poi_nodes, output_path=None):
    # reprojecting  to lat/lon for clean plotting
    G_wgs84 = ox.project_graph(G, to_crs='EPSG:4326')
    # assigning node colours based on what type of node
    node_colors = []
    for node in G_wgs84.nodes:
        if node in selected_facilities:
            node_colors.append('green')   # Facilities
        elif node in poi_nodes:
            node_colors.append('purple')  # POIs
        else:
            node_colors.append('skyblue') # Regular
    # creating plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(
        G_wgs84,
        node_color=node_colors,
        node_size=8.5,
        edge_color='gray',
        show=False,
        close=False,
        ax=ax
    )

    plt.title("MCLP Solution (Green = Facility, Purple = POI, Blue = Other)", fontsize=14)
    plt.tight_layout()
    # saving plot to a file
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
