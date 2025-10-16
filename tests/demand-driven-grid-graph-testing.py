import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib.patches import RegularPolygon
import networkx as nx
from mclp.mclp_model import build_mclp_model
import math
import random
import pandas as pd
import random
from mclp.demand_driven_dist_coverage_utils import build_coverage_sets_variable

def generate_grid(grid_type, width, height):
    if grid_type == "square":
        return nx.grid_2d_graph(width, height)
    elif grid_type == "hex":
        return generate_hex_center_graph(height, width)
    else:
        raise ValueError("Unsupported grid type")

import matplotlib.pyplot as plt
import networkx as nx
import math
from matplotlib.patches import RegularPolygon

def hex_center(row, col, size=1):
    """Return the (x, y) position of the center of a hexagon."""
    x = size * 3/2 * col
    y = size * math.sqrt(3) * (row + 0.5 * (col % 2))
    return (x, y)

def generate_hex_center_graph(rows, cols):
    G = nx.Graph()
    positions = {}

    for row in range(rows):
        for col in range(cols):
            node = (row, col)
            positions[node] = hex_center(row, col)
            G.add_node(node)

    # Define 6 directions for neighbor connections
    for row in range(rows):
        for col in range(cols):
            node = (row, col)
            if col % 2 == 0:  # even column
                directions = [(-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1), (-1, 1)]
            else:  # odd column
                directions = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 0), (1, 1)]

            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    G.add_edge(node, (r, c))

    nx.set_node_attributes(G, positions, "pos")
    return G


def draw_hex_grid(G, weights, radii, poi_nodes=None, facilities=None, title="Hexagonal Grid", show_weights=True):
    """
    Draw hexagonal grid with demand weights visible on nodes.
    Colors indicate radius (high demand = red, low demand = blue).
    """
    pos = nx.get_node_attributes(G, "pos")
    poi_nodes = poi_nodes or set()
    facilities = facilities or set()

    fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')  # Increased from 16x16

    # Get radius range for coloring
    min_radius = min(radii.values())
    max_radius = max(radii.values())

    # Calculate plot bounds
    x_coords = [pos[n][0] for n in G.nodes()]
    y_coords = [pos[n][1] for n in G.nodes()]
    x_min, x_max = min(x_coords) - 2, max(x_coords) + 2
    y_min, y_max = min(y_coords) - 2, max(y_coords) + 2

    # Draw hexagonal tiles
    for node in G.nodes():
        x, y = pos[node]
        radius_val = radii[node]
        
        # Normalize radius for color mapping
        if max_radius > min_radius:
            norm_radius = (radius_val - min_radius) / (max_radius - min_radius)
        else:
            norm_radius = 0.5
        
        # Color gradient: Red (high demand/low radius) to Blue (low demand/high radius)
        color = plt.cm.RdYlBu(norm_radius)
        
        # Facilities get thick black border
        edge_color = 'black' if node in facilities else 'gray'
        edge_width = 4 if node in facilities else 0.8  # Thicker lines
        
        hexagon = RegularPolygon(
            (x, y), numVertices=6, radius=1,
            orientation=math.radians(30),
            edgecolor=edge_color,
            facecolor=color,
            linewidth=edge_width
        )
        ax.add_patch(hexagon)
        
        # Add weight as text label (conditionally)
        if show_weights:
            weight = weights[node]
            ax.text(x, y, str(weight), ha='center', va='center', 
                    fontsize=6, fontweight='bold', color='black')  # Slightly larger font

    # Set axis limits to show all hexagons
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                                norm=plt.Normalize(vmin=min_radius, vmax=max_radius))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Max Distance D_i (hops)', shrink=0.5)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_aspect('equal')
    plt.axis('off')
    title_text = f"{title}\nRed=High Demand (small D_i) | Blue=Low Demand (large D_i) | Black Border=Facility"
    if show_weights:
        title_text = f"{title}\nNumbers=Demand Weight | " + title_text[len(title)+1:]
    plt.title(title_text, fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()


def draw_square_grid(G, weights, radii, poi_nodes=None, facilities=None, title="Square Grid", show_weights=True):
    """
    Draw square grid with demand weights visible on nodes.
    Colors indicate radius (high demand = red, low demand = blue).
    """
    pos = {node: node for node in G.nodes()}
    poi_nodes = poi_nodes or set()
    facilities = facilities or set()
    
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')  # Increased from 16x16
    
    # Get radius range for coloring
    min_radius = min(radii.values())
    max_radius = max(radii.values())
    
    # Compute node colors based on radius
    node_colors = []
    for node in G.nodes():
        radius_val = radii[node]
        if max_radius > min_radius:
            norm_radius = (radius_val - min_radius) / (max_radius - min_radius)
        else:
            norm_radius = 0.5
        node_colors.append(plt.cm.RdYlBu(norm_radius))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.8)  # Thicker edges
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=300, edgecolors='gray', linewidths=0.8)  # Larger nodes
    
    # Highlight facilities with thick border
    if facilities:
        nx.draw_networkx_nodes(G, pos, nodelist=list(facilities), 
                              ax=ax, node_color='none', edgecolors='black', 
                              linewidths=4, node_size=300)  # Thicker border
    
    # Add weight labels (conditionally)
    if show_weights:
        labels = {node: str(weights[node]) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6, font_weight='bold')  # Larger font
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                                norm=plt.Normalize(vmin=min_radius, vmax=max_radius))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Max Distance D_i (hops)', shrink=0.5)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_aspect('equal')
    plt.axis('off')
    title_text = f"{title}\nRed=High Demand (small D_i) | Blue=Low Demand (large D_i) | Thick Border=Facility"
    if show_weights:
        title_text = f"{title}\nNumbers=Demand Weight | " + title_text[len(title)+1:]
    plt.title(title_text, fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()

def run_grid_tests():
    
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    results = []
    DISTANCE_THRESHOLD = 3  # hops

    for grid_type in ['square', 'hex']:
        for size in [45]:
            G = generate_grid(grid_type, size, size)
            demand_points = list(G.nodes)
            num_nodes = len(demand_points)

            for ratio in [0.001, 0.005]:
                p = max(1, math.ceil(ratio * num_nodes))
                random.seed(42)
                facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

                # Assign weights
                random.seed(42)
                num_pois = math.ceil(0.1 * len(demand_points))
                poi_nodes = set(random.sample(demand_points, num_pois))
                random.seed(42)
                
                weights = {
                    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                    for i in demand_points
                } if weighted else {i: 1 for i in demand_points}

                # Define hop thresholds
                D_min, D_max = 3, 8  # min and max hops allowed

                w_min = min(weights.values())
                w_max = max(weights.values())

                def assign_radius(w):
                    if w_max == w_min:
                        return DISTANCE_THRESHOLD  # uniform fallback
                    ratio = (w_max - w) / (w_max - w_min)
                    return int(round(D_min + ratio * (D_max - D_min)))

                # Compute radii for each demand point
                radii = {i: assign_radius(w) for i, w in weights.items()}

                # Build coverage sets (variable radii)
                coverage_sets = build_coverage_sets_variable(
                    G, demand_points, facility_sites, radii, edge_weight=None
                )

                _, selected, covered, total = build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p)
                coverage_percent = 100 * len(covered) / num_nodes
                demand_coverage_percent = 100 * total / sum(weights.values())
                # --- Draw the grid with demand weights visible ---
                if grid_type == "square":
                    draw_square_grid(G, weights, radii, poi_nodes=poi_nodes, facilities=selected,
                     title=f"Square Grid {size}x{size}, p={p}")
                else:  # hex
                    draw_hex_grid(G, weights, radii, poi_nodes=poi_nodes, facilities=selected, 
                  title=f"Hex Grid {size}x{size}, p={p}")

                results.append({
                    "Grid Type": grid_type,
                    "Size": f"{size}x{size}",
                    "Nodes": num_nodes,
                    "p": p,
                    "Weights": "Random" if weighted else "All 1s",
                    "Demand Coverage": f"{demand_coverage_percent:.2f}%",
                    "Graph Coverage": f"{coverage_percent:.2f}%",
                    "Distance Threshold": "Demand-driven"
                })

    df = pd.DataFrame(results)
    print("\nGRID TEST RESULTS (Demand-driven distance):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_grid_tests()