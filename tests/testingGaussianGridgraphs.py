import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from mclp.mclp_model import build_mclp_model
from mclp.demand_driven_dist_coverage_utils import build_coverage_sets_variable

def generate_grid(grid_type, width, height):
    if grid_type == "square":
        return nx.grid_2d_graph(width, height)
    elif grid_type == "hex":
        return generate_hex_center_graph(height, width)
    else:
        raise ValueError("Unsupported grid type")

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
    
    for row in range(rows):
        for col in range(cols):
            node = (row, col)
            if col % 2 == 0:
                directions = [(-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1), (-1, 1)]
            else:
                directions = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 0), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    G.add_edge(node, (r, c))
    
    nx.set_node_attributes(G, positions, "pos")
    return G

def draw_hex_grid(G, weights, radii, poi_nodes=None, facilities=None, title="Hexagonal Grid"):
    """
    Draw hexagonal grid with demand weights visible on nodes.
    Colors indicate radius (high demand = red, low demand = blue).
    """
    pos = nx.get_node_attributes(G, "pos")
    poi_nodes = poi_nodes or set()
    facilities = facilities or set()

    fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')

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
        edge_width = 3 if node in facilities else 0.5
        
        hexagon = RegularPolygon(
            (x, y), numVertices=6, radius=1,
            orientation=math.radians(30),
            edgecolor=edge_color,
            facecolor=color,
            linewidth=edge_width
        )
        ax.add_patch(hexagon)
        
        # Add weight as text label (make it small)
        weight = weights[node]
        ax.text(x, y, str(weight), ha='center', va='center', 
                fontsize=5, fontweight='bold', color='black')

    # Set axis limits to show all hexagons
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                                norm=plt.Normalize(vmin=min_radius, vmax=max_radius))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Max Distance D_i (hops)', shrink=0.6)
    
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(f"{title}\nNumbers=Demand Weight | Red=High Demand (small D_i) | Blue=Low Demand (large D_i)\nBlack Border=Facility", 
              fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_square_grid(G, weights, radii, poi_nodes=None, facilities=None, title="Square Grid"):
    """
    Draw square grid with demand weights visible on nodes.
    Colors indicate radius (high demand = red, low demand = blue).
    """
    pos = {node: node for node in G.nodes()}
    poi_nodes = poi_nodes or set()
    facilities = facilities or set()
    
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')
    
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
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=250, edgecolors='gray', linewidths=0.5)
    
    # Highlight facilities with thick border
    if facilities:
        nx.draw_networkx_nodes(G, pos, nodelist=list(facilities), 
                              ax=ax, node_color='none', edgecolors='black', 
                              linewidths=3, node_size=250)
    
    # Add weight labels
    labels = {node: str(weights[node]) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5, font_weight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                                norm=plt.Normalize(vmin=min_radius, vmax=max_radius))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Max Distance D_i (hops)', shrink=0.6)
    
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(f"{title}\nNumbers=Demand Weight | Red=High Demand (small D_i) | Blue=Low Demand (large D_i)\nThick Border=Facility", 
              fontsize=11)
    plt.tight_layout()
    plt.show()


def generate_spatial_demand_gaussian(G, poi_nodes, w_min=5, w_max=25, sigma=8.0):
    """
    Gaussian decay: f(d) = e^(-(d²)/(2σ²))
    Bell curve - gentle at first, steep in middle, gentle again far away.
    """
    print(f"Generating GAUSSIAN spatial demand (σ={sigma})...")
    
    nodes = list(G.nodes())
    distances_to_nearest_poi = {node: float('inf') for node in nodes}
    
    # Calculate from each POI to all nodes
    for i, poi in enumerate(poi_nodes):
        if (i + 1) % 20 == 0:
            print(f"  Processing POI {i+1}/{len(poi_nodes)}...")
        lengths = nx.single_source_dijkstra_path_length(G, poi, weight=None)
        for node, dist in lengths.items():
            distances_to_nearest_poi[node] = min(distances_to_nearest_poi[node], dist)
    
    max_distance = max(distances_to_nearest_poi.values())
    print(f"  Max distance: {max_distance} hops")
    
    weights = {}
    for node in nodes:
        d = distances_to_nearest_poi[node]
        
        if node in poi_nodes:
            weights[node] = random.randint(15, 25)
        else:
            # Gaussian decay
            decay_factor = math.exp(-(d**2) / (2 * sigma**2))
            weights[node] = int(round(w_min + (w_max - w_min) * decay_factor))
            weights[node] = max(w_min, min(w_max, weights[node]))
    
    print(f"  Weight range: {min(weights.values())} to {max(weights.values())}")
    return weights


def run_grid_tests_gaussian():
    """Run grid tests with Gaussian spatial demand - 5 runs with averaging."""
    
    use_gaussian = input("Use Gaussian spatial demand (g) or random demand (r)? ").strip().lower() == 'g'
    results = []
    DISTANCE_THRESHOLD = 3
    NUM_RUNS = 5

    for grid_type in ['square', 'hex']:
        for size in [45]:
            print(f"\n{'='*60}")
            print(f"Testing {grid_type} grid {size}x{size} - GAUSSIAN DECAY")
            print(f"{'='*60}")
            
            G = generate_grid(grid_type, size, size)
            demand_points = list(G.nodes)
            num_nodes = len(demand_points)
            
            # Select POI nodes once (same for all runs)
            random.seed(42)
            num_pois = math.ceil(0.1 * len(demand_points))
            poi_nodes = set(random.sample(demand_points, num_pois))
            print(f"Selected {len(poi_nodes)} POI nodes")

            for ratio in [0.001, 0.005]:
                p = max(1, math.ceil(ratio * num_nodes))
                print(f"\nRatio p={ratio} (p={p} facilities) - Running {NUM_RUNS} iterations...")
                
                demand_coverages = []
                graph_coverages = []
                
                # Run 5 times with different seeds
                for run in range(NUM_RUNS):
                    seed = 42 + run  # Seeds: 42, 43, 44, 45, 46
                    
                    # Seed for facility site selection
                    random.seed(seed)
                    facility_sites = random.sample(demand_points, int(0.5 * num_nodes))
                    
                    # Generate weights based on selected mode
                    random.seed(seed)
                    if use_gaussian:
                        weights = generate_spatial_demand_gaussian(
                            G, 
                            poi_nodes=poi_nodes,
                            w_min=5, 
                            w_max=25, 
                            sigma=8.0  # Gaussian parameter
                        )
                        demand_type = "Gaussian (σ=8.0)"
                    else:
                        weights = {i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                                   for i in demand_points}
                        demand_type = "Random"
                    
                    # Define hop thresholds for demand-driven model
                    D_min, D_max = 3, 8
                    w_min = min(weights.values())
                    w_max = max(weights.values())

                    def assign_radius(w):
                        if w_max == w_min:
                            return DISTANCE_THRESHOLD
                        ratio_val = (w_max - w) / (w_max - w_min)
                        return int(round(D_min + ratio_val * (D_max - D_min)))

                    radii = {i: assign_radius(w) for i, w in weights.items()}

                    # Build coverage sets
                    coverage_sets = build_coverage_sets_variable(
                        G, demand_points, facility_sites, radii, edge_weight=None
                    )

                    # Solve MCLP
                    _, selected, covered, total = build_mclp_model(
                        demand_points, facility_sites, coverage_sets, weights, p
                    )
                    
                    coverage_percent = 100 * len(covered) / num_nodes
                    demand_coverage_percent = 100 * total / sum(weights.values())
                    
                    demand_coverages.append(demand_coverage_percent)
                    graph_coverages.append(coverage_percent)
                    
                    print(f"  Run {run+1}/{NUM_RUNS}: Demand={demand_coverage_percent:.2f}%, Graph={coverage_percent:.2f}%")

                # Calculate averages
                avg_demand = sum(demand_coverages) / NUM_RUNS
                avg_graph = sum(graph_coverages) / NUM_RUNS
                
                print(f"  AVERAGE: Demand={avg_demand:.2f}%, Graph={avg_graph:.2f}%")

                results.append({
                    "Grid Type": grid_type,
                    "Size": f"{size}x{size}",
                    "Nodes": num_nodes,
                    "p": p,
                    "Demand Type": demand_type,
                    "Demand Coverage": f"{avg_demand:.2f}%",
                    "Graph Coverage": f"{avg_graph:.2f}%"
                })

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("GAUSSIAN DECAY - GRID TEST RESULTS (5-run averages):")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("gaussian_grid_results.csv", index=False)
    print("\nResults saved to gaussian_grid_results.csv")


if __name__ == "__main__":
    run_grid_tests_gaussian()