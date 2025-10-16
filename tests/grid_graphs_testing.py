import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
from mclp.mclp_model import build_mclp_model
import math
import random
import pandas as pd
from mclp.coverage_utils import build_coverage_sets  

def generate_grid(grid_type, width, height):
    if grid_type == "square":
        return nx.grid_2d_graph(width, height)
    elif grid_type == "hex":
        return generate_hex_center_graph(height, width)
    else:
        raise ValueError("Unsupported grid type")

def generate_hex_center_graph(rows, cols):
    G = nx.Graph()
    positions = {}
    for row in range(rows):
        for col in range(cols):
            node = (row, col)
            positions[node] = (col * 1.5, row + (0.5 if col % 2 else 0))
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

def run_grid_tests():
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    results = []
    DISTANCE_THRESHOLD = 3  # hops

    for grid_type in ['square', 'hex']:
        for size in [30, 45]:
            G = generate_grid(grid_type, size, size)
            demand_points = list(G.nodes)
            num_nodes = len(demand_points)

            for ratio in [0.001, 0.005]:
                p = max(1, math.ceil(ratio * num_nodes))
                facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

                # Build coverage sets using unified function
                coverage_sets = build_coverage_sets(G, demand_points, facility_sites, DISTANCE_THRESHOLD, edge_weight=None)

                # Assign weights
                poi_nodes = set(random.sample(demand_points, int(0.1 * len(demand_points))))
                weights = {
                    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                    for i in demand_points
                } if weighted else {i: 1 for i in demand_points}

                _, selected, covered, total = build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p)
                coverage_percent = 100 * len(covered) / num_nodes
                demand_coverage_percent = 100 * total / sum(weights.values())

                results.append({
                    "Grid Type": grid_type,
                    "Size": f"{size}x{size}",
                    "Nodes": num_nodes,
                    "p": p,
                    "Weights": "Random" if weighted else "All 1s",
                    "Demand Coverage": f"{demand_coverage_percent:.2f}%",
                    "Graph Coverage": f"{coverage_percent:.2f}%",
                    "Distance Threshold": DISTANCE_THRESHOLD
                })

    df = pd.DataFrame(results)
    print("\nGRID TEST RESULTS:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_grid_tests()
