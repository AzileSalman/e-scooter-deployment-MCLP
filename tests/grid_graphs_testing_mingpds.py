import sys
import os
import math
import random
import pandas as pd
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mclp.mingpds_model import build_mingpds_model
from mclp.coverage_utils import build_coverage_sets
from pulp import PULP_CBC_CMD


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


def run_grid_mingpds_tests():
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    results = []
    DISTANCE_THRESHOLD = 3  # hops for grids

    for grid_type in ['square', 'hex']:
        size = 45
        G = generate_grid(grid_type, size, size)
        demand_points = list(G.nodes)
        num_nodes = len(demand_points)

        # 50% facility sites (same as MCLP testing)
        facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

        # Build coverage sets once
        coverage_sets = build_coverage_sets(G, demand_points, facility_sites,
                                            DISTANCE_THRESHOLD, edge_weight=None)

        # Assign weights
        if weighted:
            poi_nodes = set(random.sample(demand_points, int(0.1 * len(demand_points))))
            weights = {
                i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                for i in demand_points
            }
        else:
            weights = {i: 1 for i in demand_points}

        total_demand = sum(weights.values())

        # Only α = 0.5 and 0.8
        for alpha in [0.5, 0.8]:
            K = alpha * total_demand

            model, selected, covered, total_covered = build_mingpds_model(
                demand_points, facility_sites, coverage_sets, weights, K
            )
        
            
            demand_coverage_percent = 100 * total_covered / total_demand
            graph_coverage_percent = 100 * len(covered) / num_nodes

            results.append({
                "Grid Type": grid_type,
                "Size": f"{size}x{size}",
                "Nodes": num_nodes,
                "Alpha": alpha,
                "K (Target Demand)": f"{K:.0f}",
                "Selected Facilities": len(selected),
                "Demand Coverage %": f"{demand_coverage_percent:.2f}%",
                "Graph Coverage %": f"{graph_coverage_percent:.2f}%",
                "Weights": "Random" if weighted else "All 1s",
                "Distance Threshold": DISTANCE_THRESHOLD,
                "coverage sets": coverage_sets
                
            })

    df = pd.DataFrame(results)
    print("\n--- GRID MinGPDS TEST RESULTS (45x45 only) ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_grid_mingpds_tests()
