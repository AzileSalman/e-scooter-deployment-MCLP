import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
from mclp.mclp_model import build_mclp_model
import math
import random
import pandas as pd

def generate_grid(grid_type, width, height):
    if grid_type == "square":
        return nx.grid_2d_graph(width, height)
    elif grid_type == "hex":
        return nx.hexagonal_lattice_graph(width, height)
    else:
        raise ValueError("Unsupported grid type")

def run_grid_tests():
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    results = []

    for grid_type in ['square', 'hex']:
        for size in [30, 45]:
            G = generate_grid(grid_type, size, size)
            demand_points = list(G.nodes)
            num_nodes = len(demand_points)
            

            for ratio in [0.001, 0.005]:
                p = max(1, math.ceil(ratio * num_nodes))
                facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

                coverage_sets = {
                    i: [j for j in list(G.neighbors(i)) + [i] if j in facility_sites]
                    for i in demand_points
                }
                poi_nodes = set(random.sample(demand_points, int(0.1 * len(demand_points))))

                weights = {
                    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                    for i in demand_points
                } if weighted else {i: 1 for i in demand_points}

                _, selected, covered, total = build_mclp_model(
                    demand_points, facility_sites, coverage_sets, weights, p
                )

                coverage_percent = 100 * len(covered) / num_nodes

                results.append({
                    "Grid Type": grid_type,
                    "Size": f"{size}x{size}",
                    "Nodes": num_nodes,
                    "p": p,
                    "Weights": "Random" if weighted else "All 1s",
                    "Demand Coverage": total,
                    "Graph Coverage": f"{coverage_percent:.2f}%"
                })

    df = pd.DataFrame(results)
    print("\nGRID TEST RESULTS:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_grid_tests()
