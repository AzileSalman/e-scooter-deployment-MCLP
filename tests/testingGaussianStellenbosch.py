import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
from mclp.demand_driven_dist_coverage_utils import build_coverage_sets_variable
import networkx as nx
import random
import math

def generate_spatial_demand_gaussian(G, poi_nodes, w_min=5, w_max=25, sigma=800.0):
    """
    Generate spatially correlated demand weights using Gaussian decay
    based on road-network distance to nearest POI node.
    
    For Stellenbosch: sigma in meters (default 800.0 meters)
    """
    print(f"Generating GAUSSIAN spatial demand (σ={sigma}m) for {len(G.nodes())} nodes with {len(poi_nodes)} POIs...")

    nodes = list(G.nodes())
    distances_to_nearest_poi = {node: float('inf') for node in nodes}

    for i, poi in enumerate(poi_nodes):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(poi_nodes)} POIs...")
        lengths = nx.single_source_dijkstra_path_length(G, poi, weight="length")
        for node, dist in lengths.items():
            if dist < distances_to_nearest_poi[node]:
                distances_to_nearest_poi[node] = dist

    max_distance = max(d for d in distances_to_nearest_poi.values() if d < float('inf'))
    print(f"  Max distance from any node to nearest POI: {max_distance:.2f} m")

    weights = {}
    for node, d in distances_to_nearest_poi.items():
        if node in poi_nodes:
            weights[node] = random.randint(15, 25)
        else:
            # Gaussian decay
            decay_factor = math.exp(-(d**2) / (2 * sigma**2))
            w = int(round(w_min + (w_max - w_min) * decay_factor))
            weights[node] = max(w_min, min(w_max, w))
    
    print(f"  Weight range: {min(weights.values())} to {max(weights.values())}")
    return weights


def run_stellenbosch_tests_gaussian():
    DISTANCE_THRESHOLD = 300  # meters
    mode = input("Weighted (w), Uniform (u), or Gaussian Spatial Decay (g)? ").strip().lower()
    NUM_RUNS = 5

    G = load_stellenbosch_graph()
    poi_nodes = get_poi_nodes(G, tags={
        'amenity': ["school", "university", "library"],
        'shop': ["mall", "supermarket"],
    })

    demand_points = list(G.nodes)
    num_nodes = len(demand_points)

    print("\n" + "="*60)
    print(f"STELLENBOSCH - GAUSSIAN DECAY TEST ({NUM_RUNS} runs with averaging)")
    print("="*60)
    
    print("\n\\begin{tabular}{llllll}")
    print("\\textbf{Type} & \\textbf{Nodes} & \\textbf{p} & \\textbf{Decay} & \\textbf{Demand Cov} & \\textbf{Graph Cov} \\\\ \\hline")

    for ratio in [0.001, 0.005]:
        p = max(1, math.ceil(ratio * num_nodes))
        print(f"\nRunning p={p} - {NUM_RUNS} iterations...")
        
        demand_coverages = []
        graph_coverages = []
        
        # Run 5 times with different seeds
        for run in range(NUM_RUNS):
            seed = 42 + run  # Seeds: 42, 43, 44, 45, 46
            
            random.seed(seed)
            facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

            # Choose weighting scheme
            random.seed(seed)
            if mode == 'g':
                weights = generate_spatial_demand_gaussian(
                    G, poi_nodes, 
                    w_min=5, w_max=25, 
                    sigma=800.0  # 800 meters for Stellenbosch
                )
                weight_type = "Gaussian(σ=800)"
            elif mode == 'w':
                weights = {i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
                           for i in demand_points}
                weight_type = "Non uniform"
            else:
                weights = {i: 1 for i in demand_points}
                weight_type = "All 1s"

            D_min, D_max = 300.0, 800.0   # meters
            w_min, w_max = min(weights.values()), max(weights.values())

            def assign_radius(w):
                if w_max == w_min:
                    return DISTANCE_THRESHOLD
                ratio_val = (w_max - w) / (w_max - w_min)
                return D_min + ratio_val * (D_max - D_min)

            radii = {i: assign_radius(w) for i, w in weights.items()}

            coverage_sets = build_coverage_sets_variable(
                G, demand_points, facility_sites, radii, edge_weight="length"
            )

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
        print(f"Stellenbosch & {num_nodes} & {p} & {weight_type} & {avg_demand:.2f}\\% & {avg_graph:.2f}\\% \\\\")

    print("\\end{tabular}")


if __name__ == "__main__":
    run_stellenbosch_tests_gaussian()