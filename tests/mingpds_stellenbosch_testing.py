import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import pandas as pd
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.coverage_utils import build_coverage_sets
from mclp.mingpds_model import build_mingpds_model  

def run_mingpds_tests():
    DISTANCE_THRESHOLD = 300  # meters
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    results = []

    # Load Stellenbosch graph
    G = load_stellenbosch_graph()
    poi_nodes = get_poi_nodes(G, tags={
        "amenity": ["school", "university", "library"],
        "shop": ["mall", "supermarket"],
    })

    demand_points = list(G.nodes)
    facility_sites = random.sample(demand_points, int(0.5 * len(demand_points)))

    # Build coverage sets
    coverage_sets = build_coverage_sets(G, demand_points, facility_sites,
                                        DISTANCE_THRESHOLD, edge_weight="length")

    # Assign weights
    weights = {
        i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
        for i in demand_points
    } if weighted else {i: 1 for i in demand_points}

    total_demand = sum(weights.values())
    num_nodes = len(demand_points)

    # Loop over alpha values (coverage proportions)
    for alpha in [0.5, 0.8]:
        K = alpha * total_demand

        model, selected, covered, total_covered = build_mingpds_model(
            demand_points, facility_sites, coverage_sets, weights, K
        )

        demand_coverage_percent = 100 * total_covered / total_demand
        graph_coverage_percent = 100 * len(covered) / num_nodes

        results.append({
            "Alpha": alpha,
            "K (Target Demand)": f"{K:.0f}",
            "Selected Facilities": len(selected),
            "Demand Coverage %": f"{demand_coverage_percent:.2f}%",
            "Graph Coverage %": f"{graph_coverage_percent:.2f}%"
        })

    df = pd.DataFrame(results)
    print("\n--- MinGPDS TEST RESULTS ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_mingpds_tests()
