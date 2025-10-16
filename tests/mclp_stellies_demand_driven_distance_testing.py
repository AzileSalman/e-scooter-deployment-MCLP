import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
from mclp.demand_driven_dist_coverage_utils import build_coverage_sets_variable
import random
import math

def run_stellenbosch_tests():
    DISTANCE_THRESHOLD = 300  # meters
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    G = load_stellenbosch_graph()
    poi_nodes = get_poi_nodes(G, tags={
        'amenity': ["school", "university", "library"],
        'shop': ["mall", "supermarket"],
    })

    demand_points = list(G.nodes)
    num_nodes = len(demand_points)

    print("\n\\begin{tabular}{llllll}")
    print("\\textbf{Type} & \\textbf{Nodes} & \\textbf{p} & \\textbf{Weights} & \\textbf{Coverage} \\\\ \\hline")

    for ratio in [0.001, 0.005]:
        p = max(1, math.ceil(ratio * num_nodes))
        facility_sites = random.sample(demand_points, int(0.5 * num_nodes))

        # Assigning weights (higher for POIs and lower for regular nodes)
        weights = {
            i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
            for i in demand_points
            }if weighted else {i: 1 for i in demand_points}

        # Define min/max walking thresholds (policy choice)
        D_min, D_max = 300.0, 800.0   # meters

        w_min = min(weights.values())
        w_max = max(weights.values())

        def assign_radius(w):
            if w_max == w_min:
                return DISTANCE_THRESHOLD
            ratio = (w_max - w) / (w_max - w_min)
            return D_min + ratio * (D_max - D_min)

        # Compute radius for each demand point
        radii = {i: assign_radius(w) for i, w in weights.items()}

        coverage_sets = build_coverage_sets_variable(G, demand_points, facility_sites, radii, edge_weight="length")

        _, selected, covered, total = build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p)
        coverage_percent = 100 * len(covered) / num_nodes
        demand_coverage_percent = 100 * total / sum(weights.values())

        print(f"Stellenbosch & {num_nodes} & {p} & {'Non uniform' if weighted else 'All 1s'} & ({demand_coverage_percent:.2f}\\%) ({coverage_percent:.2f}\\%) \\\\")

    print("\\end{tabular}")

if __name__ == "__main__":
    run_stellenbosch_tests()
