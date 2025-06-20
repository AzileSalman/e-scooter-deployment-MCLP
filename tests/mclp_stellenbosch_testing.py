import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
import random
import math

def run_stellenbosch_tests():
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'

    G = load_stellenbosch_graph()
    poi_nodes = get_poi_nodes(G, tags={
        'amenity': ['school', 'university', 'library'],
        'shop': True,
        'building': ['residential', 'apartments']
    })

    demand_points = list(G.nodes)
    num_nodes = len(demand_points)

    print("\n\\begin{tabular}{llllll}")
    print("\\textbf{Type} & \\textbf{Nodes} & \\textbf{p} & \\textbf{Weights} & \\textbf{Coverage} \\\\ \\hline")

    for ratio in [0.001, 0.005]:
        p = max(1, math.ceil(ratio * num_nodes))
        facility_sites = random.sample(demand_points, int(0.5 * num_nodes))
        coverage_sets = {
            i: [j for j in list(G.neighbors(i)) + [i] if j in facility_sites]
            for i in demand_points
        }

        weights = {
            i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
            for i in demand_points
        }if weighted else {i: 1 for i in demand_points}

        _, selected, covered, total = build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p)

        coverage_percent = 100 * len(covered) / num_nodes
        print(f"Stellenbosch & {num_nodes} & {p} & {'Non uniform' if weighted else 'All 1s'} & {total} ({coverage_percent:.2f}\\%) \\\\")

    print("\\end{tabular}")

if __name__ == "__main__":
    run_stellenbosch_tests()
