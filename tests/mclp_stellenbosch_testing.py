import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
import random
import math

def run_stellenbosch_tests():
    # user input if demand is weighted or not
    weighted = input("Weighted (w) or uniform (u)? ").strip().lower() == 'w'
    # loading street network of stellenbosch
    G = load_stellenbosch_graph()
    # getting points of interests like schools
    poi_nodes = get_poi_nodes(G, tags={
        'amenity': ["school", "university", "library"],
        'shop': ["mall", "supermarket"],
    })
    
    demand_points = list(G.nodes)
    num_nodes = len(demand_points)

    print("\n\\begin{tabular}{llllll}")
    print("\\textbf{Type} & \\textbf{Nodes} & \\textbf{p} & \\textbf{Weights} & \\textbf{Coverage} \\\\ \\hline")
    # solving mclp for different values of p
    for ratio in [0.001, 0.005]:
        p = max(1, math.ceil(ratio * num_nodes)) # number of facilities to place
        # randomly choosing 50% of nodes as potential facility sites
        facility_sites = random.sample(demand_points, int(0.5 * num_nodes))
        # coverage relationship based on adjacency
        coverage_sets = {
            i: [j for j in list(G.neighbors(i)) + [i] if j in facility_sites]
            for i in demand_points
        }
        # assigning weights, higher if node is a POI & lower if not
        weights = {
            i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
            for i in demand_points
        }if weighted else {i: 1 for i in demand_points}
        # solving MCLP and extracting results
        _, selected, covered, total = build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p)
        # percentage of covered nodes
        coverage_percent = 100 * len(covered) / num_nodes
        print(f"Stellenbosch & {num_nodes} & {p} & {'Non uniform' if weighted else 'All 1s'} & {total} ({coverage_percent:.2f}\\%) \\\\")

    print("\\end{tabular}")

if __name__ == "__main__":
    run_stellenbosch_tests()
