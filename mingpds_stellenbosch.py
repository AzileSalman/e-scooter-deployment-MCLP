import random
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.visualisation import plot_solution
from mclp.mingpds_model import build_mingpds_model  
from mclp.coverage_utils import build_coverage_sets 

DISTANCE_THRESHOLD = 300  # meters

print("Loading Stellenbosch graph...")
G = load_stellenbosch_graph()

tags = {
    "amenity": ["school", "university", "library"],
    "shop": ["mall", "supermarket"],
}
poi_nodes = get_poi_nodes(G, tags)

demand_points = list(G.nodes)
facility_sites = random.sample(demand_points, int(0.5 * len(demand_points)))


coverage_sets = build_coverage_sets(G, demand_points, facility_sites, DISTANCE_THRESHOLD, edge_weight="length")


weights = {
    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
    for i in demand_points
}

alpha = 0.8
K = alpha * sum(weights.values())
print(f"Target coverage (K): {K:.2f}")

model, selected_facilities, covered_nodes, total_covered = build_mingpds_model(
    demand_points, facility_sites, coverage_sets, weights, K
)

print("\n--- MinGPDS Results ---")
print(f"Selected Facilities: {len(selected_facilities)} → {selected_facilities}")
print(f"Covered Demand: {total_covered} / {sum(weights.values())}")
print(f"Coverage %: {100 * total_covered / sum(weights.values()):.2f}%")

plot_solution(G, selected_facilities, poi_nodes)
