import random
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
from mclp.visualisation import plot_solution


MAX_FACILITIES = 70  # cap on random facility size

# loading graph
print("Loading Stellenbosch graph...")
G = load_stellenbosch_graph()

# loading points of interests (POIs)
tags = {
    "amenity": ["school", "university", "library"],
    "shop": ["mall", "supermarket"],
}
poi_nodes = get_poi_nodes(G, tags)

demand_points = list(G.nodes)
num_nodes = len(demand_points)

# randomly sampling facility locations of demand points
facility_sites = random.sample(demand_points, min(MAX_FACILITIES, len(demand_points)))

# coverage set based on neighbourhoods 
coverage_sets = {i: [j for j in list(G.neighbors(i)) + [i] if j in facility_sites] for i in demand_points}

# Assigning weights ( higher for POIs and lower for regular nodes)
weights = {
    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
    for i in demand_points
}

# facility constraint
p =  5
print(f"Total nodes: {num_nodes}, Facility budget (p): {p}")

# solving MCLP
model, selected_facilities, covered_nodes, total_covered = build_mclp_model(
    demand_points, facility_sites, coverage_sets, weights, p
)

# printing results
print("\n--- MCLP Results ---")
print(f"Selected Facilities: {selected_facilities}")
print(f"Covered Demand: {total_covered} / {sum(weights.values())}")
print(f"Graph Coverage Percentage: {100 * len(covered_nodes) / num_nodes:.2f}%")

# plotting solution
plot_solution(G, selected_facilities, poi_nodes)
