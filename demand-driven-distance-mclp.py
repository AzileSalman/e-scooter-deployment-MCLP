import random
import networkx as nx
from mclp.graph_utils import load_stellenbosch_graph, get_poi_nodes
from mclp.mclp_model import build_mclp_model
from mclp.visualisation import plot_solution
from mclp.demand_driven_dist_coverage_utils import build_coverage_sets_variable


MAX_FACILITIES = 70  # cap on random facility size
DISTANCE_THRESHOLD = 300  # meters

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

# Assigning weights (higher for POIs and lower for regular nodes)
weights = {
    i: random.randint(15, 25) if i in poi_nodes else random.randint(5, 15)
    for i in demand_points
}

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

coverage_sets = build_coverage_sets_variable(
    G, demand_points, facility_sites, radii, edge_weight="length"
)


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
# Pick 2 random demand points
sample_demand_points = random.sample(demand_points, 10)

# Show their coverage sets
for i in sample_demand_points:
    print(f"Demand point {i} is covered by facility sites: {coverage_sets[i]}")
    print(f"Number of covering facilities: {len(coverage_sets[i])}")
    print("-" * 50)

# plotting solution
plot_solution(G, selected_facilities, poi_nodes)
