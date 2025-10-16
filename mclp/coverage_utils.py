from collections import defaultdict
import networkx as nx

def build_coverage_sets(G, demand_points, facility_sites, max_distance, edge_weight=None):
    """
    Builds coverage sets for each demand point based on distance threshold.
    
    G: NetworkX graph
    demand_points: list of demand nodes
    facility_sites: list of potential facility nodes
    max_distance: max distance (hops or meters)
    edge_weight: None for unweighted graphs (grids), or "length" for weighted graphs (Stellenbosch)
    """
    coverage = defaultdict(list)

    for j in facility_sites:
        lengths = nx.single_source_dijkstra_path_length(G, j, cutoff=max_distance, weight=edge_weight)
        for i in lengths:
            if i in demand_points:
                coverage[i].append(j)

    # Ensure every demand point appears (even if empty list)
    #key = demand points (i) , values = list of facilities (j) that can cover the demand points
    return {i: coverage.get(i, []) for i in demand_points}
