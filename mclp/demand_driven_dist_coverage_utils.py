from collections import defaultdict
import networkx as nx
from mclp.coverage_utils import build_coverage_sets  # import your old facility-centric builder

def build_coverage_sets_variable(G, demand_points, facility_sites, radii, edge_weight=None, default_threshold=300.0):
    """
    Builds coverage sets for each demand point.
    - If all radii are the same -> fall back to the standard facility-centric coverage builder.
    - If radii vary -> compute demand-dependent coverage sets.

    G: NetworkX graph
    demand_points: list of demand nodes
    facility_sites: list of potential facility nodes
    radii: dict {i: D_i} mapping each demand point to its max distance
    edge_weight: None for unweighted graphs, or "length" for weighted graphs
    default_threshold: value used if all radii are equal (baseline case)
    """
    # --- Uniform case: all demand nodes share the same radius ---
    unique_radii = set(radii.values())
    if len(unique_radii) == 1:
        # Just call the baseline builder with the common threshold
        threshold = unique_radii.pop()
        return build_coverage_sets(G, demand_points, facility_sites, threshold, edge_weight=edge_weight)

    # --- Weighted case: demand-driven expansion ---
    coverage = defaultdict(list)
    for i in demand_points:
        D_i = radii[i]
        lengths = nx.single_source_dijkstra_path_length(G, i, cutoff=D_i, weight=edge_weight)
        for j in facility_sites:
            if j in lengths and lengths[j] <= D_i:
                coverage[i].append(j)

    return {i: coverage.get(i, []) for i in demand_points}
