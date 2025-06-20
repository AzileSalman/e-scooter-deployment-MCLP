import osmnx as ox

def load_stellenbosch_graph():
    G = ox.graph_from_place("Stellenbosch, South Africa", network_type="drive")
    G = ox.convert.to_undirected(G)
    return ox.project_graph(G)

def get_poi_nodes(G, tags):
    pois = ox.features_from_place("Stellenbosch, South Africa", tags)
    pois = pois.to_crs(G.graph["crs"])
    pois["geometry"] = pois["geometry"].centroid
    return list(pois["geometry"].apply(lambda g: ox.distance.nearest_nodes(G, g.x, g.y)).unique())
