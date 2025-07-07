import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point


def load_stellenbosch_graph():
    # loading stellenbosch street network
    G = ox.graph_from_place("Stellenbosch, South Africa", network_type="drive")
    # converting graph to an undirected graph
    G = ox.convert.to_undirected(G)
    # projecting graph to a metric coordinate system instead of lat/lon
    return ox.project_graph(G)


def get_poi_nodes(G, tags):
    # Getting the POIs from tags
    pois = ox.features_from_place("Stellenbosch, South Africa", tags)
    # matching the coordinate system of the POIs to graph coordinate system
    pois = pois.to_crs(G.graph["crs"])
    pois["geometry"] = pois["geometry"].centroid
    poi_nodes_from_tags = pois["geometry"].apply(lambda g: ox.distance.nearest_nodes(G, g.x, g.y)).unique()

    # Adding known university residences manually 
    known_residences = [
        "Minerva, Stellenbosch", "Academia, Stellenbosch", "Metanoia, Stellenbosch",
        "Dagbreek, Stellenbosch", "Huis ten Bosch, Stellenbosch", "Helshoogte, Stellenbosch",
        "Huis Marais, Stellenbosch", "Eendrag, Stellenbosch", "Monica, Stellenbosch",
        "Huis Visser, Stellenbosch"
    ]

    res_coords = []
    for name in known_residences:
        try:
            latlon = ox.geocode(name)
            point = Point(latlon[1], latlon[0])  # (lon, lat)
            res_coords.append(point)
        except Exception as e:
            print(f"Could not geocode {name}: {e}")

    # converting to GeoDataFrame and matching CRS
    if res_coords:
        gdf = gpd.GeoDataFrame(geometry=res_coords, crs="EPSG:4326")
        gdf = gdf.to_crs(G.graph["crs"])
        res_nodes = ox.distance.nearest_nodes(G, gdf.geometry.x, gdf.geometry.y)
    else:
        res_nodes = []

    # Combining and returning unique node IDs
    all_poi_nodes = set(poi_nodes_from_tags).union(res_nodes)
    return list(all_poi_nodes)

