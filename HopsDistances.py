import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mclp.graph_utils import load_stellenbosch_graph
import networkx as nx

def analyze_stellenbosch_distances():
    G = load_stellenbosch_graph()
    nodes = list(G.nodes())
    
    results = []
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    print(f"Analyzing {total_pairs:,} node pairs...")
    
    for i, node1 in enumerate(nodes):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(nodes)} nodes")
        for node2 in nodes[i+1:]:
            try:
                meter_distance = nx.dijkstra_path_length(G, node1, node2, weight='length')
                hop_distance = nx.dijkstra_path_length(G, node1, node2)
                results.append((meter_distance, hop_distance))
            except nx.NetworkXNoPath:
                continue
    
    distances_300 = [hops for meters, hops in results if 280 <= meters <= 320]
    distances_800 = [hops for meters, hops in results if 750 <= meters <= 850]
    
    print("=== STELLENBOSCH DISTANCE ANALYSIS ===")
    print(f"Connected pairs: {len(results):,} out of {total_pairs:,}")
    
    if distances_300:
        avg_hops_300 = sum(distances_300) / len(distances_300)
        print(f"\n~300m distances ({len(distances_300)} samples):")
        print(f"Average hops: {avg_hops_300:.1f}")
        print(f"Range: {min(distances_300)}-{max(distances_300)} hops")
        print(f"Most common: {max(set(distances_300), key=distances_300.count)} hops")
    
    if distances_800:
        avg_hops_800 = sum(distances_800) / len(distances_800)
        print(f"\n~800m distances ({len(distances_800)} samples):")
        print(f"Average hops: {avg_hops_800:.1f}")
        print(f"Range: {min(distances_800)}-{max(distances_800)} hops")
        print(f"Most common: {max(set(distances_800), key=distances_800.count)} hops")
    
    all_meters = [meters for meters, hops in results]
    all_hops = [hops for meters, hops in results]
    
    print(f"\n=== GENERAL STATISTICS ===")
    print(f"Average meters per hop: {sum(all_meters)/sum(all_hops):.0f}m")
    
    target_300_hops = []
    target_800_hops = []
    
    for meters, hops in results:
        if 250 <= meters <= 350:
            target_300_hops.append(hops)
        if 700 <= meters <= 900:
            target_800_hops.append(hops)
    
    if target_300_hops and target_800_hops:
        most_common_300 = max(set(target_300_hops), key=target_300_hops.count)
        most_common_800 = max(set(target_800_hops), key=target_800_hops.count)
        
        print(f"\n=== RECOMMENDED GRID VALUES ===")
        print(f"For 300m equivalent: {most_common_300} hops")
        print(f"For 800m equivalent: {most_common_800} hops")
        print(f"Use: D_min={most_common_300}, D_max={most_common_800}")

if __name__ == "__main__":
    analyze_stellenbosch_distances()