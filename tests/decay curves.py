import networkx as nx
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def generate_spatial_demand_exponential(G, poi_nodes, w_min=5, w_max=25, decay_rate=2.5):
    """
    Exponential decay: f(d) = e^(-λd)
    Fast initial decay, then gradual tailing off.
    Good for: phenomena that drop quickly then level off (e.g., sound, light)
    """
    print(f"Generating EXPONENTIAL spatial demand (λ={decay_rate})...")
    
    nodes = list(G.nodes())
    distances_to_nearest_poi = {node: float('inf') for node in nodes}
    
    for i, poi in enumerate(poi_nodes):
        if (i + 1) % 20 == 0:
            print(f"  Processing POI {i+1}/{len(poi_nodes)}...")
        lengths = nx.single_source_shortest_path_length(G, poi)
        for node, dist in lengths.items():
            distances_to_nearest_poi[node] = min(distances_to_nearest_poi[node], dist)
    
    max_distance = max(distances_to_nearest_poi.values())
    print(f"  Max distance: {max_distance} hops")
    
    weights = {}
    for node in nodes:
        d = distances_to_nearest_poi[node]
        
        if node in poi_nodes:
            weights[node] = random.randint(15, 25)
        else:
            norm_d = d / max_distance
            decay_factor = math.exp(-decay_rate * norm_d)  # Exponential decay
            weights[node] = int(round(w_min + (w_max - w_min) * decay_factor))
            weights[node] = max(w_min, min(w_max, weights[node]))
    
    print(f"  Weight range: {min(weights.values())} to {max(weights.values())}")
    return weights


def generate_spatial_demand_power(G, poi_nodes, w_min=5, w_max=25, alpha=2.0):
    """
    Inverse power decay: f(d) = 1 / (1 + d)^α
    Moderate decay that's not too aggressive.
    Good for: gravity models, spatial interaction
    """
    print(f"Generating INVERSE POWER spatial demand (α={alpha})...")
    
    nodes = list(G.nodes())
    distances_to_nearest_poi = {node: float('inf') for node in nodes}
    
    for i, poi in enumerate(poi_nodes):
        if (i + 1) % 20 == 0:
            print(f"  Processing POI {i+1}/{len(poi_nodes)}...")
        lengths = nx.single_source_shortest_path_length(G, poi)
        for node, dist in lengths.items():
            distances_to_nearest_poi[node] = min(distances_to_nearest_poi[node], dist)
    
    max_distance = max(distances_to_nearest_poi.values())
    print(f"  Max distance: {max_distance} hops")
    
    weights = {}
    for node in nodes:
        d = distances_to_nearest_poi[node]
        
        if node in poi_nodes:
            weights[node] = random.randint(15, 25)
        else:
            # Add 1 to avoid division by zero at d=0
            decay_factor = 1.0 / ((1 + d) ** alpha)  # Inverse power decay
            weights[node] = int(round(w_min + (w_max - w_min) * decay_factor))
            weights[node] = max(w_min, min(w_max, weights[node]))
    
    print(f"  Weight range: {min(weights.values())} to {max(weights.values())}")
    return weights


def generate_spatial_demand_gaussian(G, poi_nodes, w_min=5, w_max=25, sigma=8.0):
    """
    Gaussian decay: f(d) = e^(-(d²)/(2σ²))
    Bell curve - gentle at first, steep in middle, gentle again far away.
    Good for: normal distributions, diffusion processes
    """
    print(f"Generating GAUSSIAN spatial demand (σ={sigma})...")
    
    nodes = list(G.nodes())
    distances_to_nearest_poi = {node: float('inf') for node in nodes}
    
    for i, poi in enumerate(poi_nodes):
        if (i + 1) % 20 == 0:
            print(f"  Processing POI {i+1}/{len(poi_nodes)}...")
        lengths = nx.single_source_shortest_path_length(G, poi)
        for node, dist in lengths.items():
            distances_to_nearest_poi[node] = min(distances_to_nearest_poi[node], dist)
    
    max_distance = max(distances_to_nearest_poi.values())
    print(f"  Max distance: {max_distance} hops")
    
    weights = {}
    for node in nodes:
        d = distances_to_nearest_poi[node]
        
        if node in poi_nodes:
            weights[node] = random.randint(15, 25)
        else:
            decay_factor = math.exp(-(d**2) / (2 * sigma**2))  # Gaussian decay
            weights[node] = int(round(w_min + (w_max - w_min) * decay_factor))
            weights[node] = max(w_min, min(w_max, weights[node]))
    
    print(f"  Weight range: {min(weights.values())} to {max(weights.values())}")
    return weights


def plot_decay_curves(max_distance=20):
    """
    Visualize the three decay functions side by side.
    This helps understand their different behaviors.
    """
    distances = np.linspace(0, max_distance, 100)
    
    # Exponential (λ=2.5)
    exp_decay = np.exp(-2.5 * distances / max_distance)
    
    # Inverse power (α=2.0)
    power_decay = 1.0 / ((1 + distances) ** 2.0)
    power_decay = power_decay / power_decay[0]  # Normalize to start at 1
    
    # Gaussian (σ=8)
    gaussian_decay = np.exp(-(distances**2) / (2 * 8.0**2))
    
    plt.figure(figsize=(12, 6))
    plt.plot(distances, exp_decay, 'r-', linewidth=2, label='Exponential (λ=2.5)')
    plt.plot(distances, power_decay, 'b-', linewidth=2, label='Inverse Power (α=2.0)')
    plt.plot(distances, gaussian_decay, 'g-', linewidth=2, label='Gaussian (σ=8.0)')
    
    plt.xlabel('Distance from POI (hops)', fontsize=12)
    plt.ylabel('Decay Factor', fontsize=12)
    plt.title('Comparison of Distance Decay Functions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    # Add annotations
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(max_distance*0.7, 0.52, '50% decay line', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.show()
    
    # Print decay values at specific distances
    print("\nDecay factor at different distances:")
    print(f"{'Distance':<12} {'Exponential':<15} {'Inverse Power':<15} {'Gaussian':<15}")
    print("-" * 60)
    for d in [0, 5, 10, 15, 20]:
        if d <= max_distance:
            exp_val = math.exp(-2.5 * d / max_distance)
            pow_val = 1.0 / ((1 + d) ** 2.0) / (1.0 / 1.0)
            gauss_val = math.exp(-(d**2) / (2 * 8.0**2))
            print(f"{d:<12} {exp_val:<15.3f} {pow_val:<15.3f} {gauss_val:<15.3f}")


if __name__ == "__main__":
    # Visualize the decay curves
    print("Plotting decay function comparison...")
    plot_decay_curves(max_distance=20)