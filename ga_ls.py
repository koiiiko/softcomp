import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

class ClusterBasedDroneRouting_Hybrid:
    def __init__(self, csv_file=None, road_points=None, n_drones=None):
        self.df = pd.read_csv(csv_file)
        
        # Extract coordinates and cluster assignments
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1
        
        # Store drones per cluster
        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")
        
        # Accept road_points as dict or list
        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)
        
        # Ensure enough road points
        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])
        
        # Distance matrix
        self.all_locations = self.road_points + self.coordinates
        self.n_total_locations = len(self.all_locations)
        self.dist_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self):
        """Calculate distance matrix using Haversine formula"""
        dist = np.zeros((self.n_total_locations, self.n_total_locations))
        for i in range(self.n_total_locations):
            for j in range(self.n_total_locations):
                if i != j:
                    lat1, lon1 = self.all_locations[i]
                    lat2, lon2 = self.all_locations[j]
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    r = 6371  # km
                    dist[i][j] = c * r
        return dist
    
    def _route_distance(self, route):
        """Compute total route distance"""
        return sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    # Clustering
    def split_hotspots_for_cluster(self, cluster_id, n_drones):
        """Split hotspots in a cluster into n_drones groups using KMeans"""
        indices = [i for i, c in enumerate(self.clusters) if c == cluster_id]
        if not indices:
            return []
        coords = np.array([self.coordinates[i] for i in indices])
        if len(indices) <= n_drones:
            return [[idx] for idx in indices] + [[] for _ in range(n_drones - len(indices))]
        kmeans = KMeans(n_clusters=n_drones, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        groups = [[] for _ in range(n_drones)]
        for idx, label in zip(indices, labels):
            groups[label].append(idx)
        return groups
    
    # Local Search (2-opt)
    def _2opt_swap(self, route, i, j):
        """Perform 2-opt swap between two indices"""
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route
    
    def _local_search_2opt(self, route, max_iter=100):
        """
        Apply 2-opt local search to improve a given route. This is the LOCAL REFINEMENT step.
        Parameters:
        - route: List of location indices (without start/end points)
        - max_iter: Maximum number of improvement iterations
        Returns: Improved route (just the inner cities, no start/end)
        """
        if len(route) <= 2:  # Too small to optimize
            return route
        
        current_route = route[:]
        best_distance = self._route_distance_segment(current_route)
        
        for iteration in range(max_iter):
            improved = False
            for i in range(len(current_route) - 1):
                for j in range(i + 1, len(current_route)):
                    # Try 2-opt swap
                    new_route = current_route[:i] + current_route[i:j+1][::-1] + current_route[j+1:]
                    new_distance = self._route_distance_segment(new_route)
                    
                    if new_distance < best_distance:
                        current_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:  # Local optimum reached
                break
        
        return current_route
    
    def _route_distance_segment(self, segment):
        """Calculate distance for a route segment (without start/end points)"""
        if len(segment) <= 1:
            return 0
        return sum(self.dist_matrix[segment[i]][segment[i+1]] for i in range(len(segment)-1))
    
    # Genetic Algorithm Components
    def _tournament_selection(self, population, k=3):
        """Tournament selection for parent selection"""
        chosen = random.sample(population, min(k, len(population)))
        chosen.sort(key=lambda r: self._route_distance_segment(r))
        return chosen[0]
    
    def _order_crossover(self, parent1, parent2):
        """Order crossover (OX) for TSP"""
        if len(parent1) < 2:
            return parent1[:]
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        p2_elements = [x for x in parent2 if x not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_elements[idx]
                idx += 1
        return child
    
    def _mutate(self, route, mutation_rate=0.1):
        """Swap mutation"""
        route = route[:]
        if len(route) > 1 and random.random() < mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    # HYBRID GA + Local Search
    def solve_tsp_hybrid(self, cluster_id, hotspot_indices=None, 
                        population_size=50, generations=200, mutation_rate=0.1,
                        local_search_freq=10, local_search_intensity=50):
        """
        Hybrid Genetic Algorithm + Local Search for TSP
        Strategy:
        1. GA explores the solution space globally
        2. Every 'local_search_freq' generations, apply 2-opt to best individuals
        3. Final solution gets intensive 2-opt refinement
        Parameters:
        - cluster_id: Starting point cluster ID
        - hotspot_indices: Subset of hotspots to visit (or None for all)
        - population_size: Number of individuals in GA population
        - generations: Number of GA generations
        - mutation_rate: Probability of mutation
        - local_search_freq: Apply LS every N generations (memetic approach)
        - local_search_intensity: Max iterations for 2-opt refinement
        """
        # Setup hotspots to visit
        if hotspot_indices is None:
            cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
        else:
            cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]
        
        # Handle trivial cases
        if not cluster_hotspots:
            return []
        if len(cluster_hotspots) == 1:
            return [cluster_id] + cluster_hotspots + [cluster_id]
        
        start_point = cluster_id
        
        # Initialize population with random permutations
        population = [random.sample(cluster_hotspots, len(cluster_hotspots)) 
                     for _ in range(population_size)]
                
        # HYBRID EVOLUTION LOOP
        for gen in range(generations):
            # Standard GA operations
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            # MEMETIC COMPONENT: Periodic local search on elite individuals
            if (gen + 1) % local_search_freq == 0:
                # Sort population by fitness
                population.sort(key=lambda r: self._route_distance_segment(r))
                
                # Apply 2-opt to top 20% of population
                elite_count = max(1, population_size // 5)
                for i in range(elite_count):
                    population[i] = self._local_search_2opt(
                        population[i], 
                        max_iter=local_search_intensity
                    )
                
                # Get best distance for monitoring
                best_route_so_far = population[0]
                best_dist = self._route_distance([start_point] + best_route_so_far + [start_point])
        
        # FINAL INTENSIVE LOCAL SEARCH
        population.sort(key=lambda r: self._route_distance_segment(r))
        best_route = self._local_search_2opt(
            population[0], 
            max_iter=local_search_intensity * 3  # 3x more intensive
        )
        
        final_route = [start_point] + best_route + [start_point]
        final_distance = self._route_distance(final_route)
        
        return final_route
    
    # Optimization Wrapper
    def optimize_all_clusters(self, population_size=50, generations=200, 
                             mutation_rate=0.1, local_search_freq=10, 
                             local_search_intensity=50):
        """
        Optimize routes for all clusters and drones using Hybrid GA+LS
        Parameters:
        - population_size: GA population size
        - generations: Number of GA generations
        - mutation_rate: Mutation probability
        - local_search_freq: Apply LS every N generations
        - local_search_intensity: Max 2-opt iterations per LS application
        """
        all_routes = {}
        
        for cluster_id in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cluster_id, 1)
            cluster_routes = []
            
            if n_drones > 1:
                groups = self.split_hotspots_for_cluster(cluster_id, n_drones)
                for drone_idx, group in enumerate(groups):
                    route = self.solve_tsp_hybrid(
                        cluster_id, 
                        hotspot_indices=group,
                        population_size=population_size,
                        generations=generations,
                        mutation_rate=mutation_rate,
                        local_search_freq=local_search_freq,
                        local_search_intensity=local_search_intensity
                    )
                    cluster_routes.append(route)
            else:
                route = self.solve_tsp_hybrid(
                    cluster_id,
                    population_size=population_size,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    local_search_freq=local_search_freq,
                    local_search_intensity=local_search_intensity
                )
                cluster_routes = [route]
            
            all_routes[cluster_id] = cluster_routes
        
        return all_routes
    
    # Visualization & Output
    def visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))
        colors = ['red','blue','green','orange','purple','brown','pink','gray']
        
        # Road points
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
        
        # Hotspots
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c=colors[cid % len(colors)], s=100, alpha=0.7, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', 
                    fontsize=8, fontweight='bold')
        
        # Routes
        for cid, routes in all_routes.items():
            for d_idx, route in enumerate(routes):
                if len(route) <= 1: 
                    continue
                lats, lons = [], []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        lat, lon = self.coordinates[loc - len(self.road_points)]
                    lats.append(lat)
                    lons.append(lon)
                plt.plot(lons, lats, 'o-', color=colors[cid % len(colors)],
                        label=f'Cluster {cid} Drone {d_idx+1}', linewidth=2, markersize=6)
        
        plt.title('Drone Routes (Hybrid GA + Local Search)', fontsize=16, fontweight='bold')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_cluster_routes(self, all_routes):
        print("FINAL DRONE ROUTES (Hybrid GA + Local Search)")
        
        total_distance_all = 0
        total_time_all = 0
        
        for cid in range(len(self.road_points)):
            num_drones = self.drones_per_cluster.get(cid, 1)
            print(f"\nCluster {cid} (Drones: {num_drones}):")
            road_lat, road_lon = self.road_points[cid]
            print(f"  Start from road point: ({road_lat:.5f}, {road_lon:.5f})")
            
            routes = all_routes[cid]
            for d_idx, route in enumerate(routes):
                if len(route) <= 1:
                    print(f"  Drone {d_idx+1}: No hotspots assigned")
                    continue
                
                print(f"  Drone {d_idx+1} Route: ", end="")
                total_distance = 0
                
                for i in range(len(route)-1):
                    from_idx, to_idx = route[i], route[i+1]
                    if from_idx < len(self.road_points):
                        print(f"R{from_idx} -> ", end="")
                    else:
                        print(f"H{from_idx - len(self.road_points)} -> ", end="")
                    total_distance += self.dist_matrix[from_idx][to_idx]
                
                last = route[-1]
                print(f"R{last}" if last < len(self.road_points) 
                     else f"H{last - len(self.road_points)}")
                
                flight_time = total_distance * 1000 / 30 / 60  # 30 m/s speed
                print(f"    Total distance: {total_distance:.2f} km")
                print(f"    Est. flight time: {flight_time:.2f} minutes")
                
                total_distance_all += total_distance
                total_time_all += flight_time
        
        print(f"TOTAL DISTANCE (ALL DRONES): {total_distance_all:.2f} km")
        print(f"TOTAL FLIGHT TIME (ALL DRONES): {total_time_all:.2f} minutes")
