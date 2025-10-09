import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random, math
from ts import ClusterBasedDroneRoutingTS
from ga import ClusterBasedDroneRouting  # pastikan file GA kamu bernama ga.py
from collections import deque

class HybridGATabuDroneRouting(ClusterBasedDroneRouting):
    def __init__(self, csv_file, road_points, n_drones=None,
                 tabu_tenure=10, max_iter=200, neighborhood_size=None):
        super().__init__(csv_file, road_points, n_drones)
        self.ts_solver = ClusterBasedDroneRoutingTS(
            csv_file=csv_file, road_points=road_points, n_drones=n_drones
        )
        self.tabu_tenure = tabu_tenure
        self.max_iter = max_iter
        self.neighborhood_size = neighborhood_size

    # -------------------- Refinement Step (Tabu Search) --------------------
    def refine_with_tabu(self, route):
        locs = [self.all_locations[i] for i in route]
        dist_matrix = self.ts_solver.build_dist_matrix(locs)
        best_route, best_cost = self.ts_solver.tabu_search(
            dist_matrix,
            tabu_tenure=self.tabu_tenure,
            max_iter=self.max_iter,
            neighborhood_size=self.neighborhood_size
        )
        refined_route = [route[i] for i in best_route]
        if refined_route[-1] == refined_route[0]:
            refined_route = refined_route[:-1]
        return refined_route, best_cost

    # -------------------- Hybrid Optimization --------------------
    def optimize_all_clusters_hybrid(self):
        print("=== Running Hybrid GA + Tabu Search Optimization ===")
        all_routes_ga = super().optimize_all_clusters()
        all_routes_hybrid = {}

        for cid, routes in all_routes_ga.items():
            print(f"\n--- Refining Cluster {cid} ---")
            refined_routes = []
            for d_idx, route in enumerate(routes):
                print(f"  Drone {d_idx+1}: Refining GA route with Tabu Search...")
                refined, cost = self.refine_with_tabu(route)
                refined_routes.append(refined)
            all_routes_hybrid[cid] = refined_routes

        return all_routes_hybrid

    # -------------------- Visualization --------------------
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
            plt.scatter(lon, lat, c=colors[cid % len(colors)], s=100,
                        alpha=0.7, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center',
                     fontsize=8, fontweight='bold')

        # Routes
        for cid, routes in all_routes.items():
            for d_idx, route in enumerate(routes):
                if len(route) <= 1: continue
                lats, lons = [], []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        lat, lon = self.coordinates[loc - len(self.road_points)]
                    lats.append(lat)
                    lons.append(lon)
                plt.plot(lons, lats, 'o-', color=colors[cid % len(colors)],
                         label=f'Cluster {cid} Drone {d_idx+1}',
                         linewidth=2, markersize=6)
        plt.title('Drone Routes (Hybrid GA + Tabu Search)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # -------------------- Print Output --------------------
    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Hybrid GA + Tabu Search) ===")
        for cid, routes in all_routes.items():
            print(f"\nCluster {cid} (Drones: {len(routes)}):")
            for d_idx, route in enumerate(routes):
                # Bangun ulang matriks jarak hanya untuk rute ini
                locs = [self.all_locations[i] for i in route]
                dist_matrix = self.ts_solver.build_dist_matrix(locs)

                total_dist = 0
                for i in range(len(route)-1):
                    total_dist += dist_matrix[i][i+1]

                est_time = total_dist * 1000 / 30 / 60  # 30 m/s
                print(f"  Drone {d_idx+1} Route: {route}")
                if est_time > 170:
                    print(f"    ⚠️ WARNING: Estimated flight time {est_time:.2f} min exceeds 170 min limit!")
                print(f"    Total distance: {total_dist:.2f} km | Est. time: {est_time:.2f} min")

class ClusterBasedDroneRoutingHybridTabu:
    drone_speed = 30
    max_flight_time = 170
    max_distance = (drone_speed * max_flight_time * 60) / 1000
    penalty_factor = 100

    def __init__(self, csv_file=None, road_points=None, n_drones=None,
                 tabu_on_children=True, tabu_tenure=15, tabu_budget=50):
        self.df = pd.read_csv(csv_file)
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        # Drone distribution
        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")

        # Road points
        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)

        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])

        # Parameters for hybridization
        self.tabu_on_children = tabu_on_children
        self.tabu_tenure = tabu_tenure
        self.tabu_budget = tabu_budget

        # Distance matrix
        self.all_locations = self.road_points + self.coordinates
        self.n_total_locations = len(self.all_locations)
        self.dist_matrix = self._calculate_distance_matrix()

    # ------------------ Utility ------------------
    def _calculate_distance_matrix(self):
        dist = np.zeros((self.n_total_locations, self.n_total_locations))
        for i in range(self.n_total_locations):
            for j in range(self.n_total_locations):
                if i != j:
                    lat1, lon1 = self.all_locations[i]
                    lat2, lon2 = self.all_locations[j]
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                    c = 2*np.arcsin(np.sqrt(a))
                    dist[i][j] = 6371 * c
        return dist

    def _route_distance(self, route):
        return sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def _fitness(self, route):
        distance = self._route_distance(route)
        penalty = 0
        if distance > self.max_distance:
            penalty = (distance - self.max_distance) * self.penalty_factor
        return distance + penalty

    # ------------------ Genetic Algorithm ------------------
    def _tournament_selection(self, population, k=3):
        chosen = random.sample(population, min(k, len(population)))
        chosen.sort(key=lambda r: self._fitness(r))
        return chosen[0]

    def _order_crossover(self, parent1, parent2):
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
        route = route[:]
        if len(route) > 1 and random.random() < mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # ------------------ Tabu Search local improvement ------------------
    def _generate_neighbors(self, route):
        """Generate all swap-based neighbors."""
        neighbors = []
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route) - 1):
                new = route[:]
                new[i], new[j] = new[j], new[i]
                neighbors.append((new, (i, j)))
        return neighbors

    def _tabu_search(self, route, tabu_tenure=None, max_iter=None):
        """Basic Tabu Search using swap neighborhood."""
        if tabu_tenure is None:
            tabu_tenure = self.tabu_tenure
        if max_iter is None:
            max_iter = self.tabu_budget

        best = route
        best_fit = self._fitness(best)
        current = best[:]
        tabu_list = deque(maxlen=tabu_tenure)
        best_global = best
        best_global_fit = best_fit

        for _ in range(max_iter):
            neighbors = self._generate_neighbors(current)
            neighbors = [(r, m) for r, m in neighbors if m not in tabu_list]
            if not neighbors:
                break

            # Evaluate and choose best neighbor
            best_neighbor, best_move = min(neighbors, key=lambda x: self._fitness(x[0]))
            best_neighbor_fit = self._fitness(best_neighbor)

            # Update
            tabu_list.append(best_move)
            current = best_neighbor
            if best_neighbor_fit < best_global_fit:
                best_global = best_neighbor
                best_global_fit = best_neighbor_fit

        return best_global

    # ------------------ Hybrid GA + Tabu Search ------------------
    def solve_tsp_hybrid(self, cluster_id, hotspot_indices=None, population_size=40, generations=100, mutation_rate=0.1):
        """
        Hybrid GA + Tabu Search:
        - GA explores (selection, crossover, mutation)
        - Tabu Search refines each child if tabu_on_children=True
        """
        if hotspot_indices is None:
            cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
        else:
            cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]

        if not cluster_hotspots:
            return []
        if len(cluster_hotspots) == 1:
            return [cluster_id] + cluster_hotspots + [cluster_id]

        start_point = cluster_id
        population = [random.sample(cluster_hotspots, len(cluster_hotspots)) for _ in range(population_size)]

        for _ in range(generations):
            new_population = []
            for _ in range(population_size):
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                child = self._order_crossover(p1, p2)
                child = self._mutate(child, mutation_rate)

                # ⚡ Hybridization with Tabu Search
                if self.tabu_on_children:
                    improved = self._tabu_search([start_point] + child + [start_point])
                    child = improved[1:-1]

                new_population.append(child)

            population = new_population

        best_route = min(population, key=lambda r: self._fitness([start_point] + r + [start_point]))
        return [start_point] + best_route + [start_point]

    # ------------------ Cluster Optimization ------------------
    def optimize_all_clusters(self, population_size=40, generations=100, mutation_rate=0.1):
        all_routes = {}
        for cid in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cid, 1)
            cluster_routes = []
            if n_drones > 1:
                groups = self._split_hotspots_for_cluster(cid, n_drones)
                for g in groups:
                    route = self.solve_tsp_hybrid(cid, hotspot_indices=g,
                                                  population_size=population_size,
                                                  generations=generations,
                                                  mutation_rate=mutation_rate)
                    cluster_routes.append(route)
            else:
                route = self.solve_tsp_hybrid(cid,
                                              population_size=population_size,
                                              generations=generations,
                                              mutation_rate=mutation_rate)
                cluster_routes = [route]
            all_routes[cid] = cluster_routes
        return all_routes

    def _split_hotspots_for_cluster(self, cluster_id, n_drones):
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

        # ------------------ Print Output ------------------
    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Hybrid GA + Tabu Search) ===")
        for cid, routes in all_routes.items():
            num_drones = self.drones_per_cluster.get(cid, 1)
            print(f"\nCluster {cid} (Drones: {num_drones}):")
            road_lat, road_lon = self.road_points[cid]
            print(f"  Start from road point: ({road_lat:.5f}, {road_lon:.5f})")

            for d_idx, route in enumerate(routes):
                if len(route) <= 1:
                    print(f"  Drone {d_idx+1}: No hotspots assigned")
                    continue

                total_distance = self._route_distance(route)
                flight_time_min = total_distance * 1000 / self.drone_speed / 60

                validity = "" if flight_time_min <= self.max_flight_time else "⚠️ Flight Time Limit Exceeded!"
                print(f"  Drone {d_idx+1} Route: {route}")
                print(f"    Total distance: {total_distance:.2f} km")
                print(f"    Estimated flight time: {flight_time_min:.2f} min {validity}")

        # ------------------ Visualization ------------------
    def visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        linestyles = ['-', '--', '-.', ':']

        # Depots / road points
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center',
                     fontweight='bold', fontsize=11, color='white')

        # Hotspots
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c=colors[cid % len(colors)], s=100, alpha=0.7, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8, fontweight='bold')

        # Routes per cluster
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

                color = colors[cid % len(colors)]
                linestyle = linestyles[d_idx % len(linestyles)]
                marker = markers[d_idx % len(markers)]

                plt.plot(lons, lats, linestyle=linestyle, marker=marker,
                         color=color, linewidth=2, markersize=6,
                         label=f'Cluster {cid} Drone {d_idx+1}')

        plt.title('Drone Routes (Hybrid GA + Tabu Search)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class ClusterBasedDroneRoutingHybridTabuOptimized:
    drone_speed = 30
    max_flight_time = 170
    max_distance = (drone_speed * max_flight_time * 60) / 1000
    penalty_factor = 100

    def __init__(self, csv_file=None, road_points=None, n_drones=None,
                 tabu_on_children=True, tabu_tenure=15, tabu_budget=50):
        self.df = pd.read_csv(csv_file)
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        # Drone distribution
        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")

        # Road points (depots)
        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)

        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])

        # Parameters
        self.tabu_on_children = tabu_on_children
        self.tabu_tenure = tabu_tenure
        self.tabu_budget = tabu_budget

        # Distance matrix
        self.all_locations = self.road_points + self.coordinates
        self.n_total_locations = len(self.all_locations)
        self.dist_matrix = self._calculate_distance_matrix()

    # ------------------ Utility ------------------
    def _calculate_distance_matrix(self):
        dist = np.zeros((self.n_total_locations, self.n_total_locations))
        for i in range(self.n_total_locations):
            for j in range(self.n_total_locations):
                if i != j:
                    lat1, lon1 = self.all_locations[i]
                    lat2, lon2 = self.all_locations[j]
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                    c = 2*np.arcsin(np.sqrt(a))
                    dist[i][j] = 6371 * c
        return dist

    def _route_distance(self, route):
        return sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def _fitness(self, route):
        distance = self._route_distance(route)
        penalty = 0
        if distance > self.max_distance:
            penalty = (distance - self.max_distance) * self.penalty_factor
        return distance + penalty

    # ------------------ GA Components ------------------
    def _tournament_selection(self, population, k=3):
        chosen = random.sample(population, min(k, len(population)))
        chosen.sort(key=lambda r: self._fitness(r))
        return chosen[0]

    def _order_crossover(self, parent1, parent2):
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
        route = route[:]
        if len(route) > 1 and random.random() < mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # ------------------ Improved Tabu Search ------------------
    def _tabu_search(self, route, tabu_tenure=None, max_iter=None):
        """Improved Tabu Search using neighbor sampling + restart."""
        if tabu_tenure is None:
            tabu_tenure = self.tabu_tenure
        if max_iter is None:
            max_iter = self.tabu_budget

        best = route
        best_fit = self._fitness(best)
        current = best[:]
        tabu_list = deque(maxlen=tabu_tenure)
        best_global = best[:]
        best_global_fit = best_fit
        no_improve = 0

        for _ in range(max_iter):
            n = len(current)
            neighbors = []
            # 🔹 Evaluasi sebagian tetangga saja
            for _ in range(min(30, (n*(n-1))//2)):
                i, j = random.sample(range(1, n-1), 2)
                new = current[:]
                new[i], new[j] = new[j], new[i]
                move = (i, j)
                if move not in tabu_list:
                    neighbors.append((new, move))

            if not neighbors:
                break

            best_neighbor, best_move = min(neighbors, key=lambda x: self._fitness(x[0]))
            best_neighbor_fit = self._fitness(best_neighbor)

            tabu_list.append(best_move)
            current = best_neighbor

            if best_neighbor_fit < best_global_fit:
                best_global = best_neighbor[:]
                best_global_fit = best_neighbor_fit
                no_improve = 0
            else:
                no_improve += 1

            # 🔹 Restart jika stagnan
            if no_improve > 15:
                random.shuffle(current[1:-1])
                no_improve = 0

        return best_global

    # ------------------ Hybrid GA + Tabu Search ------------------
    def solve_tsp_hybrid(self, cluster_id, hotspot_indices=None, population_size=40, generations=100, mutation_rate=0.1):
        """
        Hybrid GA + Selective Tabu Search:
        - GA explores (selection, crossover, mutation)
        - Tabu Search refines selective offspring
        - Elitism: best individual kept across generations
        """
        if hotspot_indices is None:
            cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
        else:
            cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]

        if not cluster_hotspots:
            return []
        if len(cluster_hotspots) == 1:
            return [cluster_id] + cluster_hotspots + [cluster_id]

        start_point = cluster_id
        population = [random.sample(cluster_hotspots, len(cluster_hotspots)) for _ in range(population_size)]

        for _ in range(generations):
            new_population = []
            # 🏆 Elitism — simpan solusi terbaik
            elite = min(population, key=lambda r: self._fitness([start_point] + r + [start_point]))

            for _ in range(population_size):
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                child = self._order_crossover(p1, p2)
                child = self._mutate(child, mutation_rate)

                # ⚡ Jalankan Tabu hanya pada 20% anak terbaik
                if self.tabu_on_children and random.random() < 0.2:
                    improved = self._tabu_search([start_point] + child + [start_point])
                    child = improved[1:-1]

                new_population.append(child)

            # Masukkan individu elit agar tidak hilang
            new_population[0] = elite
            population = new_population

        best_route = min(population, key=lambda r: self._fitness([start_point] + r + [start_point]))
        return [start_point] + best_route + [start_point]

    # ------------------ Cluster Optimization ------------------
    def optimize_all_clusters(self, population_size=40, generations=100, mutation_rate=0.1):
        all_routes = {}
        for cid in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cid, 1)
            cluster_routes = []
            if n_drones > 1:
                groups = self._split_hotspots_for_cluster(cid, n_drones)
                for g in groups:
                    route = self.solve_tsp_hybrid(cid, hotspot_indices=g,
                                                  population_size=population_size,
                                                  generations=generations,
                                                  mutation_rate=mutation_rate)
                    cluster_routes.append(route)
            else:
                route = self.solve_tsp_hybrid(cid,
                                              population_size=population_size,
                                              generations=generations,
                                              mutation_rate=mutation_rate)
                cluster_routes = [route]
            all_routes[cid] = cluster_routes
        return all_routes

    def _split_hotspots_for_cluster(self, cluster_id, n_drones):
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

    # ------------------ Output ------------------
    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Optimized Hybrid GA + Tabu Search) ===")
        for cid, routes in all_routes.items():
            num_drones = self.drones_per_cluster.get(cid, 1)
            print(f"\nCluster {cid} (Drones: {num_drones}):")
            road_lat, road_lon = self.road_points[cid]
            print(f"  Start from road point: ({road_lat:.5f}, {road_lon:.5f})")

            for d_idx, route in enumerate(routes):
                if len(route) <= 1:
                    print(f"  Drone {d_idx+1}: No hotspots assigned")
                    continue
                total_distance = self._route_distance(route)
                flight_time_min = total_distance * 1000 / self.drone_speed / 60
                valid = "" if flight_time_min <= self.max_flight_time else "⚠️ Exceeds Limit!"
                print(f"  Drone {d_idx+1} Route: {route}")
                print(f"    Total distance: {total_distance:.2f} km | Time: {flight_time_min:.2f} min {valid}")

    def visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        linestyles = ['-', '--', '-.', ':']

        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center', fontweight='bold', color='white')

        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c=colors[cid % len(colors)], s=100, alpha=0.7, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8, fontweight='bold')

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
                color = colors[cid % len(colors)]
                linestyle = linestyles[d_idx % len(linestyles)]
                plt.plot(lons, lats, linestyle=linestyle, marker=markers[d_idx % len(markers)],
                         color=color, linewidth=2, markersize=6,
                         label=f'Cluster {cid} Drone {d_idx+1}')

        plt.title('Drone Routes (Optimized GA + Tabu Search with Elitism)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
