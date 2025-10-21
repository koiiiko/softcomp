import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from bs4 import BeautifulSoup
import re
from matplotlib.lines import Line2D

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
    
    # HYBRID GA + Local Search (LS Sembari GA)
    # def solve_tsp_hybrid(self, cluster_id, hotspot_indices=None, 
    #                     population_size=80, generations=150, mutation_rate=0.2,
    #                     local_search_freq=10, local_search_intensity=20):
    #     """
    #     Improved Hybrid Genetic Algorithm + Local Search (Adaptive Memetic)
    #     - LS applied less frequently and more adaptively (stronger later)
    #     - Mutation rate increased for diversity
    #     - LS intensity grows gradually with generation count
    #     """
    #     # Setup hotspots to visit
    #     if hotspot_indices is None:
    #         cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
    #     else:
    #         cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]

    #     if not cluster_hotspots:
    #         return []
    #     if len(cluster_hotspots) == 1:
    #         return [cluster_id] + cluster_hotspots + [cluster_id]

    #     start_point = cluster_id

    #     # Initialize population
    #     def _greedy_route(nodes):
    #         if len(nodes) <= 1:
    #             return nodes[:]
    #         route = [nodes[0]]
    #         unvisited = set(nodes[1:])
    #         while unvisited:
    #             last = route[-1]
    #             next_node = min(unvisited, key=lambda x: self.dist_matrix[last][x])
    #             route.append(next_node)
    #             unvisited.remove(next_node)
    #         return route

    #     # Initialize population
    #     population = []
    #     for _ in range(population_size):
    #         if random.random() < 0.2:  # 20% greedy, 80% random
    #             population.append(_greedy_route(cluster_hotspots))
    #         else:
    #             population.append(random.sample(cluster_hotspots, len(cluster_hotspots)))

    #     best_route_so_far = None
    #     best_dist = float('inf')

    #     # HYBRID EVOLUTION LOOP
    #     for gen in range(generations):
    #         # === 1. Genetic Algorithm evolution ===
    #         new_population = []
    #         for _ in range(population_size):
    #             parent1 = self._tournament_selection(population)
    #             parent2 = self._tournament_selection(population)
    #             child = self._order_crossover(parent1, parent2)
    #             child = self._mutate(child, mutation_rate)
    #             new_population.append(child)
    #         population = new_population

    #         # === 2. Adaptive Local Search (Memetic Component) ===
    #         # Apply LS only after 40% of generations have passed
    #         if gen > generations * 0.4 and (gen + 1) % local_search_freq == 0:
    #             # Sort population by fitness
    #             population.sort(key=lambda r: self._route_distance_segment(r))

    #             # Apply LS to small subset of top 10%
    #             elite_count = max(1, population_size // 10)

    #             # LS intensity grows with generation progress
    #             adaptive_intensity = int(local_search_intensity * (gen / generations) + 5)

    #             for i in random.sample(range(elite_count), k=min(3, elite_count)):
    #                 population[i] = self._local_search_2opt(
    #                     population[i], 
    #                     max_iter=adaptive_intensity
    #                 )

    #         # Track best route so far
    #         population.sort(key=lambda r: self._route_distance_segment(r))
    #         current_best = population[0]
    #         current_dist = self._route_distance([start_point] + current_best + [start_point])
    #         if current_dist < best_dist:
    #             best_dist = current_dist
    #             best_route_so_far = current_best[:]

    #     # === 3. Final Intensive Local Search ===
    #     best_route_final = self._local_search_2opt(
    #         best_route_so_far, 
    #         max_iter=local_search_intensity * 3
    #     )

    #     final_route = [start_point] + best_route_final + [start_point]
    #     final_distance = self._route_distance(final_route)

    #     return final_route
    
    #LS Setelah GA
    def solve_tsp_hybrid(self, cluster_id, hotspot_indices=None, 
                        population_size=50, generations=200, mutation_rate=0.1,
                        local_search_freq=None,
                        local_search_intensity=100):
        """
        HYBRID: Pure GA + Final 2-opt (Post-Optimization)
        - No LS during evolution → preserves GA diversity
        - LS applied ONCE at the end to best GA solution
        - Includes elitism and greedy initialization
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

        # === Greedy + Random Initialization ===
        def _greedy_route(nodes):
            if len(nodes) <= 1:
                return nodes[:]
            route = [nodes[0]]
            unvisited = set(nodes[1:])
            while unvisited:
                last = route[-1]
                next_node = min(unvisited, key=lambda x: self.dist_matrix[last][x])
                route.append(next_node)
                unvisited.remove(next_node)
            return route

        population = []
        for _ in range(population_size):
            if random.random() < 0.3:  # 30% greedy
                population.append(_greedy_route(cluster_hotspots))
            else:
                population.append(random.sample(cluster_hotspots, len(cluster_hotspots)))

        best_ever = min(population, key=lambda r: self._route_distance_segment(r))
        best_ever_dist = self._route_distance_segment(best_ever)

        # === PURE GA LOOP (NO LS DURING EVOLUTION) ===
        for gen in range(generations):
            # Elitism: keep best from previous gen
            elite = best_ever[:]

            # Create new population
            new_population = [elite]  # keep elite
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population

            # Update best_ever
            current_best = min(population, key=lambda r: self._route_distance_segment(r))
            current_dist = self._route_distance_segment(current_best)
            if current_dist < best_ever_dist:
                best_ever = current_best[:]
                best_ever_dist = current_dist

        # === FINAL LOCAL SEARCH (ONLY ONCE!) ===
        best_route_refined = self._local_search_2opt(
            best_ever,
            max_iter=local_search_intensity
        )

        return [start_point] + best_route_refined + [start_point]

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

        # Define color mapping (cluster → drone → color)
        color_map = {
            0: {0: "#FAB12F", 1: "#DD0303"},  # Cluster 1
            1: {0: "#3A6F43", 1: "#FDAAAA"},  # Cluster 2
            2: {0: "#3338A0", 1: "#C59560"},  # Cluster 3
            3: {0: "#990099", 1: "#009999"},  # Cluster 4
        }

        # --- Road points ---
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center', fontweight='bold', fontsize=12, color='white')

        # --- Hotspots ---
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c='gray', s=80, alpha=0.6, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8, fontweight='bold')

        # --- Routes ---
        for cid, routes in all_routes.items():
            for d_idx, route_info in enumerate(routes):
                if isinstance(route_info, tuple):
                    route, _ = route_info
                else:
                    route = route_info

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

                color = color_map.get(cid, {}).get(d_idx, "black")

                plt.plot(
                    lons, lats, 'o-',
                    color=color,
                    linestyle='--',           # 🔹 Dashed line style
                    linewidth=2.5,
                    markersize=6,
                    label=f'Cluster {cid+1} Drone {d_idx+1}'
                )

        plt.title('Drone Routes (Local Search 2-opt)')
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
            
            routes = all_routes.get(cid, [])
            if not routes:
                print(f"  ⚠️ No routes found for cluster {cid}. Skipping.")
                continue
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

  #Baru
    def new_visualize_cluster_routes_ketapang(self, all_routes,
                                    base_map_path="new_forest_fire_clusters_map_melawi.html",
                                    output_path="new2_forest_fire_clusters_map_melawi.html"):

        from bs4 import BeautifulSoup
        import re

        # === 1️⃣ Baca peta dasar ===
        with open(base_map_path, "r", encoding="utf-8") as f:
            html_data = f.read()

        soup = BeautifulSoup(html_data, "html.parser")

        # === 2️⃣ Deteksi variabel map Folium ===
        map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
        if not map_var_match:
            print("❌ Tidak ditemukan variabel peta di file HTML.")
            return
        map_var = map_var_match.group(1)
        print(f"✅ Variabel peta terdeteksi: {map_var}")

        # === 3️⃣ Warna kombinasi kustom per cluster ===
        cluster_color_palettes = {
            0: ["#FAB12F", "#DD0303"],
            1: ["#3A6F43", "#FDAAAA"],
            2: ["#3338A0", "#C59560"],
            3: ["#990099","#009999"]
        }

        legend_entries = []
        js_add_routes = "\n\n// === Tambahan: Drone Route Overlays ===\n"

        # === 4️⃣ Tambahkan PolyLine putus-putus per drone ===
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
            for d_idx, route in enumerate(routes):
                if len(route) <= 1:
                    continue

                latlons = []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        lat, lon = self.coordinates[loc - len(self.road_points)]
                    latlons.append([lat, lon])

                color = palette[d_idx % len(palette)]

                js_add_routes += f"""
    var droneRoute_{cid}_{d_idx} = L.polyline({latlons}, {{
        color: '{color}',
        weight: 3,
        opacity: 0.9,
        dashArray: '10, 10'
    }}).addTo({map_var});
    droneRoute_{cid}_{d_idx}.bindTooltip("Cluster {cid} - Drone {d_idx+1}");
    """
                legend_entries.append((f"Cluster {cid} - Drone {d_idx+1}", color))

        js_add_routes += "\n// === End of Drone Routes ===\n"

        # === 5️⃣ Buat HTML legenda ===
        legend_html = '''
        <div style="
            position: fixed; bottom: 30px; left: 20px; width: 270px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:13px; padding: 10px; line-height: 1.4;">
            <b>Legenda Rute Drone ✈️</b><br>
            <hr style="margin:4px 0;">
        '''
        for label, color in legend_entries:
            legend_html += f'''
        <div style="display:flex;align-items:center;margin-bottom:4px;">
            <span style="
                width:26px;
                height:0;
                border-top:2px dashed {color};
                display:inline-block;
                margin-right:8px;">
            </span>
            {label}
        </div>
        '''
        legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

        # === 6️⃣ Sisipkan JS legenda (pakai backtick untuk HTML multiline) ===
        js_add_routes += f"""
    var legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function (map) {{
        var div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `{legend_html}`;
        return div;
    }};
    legend.addTo({map_var});
    """

        # === 7️⃣ Sisipkan JS ke HTML terakhir ===
        script_tag = soup.find_all("script")[-1]
        script_content = script_tag.string or ""
        script_tag.string = script_content + js_add_routes

        # === 8️⃣ Simpan hasil ===
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"✅ Drone routes (warna kustom + legenda) berhasil ditambahkan ke peta: {output_path}")

    def new_visualize_cluster_routes_melawai(self, all_routes,
                                    base_map_path="new2_forest_fire_clusters_map_melawi.html",
                                    output_path="new3_forest_fire_clusters_map_melawi.html"):

        from bs4 import BeautifulSoup
        import re

        # === 1️⃣ Baca peta dasar ===
        with open(base_map_path, "r", encoding="utf-8") as f:
            html_data = f.read()

        soup = BeautifulSoup(html_data, "html.parser")

        # === 2️⃣ Deteksi variabel map Folium ===
        map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
        if not map_var_match:
            print("❌ Tidak ditemukan variabel peta di file HTML.")
            return
        map_var = map_var_match.group(1)
        print(f"✅ Variabel peta terdeteksi: {map_var}")

        # === 3️⃣ Warna kombinasi kustom per cluster ===
        cluster_color_palettes = {
            0: ["#FAB12F", "#DD0303"],
            1: ["#3A6F43", "#FDAAAA"],
            2: ["#3338A0", "#C59560"],
            3: ["#990099","#009999"]
        }

        legend_entries = []
        js_add_routes = "\n\n// === Tambahan: Drone Route Overlays ===\n"

        # === 4️⃣ Tambahkan PolyLine putus-putus per drone ===
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
            for d_idx, route in enumerate(routes):
                if len(route) <= 1:
                    continue

                latlons = []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        lat, lon = self.coordinates[loc - len(self.road_points)]
                    latlons.append([lat, lon])

                color = palette[d_idx % len(palette)]

                js_add_routes += f"""
    var droneRoute_{cid}_{d_idx} = L.polyline({latlons}, {{
        color: '{color}',
        weight: 3,
        opacity: 0.9,
        dashArray: '10, 10'
    }}).addTo({map_var});
    droneRoute_{cid}_{d_idx}.bindTooltip("Cluster {cid} - Drone {d_idx+1}");
    """
                legend_entries.append((f"Cluster {cid} - Drone {d_idx+1}", color))

        js_add_routes += "\n// === End of Drone Routes ===\n"

        # === 5️⃣ Buat HTML legenda ===
        legend_html = '''
        <div style="
            position: fixed; bottom: 30px; left: 20px; width: 270px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:13px; padding: 10px; line-height: 1.4;">
            <b>Legenda Rute Drone ✈️</b><br>
            <hr style="margin:4px 0;">
        '''
        for label, color in legend_entries:
            legend_html += f'''
        <div style="display:flex;align-items:center;margin-bottom:4px;">
            <span style="
                width:26px;
                height:0;
                border-top:2px dashed {color};
                display:inline-block;
                margin-right:8px;">
            </span>
            {label}
        </div>
        '''
        legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

        # === 6️⃣ Sisipkan JS legenda (pakai backtick untuk HTML multiline) ===
        js_add_routes += f"""
    var legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function (map) {{
        var div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `{legend_html}`;
        return div;
    }};
    legend.addTo({map_var});
    """

        # === 7️⃣ Sisipkan JS ke HTML terakhir ===
        script_tag = soup.find_all("script")[-1]
        script_content = script_tag.string or ""
        script_tag.string = script_content + js_add_routes

        # === 8️⃣ Simpan hasil ===
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"✅ Drone routes (warna kustom + legenda) berhasil ditambahkan ke peta: {output_path}")

    def new_visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))

        # 🎨 Define color mapping (cluster → drone → color)
        color_map = {
            0: {0: "#FAB12F", 1: "#DD0303"},  # Cluster 1
            1: {0: "#3A6F43", 1: "#FDAAAA"},  # Cluster 2
            2: {0: "#3338A0", 1: "#C59560"},  # Cluster 3
            3: {0: "#990099", 1: "#009999"},  # Cluster 4
        }

        legend_handles = []  # Custom legend container

        # 🏠 Road points (depots)
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center',
                    color='white', fontweight='bold')

        # 🔥 Hotspots
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c='gray', s=80, alpha=0.6, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # 🚁 Routes per cluster & drone
        for cid, routes in all_routes.items():
            for d_idx, route_info in enumerate(routes):
                if isinstance(route_info, tuple):
                    route, _ = route_info
                else:
                    route = route_info

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

                color = color_map.get(cid, {}).get(d_idx, "black")
                label = f'Cluster {cid+1} Drone {d_idx+1}'

                plt.plot(
                    lons, lats, 'o-',
                    color=color,
                    linestyle='--',
                    linewidth=2.5,
                    markersize=6,
                    label=label
                )

                # Add legend handle manually for better formatting
                legend_handles.append(Line2D(
                    [0], [0],
                    color=color,
                    linestyle='--',
                    linewidth=2.5,
                    label=label
                ))

        # 🧾 Custom Legend
        plt.legend(handles=legend_handles, title="Legenda Drone Routes",
                title_fontsize=11, fontsize=9, loc='best')

        # 📊 Labels & Styling
        plt.title('Drone Routes (Local Search 2-opt)', fontsize=14, fontweight='bold')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
