import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from bs4 import BeautifulSoup
import re

class ClusterBasedDroneRouting_LocalSearch:
    def __init__(self, csv_file=None, road_points=None, n_drones=None):
        self.df = pd.read_csv(csv_file)
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        # Drones per cluster setup
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

        # Distance matrix
        self.all_locations = self.road_points + self.coordinates
        self.n_total_locations = len(self.all_locations)
        self.dist_matrix = self._calculate_distance_matrix()

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
                    r = 6371  # km
                    dist[i][j] = c*r
        return dist

    def _route_distance(self, route):
        """Compute total route distance"""
        return sum(self.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def split_hotspots_for_cluster(self, cluster_id, n_drones):
        """Split hotspots using KMeans for multiple drones"""
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

    #Local search
    def _2opt_swap(self, route, i, j):
        """Perform 2-opt swap between two indices"""
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route

    def local_search_tsp(self, cluster_id, hotspot_indices=None, max_iter=1000):
        """Solve TSP for given cluster using Local Search (2-opt neighborhood)"""
        if hotspot_indices is None:
            cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
        else:
            cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]

        # Handle trivial cases
        if not cluster_hotspots:
            return []
        if len(cluster_hotspots) == 1:
            return [cluster_id] + cluster_hotspots + [cluster_id]

        # Initialize route randomly
        current_route = [cluster_id] + random.sample(cluster_hotspots, len(cluster_hotspots)) + [cluster_id]
        best_distance = self._route_distance(current_route)

        for _ in range(max_iter):
            improved = False
            for i in range(1, len(current_route)-2):
                for j in range(i+1, len(current_route)-1):
                    new_route = self._2opt_swap(current_route, i, j)
                    new_distance = self._route_distance(new_route)
                    if new_distance < best_distance:
                        current_route, best_distance = new_route, new_distance
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break  # stop if no improvement found
        return current_route
    
    #Optimization wrapper
    def optimize_all_clusters(self):
        """Run local search optimization for all clusters and drones"""
        all_routes = {}
        for cluster_id in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cluster_id, 1)
            cluster_routes = []
            if n_drones > 1:
                groups = self.split_hotspots_for_cluster(cluster_id, n_drones)
                for group in groups:
                    route = self.local_search_tsp(cluster_id, hotspot_indices=group)
                    cluster_routes.append((route, group))
            else:
                route = self.local_search_tsp(cluster_id)
                all_hotspots = list(range(len(self.coordinates)))
                cluster_routes.append((route, all_hotspots))
            all_routes[cluster_id] = cluster_routes

        # Print routes
        for cluster_id, routes in all_routes.items():
            for d_idx, route in enumerate(routes):
                print(f"Cluster {cluster_id}, Drone {d_idx+1}, Route: {route}")
        return all_routes
    
    def optimize_all_clusters_bayesian(
        self,
        max_iterations=1000,
        step_size=0.1,
        n_restarts=1,
        neighbor_swap_rate=0.2
    ):
        """
        Run local search optimization for all clusters and drones
        with tunable hyperparameters for Optuna.
        """
        all_routes = {}

        for cluster_id in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cluster_id, 1)
            cluster_routes = []

            for _ in range(n_restarts):  # allow multiple restarts to escape local minima
                if n_drones > 1:
                    groups = self.split_hotspots_for_cluster(cluster_id, n_drones)
                    for group in groups:
                        route = self.local_search_tsp(
                            cluster_id,
                            hotspot_indices=group,
                            max_iter=max_iterations
                        )
                        cluster_routes.append((route, group))
                else:
                    route = self.local_search_tsp(cluster_id, max_iter=max_iterations)
                    cluster_routes = [route]

            all_routes[cluster_id] = cluster_routes

        # for cluster_id, routes in all_routes.items():
        #     for d_idx, route in enumerate(routes):
        #         print(f"Cluster {cluster_id}, Drone {d_idx+1}, Route: {route}")

        return all_routes

    # Visualization
    def visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))
        colors = ['red','blue','green','orange','purple','brown','pink','gray']

        # Road points
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center', fontweight='bold', fontsize=12, color='white')

        # Hotspots
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(lon, lat, c=colors[cid % len(colors)], s=100, alpha=0.7, zorder=4)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8, fontweight='bold')

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
                         label=f'Cluster {cid} Drone {d_idx+1}', linewidth=2, markersize=6)
        plt.title('Drone Routes (Local Search 2-opt)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def new_visualize_cluster_routes(self, all_routes, base_map_path="forest_fire_clusters_map.html", output_path="forest_fire_clusters_with_local_search_routes.html"):
        # === 1️⃣ Baca peta dasar ===
        with open(base_map_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data, "html.parser")

        # === 2️⃣ Deteksi variabel map Leaflet ===
        map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
        if not map_var_match:
            print("❌ Tidak ditemukan variabel peta di file HTML.")
            return
        map_var = map_var_match.group(1)
        print(f"✅ Variabel peta terdeteksi: {map_var}")

        # === 3️⃣ Warna kustom per cluster ===
        cluster_color_palettes = {
            0: ["#432323", "#D7A86E"],  # merah tua & kuning lembut
            1: ["#59AC77", "#6F00FF"],  # hijau muda & ungu terang
            2: ["#F25912", "#5C3E94"],  # oranye & ungu tua
        }

        legend_entries = []
        js_add_routes = "\n\n// === Tambahan: Drone Route Overlays (Local Search) ===\n"

        # === 4️⃣ Loop per cluster dan drone ===
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
            for d_idx, (route, hotspot_indices) in enumerate(routes):
                if len(route) <= 1:
                    continue

                latlons = []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        # global hotspot index = loc - len(road_points)
                        global_hotspot_idx = loc - len(self.road_points)
                        lat, lon = self.coordinates[global_hotspot_idx]
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

        # === 5️⃣ Tambahkan legenda ===
        legend_html = '''
        <div style="
            position: fixed; bottom: 30px; left: 20px; width: 270px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:13px; padding: 10px; line-height: 1.4;">
            <b>Legenda Rute Drone ✈️</b><br>
            <hr style="margin:4px 0;">
        '''
        for label, color in legend_entries:
            legend_html += f'<div><span style="background-color:{color};width:18px;height:10px;display:inline-block;margin-right:6px;"></span>{label}</div>'
        legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

        js_add_routes += f"""
    var legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function (map) {{
        var div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `{legend_html}`;
        return div;
    }};
    legend.addTo({map_var});
    """

        # === 6️⃣ Sisipkan JS ke file HTML ===
        script_tag = soup.find_all("script")[-1]
        script_content = script_tag.string or ""
        script_tag.string = script_content + js_add_routes

        # === 7️⃣ Simpan hasil baru ===
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"✅ Rute drone (Local Search Only) berhasil ditambahkan ke peta: {output_path}")

    def melawi_visualize_cluster_routes(self, all_routes, base_map_path="forest_fire_clusters_map_melawi.html", output_path="forest_fire_clusters_with_local_search_routes_melawi.html"):
        # === 1️⃣ Baca peta dasar ===
        with open(base_map_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data, "html.parser")

        # === 2️⃣ Deteksi variabel map Leaflet ===
        map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
        if not map_var_match:
            print("❌ Tidak ditemukan variabel peta di file HTML.")
            return
        map_var = map_var_match.group(1)
        print(f"✅ Variabel peta terdeteksi: {map_var}")

        # === 3️⃣ Warna kustom per cluster ===
        cluster_color_palettes = {
            0: ["#432323", "#D7A86E"],  # merah tua & kuning lembut
            1: ["#59AC77", "#6F00FF"],  # hijau muda & ungu terang
            2: ["#F25912", "#5C3E94"],  # oranye & ungu tua
        }

        legend_entries = []
        js_add_routes = "\n\n// === Tambahan: Drone Route Overlays (Local Search) ===\n"

        # === 4️⃣ Loop per cluster dan drone ===
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
            for d_idx, (route, hotspot_indices) in enumerate(routes):
                if len(route) <= 1:
                    continue

                latlons = []
                for loc in route:
                    if loc < len(self.road_points):
                        lat, lon = self.road_points[loc]
                    else:
                        # global hotspot index = loc - len(road_points)
                        global_hotspot_idx = loc - len(self.road_points)
                        lat, lon = self.coordinates[global_hotspot_idx]
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

        # === 5️⃣ Tambahkan legenda ===
        legend_html = '''
        <div style="
            position: fixed; bottom: 30px; left: 20px; width: 270px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:13px; padding: 10px; line-height: 1.4;">
            <b>Legenda Rute Drone ✈️</b><br>
            <hr style="margin:4px 0;">
        '''
        for label, color in legend_entries:
            legend_html += f'<div><span style="background-color:{color};width:18px;height:10px;display:inline-block;margin-right:6px;"></span>{label}</div>'
        legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

        js_add_routes += f"""
    var legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function (map) {{
        var div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `{legend_html}`;
        return div;
    }};
    legend.addTo({map_var});
    """

        # === 6️⃣ Sisipkan JS ke file HTML ===
        script_tag = soup.find_all("script")[-1]
        script_content = script_tag.string or ""
        script_tag.string = script_content + js_add_routes

        # === 7️⃣ Simpan hasil baru ===
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"✅ Rute drone (Local Search Only) berhasil ditambahkan ke peta: {output_path}")

    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Local Search) ===")
        for cid in range(len(self.road_points)):
            num_drones = self.drones_per_cluster.get(cid, 1)
            print(f"\nCluster {cid} (Drones: {num_drones}):")
            road_lat, road_lon = self.road_points[cid]
            print(f"  Start from road point: ({road_lat:.5f}, {road_lon:.5f})")
            routes = all_routes[cid]
            for d_idx, (route, hotspot_indices) in enumerate(routes):
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
                print(f"R{last}" if last < len(self.road_points) else f"H{last - len(self.road_points)}")
                print(f"    Total distance: {total_distance:.2f} km")
                print(f"    Est. flight time: {total_distance*1000/30/60:.2f} minutes")
    