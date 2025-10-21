import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, random
from sklearn.cluster import KMeans
import folium
from folium import plugins
from bs4 import BeautifulSoup
import re
from IPython.display import display, display_html

class ClusterBasedDroneRoutingTS:
    def __init__(self, csv_file=None, road_points=None, n_drones=None):
        self.df = pd.read_csv(csv_file)
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        # Drone per cluster
        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")

        # Road points (depot)
        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)

        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])

    # ---------- Distance & Cost ----------
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def route_cost(self, route, dist_matrix, speed=30, max_time=170):
        dist = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        time_minutes = (dist*1000/speed)/60  # convert km → m, divide by m/s, convert to minutes
        if time_minutes > max_time:
            dist += 1000 * (time_minutes - max_time)  # penalty
        return dist, time_minutes

    # ---------- Build distance matrix ----------
    def build_dist_matrix(self, locations):
        n = len(locations)
        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    dist_matrix[i][j] = self.haversine(*locations[i], *locations[j])
        return dist_matrix

    # ---------- Tabu Search core ----------
    def tabu_search(self, dist_matrix, start=0, tabu_tenure=10, max_iter=200, neighborhood_size=None):
        n = len(dist_matrix)
        route = list(range(n))
        random.shuffle(route[1:])
        route.append(start)

        best_route = route[:]
        best_cost, _ = self.route_cost(best_route, dist_matrix)
        current_route = route[:]
        tabu_list = []

        for _ in range(max_iter):
            neighborhood = []
            all_moves = [(i, j) for i in range(1, n-2) for j in range(i+1, n-1)]

            # sampling move sesuai neighborhood_size
            if neighborhood_size is not None and neighborhood_size < len(all_moves):
                moves = random.sample(all_moves, neighborhood_size)
            else:
                moves = all_moves

            for i, j in moves:
                neighbor = current_route[:]
                neighbor[i:j] = reversed(neighbor[i:j])  # 2-opt
                cost, _ = self.route_cost(neighbor, dist_matrix)
                neighborhood.append((neighbor, cost, (i, j)))

            neighborhood.sort(key=lambda x: x[1])

            for neighbor, cost, move in neighborhood:
                if move not in tabu_list or cost < best_cost:  # aspiration
                    current_route = neighbor
                    if cost < best_cost:
                        best_route, best_cost = neighbor, cost
                    tabu_list.append(move)
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
                    break

        return best_route, best_cost

    # ---------- Optimization per cluster ----------
    def optimize_all_clusters(self, tabu_tenure=10, max_iter=200, neighborhood_size=None):
        all_routes = {}
        for cid in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cid, 1)
            cluster_indices = [i for i, c in enumerate(self.clusters) if c == cid]
            coords = [self.coordinates[i] for i in cluster_indices]
            depot = self.road_points[cid]

            # bagi hotspot ke n_drones
            if n_drones > 1 and len(coords) > n_drones:
                kmeans = KMeans(n_clusters=n_drones, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords)
                groups = [[] for _ in range(n_drones)]
                for idx, label in zip(cluster_indices, labels):
                    groups[label].append(idx)
            else:
                groups = [cluster_indices]

            cluster_routes = []
            for g in groups:
                locs = [depot] + [self.coordinates[i] for i in g]
                dist_matrix = self.build_dist_matrix(locs)
                route, cost = self.tabu_search(
                    dist_matrix,
                    tabu_tenure=tabu_tenure,
                    max_iter=max_iter,
                    neighborhood_size=neighborhood_size
                )
                cluster_routes.append((route, g))
            all_routes[cid] = cluster_routes
        return all_routes

    # ---------- Visualization ----------
    def visualize_cluster_routes(self, all_routes):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 10))

        # 🎨 Definisi palet warna kontras per cluster
        cluster_color_palettes = {
            0: ["#432323", "#D7A86E", "#AA4A44", "#FFD580"],  # merah tua, emas lembut, coklat tua, kuning muda
            1: ["#59AC77", "#6F00FF", "#00CED1", "#FFD700"],  # hijau muda, ungu, turquoise, emas
            2: ["#F25912", "#5C3E94", "#008B8B", "#FFC300"],  # oranye, ungu tua, hijau toska, kuning terang
            3: ["#3D85C6", "#E6194B", "#F58231", "#911EB4"],  # biru, merah, oranye, ungu
            4: ["#6AA84F", "#C90076", "#FFB6C1", "#674EA7"],  # hijau, magenta, pink muda, ungu
        }

        # 🏠 depot
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s')
            plt.text(lon, lat, f'R{i}', ha='center', va='center', color='white', fontweight='bold')

        # 🔥 hotspots
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            base_color = cluster_color_palettes.get(cid, ['gray'])[0]
            plt.scatter(lon, lat, c=base_color, s=100, alpha=0.7)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8)

        # 🚁 routes per cluster dan per drone
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ['gray', 'lightgray', 'black'])
            for d_idx, (route, hotspot_indices) in enumerate(routes):
                if len(route) <= 1:
                    continue
                color = palette[d_idx % len(palette)]  # pilih warna drone berdasarkan indeks
                lats, lons = [], []
                locs = [self.road_points[cid]] + [self.coordinates[i] for i in hotspot_indices]
                for loc in route:
                    lat, lon = locs[loc]
                    lats.append(lat)
                    lons.append(lon)
                plt.plot(lons, lats, 'o-', color=color, label=f'Cluster {cid} Drone {d_idx+1}')

        plt.title("Drone Routes (Tabu Search)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # def visualize_cluster_routes(self, all_routes, base_map_path="forest_fire_clusters_map.html", output_path="forest_fire_clusters_with_tabu_routes.html"):
    #     # === 1️⃣ Baca peta dasar ===
    #     with open(base_map_path, "r", encoding="utf-8") as f:
    #         html_data = f.read()
    #     soup = BeautifulSoup(html_data, "html.parser")

    #     # === 2️⃣ Deteksi variabel map Leaflet ===
    #     map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
    #     if not map_var_match:
    #         print("❌ Tidak ditemukan variabel peta di file HTML.")
    #         return
    #     map_var = map_var_match.group(1)
    #     print(f"✅ Variabel peta terdeteksi: {map_var}")

    #     # === 3️⃣ Warna kustom per cluster ===
    #     cluster_color_palettes = {
    #         0: ["#432323", "#D7A86E"],  # merah tua & kuning lembut
    #         1: ["#59AC77", "#6F00FF"],  # hijau muda & ungu terang
    #         2: ["#F25912", "#5C3E94"],  # oranye & ungu tua
    #     }

    #     legend_entries = []
    #     js_add_routes = "\n\n// === Tambahan: Drone Route Overlays (Tabu Search) ===\n"

    #     # === 4️⃣ Loop per cluster dan drone ===
    #     for cid, routes in all_routes.items():
    #         palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
    #         for d_idx, (route, hotspot_indices) in enumerate(routes):
    #             if len(route) <= 1:
    #                 continue

    #             latlons = []
    #             for loc in route:
    #                 if loc == 0:  # depot
    #                     lat, lon = self.road_points[cid]
    #                 else:
    #                     # gunakan indeks hotspot sebenarnya
    #                     hotspot_idx = hotspot_indices[loc - 1]
    #                     lat, lon = self.coordinates[hotspot_idx]
    #                 latlons.append([lat, lon])

    #             color = palette[d_idx % len(palette)]

    #             js_add_routes += f"""
    # var droneRoute_{cid}_{d_idx} = L.polyline({latlons}, {{
    #     color: '{color}',
    #     weight: 3,
    #     opacity: 0.9,
    #     dashArray: '10, 10'
    # }}).addTo({map_var});
    # droneRoute_{cid}_{d_idx}.bindTooltip("Cluster {cid} - Drone {d_idx+1}");
    # """
    #             legend_entries.append((f"Cluster {cid} - Drone {d_idx+1}", color))

    #     js_add_routes += "\n// === End of Drone Routes ===\n"

    #     # === 5️⃣ Tambahkan legenda ===
    #     legend_html = '''
    #     <div style="
    #         position: fixed; bottom: 30px; left: 20px; width: 270px;
    #         background-color: white; border:2px solid grey; z-index:9999;
    #         font-size:13px; padding: 10px; line-height: 1.4;">
    #         <b>Legenda Rute Drone ✈️</b><br>
    #         <hr style="margin:4px 0;">
    #     '''
    #     for label, color in legend_entries:
    #         legend_html += f'<div><span style="background-color:{color};width:18px;height:10px;display:inline-block;margin-right:6px;"></span>{label}</div>'
    #     legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

    #     js_add_routes += f"""
    # var legend = L.control({{position: 'bottomleft'}});
    # legend.onAdd = function (map) {{
    #     var div = L.DomUtil.create('div', 'info legend');
    #     div.innerHTML = `{legend_html}`;
    #     return div;
    # }};
    # legend.addTo({map_var});
    # """

    #     # === 6️⃣ Sisipkan JS ke file HTML ===
    #     script_tag = soup.find_all("script")[-1]
    #     script_content = script_tag.string or ""
    #     script_tag.string = script_content + js_add_routes

    #     # === 7️⃣ Simpan hasil baru ===
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         f.write(str(soup))

    #     print(f"✅ Rute drone (Tabu Search Only) berhasil ditambahkan ke peta: {output_path}")


    # ---------- Print ----------
    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Tabu Search) ===")
        for cid, routes in all_routes.items():
            for d_idx,(route, hotspot_indices) in enumerate(routes):
                # Konversi ke indeks global (pakai ID cluster untuk depot, ID asli untuk hotspot)
                mapped_route = []
                for loc in route:
                    if loc == 0:
                        mapped_route.append(cid)  # depot pakai ID cluster
                    else:
                        mapped_route.append(hotspot_indices[loc-1])
                print(f"Cluster {cid}, Drone {d_idx+1}, Route: {mapped_route}")

        # lalu detail versi lama
        for cid, routes in all_routes.items():
            print(f"\nCluster {cid} (Drones: {len(routes)})")
            for d_idx,(route, hotspot_indices) in enumerate(routes):
                locs = [self.road_points[cid]]+[self.coordinates[i] for i in hotspot_indices]
                dist_matrix = self.build_dist_matrix(locs)

                total_dist=0
                print(f"  Drone {d_idx+1} Route: ",end="")
                for i in range(len(route)-1):
                    from_idx,to_idx=route[i],route[i+1]
                    if from_idx==0: 
                        print(f"R{cid} -> ",end="")
                    else: 
                        print(f"H{hotspot_indices[from_idx-1]} -> ",end="")
                    total_dist+=dist_matrix[from_idx][to_idx]
                print(f"R{cid}")
                est_time=total_dist*1000/30/60
                print(f"    Total distance: {total_dist:.2f} km | Est. time: {est_time:.2f} min")

class ClusterBasedDroneRoutingTSMelawi:
    def __init__(self, csv_file=None, road_points=None, n_drones=None):
        self.df = pd.read_csv(csv_file)

        # --- Konversi cluster 1-based → 0-based agar konsisten dengan indeks road_points
        if self.df["Cluster"].min() == 1:
            print("⚙️ Cluster numbering detected as 1-based → converting to 0-based internally.")
            self.df["Cluster"] = self.df["Cluster"] - 1

        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        # Drone per cluster
        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")

        # Road points (depot)
        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)

        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])

    # ---------- Distance & Cost ----------
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def route_cost(self, route, dist_matrix, speed=30, max_time=170):
        dist = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        time_minutes = (dist * 1000 / speed) / 60  # km → m / (m/s) → minutes
        if time_minutes > max_time:
            dist += 1000 * (time_minutes - max_time)  # penalty
        return dist, time_minutes

    # ---------- Build distance matrix ----------
    def build_dist_matrix(self, locations):
        n = len(locations)
        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    dist_matrix[i][j] = self.haversine(*locations[i], *locations[j])
        return dist_matrix

    # ---------- Tabu Search core ----------
    def tabu_search(self, dist_matrix, start=0, tabu_tenure=10, max_iter=200, neighborhood_size=None):
        n = len(dist_matrix)
        route = list(range(n))
        random.shuffle(route[1:])
        route.append(start)

        best_route = route[:]
        best_cost, _ = self.route_cost(best_route, dist_matrix)
        current_route = route[:]
        tabu_list = []

        for _ in range(max_iter):
            neighborhood = []
            all_moves = [(i, j) for i in range(1, n-2) for j in range(i+1, n-1)]

            if neighborhood_size is not None and neighborhood_size < len(all_moves):
                moves = random.sample(all_moves, neighborhood_size)
            else:
                moves = all_moves

            for i, j in moves:
                neighbor = current_route[:]
                neighbor[i:j] = reversed(neighbor[i:j])  # 2-opt
                cost, _ = self.route_cost(neighbor, dist_matrix)
                neighborhood.append((neighbor, cost, (i, j)))

            neighborhood.sort(key=lambda x: x[1])

            for neighbor, cost, move in neighborhood:
                if move not in tabu_list or cost < best_cost:  # aspiration
                    current_route = neighbor
                    if cost < best_cost:
                        best_route, best_cost = neighbor, cost
                    tabu_list.append(move)
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
                    break

        return best_route, best_cost

    # ---------- Optimization per cluster ----------
    def optimize_all_clusters(self, tabu_tenure=10, max_iter=200, neighborhood_size=None):
        all_routes = {}
        for cid in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cid, 1)
            cluster_indices = [i for i, c in enumerate(self.clusters) if c == cid]
            coords = [self.coordinates[i] for i in cluster_indices]
            depot = self.road_points[cid]

            if len(coords) == 0:
                print(f"⚠️ Cluster {cid} kosong — dilewati.")
                continue

            # Bagi hotspot ke n_drones
            if n_drones > 1 and len(coords) > n_drones:
                kmeans = KMeans(n_clusters=n_drones, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords)
                groups = [[] for _ in range(n_drones)]
                for idx, label in zip(cluster_indices, labels):
                    groups[label].append(idx)
            else:
                groups = [cluster_indices]

            cluster_routes = []
            for g in groups:
                locs = [depot] + [self.coordinates[i] for i in g]
                dist_matrix = self.build_dist_matrix(locs)
                route, cost = self.tabu_search(
                    dist_matrix,
                    tabu_tenure=tabu_tenure,
                    max_iter=max_iter,
                    neighborhood_size=neighborhood_size
                )
                cluster_routes.append((route, g))
            all_routes[cid] = cluster_routes
        return all_routes

    # ---------- Visualization ----------
    def visualize_cluster_routes(self, all_routes):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 10))

        # 🎨 Palet warna per cluster
        cluster_color_palettes = {
            0: ["#A93226", "#52BE80"],  # Cluster 1: merah tua & merah muda
            1: ["#1E8449", "#E74C3C"],  # Cluster 2: hijau tua & hijau muda
            2: ["#2471A3", "#CCD1D1"],  # Cluster 3: biru tua & biru muda
            3: ["#AF7AC5", "#fa9725"],  # Cluster 4: ungu tua & oranye lembut
            4: ["#5D6D7E", "#85C1E9"],  # Cluster 5: abu tua & abu muda
        }

        # 🏠 Road points (Depot)
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i+1}', ha='center', va='center', fontweight='bold', fontsize=12, color='white')

        # 🔥 Hotspots per cluster
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            base_color = cluster_color_palettes.get(cid, ['gray'])[0]
            plt.scatter(lon, lat, c=base_color, s=80, alpha=0.7, zorder=4)

        # 🚁 Routes per cluster dan per drone
        for cid, routes in all_routes.items():
            palette = cluster_color_palettes.get(cid, ['gray', 'lightgray'])
            for d_idx, (route, hotspot_indices) in enumerate(routes):
                if len(route) <= 1:
                    continue

                # warna drone berdasarkan urutan
                color = palette[d_idx % len(palette)]

                lats, lons = [], []
                for loc in route:
                    if loc == 0:  # depot
                        lat, lon = self.road_points[cid]
                    else:
                        hotspot_idx = hotspot_indices[loc - 1]
                        lat, lon = self.coordinates[hotspot_idx]
                    lats.append(lat)
                    lons.append(lon)

                plt.plot(lons, lats, 'o-', color=color,
                        label=f'Cluster {cid+1} Drone {d_idx+1}',
                        linewidth=2.2, markersize=6, alpha=0.9)

        plt.title('Drone Routes (Tabu Search per Cluster)', fontsize=15, fontweight='bold')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # def visualize_cluster_routes(self, all_routes, base_map_path="forest_fire_clusters_map_melawi.html", output_path="forest_fire_clusters_with_tabu_routes_melawi.html"):
    #     from bs4 import BeautifulSoup
    #     import re

    #     # === 1️⃣ Baca peta dasar ===
    #     try:
    #         with open(base_map_path, "r", encoding="utf-8") as f:
    #             html_data = f.read()
    #     except FileNotFoundError:
    #         print(f"❌ File peta dasar tidak ditemukan: {base_map_path}")
    #         return

    #     soup = BeautifulSoup(html_data, "html.parser")

    #     # === 2️⃣ Deteksi variabel map Leaflet ===
    #     map_var_match = re.search(r"var\s+(map_[a-z0-9]+)\s*=", html_data)
    #     if not map_var_match:
    #         print("❌ Tidak ditemukan variabel peta di file HTML.")
    #         return
    #     map_var = map_var_match.group(1)
    #     print(f"✅ Variabel peta terdeteksi: {map_var}")

    #     # === 3️⃣ Warna kustom per cluster (Melawi punya 4 cluster) ===
    #     cluster_color_palettes = {
    #         0: ["#A93226", "#E74C3C"],  # Cluster 1: merah tua & merah muda
    #         1: ["#1E8449", "#52BE80"],  # Cluster 2: hijau tua & hijau muda
    #         2: ["#2471A3", "#85C1E9"],  # Cluster 3: biru tua & biru muda
    #         3: ["#AF7AC5", "#fa9725"]   # Cluster 4: ungu tua & ungu muda
    #     }

    #     legend_entries = []
    #     js_add_routes = "\n\n// === Tambahan: Drone Route Overlays (Tabu Search, Melawi) ===\n"

    #     # === 4️⃣ Loop per cluster dan drone ===
    #     for cid, routes in all_routes.items():
    #         palette = cluster_color_palettes.get(cid, ["#555555", "#AAAAAA"])
    #         for d_idx, (route, hotspot_indices) in enumerate(routes):
    #             if len(route) <= 1:
    #                 continue

    #             latlons = []
    #             for loc in route:
    #                 if loc == 0:  # depot
    #                     lat, lon = self.road_points[cid]
    #                 else:
    #                     # gunakan indeks hotspot sebenarnya
    #                     hotspot_idx = hotspot_indices[loc - 1]
    #                     lat, lon = self.coordinates[hotspot_idx]
    #                 latlons.append([lat, lon])

    #             color = palette[d_idx % len(palette)]

    #             js_add_routes += f"""
    # var droneRoute_{cid}_{d_idx} = L.polyline({latlons}, {{
    #     color: '{color}',
    #     weight: 3,
    #     opacity: 0.9,
    #     dashArray: '10, 10'
    # }}).addTo({map_var});
    # droneRoute_{cid}_{d_idx}.bindTooltip("Cluster {cid+1} - Drone {d_idx+1}");
    # """
    #             legend_entries.append((f"Cluster {cid+1} - Drone {d_idx+1}", color))

    #     js_add_routes += "\n// === End of Drone Routes ===\n"

    #     # === 5️⃣ Tambahkan legenda ===
    #     legend_html = '''
    #     <div style="
    #         position: fixed; bottom: 30px; left: 20px; width: 270px;
    #         background-color: white; border:2px solid grey; z-index:9999;
    #         font-size:13px; padding: 10px; line-height: 1.4;">
    #         <b>Legenda Rute Drone ✈️ (Melawi)</b><br>
    #         <hr style="margin:4px 0;">
    #     '''
    #     for label, color in legend_entries:
    #         legend_html += f'<div><span style="background-color:{color};width:18px;height:10px;display:inline-block;margin-right:6px;"></span>{label}</div>'
    #     legend_html += "<hr style='margin:6px 0;'>Garis putus-putus: Jalur Udara</div>"

    #     js_add_routes += f"""
    # var legend = L.control({{position: 'bottomleft'}});
    # legend.onAdd = function (map) {{
    #     var div = L.DomUtil.create('div', 'info legend');
    #     div.innerHTML = `{legend_html}`;
    #     return div;
    # }};
    # legend.addTo({map_var});
    # """

    #     # === 6️⃣ Sisipkan JS ke file HTML ===
    #     script_tag = soup.find_all("script")[-1]
    #     script_content = script_tag.string or ""
    #     script_tag.string = script_content + js_add_routes

    #     # === 7️⃣ Simpan hasil baru ===
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         f.write(str(soup))

    #     print(f"✅ Rute drone (Tabu Search, Melawi) berhasil ditambahkan ke peta: {output_path}")


    # ---------- Print ----------
    def print_cluster_routes(self, all_routes):
        print("\n=== DRONE ROUTES (Tabu Search, Fixed Version) ===")
        for cid, routes in all_routes.items():
            print(f"\nCluster {cid+1} (Drones: {len(routes)})")
            for d_idx, (route, hotspot_indices) in enumerate(routes):
                locs = [self.road_points[cid]] + [self.coordinates[i] for i in hotspot_indices]
                dist_matrix = self.build_dist_matrix(locs)

                total_dist = 0
                print(f"  Drone {d_idx+1} Route: ", end="")
                for i in range(len(route)-1):
                    from_idx, to_idx = route[i], route[i+1]
                    if from_idx == 0:
                        print(f"R{cid+1} -> ", end="")
                    else:
                        print(f"H{hotspot_indices[from_idx-1]} -> ", end="")
                    total_dist += dist_matrix[from_idx][to_idx]
                print(f"R{cid+1}")

                est_time = total_dist * 1000 / 30 / 60  # 30 m/s = 108 km/h
                print(f"    Total distance: {total_dist:.2f} km | Est. time: {est_time:.2f} min")

