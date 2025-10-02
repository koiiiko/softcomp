import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, random
from sklearn.cluster import KMeans

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
        plt.figure(figsize=(14, 10))
        colors = ['red','blue','green','orange','purple','brown','pink','gray']

        # depot
        for i,(lat,lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s')
            plt.text(lon, lat, f'R{i}', ha='center', va='center', color='white', fontweight='bold')

        # hotspots
        for i,(lat,lon) in enumerate(self.coordinates):
            cid=self.clusters[i]
            plt.scatter(lon, lat, c=colors[cid%len(colors)], s=100, alpha=0.7)
            plt.text(lon, lat, f'H{i}', ha='center', va='center', fontsize=8)

        # routes
        for cid,routes in all_routes.items():
            for d_idx,(route, hotspot_indices) in enumerate(routes):
                if len(route)<=1: continue
                lats,lons=[],[]
                locs = [self.road_points[cid]]+[self.coordinates[i] for i in hotspot_indices]
                for loc in route:
                    lat,lon = locs[loc]
                    lats.append(lat); lons.append(lon)
                plt.plot(lons,lats,'o-',color=colors[cid%len(colors)],
                         label=f'Cluster {cid} Drone {d_idx+1}')

        plt.title("Drone Routes (Tabu Search)")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.legend(); plt.grid(True,alpha=0.3); plt.show()

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
