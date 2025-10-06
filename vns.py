import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ClusterBasedDroneRoutingVNS:
    drone_speed = 30
    max_flight_time = 170
    max_distance = (drone_speed * max_flight_time * 60) / 1000
    penalty_factor = 100

    def __init__(self, csv_file=None, road_points=None, n_drones=None):
        self.df = pd.read_csv(csv_file)
        self.coordinates = list(zip(self.df['Latitude'], self.df['Longitude']))
        self.clusters = self.df['Cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        if n_drones is None:
            self.drones_per_cluster = {cid: 1 for cid in range(self.n_clusters)}
        elif isinstance(n_drones, int):
            self.drones_per_cluster = {cid: n_drones for cid in range(self.n_clusters)}
        elif isinstance(n_drones, dict):
            self.drones_per_cluster = {cid: n_drones.get(cid, 1) for cid in range(self.n_clusters)}
        else:
            raise ValueError("n_drones must be None, int, or dict")

        if isinstance(road_points, dict):
            self.road_points = [road_points[k] for k in sorted(road_points.keys())]
        else:
            self.road_points = list(road_points)

        while len(self.road_points) < self.n_clusters:
            self.road_points.append(self.road_points[0])

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

    def _swap(self, route):
        r = route[:]
        if len(r) > 2:
            i, j = sorted(random.sample(range(1, len(r)-1), 2))
            r[i], r[j] = r[j], r[i]
        return r

    def _insert(self, route):
        r = route[:]
        if len(r) > 3:
            i, j = random.sample(range(1, len(r)-1), 2)
            elem = r.pop(i)
            r.insert(j, elem)
        return r

    def _reverse_subroute(self, route):
        r = route[:]
        if len(r) > 3:
            i, j = sorted(random.sample(range(1, len(r)-1), 2))
            r[i:j] = reversed(r[i:j])
        return r

    def _local_search(self, route):
        """Try small 2-opt-like improvements until no improvement found."""
        improved = True
        best = route
        best_fit = self._fitness(best)
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new = route[:]
                    new[i:j] = reversed(new[i:j])
                    new_fit = self._fitness(new)
                    if new_fit < best_fit:
                        best = new
                        best_fit = new_fit
                        improved = True
            route = best
        return best

    def solve_vrp_vns(self, cluster_id, hotspot_indices=None, max_iter=200):
        if hotspot_indices is None:
            cluster_hotspots = [i + len(self.road_points) for i, c in enumerate(self.clusters) if c == cluster_id]
        else:
            cluster_hotspots = [i + len(self.road_points) for i in hotspot_indices]

        if not cluster_hotspots:
            return []
        if len(cluster_hotspots) == 1:
            return [cluster_id] + cluster_hotspots + [cluster_id]

        start_point = cluster_id
        current = [start_point] + random.sample(cluster_hotspots, len(cluster_hotspots)) + [start_point]
        best = current
        best_fit = self._fitness(best)
        neighborhoods = [self._swap, self._insert, self._reverse_subroute]

        for _ in range(max_iter):
            k = 0
            while k < len(neighborhoods):
                new = neighborhoods[k](current)
                new = self._local_search(new)
                new_fit = self._fitness(new)
                if new_fit < best_fit:
                    best, best_fit = new, new_fit
                    current = new
                    k = 0
                else:
                    k += 1
        return best

    def optimize_all_clusters(self, max_iter=200):
        all_routes = {}
        for cluster_id in range(self.n_clusters):
            n_drones = self.drones_per_cluster.get(cluster_id, 1)
            cluster_routes = []
            if n_drones > 1:
                groups = self._split_hotspots_for_cluster(cluster_id, n_drones)
                for group in groups:
                    route = self.solve_vrp_vns(cluster_id, hotspot_indices=group, max_iter=max_iter)
                    cluster_routes.append(route)
            else:
                route = self.solve_vrp_vns(cluster_id, max_iter=max_iter)
                cluster_routes = [route]
            all_routes[cluster_id] = cluster_routes
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
    
    def visualize_cluster_routes(self, all_routes):
        plt.figure(figsize=(14, 10))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        linestyles = ['-', '--', '-.', ':']  # Different line styles

        # Road points (depots)
        for i, (lat, lon) in enumerate(self.road_points):
            plt.scatter(lon, lat, c='black', s=200, marker='s', zorder=5)
            plt.text(lon, lat, f'R{i}', ha='center', va='center',
                    fontweight='bold', fontsize=12, color='white')

        # Hotspot points
        for i, (lat, lon) in enumerate(self.coordinates):
            cid = self.clusters[i]
            plt.scatter(
                lon, lat,
                c=colors[cid % len(colors)],
                s=100, alpha=0.7, zorder=4
            )
            plt.text(lon, lat, f'H{i}', ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # Routes
        for cid, routes in all_routes.items():
            num_drones = len(routes)
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

                # Change line + marker style if multiple drones per cluster
                if num_drones > 1:
                    linestyle = linestyles[d_idx % len(linestyles)]
                    marker = markers[d_idx % len(markers)]
                else:
                    linestyle = '-'
                    marker = 'o'

                plt.plot(
                    lons, lats,
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    label=f'Cluster {cid} Drone {d_idx+1}',
                    linewidth=2,
                    markersize=6
                )

        plt.title('Drone Routes (VNS)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def print_cluster_routes(self, all_routes):
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
                valid = " " if flight_time_min <= self.max_flight_time else "Flight Time Limit Exceeded!"
                print(f"  Drone {d_idx+1} Route: {route}")
                print(f"    Total distance: {total_distance:.2f} km")
                print(f"    Flight time: {flight_time_min:.2f} min {valid}")
