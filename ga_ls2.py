import random
import numpy as np
from ls import ClusterBasedDroneRouting_LocalSearch

class HybridGA_LocalSearch:
    def __init__(
        self,
        csv_file,
        road_points,
        n_drones,
        pop_size=20,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        local_search_rate=0.3,
        max_ls_iter=300
    ):
        self.routing = ClusterBasedDroneRouting_LocalSearch(
            csv_file=csv_file,
            road_points=road_points,
            n_drones=n_drones
        )
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.max_ls_iter = max_ls_iter
        self.dist_matrix = self.routing.dist_matrix

    # Core GA + LS Components
    def initialize_population(self, cluster_id):
        """Create random route population for a cluster"""
        cluster_hotspots = [
            i + len(self.routing.road_points)
            for i, c in enumerate(self.routing.clusters)
            if c == cluster_id
        ]
        population = []
        for _ in range(self.pop_size):
            route = [cluster_id] + random.sample(cluster_hotspots, len(cluster_hotspots)) + [cluster_id]
            population.append(route)
        return population

    def route_distance(self, route):
        return sum(self.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def fitness(self, route):
        return 1 / (1 + self.route_distance(route))

    def selection(self, population):
        """Tournament selection"""
        k = 3
        selected = random.sample(population, k)
        selected.sort(key=lambda r: self.route_distance(r))
        return selected[0]

    def crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        if random.random() > self.crossover_rate:
            return parent1[:]

        start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        pos = end
        for gene in parent2:
            if gene not in child:
                if pos >= len(child) - 1:
                    pos = 1
                child[pos] = gene
                pos += 1
        return child

    def mutate(self, route):
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(1, len(route) - 1), 2))
            route[i], route[j] = route[j], route[i]
        return route

    def local_search_refine(self, route):
        """Apply 2-opt local search for refinement"""
        cluster_id = route[0]
        best_route = self.routing.local_search_tsp(
            cluster_id=cluster_id,
            hotspot_indices=[idx - len(self.routing.road_points) for idx in route[1:-1]],
            max_iter=self.max_ls_iter
        )
        return best_route

    # Hybrid Optimization
    def optimize_cluster(self, cluster_id):
        population = self.initialize_population(cluster_id)
        best_route = min(population, key=self.route_distance)

        for gen in range(self.generations):
            new_population = []

            for _ in range(self.pop_size):
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                # Apply local search refinement probabilistically
                if random.random() < self.local_search_rate:
                    child = self.local_search_refine(child)

                new_population.append(child)

            population = new_population
            gen_best = min(population, key=self.route_distance)

            if self.route_distance(gen_best) < self.route_distance(best_route):
                best_route = gen_best

            print(f"Cluster {cluster_id} | Gen {gen+1}/{self.generations} | Best Distance: {self.route_distance(best_route):.2f}")

        return best_route

    def optimize_all_clusters(self):
        all_routes = {}
        for cluster_id in range(self.routing.n_clusters):
            n_drones = self.routing.drones_per_cluster.get(cluster_id, 1)
            cluster_routes = []
            if n_drones > 1:
                groups = self.routing.split_hotspots_for_cluster(cluster_id, n_drones)
                for _ in range(n_drones):
                    best_route = self.optimize_cluster(cluster_id)
                    cluster_routes.append(best_route)
            else:
                best_route = self.optimize_cluster(cluster_id)
                cluster_routes = [best_route]

            all_routes[cluster_id] = cluster_routes

        return all_routes
