import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable
import math
from osm_parser import parse_osm_file
import folium
from folium import plugins
import random
from collections import deque, defaultdict

class CampusGraph:
    def __init__(self, osm_file: str = "giki.osm"):
        self.graph = nx.Graph()
        self.node_positions = {}  # For visualization
        self.obstacles = set()    # Set of nodes that are obstacles
        self._initialize_campus_graph(osm_file)
        self._add_complex_topology()
        self._add_obstacles()
        self._add_varied_weights()

    def _initialize_campus_graph(self, osm_file: str):
        # Parse OSM file to get locations and edges
        locations, edges = parse_osm_file(osm_file)
        
        # Add nodes with positions
        for loc, coords in locations.items():
            self.graph.add_node(loc)
            self.node_positions[loc] = coords

        # Add edges with weights (distances in meters)
        for edge in edges:
            loc1, loc2, distance = edge
            # Add some random variation to edge weights
            variation = random.uniform(0.8, 1.2)
            self.graph.add_edge(loc1, loc2, weight=distance * variation)

    def _add_complex_topology(self):
        """Add additional nodes and edges to create a more complex topology"""
        # Get existing nodes and their positions
        existing_nodes = list(self.graph.nodes())
        if len(existing_nodes) < 2:
            return

        # Add new nodes with connections to existing ones
        for _ in range(len(existing_nodes) // 2):
            # Select two random existing nodes
            node1 = random.choice(existing_nodes)
            node2 = random.choice(existing_nodes)
            if node1 == node2:
                continue

            # Create a new node between them
            lat1, lon1 = self.node_positions[node1]
            lat2, lon2 = self.node_positions[node2]
            new_lat = (lat1 + lat2) / 2
            new_lon = (lon1 + lon2) / 2
            new_node = f"junction_{len(self.graph.nodes())}"
            
            # Add the new node and its connections
            self.graph.add_node(new_node)
            self.node_positions[new_node] = (new_lat, new_lon)
            
            # Calculate distances and add edges
            dist1 = self._haversine_distance(lat1, lon1, new_lat, new_lon)
            dist2 = self._haversine_distance(new_lat, new_lon, lat2, lon2)
            
            self.graph.add_edge(node1, new_node, weight=dist1)
            self.graph.add_edge(new_node, node2, weight=dist2)

    def _add_obstacles(self):
        """Add random obstacles to the graph"""
        nodes = list(self.graph.nodes())
        num_obstacles = len(nodes) // 10  # 10% of nodes are obstacles
        
        for _ in range(num_obstacles):
            node = random.choice(nodes)
            self.obstacles.add(node)
            # Remove edges connected to this node
            self.graph.remove_edges_from(list(self.graph.edges(node)))

    def _add_varied_weights(self):
        """Add varied edge weights to create interesting path choices"""
        for u, v in self.graph.edges():
            # Get current weight
            current_weight = self.graph[u][v]['weight']
            
            # Add random variation based on edge properties
            if random.random() < 0.3:  # 30% chance of significant weight change
                # Create a "highway" effect - some edges are much faster
                if random.random() < 0.5:
                    self.graph[u][v]['weight'] = current_weight * 0.5  # Fast path
                else:
                    self.graph[u][v]['weight'] = current_weight * 2.0  # Slow path
            else:
                # Normal variation
                self.graph[u][v]['weight'] = current_weight * random.uniform(0.8, 1.2)

    def _heuristic(self, node1: str, node2: str, heuristic_type: str = "haversine") -> float:
        """Calculate heuristic distance between two nodes using different methods"""
        lat1, lon1 = self.node_positions[node1]
        lat2, lon2 = self.node_positions[node2]
        
        if heuristic_type == "haversine":
            return self._haversine_distance(lat1, lon1, lat2, lon2)
        elif heuristic_type == "manhattan":
            # Manhattan distance (grid-based)
            return abs(lat2 - lat1) + abs(lon2 - lon1)
        elif heuristic_type == "euclidean":
            # Euclidean distance
            return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        elif heuristic_type == "weighted":
            # Consider edge weights in the heuristic
            base_distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            # Add some weight consideration
            return base_distance * random.uniform(0.8, 1.2)
        else:
            return 0  # No heuristic (equivalent to Dijkstra's)

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the Haversine distance between two points in meters."""
        R = 6371000  # Earth's radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda/2) * math.sin(delta_lambda/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def _bfs_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Breadth-First Search implementation"""
        if start == end:
            return [start], 0

        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.graph.neighbors(current):
                if neighbor == end:
                    final_path = path + [neighbor]
                    distance = sum(self.graph[final_path[i]][final_path[i+1]]['weight'] 
                                 for i in range(len(final_path)-1))
                    return final_path, distance
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return [], float('inf')

    def _dfs_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Depth-First Search implementation"""
        if start == end:
            return [start], 0

        visited = set()
        path = []
        
        def dfs(current: str) -> bool:
            visited.add(current)
            path.append(current)
            
            if current == end:
                return True
                
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            
            path.pop()
            return False
        
        if dfs(start):
            distance = sum(self.graph[path[i]][path[i+1]]['weight'] 
                         for i in range(len(path)-1))
            return path, distance
        return [], float('inf')

    def _greedy_best_first(self, start: str, end: str) -> Tuple[List[str], float]:
        """Greedy Best-First Search implementation"""
        if start == end:
            return [start], 0

        visited = set()
        path = []
        current = start
        
        while current != end:
            path.append(current)
            visited.add(current)
            
            # Get unvisited neighbors
            neighbors = [(n, self._heuristic(n, end, "haversine")) 
                        for n in self.graph.neighbors(current) 
                        if n not in visited]
            
            if not neighbors:
                return [], float('inf')
            
            # Choose neighbor with minimum heuristic value
            current = min(neighbors, key=lambda x: x[1])[0]
        
        path.append(end)
        distance = sum(self.graph[path[i]][path[i+1]]['weight'] 
                      for i in range(len(path)-1))
        return path, distance

    def find_path(self, start: str, end: str, algorithm: str = "dijkstra", 
                 heuristic_type: str = "haversine") -> Tuple[List[str], float]:
        """Find shortest path using specified algorithm and heuristic"""
        if algorithm == "astar":
            # Use A* with specified heuristic
            path = nx.astar_path(self.graph, start, end, 
                               heuristic=lambda n1, n2: self._heuristic(n1, n2, heuristic_type),
                               weight="weight")
            distance = sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return path, distance
        elif algorithm == "dijkstra":
            # Use Dijkstra's algorithm
            path = nx.dijkstra_path(self.graph, start, end, weight="weight")
            distance = sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return path, distance
        elif algorithm == "bfs":
            return self._bfs_path(start, end)
        elif algorithm == "dfs":
            return self._dfs_path(start, end)
        elif algorithm == "greedy":
            return self._greedy_best_first(start, end)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def visualize_path(self, path: List[str] = None) -> folium.Map:
        """Visualize the graph and highlight the given path using folium"""
        # Calculate center of the map
        lats = [lat for lat, _ in self.node_positions.values()]
        lons = [lon for _, lon in self.node_positions.values()]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # Add all nodes as markers
        for node, (lat, lon) in self.node_positions.items():
            if node in self.obstacles:
                color = 'black'  # Obstacles are black
            elif path and node in path:
                color = 'green'  # Path nodes are green
            else:
                color = 'blue'   # Other nodes are blue
                
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                popup=node
            ).add_to(m)
        
        # Add all edges
        for edge in self.graph.edges():
            node1, node2 = edge
            lat1, lon1 = self.node_positions[node1]
            lat2, lon2 = self.node_positions[node2]
            
            # Create a line between the nodes
            line = folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color='gray',
                weight=2,
                opacity=0.5
            )
            line.add_to(m)
        
        # Highlight the path if provided
        if path:
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                lat1, lon1 = self.node_positions[node1]
                lat2, lon2 = self.node_positions[node2]
                
                # Create a highlighted line for the path
                path_line = folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color='red',
                    weight=4,
                    opacity=1
                )
                path_line.add_to(m)
        
        return m 