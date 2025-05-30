import osmium as osm
import networkx as nx
from typing import Dict, List, Tuple
import math

class GIKIHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.locations = {}  # name -> (lat, lon)
        self.buildings = {}  # name -> (lat, lon)
        self.ways = []  # List of ways (paths)
        self.nodes = {}  # node_id -> (lat, lon)
        
    def node(self, n):
        # Store all nodes
        self.nodes[n.id] = (n.location.lat, n.location.lon)
        
        # Check if it's a named location
        if 'name' in n.tags:
            name = n.tags['name']
            if 'building' in n.tags:
                self.buildings[name] = (n.location.lat, n.location.lon)
            else:
                self.locations[name] = (n.location.lat, n.location.lon)
    
    def way(self, w):
        # Store ways (paths) between nodes
        if len(w.nodes) >= 2:
            self.ways.append([n.ref for n in w.nodes])

def parse_osm_file(file_path: str) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, float]]]:
    """
    Parse the OSM file and return locations and edges with distances.
    Returns:
        - locations: Dictionary mapping location names to their (lat, lon) coordinates
        - edges: List of tuples (location1, location2, distance_in_meters)
    """
    handler = GIKIHandler()
    handler.apply_file(file_path)
    
    # Create a graph from the ways
    G = nx.Graph()
    
    # Add all nodes
    for node_id, coords in handler.nodes.items():
        G.add_node(node_id, pos=coords)
    
    # Add edges from ways
    for way in handler.ways:
        for i in range(len(way) - 1):
            node1 = way[i]
            node2 = way[i + 1]
            if node1 in handler.nodes and node2 in handler.nodes:
                # Calculate distance in meters using Haversine formula
                lat1, lon1 = handler.nodes[node1]
                lat2, lon2 = handler.nodes[node2]
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                G.add_edge(node1, node2, weight=distance)
    
    # Combine all named locations
    all_locations = {**handler.locations, **handler.buildings}
    
    # Create edges between named locations
    edges = []
    location_names = list(all_locations.keys())
    
    # Connect each location to its nearest neighbors
    for i in range(len(location_names)):
        name1 = location_names[i]
        coords1 = all_locations[name1]
        
        # Find distances to all other locations
        distances = []
        for j in range(len(location_names)):
            if i != j:
                name2 = location_names[j]
                coords2 = all_locations[name2]
                distance = haversine_distance(coords1[0], coords1[1], coords2[0], coords2[1])
                distances.append((distance, j))
        
        # Sort by distance and connect to the 3 nearest neighbors
        distances.sort()
        for dist, j in distances[:3]:  # Connect to 3 nearest neighbors
            name2 = location_names[j]
            edges.append((name1, name2, dist))
    
    return all_locations, edges

def find_closest_node(graph: nx.Graph, coords: Tuple[float, float]) -> int:
    """Find the closest node in the graph to the given coordinates."""
    min_dist = float('inf')
    closest_node = None
    
    for node, data in graph.nodes(data=True):
        node_coords = data['pos']
        dist = haversine_distance(coords[0], coords[1], node_coords[0], node_coords[1])
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    
    return closest_node

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points in meters.
    """
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