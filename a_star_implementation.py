'''
The A* algorithm is a pathfinding algorithm used to calculate the optimal path between two points, known as nodes.

It is considered an extension of Dijkstras algorithm. One of the advantages of A* over other pathfinding algorithms is that it uses a heuristic function to reduce the runtime.

Heuristic

The heuristic estimates the cost of a sequence of operations to reach a target state. These are strategies that help make quick and efficient decisions with minimal effort.

A* always explores the nodes that are most likely to lead quickly to the goal first. These nodes are determined using a heuristic function, 
which indicates the order in which the nodes are traversed.

f(x) = g(x) + h(x)

For a node  x:
	g(x) : The exact cost of the path from the start node to the current node.
	h(x) : An admissible (non-overestimated) cost to reach the goal from the current node.

If an overestimated value is chosen for  h(x) , the goal node may be found faster, but at the cost of optimality. In the worst case, 
this can result in no path being found, even if one exists.

If an underestimated value is chosen for  h(x) , A* will always find the best possible path. 
The smaller the value, the longer it may take to find the path, and this can lead to A* performing the same as Dijkstras algorithm.

f(x) = g(x) + h(x)

This represents the cost of reaching the goal if the current node is chosen as part of the path.

A crucial factor in ensuring the best performance of A* is the selection of the heuristic function. Ideally,  
h(x)  should equal the actual cost required to reach the goal node. A suitable example for path estimation is the straight-line distance, 
as the actual path can never be shorter than the direct connection


In the following f(x) is refered to as f_score, g(x) as g_score and h(x) as h_score.

https://www.hs-augsburg.de/homes/mhau1/Pathfinding/
'''
import heapq
import math
from typing import Tuple, Callable, List
import networkx as nx


class AStar:
    def __init__(self) -> None:
        pass

    def euclidean_distance(self, graph: nx.DiGraph, node1: str, node2: str) -> float:
        """Calculate the Euclidean distance between two nodes based on their (x, y) coordinates."""
        x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
        x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def manhatten_distance(self, graph: nx.DiGraph, node1: str, node2: str) -> float:
        """Calculate the Manhatten distance between two nodes based on their (x, y) coordinates."""
        x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
        x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
        return abs(x2 - x1) + abs(y2 - y1)
    
    def chebyshev_distance(self, graph: nx.DiGraph, node1: str, node2: str) -> float:
        """Calculate the Manhatten distance between two nodes based on their (x, y) coordinates."""
        x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
        x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
        return max(abs(x2 - x1), abs(y2 - y1))

    def get_elevation_difference(self, graph: nx.DiGraph, node1: str, node2: str) -> float:
        elevation_current = graph.nodes[node1].get('elevation', 0)
        elevation_goal = graph.nodes[node2].get('elevation', 0)
        # Penalize uphill only
        return max(0, elevation_goal - elevation_current)

    def a_star(self, graph: nx.DiGraph, start: str, goal: str, custom_heuristic: bool, heuristic: Callable[[nx.DiGraph, str, str], float], attribute, alpha) -> Tuple[List[str], float]:
        """Perform A* pathfinding on a networkx graph."""
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}  # To reconstruct the path
        g_score = {node: float("inf") for node in graph.nodes}
        g_score[start] = 0
        f_score = {node: float("inf") for node in graph.nodes}
        f_score[start] = heuristic(graph, start, goal)

        visited_nodes = 0

        while open_set:
            # Pop the node with the smallest f_score
            current_f, current_node = heapq.heappop(open_set)
            visited_nodes += 1

            # Check if we have reached the goal
            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                # we return the cummulative cost, to reach the goal from the start
                cum_atttribute: float = 0
                for a, b in zip(path[::-1], path[::-1][1:]):
                    cum_atttribute += graph[a][b].get("travel_time")
                return path[::-1], cum_atttribute, visited_nodes

            # Explore neighbors of the current node
            for neighbor in graph.neighbors(current_node):
                tentative_g_score = g_score[current_node] + \
                    graph[current_node][neighbor].get(attribute)

                # If this path is better, update the path to the neighbor
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score

                    if custom_heuristic:
                        f_score[neighbor] = alpha * heuristic(graph, neighbor, goal) * 100 + (1 - alpha) * self.get_elevation_difference(
                            graph, neighbor, goal)
                    else:
                        f_score[neighbor] = g_score[neighbor] + \
                            heuristic(graph, neighbor, goal)
                    # tentative_g_score +  h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If the goal was never reached, return None or a large cost
        return None, float("inf"), visited_nodes
