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
from typing import Tuple, Callable, List, Dict
import networkx as nx


class A_star():
    def __init__(self) -> None:
        pass

    def euclidean_distance(self, G: nx.DiGraph, node1: str, node2: str) -> float:
        """Calculate the Euclidean distance between two nodes based on their (x, y) coordinates."""
        # grab the positons first
        x1, y1 = G.nodes[node1]['x'], G.nodes[node1]['y']
        x2, y2 = G.nodes[node2]['x'], G.nodes[node2]['y']
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def a_star(self, graph: nx.DiGraph, start: int, goal: int, heuristic: Callable[[str, str, Dict[str, Tuple[float, float]]], float], attribute = "travel_time") -> Tuple[List[str], float]:
        """Perform A* pathfinding on a networkx graph."""
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}  # To reconstruct the path
        g_score = {node: float("inf") for node in graph.nodes}
        g_score[start] = 0
        f_score = {node: float("inf") for node in graph.nodes}
        f_score[start] = heuristic(graph, start, goal)

        while open_set:
            # Pop the node with the smallest f_score
            current_f, current_node = heapq.heappop(open_set)

            # Check if we have reached the goal
            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                return path[::-1], g_score[goal]

            # Explore neighbors of the current node
            for neighbor in graph.neighbors(current_node):
                weight = graph[current_node][neighbor].get(attribute)  # Default to 1 if weight is not provided
                tentative_g_score = g_score[current_node] + weight

                # If this path is better, update the path to the neighbor
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    h_score = heuristic(graph, neighbor, goal)
                    f_score[neighbor] = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If the goal was never reached, return None or a large cost
        return None, float("inf")

# Example usage
if __name__ == '__main__':
    # Create a directed graph with weights
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('A', 'C', weight=4)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('B', 'D', weight=5)
    G.add_edge('C', 'D', weight=1)
    G.add_edge('D', 'E', weight=3)

    # Positions of the nodes in (x, y) coordinates for the heuristic function
    positions = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 2),
        'D': (3, 1),
        'E': (4, 0)
    }

    nx.set_node_attributes(G, positions, 'pos')
    # Find the shortest path from A to E
    path, cost = A_star.a_star(G, 'A', 'E', A_star.euclidean_distance)
    print(f"Shortest path: {path} with cost: {cost}")