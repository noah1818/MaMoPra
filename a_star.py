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


def euclidean_distance(node1: str, node2: str, positions: str) -> float:
    # Calculate the Euclidean distance between two nodes, positions is only there to
    # get the (x,y) coordinates of the two nodes
    x1, y1 = positions[node1]
    x2, y2 = positions[node2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def a_star(graph: dict, start: str, goal: str, positions: dict, heuristic: Callable[[str, str, dict], float]) -> Tuple[List[str], float]:
    # Initialize distances and heuristic costs, could also use float("infinity")
    open_set = [(0, start)]  # (f_score, node)
    came_from = {}  # To reconstruct the path
    g_score = {node: 1e100 for node in graph}
    g_score[start] = 0
    f_score = {node: 1e100 for node in graph}
    f_score[start] = 0

    while open_set:
        # Here we just pop the node with the smallest f_score, we dont care about current_f
        # We dont need current_f
        current_f, current_node = heapq.heappop(open_set)

        # In case the goal is reached, reconstruct the path
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1], g_score[goal]

        # Explore all other neighbors
        for neighbor, weight in graph[current_node].items():
            tentative_g_score = g_score[current_node] + weight

            if tentative_g_score < g_score[neighbor]:
                # This path to neighbor is the best so far
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                h_score = heuristic(neighbor, goal, positions)
                f_score[neighbor] = tentative_g_score + h_score

                # Add the neighbor to the priority queue
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float(1e100)  # If no path is found


# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1, 'E': 3},
    'E': {'D': 3}
}

# Positions of the nodes in (x, y) coordinates, for example a street network
positions = {
    'A': (0, 0),
    'B': (1, 1),
    'C': (2, 2),
    'D': (3, 1),
    'E': (4, 0)
}

path, cost = a_star(graph, 'A', 'E', positions, euclidean_distance)
print(f"Shortest path: {path} with cost: {cost}")
