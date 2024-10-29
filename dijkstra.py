'''
Dijkstra Algorithm Implementation for Pathfinding

The algorithm can be described through the following steps:

Initialization

	1.	Create a Table
		Set up a table for tracking nodes, costs, and iterations.
	2.	Input the Iteration Number
		Enter the current iteration number in the first column.
	3.	Specify Costs for Each Node
		For each node, specify the associated cost (the cost for the starting node is zero).
	4.	Assign Infinite Cost for All Other Nodes
		Since no paths to other nodes exist yet, assign them all an initial cost of infinity.
	5.	Create a Queue
		Set up a queue that displays all nodes found so far.

Iteration

	1.	Select the First Node and Check Its Direct Successors
		Choose the first node and look at its direct successors (adjacent nodes).
	2.	Record Costs and Mark the Start Node as Complete
		After recording the costs of the successors, mark the start node as completed.
	3.	Add Successors to the Queue
		Add the successors to the queue.

Iteration 2

	1.	Choose the Node with the Lowest Cost
		Now, select the node with the lowest cost from the queue.
	2.	Check Its Successors and Record the Costs
		Look at its successors and update the costs accordingly.

This process is repeated until all nodes have been processed. In the end, you obtain the path with the lowest total cost through the selected nodes.

O((V + E) log V) 
'''
import heapq
from typing import Tuple


def dijkstra(graph: dict, start: str) -> Tuple[list, list]:
    # First we initialize all distances with a large number, could also use float("infinity") for all nodes besides the start node
    distances = {node: 1e100 for node in graph}
    distances[start] = 0

    # We also need a priority queue to store nodes and their distances
    priority_queue = [(0, start)]

    # As well as a dictionary to keep track of the shortest path to all nodes
    previous_nodes = {node: None for node in graph}

    while priority_queue:
        # Here we pop the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the distance is greater than the recorded one, skip processing
        # Dont really need this tbh, just looks nice
        if current_distance > distances[current_node]:
            continue

        # Now we iterate over neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # ofc  we only update if the new distance is smaller
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

            # not necessary either
            else:
                continue

    return distances, previous_nodes


def get_shortest_path(graph: dict, start: str, end: str) -> Tuple[list, list]:
    distances, previous_nodes = dijkstra(graph, start)
    path = []
    current_node = end

    # Now we backtrack from the end node to the start node
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]

    # We reverse the path to get the correct order from start to end
    path = path[::-1]

    if path[0] == start:
        return path, distances[end]
    else:
        # we didnt find a path, we should raise a error or return None, large number
        return None, 1e100


# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

path, cost = get_shortest_path(graph, 'A', 'D')
# flush = True, in case we run on raspbery, etc.
print(f"Shortest path: {path} with cost: {cost}", flush=True)
