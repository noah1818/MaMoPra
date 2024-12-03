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
import networkx as nx


def dijkstra(G: nx.DiGraph, start: str, end: str, attribute: str = "weight") -> Tuple[list[str], float]:
    # check if attribute is in the attributes of the graph
    if attribute not in list(G.edges(data=True))[0][2].keys():
        raise ValueError("attribute not in graphs attributes", flush = True)
    # Initialize distances with infinity for all nodes except the start node
    distances = {node: float("inf") for node in G.nodes}
    distances[start] = 0

    # Priority queue to store nodes with their f_score
    priority_queue = [(0, start)]
    
    # To reconstruct the path
    previous_nodes = {node: None for node in G.nodes}

    while priority_queue:
        # Pop the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we reached the end node, reconstruct and return the path
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.reverse()
            return path, distances[end]

        # Process each neighbor
        for neighbor, attributes in G[current_node].items():
            weight = attributes[0].get(attribute, 1)  # Default weight is 1 if not specified
            distance = current_distance + weight

            # Only update if the new distance is shorter
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # If there's no path from start to end
    return None, float("inf")


def get_shortest_path(graph: nx.DiGraph, start: str, end: str) -> Tuple[list, int]:
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

'''if __name__ == "__main__":
    # Create a directed graph with weights
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=4)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('B', 'C', weight=5)
    G.add_edge('B', 'D', weight=10)
    G.add_edge('C', 'E', weight=3)
    G.add_edge('E', 'D', weight=4)
    G.add_edge('D', 'F', weight=11)

    # Find the shortest path from 'A' to 'D'
    path, total_distance = dijkstra(G, 'A', 'D')
    print(f"Shortest path from 'A' to 'D': {path} with total distance: {total_distance}")

    # Find the shortest path from 'A' to 'F'
    path, total_distance = dijkstra(G, 'A', 'F')
    print(f"Shortest path from 'A' to 'F': {path} with total distance: {total_distance}")'''