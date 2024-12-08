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
from typing import Tuple, Optional
import networkx as nx

class Dijkstra():
    def __init__(self) -> None:
        pass

    def dijkstra(self, G: nx.DiGraph, start: int, end: int, attribute: str = "travel_time") -> Tuple[Optional[list[int]], float]:
        # Check if attribute exists in the graph edges
        if not any(attribute in data for _, _, data in G.edges(data=True)):
            raise ValueError(f"Attribute '{attribute}' not found in graph edges.")

        # Initialize distances with infinity for all nodes except the start node
        distances = {node: float("inf") for node in G.nodes}
        distances[start] = 0

        # Priority queue to store nodes with their distances
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
                weight = attributes.get(attribute, 1)  # Default weight is 1 if not specified
                distance = current_distance + weight

                # Only update if the new distance is shorter
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        # If there's no path from start to end
        return None, float("inf")

    def get_shortest_path(self, graph: nx.DiGraph, start: int, goal: int, attribute: str = "travel_time") -> Tuple[Optional[list[int]], float]:
        path, distance = self.dijkstra(graph, start, goal, attribute)
        if path is None or path[0] != start:
            # No valid path found
            return None, float("inf")
        return path, distance