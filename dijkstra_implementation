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
	1.	Select the Node with the Lowest Cost:
		Extract the node with the smallest cost from the priority queue. Initially, this will be the starting node.

	2.	Process its direct Successors:
		For each adjacent node (successor), calculate the potential new cost of reaching that node:
		New Cost = Current Nodes Cost + Weight of Edge to Successor
		If this new cost is smaller than the previously recorded cost for the successor:
		Update the successors cost in the table.
		Record the current node as the predecessor of the successor.
		Add the successor to the priority queue with its updated cost.

	3.	Mark the Node as Processed:
		Once all its successors are updated, the current node is considered �processed� and will not be revisited.

This process is repeated until all nodes have been processed. In the end, you obtain the path with the lowest total cost through the selected nodes.

O((V + E) log V) 
'''
import heapq
from typing import Tuple, Optional
import networkx as nx


class Dijkstra():
    def __init__(self) -> None:
        pass

    def dijkstra(self, graph: nx.DiGraph, start: int, end: int, attribute: str = "travel_time") -> Tuple[Optional[list[int]], float]:
        # Check if attribute exists in the graph edges
        if not any(attribute in data for _, _, data in graph.edges(data=True)):
            raise ValueError(
                f"Attribute '{attribute}' not found in graph edges.")

        # Initialize distances with infinity for all nodes except the start node
        distances = {node: float("inf") for node in graph.nodes}
        distances[start] = 0

        # Priority queue to store nodes with their distances
        priority_queue = [(0, start)]

        # To reconstruct the path
        previous_nodes = {node: None for node in graph.nodes}

        while priority_queue:
            # Pop the node with the smallest distance
            current_distance, current_node = heapq.heappop(priority_queue)

            # If we reached the end node, reconstruct and return the path
            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                # we return the cum cost
                cum_atttribute: float = 0
                for a, b in zip(path[::-1], path[::-1][1:]):
                    cum_atttribute += graph[a][b].get(attribute)
                return path[::-1], cum_atttribute

            # Process each neighbor
            for neighbor, attributes in graph[current_node].items():
                # Default weight is 1 if not specified
                weight = attributes.get(attribute, 1)
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
