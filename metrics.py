from graph.graph import Graph
from typing import Dict
import networkx as nx
import matplotlib.pyplot as plt

def calculate_metrics(graph: Graph) -> Dict[str, float]:
    """
    Calculate various metrics for the given graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary containing the calculated metrics.
    """
    G = graph.convert_to_networkx()
    metrics = {
        # degree metrics
        "num_vertices": len(graph.vertices),
        "num_edges": len(graph.edges),
        "average_degree": (2 * len(graph.edges)) / len(graph.vertices) if graph.vertices else 0,
        "average_in_degree": sum(dict(G.in_degree()).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_out_degree": sum(dict(G.out_degree()).values()) / len(graph.vertices) if graph.vertices else 0,
        "density": (2 * len(graph.edges)) / (len(graph.vertices) * (len(graph.vertices) - 1)) if graph.vertices else 0,
        # other metrics
        "average_clustering_coefficient": nx.average_clustering(G) if graph.vertices else 0,
        "assortativity": nx.degree_assortativity_coefficient(G) if graph.vertices else 0,
        "connected_components": nx.number_connected_components(G) if graph.vertices else 0,
        "giant_component_size": len(max(nx.connected_components(G), key=len)) if graph.vertices else 0,
        # path metrics
        "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
        "diameter": nx.diameter(G) if nx.is_connected(G) else float('inf'),
        # centrality metrics
        "average_degree_centrality": sum(nx.degree_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_betweenness_centrality": sum(nx.betweenness_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_closeness_centrality": sum(nx.closeness_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_eigenvector_centrality": sum(nx.eigenvector_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
    }

    return metrics


def distribution_of_degrees(graph: Graph, log_scale: bool = False) -> Dict[int, int]:
    """
    Calculate the distribution of vertex degrees in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    degree_distribution = {}
    
    for node in G.nodes():
        degree = G.degree(node)
        if degree not in degree_distribution:
            degree_distribution[degree] = 0
        degree_distribution[degree] += 1

    # Visualize the degree distribution
    if log_scale:
        plt.yscale("log")
    plt.bar(degree_distribution.keys(), degree_distribution.values())
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Distribution of Vertex Degrees")
    plt.show()

    return degree_distribution


def distribution_of_clustering_coefficients(graph: Graph, log_scale: bool = False) -> Dict[float, int]:
    """
    Calculate the distribution of clustering coefficients in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    clustering_distribution = {}
    
    for node in G.nodes():
        clustering_coeff = nx.clustering(G, node)
        if clustering_coeff not in clustering_distribution:
            clustering_distribution[clustering_coeff] = 0
        clustering_distribution[clustering_coeff] += 1

    # Visualize the clustering coefficient distribution
    if log_scale:
        plt.yscale("log")
    plt.bar(clustering_distribution.keys(), clustering_distribution.values())
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Count")
    plt.title("Distribution of Clustering Coefficients")
    plt.show()

    return clustering_distribution


def distribution_of_shortest_path_lengths(graph: Graph, log_scale: bool = False) -> Dict[int, int]:
    """
    Calculate the distribution of shortest path lengths in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    path_length_distribution = {}
    
    if nx.is_connected(G):
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            for length in lengths.values():
                if length not in path_length_distribution:
                    path_length_distribution[length] = 0
                path_length_distribution[length] += 1

        # Visualize the shortest path length distribution
        if log_scale:
            plt.yscale("log")
        plt.bar(path_length_distribution.keys(), path_length_distribution.values())
        plt.xlabel("Shortest Path Length")
        plt.ylabel("Count")
        plt.title("Distribution of Shortest Path Lengths")
        plt.show()

    return path_length_distribution


def betweenness_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the betweenness centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their betweenness centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.betweenness_centrality(G)


def closeness_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the closeness centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their closeness centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.closeness_centrality(G)


def eigenvector_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the eigenvector centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their eigenvector centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.eigenvector_centrality(G)


def degree_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the degree centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their degree centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.degree_centrality(G)