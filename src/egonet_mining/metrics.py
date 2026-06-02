from __future__ import annotations

from collections.abc import Iterable

import networkx as nx


def normalize_communities(communities: Iterable[Iterable[str]]) -> list[set[str]]:
    return [set(comm) for comm in communities if comm]


def mean_density(graph: nx.Graph, communities: Iterable[Iterable[str]]) -> float:
    values: list[float] = []
    for community in normalize_communities(communities):
        if len(community) < 2:
            continue
        subgraph = graph.subgraph(community)
        possible = len(community) * (len(community) - 1) / 2
        values.append(subgraph.number_of_edges() / possible if possible else 0.0)
    return sum(values) / len(values) if values else 0.0


def conductance(graph: nx.Graph, community: set[str]) -> float:
    if not community or len(community) == graph.number_of_nodes():
        return 0.0
    cut_edges = nx.cut_size(graph, community)
    volume = sum(dict(graph.degree(community)).values())
    return cut_edges / volume if volume else 0.0


def mean_conductance(graph: nx.Graph, communities: Iterable[Iterable[str]]) -> float:
    values = [conductance(graph, set(comm)) for comm in communities if len(set(comm)) >= 2]
    return sum(values) / len(values) if values else 0.0


def modularity_safe(graph: nx.Graph, communities: Iterable[Iterable[str]]) -> float:
    partition = normalize_communities(communities)
    covered: set[str] = set()
    disjoint: list[set[str]] = []
    for community in partition:
        fresh = community - covered
        if fresh:
            disjoint.append(fresh)
            covered.update(fresh)
    for node in graph.nodes:
        if node not in covered:
            disjoint.append({node})
    if graph.number_of_edges() == 0 or not disjoint:
        return 0.0
    return nx.algorithms.community.quality.modularity(graph, disjoint)


def coverage(graph: nx.Graph, communities: Iterable[Iterable[str]]) -> float:
    partition = normalize_communities(communities)
    node_to_comm: dict[str, int] = {}
    for idx, community in enumerate(partition):
        for node in community:
            node_to_comm.setdefault(node, idx)
    if graph.number_of_edges() == 0:
        return 0.0
    internal = sum(
        1
        for source, target in graph.edges
        if node_to_comm.get(source) == node_to_comm.get(target)
    )
    return internal / graph.number_of_edges()
