from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx


def load_graph_csv(path: str | Path) -> nx.Graph:
    """Load an undirected graph from a CSV file with source,target columns."""
    graph = nx.Graph()
    with Path(path).open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            source = row["source"].strip()
            target = row["target"].strip()
            if source and target and source != target:
                graph.add_edge(source, target)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def write_graph_csv(graph: nx.Graph, path: str | Path) -> None:
    """Write graph edges as source,target CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["source", "target"])
        for source, target in sorted(graph.edges()):
            writer.writerow([source, target])
