from __future__ import annotations

import gzip
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from egonet_mining.data import write_graph_csv


RAW_DIR = ROOT / "data" / "raw"

DATASETS = {
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "astro_ph": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
}


def download(url: str, path: pathlib.Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def load_snap_edges(path: pathlib.Path, prefix: str) -> nx.Graph:
    graph = nx.Graph()
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip() or line.startswith("#"):
                continue
            source, target, *_ = line.split()
            if source != target:
                graph.add_edge(f"{prefix}_{source}", f"{prefix}_{target}")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def main() -> None:
    for name, url in DATASETS.items():
        raw_path = RAW_DIR / pathlib.Path(url).name
        download(url, raw_path)
        graph = load_snap_edges(raw_path, name)
        write_graph_csv(graph, ROOT / "data" / f"{name}.csv")
        print(f"{name}: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


if __name__ == "__main__":
    main()
