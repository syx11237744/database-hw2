# Project 2 Topic 1 - Ego-net Community Mining

This repository contains a runnable implementation for topic 1, "Community Detection and Community Search".

The reproduced paper is:

Epasto, Lattanzi, Mirrokni, Sebe, Taei, and Verma, "Ego-net Community Mining Applied to Friend Suggestion", PVLDB 2015.

## What Is Implemented

- Ego-net construction without ego node `Z_u`.
- Triangle-enumeration ego-net construction inspired by the paper's fast
  ego-network construction algorithm.
- Optional local PySpark MapReduce-style ego-net construction that mirrors the
  paper's partition-triple shuffle at a small local scale.
- Absolute-Potts-style label propagation and connected-components baselines used inside each ego-net.
- Paper-aligned ego-network friendship features `W1` to `W4`.
- Louvain baseline on the same train graph.
- Metrics: runtime, local ego-cluster density/conductance, link-prediction AUC, average precision, and Louvain global partition metrics.

## Datasets

The project uses three public datasets that are listed in the reproduced paper and downloadable from SNAP:

- Facebook social circles: 4,039 nodes and 88,234 edges
- Enron email network: 36,692 nodes and 183,831 edges
- Astro Physics collaboration network: 18,771 nodes and 198,050 edges after self-loop removal

These are the smallest practical choices among the paper's datasets. The script downloads the raw `.txt.gz` files to `data/raw/` and converts them to `data/*.csv`.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_datasets.py
python run_experiments.py
python plot_results.py
python run_dynamic_experiments.py
python plot_dynamic_results.py
```

The main results are saved to `results/experiment_results.csv`, and the plot is saved to `results/quality_runtime.png`.

The optional dynamic-graph results are saved to `results/dynamic_experiment_results.csv`, and the dynamic speedup plot is saved to `results/dynamic_speedup.png`.

Optional local Spark reproduction:

```bash
pip install -r requirements-spark.txt
python run_spark_experiments.py --datasets facebook enron astro_ph --rho 4 --rdd-partitions 4
```

The Spark results are saved to `results/spark_experiment_results.csv`. The script uses Spark local mode, so it validates the MapReduce dataflow but is not expected to be faster than the single-machine constructor on these small datasets.

## Reproduction Notes

The original paper focuses on large-scale MapReduce construction of all ego-nets. This implementation now uses a single-machine triangle-enumeration constructor for all `Z_u` edge sets. It also includes a local PySpark version of the paper's partition-triple shuffle, but the Spark local result is a functional reproduction of the dataflow rather than a proof of cluster-scale speedup. For debugging, `run_experiments.py --construction neighbor` switches to per-node neighborhood intersection construction.

The APM label-propagation update follows the paper appendix:

```text
score(u, label) = C_u(label) - alpha * (T(label) - C_u(label))
```

where `C_u(label)` is the number of neighbors of `u` currently carrying `label`, and `T(label)` is the current size of that label group.

## Optional Dynamic Extension

The optional extension maintains ego-net scores under edge insertions and deletions. When edge `(u,v)` changes, only the ego-nets of `u`, `v`, and their common neighbors can change. The implementation removes the old score contributions of these affected ego-nets, applies the edge updates, recomputes only those ego-nets, and then rebuilds communities from the maintained score graph.

Run:

```bash
python run_dynamic_experiments.py
python plot_dynamic_results.py
```
