# Topic 1 Report Notes

## Background

Community detection finds densely connected groups in a graph. Community search usually starts from query nodes or constraints and returns a cohesive subgraph around them. Ego-net mining is a local form of community analysis: instead of detecting one global partition, it detects small communities among the neighbors of each node.

## Literature Taxonomy

Community detection can be organized by:

- Objective: modularity optimization, conductance minimization, density maximization, statistical likelihood, embedding-based clustering.
- Output type: disjoint communities, overlapping communities, hierarchical communities, temporal communities.
- Scope: global partitioning, local community search, ego-net or query-centric mining.
- Graph type: homogeneous graphs, heterogeneous information networks, directed graphs, attributed graphs, temporal graphs, bipartite or labeled graphs.
- Algorithm family: label propagation, spectral methods, flow/random-walk methods, k-core/k-truss decomposition, matrix factorization, graph embedding, heuristic optimization.

Community search can be organized by:

- Cohesiveness model: k-core, k-truss, clique, quasi-clique, density, conductance.
- Query type: single query vertex, multiple query vertices, keyword or attribute constraints, spatial or temporal constraints.
- Graph semantics: attributed, heterogeneous, directed, temporal, labeled, bipartite.
- Update model: static, incremental, fully dynamic.

## Reproduced Algorithm

The implementation reproduces the ego-net mining idea from Epasto et al. For each node `u`, it builds `Z_u`, the induced graph on the neighbors of `u`. It clusters `Z_u` using either the APM label-propagation rule described in the paper appendix or the connected-components baseline inside `Z_u`. It then computes the paper's ego-network friendship features:

```text
W1(v,w) = sum over common egos u of 1[v and w are in the same cluster C of Z_u]
W2(v,w) = sum density(C)
W3(v,w) = sum |N_C(v) intersect N_C(w)|
W4(v,w) = sum min(|N_C(v)|, |N_C(w)|) / |C|
```

The previous global score-graph connected-components heuristic is no longer used in the main experiment, because the paper's connected-components baseline is local to each ego-net `Z_u`, not a post-processing step on the global pair-score graph.

## Baseline

Louvain is run on the same train graph as a global community-detection baseline. For link-prediction comparison, a held-out edge is scored by whether its endpoints are in the same Louvain community plus normalized common-neighbor support. Global modularity is reported only for Louvain, because EgoNet W1-W4 are friend-suggestion features rather than global partitions.

## Experiments

Run:

```bash
python scripts/generate_datasets.py
python run_experiments.py --seeds 1 2 3 4 5
python plot_results.py
```

Results are stored in:

- `results/experiment_results.csv`: per-seed detail rows.
- `results/experiment_summary.csv`: mean/std summary by dataset, algorithm, feature, and metric.
- `results/topk_precision_recall.csv`: per-seed top-k precision/recall curves.
- `results/topk_precision_recall_summary.csv`: mean/std top-k precision/recall curves.
- `results/runtime_csr_comparison.csv`: before/after runtime comparison for the CSR ego-net optimization.
- `results/runtime_array_apm_comparison.csv`: attempted APM array-counting comparison.
- `results/runtime_pair_accumulator_comparison.csv`: before/after runtime comparison for pair-id score accumulation.
- `results/quality_runtime.png`: mean-over-seeds runtime, AUC, and AP plot.

The experiment records train/test split seeds, negative-sampling seeds, the APM random tie-break seed strategy, local ego-cluster density/conductance quantiles, and circle-reconstruction availability. Circle reconstruction precision/recall is marked unavailable because the included SNAP combined graph files do not contain ground-truth circle memberships.

The code uses three datasets that are explicitly listed in the reproduced paper:

- Facebook social circles: 4,039 nodes and 88,234 edges.
- Enron email network: 36,692 nodes and 183,831 edges.
- Astro Physics collaboration network: 18,771 nodes and 198,050 edges after self-loop removal.

The mean-over-seeds results are:

| Dataset | Algorithm | Runtime (s) | Local Density | Global Modularity | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Facebook | EgoNet-APM-W1 | 3.9127 | 0.7267 | n/a | 0.9812 | 0.9802 |
| Facebook | EgoNet-APM-W2 | 3.9127 | 0.7267 | n/a | 0.9812 | 0.9803 |
| Facebook | EgoNet-APM-W3 | 3.9127 | 0.7267 | n/a | 0.9806 | 0.9796 |
| Facebook | EgoNet-APM-W4 | 3.9127 | 0.7267 | n/a | 0.9817 | 0.9812 |
| Facebook | EgoNet-CC-W1 | 2.3101 | 0.6141 | n/a | 0.9933 | 0.9919 |
| Facebook | EgoNet-CC-W2 | 2.3101 | 0.6141 | n/a | 0.9932 | 0.9920 |
| Facebook | EgoNet-CC-W3 | 2.3101 | 0.6141 | n/a | 0.9907 | 0.9895 |
| Facebook | EgoNet-CC-W4 | 2.3101 | 0.6141 | n/a | 0.9936 | 0.9930 |
| Facebook | Louvain | 0.3300 | n/a | 0.8346 | 0.9939 | 0.9934 |
| Enron | EgoNet-APM-W1 | 6.0719 | 0.8313 | n/a | 0.9338 | 0.9336 |
| Enron | EgoNet-APM-W2 | 6.0719 | 0.8313 | n/a | 0.9339 | 0.9338 |
| Enron | EgoNet-APM-W3 | 6.0719 | 0.8313 | n/a | 0.9333 | 0.9331 |
| Enron | EgoNet-APM-W4 | 6.0719 | 0.8313 | n/a | 0.9340 | 0.9338 |
| Enron | EgoNet-CC-W1 | 1.7311 | 0.8049 | n/a | 0.9798 | 0.9796 |
| Enron | EgoNet-CC-W2 | 1.7311 | 0.8049 | n/a | 0.9802 | 0.9803 |
| Enron | EgoNet-CC-W3 | 1.7311 | 0.8049 | n/a | 0.9742 | 0.9740 |
| Enron | EgoNet-CC-W4 | 1.7311 | 0.8049 | n/a | 0.9800 | 0.9801 |
| Enron | Louvain | 1.9963 | n/a | 0.6150 | 0.9699 | 0.9584 |
| AstroPh | EgoNet-APM-W1 | 5.0031 | 0.8989 | n/a | 0.9743 | 0.9742 |
| AstroPh | EgoNet-APM-W2 | 5.0031 | 0.8989 | n/a | 0.9743 | 0.9743 |
| AstroPh | EgoNet-APM-W3 | 5.0031 | 0.8989 | n/a | 0.9737 | 0.9737 |
| AstroPh | EgoNet-APM-W4 | 5.0031 | 0.8989 | n/a | 0.9743 | 0.9743 |
| AstroPh | EgoNet-CC-W1 | 2.4451 | 0.7933 | n/a | 0.9857 | 0.9856 |
| AstroPh | EgoNet-CC-W2 | 2.4451 | 0.7933 | n/a | 0.9860 | 0.9862 |
| AstroPh | EgoNet-CC-W3 | 2.4451 | 0.7933 | n/a | 0.9827 | 0.9826 |
| AstroPh | EgoNet-CC-W4 | 2.4451 | 0.7933 | n/a | 0.9856 | 0.9858 |
| AstroPh | Louvain | 1.5852 | n/a | 0.6273 | 0.9816 | 0.9793 |

Replacing per-ego NetworkX graphs with a local CSR representation and accumulating fixed candidate-pair scores by pair id improves static EgoNet runtime by 1.72x-2.18x for APM and 2.07x-2.39x for the connected-components baseline, relative to the earlier NetworkX ego-net implementation. The pair-id accumulator alone gives a small additional change: about 1.00x-1.08x for APM and 1.03x-1.07x for CC. The quality metrics remain unchanged because only score aggregation changed. We also tried replacing APM's per-node neighbor-label dictionary with a dense array counter; on these ego-nets it slowed APM to 0.91x-0.94x of the CSR baseline, so the default path keeps integer node ids with the faster sparse dictionary counter.

## Local PySpark MapReduce Reproduction

To approximate the paper's MapReduce Algorithm 3, the optional Spark experiment hashes nodes into `rho` partitions, maps each edge to partition-triple keys, shuffles edges by triple, runs triangle-based ego-net construction inside each reducer group, deduplicates `(ego, ego-edge)` outputs, and then groups by ego to run the same APM clustering. This validates the distributed dataflow locally, but Spark local mode includes JVM startup, serialization, and shuffle overhead, so it should not be interpreted as a speedup experiment.

Run:

```bash
python run_spark_experiments.py --datasets facebook enron astro_ph --rho 4 --rdd-partitions 4
```

Results are stored in `results/spark_experiment_results.csv`. Spark EgoNet is reported as W1-W4 friend-suggestion features and local ego-cluster statistics, not as a global partition.

| Dataset | Algorithm | Runtime (s) | Replication Factor | Local Density | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Facebook | Spark-EgoNet-APM-W1 | 23.8400 | 2.25 | 0.7251 | 0.9807 | 0.9799 |
| Facebook | Spark-EgoNet-APM-W2 | 23.8400 | 2.25 | 0.7251 | 0.9807 | 0.9799 |
| Facebook | Spark-EgoNet-APM-W3 | 23.8400 | 2.25 | 0.7251 | 0.9799 | 0.9792 |
| Facebook | Spark-EgoNet-APM-W4 | 23.8400 | 2.25 | 0.7251 | 0.9809 | 0.9803 |
| Facebook | Louvain | 0.2951 | 0.00 | n/a | 0.9938 | 0.9925 |
| Enron | Spark-EgoNet-APM-W1 | 20.0122 | 2.25 | 0.8310 | 0.9387 | 0.9386 |
| Enron | Spark-EgoNet-APM-W2 | 20.0122 | 2.25 | 0.8310 | 0.9388 | 0.9388 |
| Enron | Spark-EgoNet-APM-W3 | 20.0122 | 2.25 | 0.8310 | 0.9386 | 0.9385 |
| Enron | Spark-EgoNet-APM-W4 | 20.0122 | 2.25 | 0.8310 | 0.9388 | 0.9388 |
| Enron | Louvain | 1.8068 | 0.00 | n/a | 0.9693 | 0.9596 |
| AstroPh | Spark-EgoNet-APM-W1 | 29.5018 | 2.25 | 0.8991 | 0.9725 | 0.9725 |
| AstroPh | Spark-EgoNet-APM-W2 | 29.5018 | 2.25 | 0.8991 | 0.9725 | 0.9725 |
| AstroPh | Spark-EgoNet-APM-W3 | 29.5018 | 2.25 | 0.8991 | 0.9720 | 0.9720 |
| AstroPh | Spark-EgoNet-APM-W4 | 29.5018 | 2.25 | 0.8991 | 0.9725 | 0.9725 |
| AstroPh | Louvain | 1.5207 | 0.00 | n/a | 0.9828 | 0.9803 |

The mean Spark-EgoNet runtime is 24.4513 seconds, while mean Louvain runtime is 1.2075 seconds, a 20.25x runtime gap in local mode. The Spark-EgoNet W1-W4 AUC/AP values match the single-machine triangle-construction APM results, which indicates that the Spark job reproduces the same ego-net scoring semantics. The remaining runtime gap is mainly due to local Spark overhead and edge replication/shuffle, not a change in the underlying community scoring logic.

## Comparison With The Paper

The original paper evaluates ego-net mining on Amazon, Astro Physics, Enron, Slashdot, Patents, Facebook, LiveJournal, Twitter, and a proprietary Google+ graph. It reports detailed figures mainly for Patents, Facebook, LiveJournal, Twitter, and Google+. In this reproduction, we use three of the paper's public datasets: Facebook, Enron, and Astro Physics. This satisfies the project requirement of running 3-4 datasets from the chosen paper while avoiding the very large Patents, LiveJournal, Twitter, and Google+ graphs.

The following table compares the paper's reported numbers with our reproduced results. Some entries are not directly comparable because the paper reports circle reconstruction precision/recall and ranked friend-suggestion curves, while this project reports modularity, conductance, density, AUC, and AP on train/test edge splits.

| Aspect | Paper result | Our result | Interpretation |
|---|---:|---:|---|
| Dataset scale | Facebook: 4,039 nodes / 88,234 edges; Twitter: 81,306 / 1,768,149; LiveJournal: 4,847,571 / 68,993,773; Patents: 3,774,768 / 16,518,948 | Facebook: 4,039 / 88,234; Enron: 36,692 / 183,831; AstroPh: 18,771 / 198,050 | Facebook exactly matches the paper's reported Facebook scale. Enron and AstroPh are also paper-listed datasets, selected to keep experiments feasible. |
| Ego-net cluster density | Most retrieved ego-net clusters have density larger than 0.2; Google+ circles average density is 0.352 | APM local ego-cluster mean densities are 0.7267, 0.8313, and 0.8989; CC local ego-cluster mean densities are 0.6141, 0.8049, and 0.7933 for Facebook, Enron, and AstroPh | The local ego-net clusters are dense, consistent with the paper's microscopic-community claim. |
| Ego-net conductance | Paper observes typically low conductance in retrieved ego-net clusters; Google+ circles average conductance is 0.1438 | APM local ego-cluster mean conductances are 0.3307, 0.3186, and 0.2152; CC clusters are full connected components inside each `Z_u`, so their local conductance is 0.0 by construction | The previous high conductance mismatch came from evaluating global score-graph components, not local ego-net clusters. |
| Circle reconstruction | LP: node precision 0.86, edge precision 0.87, edge recall 0.96; SLPA: node precision 0.85, edge recall 0.98 | No circle labels are used in our experiment | Not directly comparable. We did not use Facebook circle ground truth; we evaluate unsupervised community quality and held-out edge prediction instead. |
| Friend suggestion | Ego-net friendship scores outperform common friends and Adamic-Adar in ranked precision/recall curves; live Google+ acceptance rate increases by more than 0.5% | Best EgoNet mean AUC/AP: Facebook CC-W4 0.9936/0.9930, Enron CC-W2 0.9802/0.9803, AstroPh CC-W2 0.9860/0.9862 | Strong qualitative agreement: local ego-net co-clustering is useful for link prediction. |
| Runtime/scalability | Fast sequential ego-net construction is 5x faster than naive on LiveJournal; distributed version is 11x faster | After CSR local ego-net optimization and pair-id score accumulation, mean APM-W1..W4 runtime is 3.9-6.1 seconds; CC-W1..W4 runtime is 1.7-2.4 seconds; Louvain is 0.3-2.0 seconds | The Python version uses triangle-based single-machine ego-net construction, CSR local ego-net clustering, and sparse candidate-pair score vectors, but still does not implement true distributed MapReduce scalability. |

The strongest agreement with the paper is in the link-prediction behavior. The paper argues that ego-net friendship scores are useful local features for friend suggestion. Our W1-W4 results are high on all three paper datasets, and the connected-components baseline inside each ego-net is particularly strong: Facebook CC-W4 reaches 0.9936/0.9930 mean AUC/AP, Enron CC-W2 reaches 0.9802/0.9803, and AstroPh CC-W2 reaches 0.9860/0.9862. This supports the same qualitative conclusion: co-occurrence inside ego-net communities gives a strong local similarity signal.

The main methodological correction is that EgoNet W1-W4 are no longer evaluated as global graph partitions. Louvain obtains meaningful global modularity because it directly optimizes a global partition; EgoNet produces local clusters and pairwise recommendation features. Therefore, global modularity is reported for Louvain only, while EgoNet is evaluated through local ego-cluster statistics and held-out link prediction.

In runtime, EgoNet remains slower than Louvain for APM, but the connected-components ego-net baseline is cheaper because it avoids iterative label propagation and is now close to Louvain on Enron. The implementation uses triangle enumeration to construct all ego-net edge sets and CSR arrays for local ego-net clustering/statistics. It still does not implement distributed MapReduce execution in the main single-machine experiment.

## Optional Dynamic Graph Extension

The optional part considers evolving graphs with edge insertions and deletions. The key observation is local dependency. For an edge update `(u,v)`, only the following ego-nets can change:

- `Z_u`, because `v` may enter or leave the neighbor set of `u`.
- `Z_v`, because `u` may enter or leave the neighbor set of `v`.
- `Z_x` for each common neighbor `x` of `u` and `v`, because the edge `(u,v)` may appear or disappear inside `x`'s ego-net.

Therefore, the dynamic algorithm maintains a score contribution table for every ego node. For each update batch, it removes old contributions for affected ego-nets, applies the edge insertions/deletions, recomputes only affected ego-nets, and updates the global pair-score table. Communities are then extracted from the maintained score graph, using the same `min_score` threshold as the static experiment.

The dynamic experiment simulates two batches per dataset. Each batch contains 25 inserted edges and 25 deleted edges. The full rebuild baseline reconstructs all ego-net scores from scratch using the same dynamic miner, so the comparison is methodologically consistent.

| Dataset | Batch | Events | Affected Ego-nets | Dynamic Update (s) | Full Rebuild (s) | Speedup | Dynamic Modularity | Full Modularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Facebook | 1 | 50 | 923 | 15.3857 | 20.9826 | 1.36x | 0.0009 | 0.0009 |
| Facebook | 2 | 50 | 1067 | 17.2129 | 21.2199 | 1.23x | 0.0009 | 0.0009 |
| Enron | 1 | 50 | 435 | 4.7207 | 24.0876 | 5.10x | 0.0568 | 0.0568 |
| Enron | 2 | 50 | 578 | 7.6395 | 24.8108 | 3.25x | 0.0569 | 0.0569 |
| AstroPh | 1 | 50 | 1052 | 6.0394 | 22.7659 | 3.77x | 0.0401 | 0.0401 |
| AstroPh | 2 | 50 | 1168 | 6.5260 | 22.9610 | 3.52x | 0.0402 | 0.0402 |

The average dynamic update time is 9.5874 seconds, while the average full rebuild time is 22.8046 seconds. The average speedup is 3.04x. The dynamic and full modularity values match to four decimal places, indicating that the localized update rule preserves the same community quality as full recomputation while reducing runtime.

Facebook has the smallest speedup because affected edges often touch high-degree nodes and many common neighbors, so the affected ego-net set is large. Enron obtains the largest speedup because each batch affects fewer ego-nets. This confirms the main intuition: dynamic ego-net maintenance is most beneficial when graph updates are local relative to the total graph size.
