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

The implementation reproduces the ego-net mining idea from Epasto et al. For each node `u`, it builds `Z_u`, the induced graph on the neighbors of `u`. It clusters `Z_u` using the APM label-propagation rule described in the paper appendix. Then it computes the ego-network friendship score:

```text
W(v,w) = sum over common egos u of 1[v and w are in the same cluster of Z_u]
```

The pair scores are also aggregated into a score graph, whose connected components are used as the EgoNet-APM community output for modularity, coverage, density, and conductance reporting.

## Baseline

Louvain is run on the same train graph and evaluated with the same community-quality metrics. For link-prediction comparison, a held-out edge is scored by whether its endpoints are in the same Louvain community plus normalized common-neighbor support.

## Experiments

Run:

```bash
python scripts/generate_datasets.py
python run_experiments.py
python plot_results.py
```

Results are stored in:

- `results/experiment_results.csv`
- `results/quality_runtime.png`

The code uses three datasets that are explicitly listed in the reproduced paper:

- Facebook social circles: 4,039 nodes and 88,234 edges.
- Enron email network: 36,692 nodes and 183,831 edges.
- Astro Physics collaboration network: 18,771 nodes and 198,050 edges after self-loop removal.

The results are:

| Dataset | Algorithm | Runtime (s) | Modularity | Coverage | Mean Density | Mean Conductance | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Facebook | EgoNet-APM | 6.7847 | 0.0009 | 0.9970 | 0.1130 | 0.7682 | 0.9807 | 0.9799 |
| Facebook | Louvain | 0.3337 | 0.8346 | 0.9611 | 0.2574 | 0.0545 | 0.9938 | 0.9925 |
| Enron | EgoNet-APM | 8.8199 | 0.0556 | 0.8204 | 0.1688 | 0.8571 | 0.9387 | 0.9386 |
| Enron | Louvain | 1.7220 | 0.6104 | 0.7407 | 0.8817 | 0.0189 | 0.9693 | 0.9596 |
| AstroPh | EgoNet-APM | 7.6114 | 0.0390 | 0.8892 | 0.1529 | 0.8938 | 0.9725 | 0.9725 |
| AstroPh | Louvain | 1.5783 | 0.6267 | 0.6979 | 0.8603 | 0.0244 | 0.9828 | 0.9803 |

## Comparison With The Paper

The original paper evaluates ego-net mining on Amazon, Astro Physics, Enron, Slashdot, Patents, Facebook, LiveJournal, Twitter, and a proprietary Google+ graph. It reports detailed figures mainly for Patents, Facebook, LiveJournal, Twitter, and Google+. In this reproduction, we use three of the paper's public datasets: Facebook, Enron, and Astro Physics. This satisfies the project requirement of running 3-4 datasets from the chosen paper while avoiding the very large Patents, LiveJournal, Twitter, and Google+ graphs.

The following table compares the paper's reported numbers with our reproduced results. Some entries are not directly comparable because the paper reports circle reconstruction precision/recall and ranked friend-suggestion curves, while this project reports modularity, conductance, density, AUC, and AP on train/test edge splits.

| Aspect | Paper result | Our result | Interpretation |
|---|---:|---:|---|
| Dataset scale | Facebook: 4,039 nodes / 88,234 edges; Twitter: 81,306 / 1,768,149; LiveJournal: 4,847,571 / 68,993,773; Patents: 3,774,768 / 16,518,948 | Facebook: 4,039 / 88,234; Enron: 36,692 / 183,831; AstroPh: 18,771 / 198,050 | Facebook exactly matches the paper's reported Facebook scale. Enron and AstroPh are also paper-listed datasets, selected to keep experiments feasible. |
| Ego-net cluster density | Most retrieved ego-net clusters have density larger than 0.2; Google+ circles average density is 0.352 | EgoNet-APM mean density is 0.1449; ego-cluster mean densities are 0.7251, 0.8310, and 0.8991 for Facebook, Enron, and AstroPh | Our final aggregated communities are less dense, but the local ego-net clusters themselves are very dense, consistent with the paper's microscopic-community claim. |
| Ego-net conductance | Paper observes typically low conductance in retrieved ego-net clusters; Google+ circles average conductance is 0.1438 | Aggregated EgoNet-APM mean conductance is 0.8397 | This is the main numerical mismatch. Our conductance is measured after aggregating pair-score components into global communities, while the paper primarily analyzes local ego-net clusters and Google+ circles. |
| Circle reconstruction | LP: node precision 0.86, edge precision 0.87, edge recall 0.96; SLPA: node precision 0.85, edge recall 0.98 | No circle labels are used in our experiment | Not directly comparable. We did not use Facebook circle ground truth; we evaluate unsupervised community quality and held-out edge prediction instead. |
| Friend suggestion | Ego-net friendship scores outperform common friends and Adamic-Adar in ranked precision/recall curves; live Google+ acceptance rate increases by more than 0.5% | EgoNet-APM AUC/AP: Facebook 0.9807/0.9799, Enron 0.9387/0.9386, AstroPh 0.9725/0.9725 | Strong qualitative agreement: ego-net co-clustering gives a useful local link-prediction signal. |
| Runtime/scalability | Fast sequential ego-net construction is 5x faster than naive on LiveJournal; distributed version is 11x faster | EgoNet-APM runtime is 6.8-8.8 seconds; Louvain is 0.3-1.7 seconds | Our Python version now uses triangle-based single-machine ego-net construction, but still does not implement the paper's MapReduce scalability optimization. |

The strongest agreement with the paper is in the link-prediction behavior. The paper argues that ego-net friendship scores are useful local features for friend suggestion. Our AUC/AP results are high on all three paper datasets: Facebook 0.9807/0.9799, Enron 0.9387/0.9386, and AstroPh 0.9725/0.9725. This supports the same qualitative conclusion: co-occurrence inside ego-net communities gives a strong local similarity signal.

The main difference is in global community quality. Louvain obtains much higher modularity and lower conductance because it directly optimizes a global partition. EgoNet-APM in this implementation first creates local ego-net clusters and then aggregates pair scores into connected components; this is more conservative and not designed to maximize global modularity. Therefore, lower modularity does not contradict the paper's main idea, which emphasizes microscopic ego-net structure and friend suggestion rather than global modularity optimization.

In runtime, EgoNet-APM is slower than Louvain in this single-machine Python implementation: 6.8-8.8 seconds versus 0.3-1.7 seconds. The implementation now uses triangle enumeration to construct all ego-net edge sets, which is closer to the paper's fast construction idea than repeated per-node subgraph extraction. It still does not implement distributed MapReduce execution.

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
