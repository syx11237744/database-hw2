结论：当前项目更像是“论文核心思想的单机简化实验”，不是 Epasto et al. 的完整技术复现。论文重点是对所有 ego-net 的高效/并行构造、在每个 `Z_u` 上独立聚类，并把 ego-net co-clustering 作为好友推荐特征；动态图也只是论文提出的未来方向。论文定义 `Z_u` 为去掉 ego 后由邻居诱导出的图，并强调大规模 ego-net 构造与 MapReduce 并行框架。([vldb.org](https://www.vldb.org/pvldb/vol9/p324-epasto.pdf)) 

**需要改进清单**

1. **补齐 scalable ego-net construction**
   当前 [algorithms.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/algorithms.py:80) 是对每个节点顺序 `subgraph(neighbors).copy()`，属于单机内存版；[dynamic.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/dynamic.py:45) 初始化也逐 ego 顺序重算。建议实现论文 Algorithm 2/3 风格的三角形枚举/分区构造，至少先做单机高效版，再考虑 Spark/MapReduce 风格并行版本。

2. **修正 `ego_score_partition` 的语义**
   当前 [algorithms.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/algorithms.py:104) 把 pair score graph 的 connected components 当社区。论文里的 CC 是 ego-net 内部聚类 baseline，不是把全局 pair-score 图再取连通分量；论文的 W1-W4 是好友推荐特征。([vldb.org](https://www.vldb.org/pvldb/vol9/p324-epasto.pdf))  
   建议把输出拆清楚：`ego_clusters_by_ego`、`friendship_scores`、可选 global merge。若要全局社区，应实现论文相关工作里的 overlap/Jaccard merge，或明确标注当前做法为额外 heuristic。

3. **不要用“非边 score graph 社区”做论文社区质量对比**
   [algorithms.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/algorithms.py:98) 只给原图中不存在的 pair 加分，这对 link prediction 合理，但拿这些非边构成的 connected components 去算 modularity/conductance 含义错位。建议社区质量直接在每个 `Z_u` 的局部簇上评估 density/conductance 分布，并把 pair score 只用于推荐任务。

4. **保留多 seed 实验**
   当前 [run_experiments.py](/Users/alan671/Documents/database-hw2/run_experiments.py:151) 和 [run_dynamic_experiments.py](/Users/alan671/Documents/database-hw2/run_dynamic_experiments.py:197) 都是单 seed。建议支持 `--seeds 1 2 3 4 5`，输出 per-seed 明细和 mean/std；train/test split、negative sampling、APM 随机 tie-break、Louvain 都要按 seed 记录。

5. **动态图拓展需要重新定位**
   论文只把 dynamic graph streams 作为未来工作，并建议结合 streaming triangle counting 和 dynamic community detection。 当前 [dynamic.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/dynamic.py:51) 是“找受影响 ego-net 后整 ego 重算”，不是流式三角形维护，也没有 warm-start 社区检测。建议维护 triangle/ego-edge 索引、增量更新局部 score delta、复用上一轮 label，并避免每次 `partition()` 全量 connected components。

6. **补齐论文中的算法对照**
   当前只实现 APM-LP，外加 Louvain baseline。论文比较 PPR、Over-PPR、LP/APM、SLPA、K-core、CC 等 ego-net 内聚类算法。建议至少补 CC、SLPA、K-core，再加 common neighbors、Adamic-Adar 作为推荐 baseline。

7. **调整评价指标与论文口径**
   当前 [metrics.py](/Users/alan671/Documents/database-hw2/src/egonet_mining/metrics.py:31) 用平均 conductance；论文对 cluster/clustering 的 conductance、density 关注局部 ego-net 簇质量和分布。建议报告 ego-net 簇级别的 density/conductance 分位数、top-k precision/recall 曲线、circle reconstruction precision/recall，而不是只报全局 modularity。

8. **增加验证和可复现实验元数据**
   仓库没有测试文件。建议加 toy graph 单测：`Z_u` 构造、APM 收敛、W 分数、动态更新等价于静态重算、seed 确定性。CSV 也应记录 alpha/beta/max_iter/min_score/seed/dataset hash，避免结果覆盖后不可追踪。

- 用数组/CSR 邻接表代替 NetworkX 小图；
- 不为每个 ego 创建 nx.Graph()；
- APM 聚类用整数 node id 和数组计数；
- 多进程并行每个 ego 的聚类；
- 对高 degree ego 做采样、截断或阈值过滤；
- pair score 聚合用稀疏矩阵或排序归并；
- 只计算 link-prediction candidate pairs，而不是所有 cluster 内非边。