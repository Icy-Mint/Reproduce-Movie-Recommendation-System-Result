#!/usr/bin/env python
# plots_graph.py – build the similarity graph for a given α and
# dump (i) a PNG visualisation (spring layout) and (ii) a CSV with
# centrality statistics that the paper tables read.

import argparse, itertools, collections, networkx as nx, pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--out",   default="Figs/graph_alpha.png")
args = parser.parse_args()

ratings = pd.read_csv("datasets/ml-100k/ua.base", sep=r"\t",
                      names=["uid","iid","r","ts"])
pair_cnt = collections.Counter()
for (_, r), g in ratings.groupby(["iid","r"]):
    pair_cnt.update(itertools.combinations(g.uid, 2))

thr = args.alpha * ratings.iid.nunique()
E   = [p for p,c in pair_cnt.items() if c >= thr]
G   = nx.Graph(); G.add_edges_from(E)

# -------- centrality CSV --------------------------------------------
cent = pd.DataFrame({
    "uid": list(G.nodes),
    "pagerank":        nx.pagerank(G.to_directed()).values(),
    "deg_cent":        nx.degree_centrality(G).values(),
    "close_cent":      nx.closeness_centrality(G).values(),
    "bet_cent":        nx.betweenness_centrality(G).values(),
    "load_cent":       nx.load_centrality(G).values(),
    "avg_neigh_deg":   nx.average_neighbor_degree(G).values()
})
cent.to_csv("Figs/graph_centrality.csv", index=False)

# -------- picture ----------------------------------------------------
plt.figure(figsize=(6,6))
nx.draw_spring(G, node_size=8, linewidths=0, edge_color="#CCCCCC")
plt.axis("off"); plt.tight_layout()
plt.savefig(args.out, dpi=300)
print("✓ graph + stats saved in Figs/")
