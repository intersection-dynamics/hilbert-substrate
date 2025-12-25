import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# ============================================================
# CONFIG
# ============================================================

N_NODES = 16
EDGE_PROB = 0.25
TIMES = np.linspace(0.5, 6.0, 25)
FIG_DIR = "figures"
TOP_K = 5
RNG_SEED = 1

np.random.seed(RNG_SEED)

# ============================================================
# SUBSTRATE + MOCK PERMEABILITY
# (Replace internals later without touching plotting logic)
# ============================================================

def generate_interaction_graph(n, p):
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    return G

def directed_permeability(i, j, t):
    base = 0.5 + 0.5 * np.sin(0.7 * t + 0.3 * i - 0.2 * j)
    noise = 0.05 * np.random.randn()
    return np.clip(abs(base + noise), 0.0, 1.0)

# ============================================================
# COMPUTATION
# ============================================================

def compute_edge_permeabilities(G, t):
    P = {}
    for i, j in G.edges():
        pij = directed_permeability(i, j, t)
        pji = directed_permeability(j, i, t)
        P[(i, j)] = max(pij, pji)
    return P

def compute_node_intensity(G, t):
    I = np.zeros(G.number_of_nodes())
    for i in G.nodes():
        for j in G.neighbors(i):
            I[i] += directed_permeability(i, j, t)
    return I

# ============================================================
# FIGURE 1 — EDGE-LEVEL HIP SNAPSHOT
# ============================================================

def make_figure_1(G, t_star):
    P_edge = compute_edge_permeabilities(G, t_star)
    pos = nx.spring_layout(G, seed=RNG_SEED)

    edges = list(G.edges())
    edge_colors = [P_edge[e] for e in edges]

    fig, ax = plt.subplots(figsize=(6, 6))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=200,
        node_color="lightgray"
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edges,
        edge_color=edge_colors,
        edge_cmap=plt.cm.viridis,
        width=2
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Edge permeability $P_{\{i,j\}}(t^*)$")

    ax.set_title("Figure 1: Edge-level HIP snapshot")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure1_edge_hip.png", dpi=300)
    fig.savefig(f"{FIG_DIR}/figure1_edge_hip.pdf")
    plt.close(fig)

# ============================================================
# FIGURE 2 — PERSISTENCE OF HIP
# ============================================================

def make_figure_2(G):
    intensities = np.array([compute_node_intensity(G, t) for t in TIMES])
    mean_intensity = intensities.mean(axis=0)
    top_nodes = np.argsort(mean_intensity)[-TOP_K:]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i in top_nodes:
        ax.plot(TIMES, intensities[:, i], label=f"Node {i}")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Outgoing intensity $I_i(t)$")
    ax.set_title("Figure 2: Persistence of HIP (top-k nodes)")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure2_persistence_timeseries.png", dpi=300)
    fig.savefig(f"{FIG_DIR}/figure2_persistence_timeseries.pdf")
    plt.close(fig)

    # ---- Rank stability diagnostic ----

    ranks = np.argsort(np.argsort(-intensities, axis=1), axis=1)
    stability = []

    for t in range(len(TIMES) - 1):
        r, _ = spearmanr(ranks[t], ranks[t + 1])
        stability.append(r)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(TIMES[:-1], stability, marker="o")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Rank stability of node intensities")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure2_rank_stability.png", dpi=300)
    fig.savefig(f"{FIG_DIR}/figure2_rank_stability.pdf")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    G = generate_interaction_graph(N_NODES, EDGE_PROB)
    t_star = TIMES[len(TIMES) // 2]

    make_figure_1(G, t_star)
    make_figure_2(G)

    print("Figures generated:")
    print(" - figures/figure1_edge_hip.(png|pdf)")
    print(" - figures/figure2_persistence_timeseries.(png|pdf)")
    print(" - figures/figure2_rank_stability.(png|pdf)")

if __name__ == "__main__":
    main()
