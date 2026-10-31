import itertools
import numpy as np
import networkx as nx


def state_tree(edges, root):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if root not in G:
        G.add_node(root)
    return G


def is_arborescence(G, root):
    return nx.is_arborescence(G) and G.in_degree(root) == 0


def rooted_from_undirected(edges, root):
    U = nx.Graph()
    U.add_edges_from(edges)
    return nx.bfs_tree(U, root)


def clusters(G, root):
    return {frozenset({v} | nx.descendants(G, v)) for v in G.nodes if v != root}


def splits(G):
    U = G.to_undirected()
    nodes = frozenset(U.nodes)
    out = set()
    for u, v in U.edges:
        H = U.copy()
        H.remove_edge(u, v)
        side = frozenset(nx.node_connected_component(H, u))
        out.add(min(side, nodes - side, key=lambda s: (len(s), sorted(s))))
    return out


def rf_rooted(G1, G2, root):
    c1, c2 = clusters(G1, root), clusters(G2, root)
    return len(c1 ^ c2) / (len(c1) + len(c2))


def rf_unrooted(G1, G2):
    s1, s2 = splits(G1), splits(G2)
    return len(s1 ^ s2) / (len(s1) + len(s2))


def edge_diff(G1, G2):
    e1 = {frozenset(e) for e in G1.to_undirected().edges}
    e2 = {frozenset(e) for e in G2.to_undirected().edges}
    fmt = lambda es: sorted("-".join(sorted(e)) for e in es)
    return fmt(e1 & e2), fmt(e1 - e2), fmt(e2 - e1)


def case_name(G, root="NPC"):
    kids = sorted(G.successors(root))
    if len(G) != 4:
        return newick_states(G, root)
    if len(kids) == 3:
        return "Star"
    if len(kids) == 1:
        x = kids[0]
        gk = sorted(G.successors(x))
        if len(gk) == 2:
            return f"Relay-{x}"
        y = gk[0]
        return f"Chain-{x}-{y}-{next(iter(G.successors(y)))}"
    x = [k for k in kids if G.out_degree(k) == 1][0]
    return f"Fork-{x}-{next(iter(G.successors(x)))}"


def newick_states(G, node):
    kids = sorted(G.successors(node))
    if not kids:
        return node
    return f"{node}>(" + ", ".join(newick_states(G, k) for k in kids) + ")"


def all_rooted_trees(states, root):
    others = [s for s in states if s != root]
    out = []
    for parents in itertools.product(states, repeat=len(others)):
        edges = [(p, c) for p, c in zip(parents, others)]
        if any(p == c for p, c in edges):
            continue
        G = state_tree(edges, root)
        if is_arborescence(G, root) and len(G) == len(states):
            out.append(G)
    return out


def neighbor_joining(D, labels):
    D = np.array(D, dtype=float)
    nodes = list(labels)
    edges = []
    nxt = 0
    while len(nodes) > 2:
        n = len(nodes)
        r = D.sum(1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        li = 0.5 * D[i, j] + (r[i] - r[j]) / (2 * (n - 2))
        lj = D[i, j] - li
        u = f"_n{nxt}"
        nxt += 1
        edges += [(u, nodes[i], max(li, 0.0)), (u, nodes[j], max(lj, 0.0))]
        du = 0.5 * (D[i] + D[j] - D[i, j])
        keep = [k for k in range(n) if k not in (i, j)]
        D = np.vstack([np.hstack([D[np.ix_(keep, keep)], du[keep][:, None]]),
                       np.hstack([du[keep], [0.0]])])
        nodes = [nodes[k] for k in keep] + [u]
    edges.append((nodes[0], nodes[1], max(D[0, 1], 0.0)))
    T = nx.Graph()
    for a, b, w in edges:
        T.add_edge(a, b, length=w)
    return T


def nj_leaf_splits(T, labels):
    leaves = frozenset(labels)
    out = set()
    for u, v in T.edges:
        H = T.copy()
        H.remove_edge(u, v)
        side = frozenset(x for x in nx.node_connected_component(H, u) if x in leaves)
        if 1 < len(side) < len(leaves) - 1:
            out.add(min(side, leaves - side, key=lambda s: (len(s), sorted(s))))
    return out


def patristic(T, labels):
    d = dict(nx.all_pairs_dijkstra_path_length(T, weight="length"))
    return np.array([[d[a][b] for b in labels] for a in labels])


def distance_state_graph(D, labels, root):
    U = nx.Graph()
    for i, j in itertools.combinations(range(len(labels)), 2):
        U.add_edge(labels[i], labels[j], weight=D[i, j])
    return rooted_from_undirected(nx.minimum_spanning_tree(U).edges, root)


def nj_state_graph(T, labels, root):
    return distance_state_graph(patristic(T, labels), labels, root)


def euclid(X):
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def zscore_cols(X):
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


def nj_bootstrap(X, labels, root, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    T0 = neighbor_joining(euclid(zscore_cols(X)), labels)
    ref_splits = nj_leaf_splits(T0, labels)
    ref_edges = {frozenset(e) for e in nj_state_graph(T0, labels, root).to_undirected().edges}
    split_hits = {s: 0 for s in ref_splits}
    edge_hits = {e: 0 for e in ref_edges}
    for _ in range(B):
        Xb = X[:, rng.integers(0, X.shape[1], X.shape[1])]
        Tb = neighbor_joining(euclid(zscore_cols(Xb)), labels)
        sb = nj_leaf_splits(Tb, labels)
        eb = {frozenset(e) for e in nj_state_graph(Tb, labels, root).to_undirected().edges}
        for s in split_hits:
            split_hits[s] += s in sb
        for e in edge_hits:
            edge_hits[e] += e in eb
    return T0, {s: h / B for s, h in split_hits.items()}, {e: h / B for e, h in edge_hits.items()}


def greedy_arborescence(T, labels, root):
    idx = {s: i for i, s in enumerate(labels)}
    intree, edges = {root}, []
    while len(intree) < len(labels):
        s, d = max(((a, b) for a in labels if a in intree for b in labels if b not in intree),
                   key=lambda e: T[idx[e[0]], idx[e[1]]])
        edges.append((s, d))
        intree.add(d)
    return state_tree(edges, root)


def optimal_arborescence(T, labels, root, eps=1e-9):
    idx = {s: i for i, s in enumerate(labels)}
    G = nx.DiGraph()
    for a in labels:
        for b in labels:
            if a != b and b != root:
                G.add_edge(a, b, weight=float(np.log(T[idx[a], idx[b]] + eps)))
    A = nx.maximum_spanning_arborescence(G, attr="weight")
    return state_tree(list(A.edges), root)


def rank_arborescences(T, labels, root):
    idx = {s: i for i, s in enumerate(labels)}
    rows = []
    for G in all_rooted_trees(labels, root):
        w = [T[idx[a], idx[b]] for a, b in G.edges]
        rows.append((case_name(G, root), float(np.prod(w)), G))
    rows.sort(key=lambda r: -r[1])
    return rows


def that_mst(T, labels, root):
    Dm = 1 - (T + T.T) / 2
    np.fill_diagonal(Dm, 0)
    return distance_state_graph(Dm, labels, root), Dm


def compare(G1, G2, root):
    shared, only1, only2 = edge_diff(G1, G2)
    return {"rf_rooted": round(rf_rooted(G1, G2, root), 3),
            "rf_unrooted": round(rf_unrooted(G1, G2), 3),
            "shared_edges": "; ".join(shared),
            "only_first": "; ".join(only1),
            "only_second": "; ".join(only2)}


def self_test():
    S = ["NPC", "AC", "OPC", "MES"]
    trees = all_rooted_trees(S, "NPC")
    names = sorted(case_name(G) for G in trees)
    assert len(trees) == 16 and len(set(names)) == 16, names
    for G in trees:
        assert rf_rooted(G, G, "NPC") == 0 and rf_unrooted(G, G) == 0
    relay = state_tree([("NPC", "OPC"), ("OPC", "AC"), ("OPC", "MES")], "NPC")
    star = state_tree([("NPC", "OPC"), ("NPC", "AC"), ("NPC", "MES")], "NPC")
    chain = state_tree([("NPC", "MES"), ("MES", "OPC"), ("OPC", "AC")], "NPC")
    assert case_name(relay) == "Relay-OPC" and case_name(star) == "Star" and case_name(chain) == "Chain-MES-OPC-AC"
    assert abs(rf_rooted(relay, star, "NPC") - 1 / 3) < 1e-9
    assert abs(rf_unrooted(relay, star) - 1 / 3) < 1e-9
    for G in trees:
        H = G.to_undirected()
        L = dict(nx.all_pairs_shortest_path_length(H))
        D = np.array([[L[a][b] for b in S] for a in S], dtype=float)
        assert rf_rooted(distance_state_graph(D, S, "NPC"), G, "NPC") == 0
    for K in (5, 6):
        states = [f"S{i}" for i in range(K)]
        assert len(all_rooted_trees(states, "S0")) == K ** (K - 2)
    return names


if __name__ == "__main__":
    names = self_test()
    print("All self-tests passed.")
    print("The 16 four-state trees:", ", ".join(names))