import torch
import numpy as np
from itertools import combinations
from tqdm import trange

def _neighbors(adj, i):
    """Return indices of neighbors of node i in an undirected graph."""
    return np.where(adj[i] != 0)[0]


def find_triangles(adj: np.ndarray) -> np.ndarray:
    """
    Find all triangles {i, j, k} with i < j < k.
    Returns: array of shape (M, 3)
    """
    N = adj.shape[0]
    triangles = []

    for i in range(N):
        # neighbors of i with index > i to avoid duplicates
        nbrs_i = np.where(adj[i] != 0)[0]
        nbrs_i = nbrs_i[nbrs_i > i]
        for idx, j in enumerate(nbrs_i):
            # common neighbors between i and j with index > j
            # we only look among nbrs_i[idx+1:] to ensure i < j < k
            cand_k = nbrs_i[idx+1:]
            if cand_k.size == 0:
                continue
            # keep those k where edge j-k exists
            mask = adj[j, cand_k] != 0
            ks = cand_k[mask]
            for k in ks:
                triangles.append((i, j, k))

    return np.array(triangles, dtype=int) if triangles else np.zeros((0, 3), dtype=int)


def find_3_stars(adj: np.ndarray) -> np.ndarray:
    """
    Find all 3-stars: center + 3 distinct leaves.
    Returns rows [center, leaf1, leaf2, leaf3] (order of leaves arbitrary).
    Shape: (M, 4)  (N' = 4 nodes in the star)
    """
    N = adj.shape[0]
    stars = []

    for c in range(N):
        nbrs = _neighbors(adj, c)
        if len(nbrs) < 3:
            continue
        for leaves in combinations(nbrs, 3):
            stars.append((c, *leaves))

    return np.array(stars, dtype=int) if stars else np.zeros((0, 4), dtype=int)


def find_4_stars(adj: np.ndarray) -> np.ndarray:
    """
    Find all 4-stars: center + 4 distinct leaves.
    Returns rows [center, leaf1, leaf2, leaf3, leaf4].
    Shape: (M, 5)
    """
    N = adj.shape[0]
    stars = []

    for c in range(N):
        nbrs = _neighbors(adj, c)
        if len(nbrs) < 4:
            continue
        for leaves in combinations(nbrs, 4):
            stars.append((c, *leaves))

    return np.array(stars, dtype=int) if stars else np.zeros((0, 5), dtype=int)


def find_3_paths(adj: np.ndarray, induced: bool = True) -> np.ndarray:
    """
    Find all simple 3-node paths i - j - k.
    j is the middle node. i != k.
    If induced=True, require no edge between i and k (i.e., exclude triangles).
    Returns rows [i, j, k] with i < k for canonical orientation.
    Shape: (M, 3)
    """
    N = adj.shape[0]
    paths = []

    for j in range(N):
        nbrs = _neighbors(adj, j)
        if len(nbrs) < 2:
            continue
        for i, k in combinations(nbrs, 2):
            if induced and adj[i, k] != 0:
                continue  # this would be part of a triangle, not a pure path
            # canonical orientation: smaller endpoint first
            if i < k:
                paths.append((i, j, k))
            else:
                paths.append((k, j, i))

    # deduplicate
    if not paths:
        return np.zeros((0, 3), dtype=int)
    paths = np.unique(np.array(paths, dtype=int), axis=0)
    return paths


def find_4_paths(adj: np.ndarray, induced: bool = True) -> np.ndarray:
    """
    Find all simple 4-node paths v0 - v1 - v2 - v3.
    We enforce all nodes distinct.
    If induced=True, we require no extra edges between non-consecutive nodes:
        no edges (v0,v2), (v0,v3), (v1,v3).
    Returns rows [v0, v1, v2, v3] in a canonical orientation
    (lexicographically smallest between path and its reverse).
    Shape: (M, 4)
    """
    N = adj.shape[0]
    path_set = set()

    for v1 in range(N):
        nbrs1 = _neighbors(adj, v1)
        for v2 in nbrs1:
            if v2 == v1:
                continue
            nbrs0 = _neighbors(adj, v1)
            nbrs0 = nbrs0[nbrs0 != v2]
            nbrs3 = _neighbors(adj, v2)
            nbrs3 = nbrs3[nbrs3 != v1]

            for v0 in nbrs0:
                if v0 == v2:
                    continue
                for v3 in nbrs3:
                    if v3 in (v0, v1, v2):
                        continue
                    # induced path constraints
                    if induced:
                        if adj[v0, v2] != 0:
                            continue
                        if adj[v0, v3] != 0:
                            continue
                        if adj[v1, v3] != 0:
                            continue

                    path = (v0, v1, v2, v3)
                    rev = (v3, v2, v1, v0)
                    canon = min(path, rev)
                    path_set.add(canon)

    if not path_set:
        return np.zeros((0, 4), dtype=int)
    paths = np.array(sorted(path_set), dtype=int)
    return paths


def find_4_cycles(adj: np.ndarray, induced: bool = True) -> np.ndarray:
    """
    Find all simple 4-cycles (i - j - k - l - i) up to rotation / reflection.
    Implementation:
        - For each unordered pair (i,k) with i < k and NOT adjacent:
          find common neighbors S = N(i) ∩ N(k).
          Each unordered pair {j,l} ⊂ S gives cycle i-j-k-l-i.
        - If induced=True, require no chords: no edges (i,k) and (j,l).
    Returns rows [i, j, k, l] in a canonical orientation:
        smallest node first, second node = smallest neighbor of i in the cycle.
    Shape: (M, 4)
    """
    N = adj.shape[0]
    cycle_set = set()

    for i in range(N):
        nbrs_i = _neighbors(adj, i)
        for k in range(i + 1, N):
            if adj[i, k] != 0:
                # for an induced 4-cycle, i and k should be non-adjacent (opposite nodes)
                if induced:
                    continue
            # common neighbors of i and k
            nbrs_k = _neighbors(adj, k)
            common = np.intersect1d(nbrs_i, nbrs_k, assume_unique=False)
            if len(common) < 2:
                continue
            # choose unordered pairs {j, l} in common
            for j, l in combinations(common, 2):
                if j == l:
                    continue
                # Induced: require j and l not adjacent
                if induced and adj[j, l] != 0:
                    continue
                # Now we have a 4-cycle i - j - k - l - i, up to rotation / reflection.
                cycle = (i, j, k, l)
                # Canonicalize cycle: rotate so smallest node is first,
                # then choose orientation (clockwise/counter) that is lexicographically smallest.
                cyc_variants = [
                    cycle,
                    (j, k, l, i),
                    (k, l, i, j),
                    (l, i, j, k),
                ]
                # reverse (reflections)
                cyc_variants += [tuple(reversed(c)) for c in cyc_variants]
                canon = min(cyc_variants)
                cycle_set.add(canon)

    if not cycle_set:
        return np.zeros((0, 4), dtype=int)
    cycles = np.array(sorted(cycle_set), dtype=int)
    return cycles


def build_motif_hypergraphs(adj: np.ndarray, motif_types: list):
    """
    Convenience wrapper that returns a dict of motif tables:
    {
        'triangles': (M1, 3),
        '3-stars':   (M2, 4),
        '4-stars':   (M3, 5),
        '3-paths':   (M4, 3),
        '4-paths':   (M5, 4),
        '4-cycles':  (M6, 4),
    }
    """
    motif_dict =  {
        'triangles': find_triangles(adj),
        '3-stars':   find_3_stars(adj),
        '4-stars':   find_4_stars(adj),
        '3-paths':   find_3_paths(adj, induced=True),
        '4-paths':   find_4_paths(adj, induced=True),
        '4-cycles':  find_4_cycles(adj, induced=True),
    }
    vals = [motif_dict[mt] for mt in motif_types]
    output = dict(zip(motif_types, vals))
    return output


def build_motif_features_from_dict(motif_dict: dict, num_nodes: int):
    """
    motif_list[k]: LongTensor of shape (M_k, N_k),
        each row is a motif instance, elements are node indices in [0, num_nodes-1].

    motif_types[k]: List of string (N_k), indicating the type of motif for different representation methods

    Returns
    -------
    feat : (num_nodes, D_motif) per-node motif feature matrix
    """

    feats = []
    for mtype in motif_dict:
        motifs_k = motif_dict[mtype]
        # motifs_k: (M_k, N_k)
        M_k, N_k = motifs_k.shape

        # Orbit encoding
        per_pos_counts = []
        for p in range(N_k):
            nodes_p = motifs_k[:, p]  # node at position p in each instance
            c_p = np.bincount(nodes_p,
                                 minlength=num_nodes)  # (N,)
            per_pos_counts.append(c_p)

        # stack per-position features: (N, N_k)
        per_pos_counts = np.stack(per_pos_counts, axis=-1)  # (N, N_k)


        if mtype == '3-stars' or mtype == '4-stars':
            center_counts = per_pos_counts[:,0]
            leaf_counts = np.sum(per_pos_counts[:,0:], axis=1)
            feats.append(center_counts)
            feats.append(leaf_counts)
        
        elif mtype == '4-cycles':
            leaf_counts = np.sum(per_pos_counts, axis=1)
            feats.append(leaf_counts)

    # concat all motif types along feature dim: (N, D_motif)W
    feat = np.stack(feats, axis=1)  # (num_nodes, sum_k (N_k))

    return feat


def build_motif_features(adj: np.ndarray, motif_types: list, device='None'):
    N = adj.shape[0]
    motif_dict = build_motif_hypergraphs(adj, motif_types)
    _feat = build_motif_features_from_dict(motif_dict, N)
    road_source = np.sum(adj, axis=0)
    road_target = np.sum(adj, axis=1)
    motif_2l = np.stack([road_source, road_target], axis=1)
    feat = np.concatenate([motif_2l, _feat], axis=1)
    if device != 'None':
        feat = torch.tensor(feat, device=device, dtype=torch.float32)
    return feat
