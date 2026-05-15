import torch
import numpy as np
from scipy.spatial import KDTree


# ── 顶点焊接辅助 ─────────────────────────────────────────────────────────────

def _weld_vertices(V: np.ndarray, eps_ratio: float = 1e-4) -> np.ndarray:
    """
    Union-Find 顶点焊接：将空间距离 < eps 的顶点合并为同一 canonical ID。
    eps = eps_ratio × bbox 对角线长度（对不同场景尺度自适应）。

    返回 [V] int64 数组，每个原始顶点对应其 canonical ID（连续整数 0..N_unique-1）。
    """
    n = len(V)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    eps = eps_ratio * diag if diag > 1e-8 else 1e-6

    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for i, j in KDTree(V).query_pairs(eps):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    for i in range(n):          # final path compression
        parent[i] = find(i)

    _, canonical = np.unique(parent, return_inverse=True)
    return canonical.astype(np.int64)


# ── 边候选图构建（含拓扑 mask）────────────────────────────────────────────────

def build_edge_candidates(
    vertices: torch.Tensor,
    triangle_indices: torch.Tensor,
    k: int = 5,
    eps_ratio: float = 1e-4,
) -> tuple:
    """
    为每条三角形边（共 3P 条）找 k 个最近邻边（按边中点 3D 距离）。
    同时做顶点焊接，检测空洞边缘三角形和悬空顶点三角形。

    Returns:
        candidates:        [3P, k] LongTensor (CPU)
        boundary_mask:     [P]  BoolTensor (CPU) — 三角形含空洞边缘
        low_valence_mask:  [P]  BoolTensor (CPU) — 三角形含悬空顶点(valence<3)
    """
    V = vertices.detach().cpu().numpy()
    T = triangle_indices.long().cpu().numpy()
    P = T.shape[0]

    _empty_bool = torch.zeros(P, dtype=torch.bool)
    if P == 0 or not np.isfinite(V).all():
        return torch.zeros((3 * P, k), dtype=torch.long), _empty_bool, _empty_bool

    # ── 1. KDTree 边候选搜索（原有逻辑，不变）────────────────────────
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    edge_a = np.concatenate([v0, v1, v2], axis=0)   # [3P, 3]
    edge_b = np.concatenate([v1, v2, v0], axis=0)   # [3P, 3]
    midpoints = (edge_a + edge_b) / 2               # [3P, 3]

    tri_owner = np.tile(np.arange(P), 3)            # [3P]

    query_k = min(k * 4 + 3, 3 * P)
    tree = KDTree(midpoints)
    _, nn_idx = tree.query(midpoints, k=query_k)

    candidates = np.full((3 * P, k), -1, dtype=np.int64)
    for i in range(3 * P):
        valid = [
            j for j in nn_idx[i]
            if j != i and tri_owner[j] != tri_owner[i]
        ]
        n = min(k, len(valid))
        candidates[i, :n] = valid[:n]
        if 0 < len(valid) < k:
            candidates[i, len(valid):] = valid[-1]
        elif len(valid) == 0:
            candidates[i, :] = i

    # ── 2. 顶点焊接 → canonical 索引 ─────────────────────────────────
    canonical = _weld_vertices(V, eps_ratio)        # [V] original → canonical
    T_weld    = canonical[T]                        # [P, 3] canonical indices
    n_can     = int(canonical.max()) + 1

    # ── 3. 边计数（空洞检测）─────────────────────────────────────────
    # 构建无序边：排序使 (a,b) == (b,a)，编码为标量 key
    edges = np.concatenate([
        T_weld[:, [0, 1]],
        T_weld[:, [1, 2]],
        T_weld[:, [2, 0]],
    ], axis=0)                                      # [3P, 2]
    edges_sorted = np.sort(edges, axis=1)
    edge_keys = edges_sorted[:, 0] * (n_can + 1) + edges_sorted[:, 1]
    _, inv, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    is_boundary = (counts[inv] == 1)               # [3P] — 只被 1 个三角形用的边

    # 每条边对应哪个三角形（按 [e0..eP, e1..e2P, e2..e3P] 顺序）
    boundary_per_tri = (
        is_boundary[0:P].astype(np.int32) +
        is_boundary[P:2*P].astype(np.int32) +
        is_boundary[2*P:3*P].astype(np.int32)
    )
    boundary_mask = boundary_per_tri > 0           # [P]

    # ── 4. 顶点 valence（悬空顶点检测）───────────────────────────────
    valence = np.bincount(T_weld.flatten(), minlength=n_can)  # [n_can]
    low_valence_mask = (valence < 3)[T_weld].any(axis=1)      # [P]

    return (
        torch.tensor(candidates,       dtype=torch.long),
        torch.tensor(boundary_mask,    dtype=torch.bool),
        torch.tensor(low_valence_mask, dtype=torch.bool),
    )


# ── Connectivity loss ────────────────────────────────────────────────────────

def connectivity_loss(
    vertices: torch.Tensor,
    triangle_indices: torch.Tensor,
    edge_candidates: torch.Tensor,
    importance_score: torch.Tensor,
    vertex_weight: torch.Tensor,
    tau: float = 0.1,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """
    边级软连通 loss（分块计算，避免 [3P,k,3] 张量 OOM）：

        L_conn = mean_e [ omega_e * sum_f P_ef * D_edge(e, f) ]

    D_edge(e=(a,b), f=(c,d)) = min( ||a-c||^2+||b-d||^2, ||a-d||^2+||b-c||^2 )
    P_ef = softmax(-D / tau)    （matching 权重，stop-grad）
    omega_e = importance[tri] * min_opacity[tri]   （置信权重，stop-grad）

    梯度仅通过 D_edge 中的顶点坐标传播。
    """
    T = triangle_indices.long()
    P = T.shape[0]
    dev = vertices.device

    # Shape guard：candidates 与当前三角形数不一致时跳过（prune/densify 后可能过期）
    if edge_candidates.shape[0] != 3 * P:
        return torch.tensor(0.0, device=dev)

    v0 = vertices[T[:, 0]]
    v1 = vertices[T[:, 1]]
    v2 = vertices[T[:, 2]]
    edge_a = torch.cat([v0, v1, v2], dim=0)   # [3P, 3]
    edge_b = torch.cat([v1, v2, v0], dim=0)   # [3P, 3]

    opacity = torch.sigmoid(vertex_weight.squeeze(-1)[T]).min(dim=1).values  # [P]
    imp = importance_score.squeeze()[:P]                                       # [P]
    omega = (imp * opacity).detach()
    tri_owner = torch.arange(P, device=dev).repeat(3)
    omega_per_edge = omega[tri_owner]                                          # [3P]

    cands = edge_candidates.to(dev)   # [3P, k]
    n_edges = 3 * P
    total = torch.tensor(0.0, device=dev)

    for start in range(0, n_edges, chunk_size):
        end = min(start + chunk_size, n_edges)

        ea = edge_a[start:end].unsqueeze(1)   # [C, 1, 3]
        eb = edge_b[start:end].unsqueeze(1)   # [C, 1, 3]
        ne_a = edge_a[cands[start:end]]       # [C, k, 3]
        ne_b = edge_b[cands[start:end]]       # [C, k, 3]
        w    = omega_per_edge[start:end]      # [C]

        d1 = (ea - ne_a).pow(2).sum(-1) + (eb - ne_b).pow(2).sum(-1)   # [C, k]
        d2 = (ea - ne_b).pow(2).sum(-1) + (eb - ne_a).pow(2).sum(-1)   # [C, k]
        D  = torch.minimum(d1, d2)                                       # [C, k]

        P_ef = torch.softmax(-D.detach() / tau, dim=-1)
        total = total + (w * (P_ef * D).sum(-1)).sum()

    return total / n_edges


# ── Opacity solidification loss ──────────────────────────────────────────────

def opacity_solidification_loss(
    vertex_weight: torch.Tensor,
    triangle_indices: torch.Tensor,
    importance_score: torch.Tensor,
    importance_threshold: float,
) -> torch.Tensor:
    """
    实体化 loss：

        L_opaque = mean_i [ m_i * (1 - o_i)^2 ]

    m_i = 1 若 importance_score[i] > importance_threshold，否则 0。
    只对高置信三角形施加"变不透明"的压力。
    """
    T = triangle_indices.long()
    P = T.shape[0]
    o_per_tri = torch.sigmoid(vertex_weight.squeeze(-1)[T]).min(dim=1).values  # [P]
    mask = (importance_score.squeeze()[:P] > importance_threshold).float().detach()
    return (mask * (1.0 - o_per_tri).pow(2)).mean()


# ── Topology loss ────────────────────────────────────────────────────────────

def topology_loss(
    triangle_indices: torch.Tensor,
    vertex_weight: torch.Tensor,
    boundary_mask: torch.Tensor,
    low_valence_mask: torch.Tensor,
) -> torch.Tensor:
    """
    拓扑惩罚 loss：对空洞边缘和悬空顶点三角形施加 opacity 压制。

    boundary_mask / low_valence_mask 由 build_edge_candidates 预计算，
    每次拓扑变化后（prune/densify/Delaunay）重建，每 iter 直接复用。

    梯度：loss = mean(bad * opacity) → opacity 梯度向 0，推动透明化 → pruning 清除。
    """
    T = triangle_indices.long()
    P = T.shape[0]
    dev = T.device

    if P == 0 or boundary_mask.shape[0] != P:
        return torch.tensor(0.0, device=dev)

    bad = (boundary_mask.to(dev) | low_valence_mask.to(dev)).float().detach()
    opacity = torch.sigmoid(vertex_weight.squeeze(-1)[T]).min(dim=1).values  # [P]
    return (bad * opacity).mean()
